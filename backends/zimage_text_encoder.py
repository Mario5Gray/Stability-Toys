"""Content-keyed store for the int8 Z-Image text encoder.

The worker quantizes the Qwen text encoder to int8 one time per weight content.
The store keeps the result on disk. Later loads read the stored entry and skip
the bf16 load and the quantize step.

The store key binds to content only: the config digest, the shard digests in
digest order, the quantization tag, and the quanto version. Shard names, sizes,
times, and the folder path are diagnostic data. They are not key inputs.

A per-key ``fcntl.flock`` lock serializes builds across threads and processes.
The lock protects the construction of one entry. It does not limit total GPU
memory across different models.

This module imports torch, transformers, optimum.quanto, safetensors, and
accelerate only inside the private seams. It imports clean without them.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STORE_ENV = "ZIMAGE_ENCODER_STORE"
QUANT_TAG = "qint8"
ENTRY_FILES = ("model.safetensors", "quanto_qmap.json", "config.json", "store_manifest.json")

_MANIFEST_FILE = ENTRY_FILES[3]
_CHUNK_BYTES = 8 * 1024 * 1024
_DEFAULT_SUBDIR = Path(".cache") / "stability-toys" / "zimage-text-encoder"


class ZImageEncoderStoreError(RuntimeError):
    """The int8 text-encoder store cannot give a valid encoder."""


# --- Key -------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_store_manifest(encoder_dir: Path, quanto_version: str) -> dict:
    """Return the manifest for one encoder folder.

    Raise ``ZImageEncoderStoreError`` when the folder has no weight shard.
    """

    encoder_dir = Path(encoder_dir)
    shards = sorted(encoder_dir.glob("*.safetensors")) if encoder_dir.is_dir() else []
    if not shards:
        raise ZImageEncoderStoreError(
            f"text encoder folder has no weight shards (*.safetensors): {encoder_dir}"
        )
    config_path = encoder_dir / "config.json"
    if not config_path.is_file():
        raise ZImageEncoderStoreError(f"text encoder folder has no config.json: {encoder_dir}")

    shard_rows = [
        {"name": shard.name, "bytes": shard.stat().st_size, "sha256": _sha256_file(shard)}
        for shard in shards
    ]
    return {
        "key_inputs": {
            "config_sha256": _sha256_file(config_path),
            "shard_sha256": sorted(row["sha256"] for row in shard_rows),
            "quantization": QUANT_TAG,
            "quanto_version": quanto_version,
        },
        "diagnostic": {"shards": shard_rows, "encoder_dir": str(encoder_dir)},
    }


def store_key(manifest: dict) -> str:
    """Return the SHA-256 hex key of the manifest key inputs only."""

    canonical = json.dumps(manifest["key_inputs"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def encoder_store_root(model_path: str) -> Path:
    """Return the store root.

    Use ``STORE_ENV`` when it is set. Else use a folder beside the model folder.
    """

    override = os.environ.get(STORE_ENV)
    if override:
        return Path(override)
    return Path(model_path).parent / _DEFAULT_SUBDIR


# --- Private seams ---------------------------------------------------------


def _quanto_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("optimum-quanto")
    except PackageNotFoundError as exc:
        raise ZImageEncoderStoreError("optimum-quanto is not installed") from exc


def _load_bf16_encoder(encoder_dir: Path, dtype: Any) -> Any:
    from transformers import AutoModel

    return AutoModel.from_pretrained(encoder_dir, torch_dtype=dtype, local_files_only=True)


def _quantize_and_freeze(encoder: Any, device: str) -> Any:
    from optimum.quanto import freeze, qint8, quantize

    quantize(encoder, weights=qint8)
    encoder.to(device)
    freeze(encoder)
    encoder.eval()
    return encoder


def _save_entry(encoder: Any, entry_dir: Path) -> None:
    from optimum.quanto import quantization_map
    from safetensors.torch import save_file

    entry_dir = Path(entry_dir)
    state = {k: v.cpu().contiguous() for k, v in encoder.state_dict().items()}
    save_file(state, entry_dir / ENTRY_FILES[0])
    (entry_dir / ENTRY_FILES[1]).write_text(json.dumps(quantization_map(encoder)))
    encoder.config.save_pretrained(entry_dir)


def _load_entry(entry_dir: Path, device: str) -> Any:
    from accelerate import init_empty_weights
    from optimum.quanto import requantize
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    entry_dir = Path(entry_dir)
    config = AutoConfig.from_pretrained(entry_dir, local_files_only=True)
    with init_empty_weights():
        encoder = AutoModel.from_config(config)
    state = load_file(entry_dir / ENTRY_FILES[0])
    qmap = json.loads((entry_dir / ENTRY_FILES[1]).read_text())
    requantize(encoder, state, qmap, device=device)
    encoder.eval()
    return encoder


# --- Load or build ---------------------------------------------------------


def _remove_stale_temp_dirs(root: Path, key: str) -> None:
    # The key lock is held, so no live writer owns a temp dir for this key.
    for stale in root.glob(f".tmp-{key}-*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)


def _load_hit(entry: Path, manifest: dict, key: str, device: str) -> Any:
    try:
        stored = json.loads((entry / _MANIFEST_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise ZImageEncoderStoreError(
            f"store entry {entry} has an unreadable manifest: {exc}"
        ) from exc
    if not isinstance(stored, dict) or not isinstance(stored.get("key_inputs"), dict):
        raise ZImageEncoderStoreError(
            f"store entry {entry} has a malformed manifest: expected an object with key_inputs"
        )
    if stored["key_inputs"] != manifest["key_inputs"]:
        raise ZImageEncoderStoreError(
            f"store entry {entry} manifest key_inputs differ from the model folder; "
            f"remove the entry to rebuild it"
        )
    try:
        encoder = _load_entry(entry, device)
    except Exception as exc:
        raise ZImageEncoderStoreError(f"store entry {entry} failed to load: {exc}") from exc
    logger.info(f"[ZImageEncoderStore] store hit key={key}")
    return encoder


def _build_entry(
    root: Path, entry: Path, manifest: dict, key: str, encoder_dir: Path, device: str, dtype: Any
) -> Any:
    _remove_stale_temp_dirs(root, key)
    encoder = _quantize_and_freeze(_load_bf16_encoder(encoder_dir, dtype), device)
    tmp_dir = Path(tempfile.mkdtemp(dir=root, prefix=f".tmp-{key}-"))
    try:
        _save_entry(encoder, tmp_dir)
        # The manifest is written last. An entry without it is never published.
        (tmp_dir / _MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(tmp_dir, entry)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    logger.info(f"[ZImageEncoderStore] store miss key={key}; stored entry {entry}")
    return encoder


def load_int8_text_encoder(
    model_path: str, *, device: str, dtype: Any, store_root: Path | None = None
) -> Any:
    """Return the int8 text encoder for ``model_path``.

    Load the stored entry on a hit. Build and store it on a miss.
    """

    encoder_dir = Path(model_path) / "text_encoder"
    started = time.monotonic()
    manifest = compute_store_manifest(encoder_dir, _quanto_version())
    key = store_key(manifest)
    logger.info(
        f"[ZImageEncoderStore] key={key} computed in {time.monotonic() - started:.1f} s"
    )

    root = Path(store_root) if store_root is not None else encoder_store_root(model_path)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ZImageEncoderStoreError(f"cannot create encoder store root {root}: {exc}") from exc

    lock_path = root / f"{key}.lock"
    try:
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        raise ZImageEncoderStoreError(f"cannot open store lock {lock_path}: {exc}") from exc
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # This recheck under the lock is the only hit test.
            entry = root / key
            if entry.is_dir():
                return _load_hit(entry, manifest, key, device)
            if os.path.lexists(entry):
                # A file or a broken link at the entry path is not a miss.
                raise ZImageEncoderStoreError(
                    f"store entry {entry} exists but is not a directory; remove it to rebuild"
                )
            return _build_entry(root, entry, manifest, key, encoder_dir, device, dtype)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        # The lock file is never unlinked. Its path stays stable for each call.
        os.close(lock_fd)
