"""Tests for the content-keyed int8 Z-Image text-encoder store.

No test loads real weights. Each test patches the private seams with fakes.
"""

import json
import multiprocessing
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import backends.zimage_text_encoder as zte
from backends.zimage_text_encoder import (
    ENTRY_FILES,
    QUANT_TAG,
    STORE_ENV,
    ZImageEncoderStoreError,
    compute_store_manifest,
    encoder_store_root,
    load_int8_text_encoder,
    store_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
QV = "0.2.7"
SHARD_1 = "model-00001-of-00002.safetensors"
SHARD_2 = "model-00002-of-00002.safetensors"


def _model(parent: Path, name: str = "Z-Image-Turbo-BF16", shards=None) -> Path:
    if shards is None:
        shards = {SHARD_1: b"A" * 64, SHARD_2: b"B" * 64}
    root = parent / name
    te = root / "text_encoder"
    te.mkdir(parents=True)
    (te / "config.json").write_text('{"hidden_size": 2560}')
    for shard_name, data in shards.items():
        (te / shard_name).write_bytes(data)
    return root


def _key(model_root: Path, quanto_version: str = QV) -> str:
    return store_key(compute_store_manifest(model_root / "text_encoder", quanto_version))


def _install_fake_seams(state, quant_hook=None):
    """Replace the five seams in this process. Return the state object."""

    def load_bf16(encoder_dir, dtype):
        state.bf16 += 1
        return {"encoder_dir": str(encoder_dir), "dtype": dtype}

    def quantize_and_freeze(encoder, device):
        state.quant += 1
        if quant_hook is not None:
            quant_hook()
        return {"quantized": encoder, "device": device}

    def save_entry(encoder, entry_dir):
        state.save += 1
        if state.save_error is not None:
            (Path(entry_dir) / ENTRY_FILES[0]).write_text("partial")
            raise state.save_error
        for name in ENTRY_FILES[:3]:
            (Path(entry_dir) / name).write_text("x")

    def load_entry(entry_dir, device):
        state.load += 1
        if state.load_error is not None:
            raise state.load_error
        return {"loaded": str(entry_dir), "device": device}

    zte._load_bf16_encoder = load_bf16
    zte._quantize_and_freeze = quantize_and_freeze
    zte._save_entry = save_entry
    zte._load_entry = load_entry
    zte._quanto_version = lambda: QV
    return state


def _new_state():
    return SimpleNamespace(bf16=0, quant=0, save=0, load=0, save_error=None, load_error=None)


@pytest.fixture
def seams(monkeypatch):
    for name in ("_load_bf16_encoder", "_quantize_and_freeze", "_save_entry", "_load_entry", "_quanto_version"):
        monkeypatch.setattr(zte, name, getattr(zte, name))
    return _install_fake_seams(_new_state())


def _tmp_dirs(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if p.name.startswith(".tmp-")]


# --- Key -------------------------------------------------------------------


def test_key_is_stable_for_equal_content(tmp_path):
    a = _model(tmp_path / "a")
    b = _model(tmp_path / "b")
    assert _key(a) == _key(b)


def test_key_changes_when_content_changes_and_size_name_time_stay_equal(tmp_path):
    root = _model(tmp_path)
    shard = root / "text_encoder" / SHARD_1
    before = _key(root)
    stat = shard.stat()
    shard.write_bytes(b"C" * 64)
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert shard.stat().st_size == stat.st_size
    assert _key(root) != before


def test_key_ignores_shard_time(tmp_path):
    root = _model(tmp_path)
    before = _key(root)
    os.utime(root / "text_encoder" / SHARD_1, (1_000_000, 1_000_000))
    assert _key(root) == before


def test_key_changes_with_quanto_version(tmp_path):
    root = _model(tmp_path)
    assert _key(root, "0.2.7") != _key(root, "0.2.8")


def test_key_is_equal_after_shard_rename(tmp_path):
    root = _model(tmp_path)
    before = _key(root)
    te = root / "text_encoder"
    (te / SHARD_1).rename(te / "a.safetensors")
    assert _key(root) == before


def test_key_ignores_folder_path(tmp_path):
    a = _model(tmp_path / "one" / "deep")
    b = _model(tmp_path / "two")
    assert _key(a) == _key(b)


def test_shard_names_are_diagnostic_only(tmp_path):
    root = _model(tmp_path)
    manifest = compute_store_manifest(root / "text_encoder", QV)
    names = {s["name"] for s in manifest["diagnostic"]["shards"]}
    assert names == {SHARD_1, SHARD_2}
    key_text = json.dumps(manifest["key_inputs"])
    assert "name" not in key_text
    assert SHARD_1 not in key_text
    assert str(root) not in key_text
    assert manifest["key_inputs"]["quantization"] == QUANT_TAG
    assert manifest["key_inputs"]["quanto_version"] == QV


def test_shard_digests_are_sorted_by_digest_value(tmp_path):
    # "Z" content first by name, "A" content second by name.
    root = _model(tmp_path, shards={"a.safetensors": b"Z" * 32, "b.safetensors": b"A" * 32})
    digests = compute_store_manifest(root / "text_encoder", QV)["key_inputs"]["shard_sha256"]
    assert len(digests) == 2
    assert digests == sorted(digests)


def test_no_shards_raises(tmp_path):
    root = _model(tmp_path, shards={})
    with pytest.raises(ZImageEncoderStoreError, match="no weight shards"):
        compute_store_manifest(root / "text_encoder", QV)


def test_absent_text_encoder_folder_raises(tmp_path):
    with pytest.raises(ZImageEncoderStoreError, match="no weight shards"):
        compute_store_manifest(tmp_path / "missing" / "text_encoder", QV)


# --- Store root ------------------------------------------------------------


def test_store_root_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "store"))
    assert encoder_store_root("/models/repos/X") == tmp_path / "store"


def test_store_root_default_is_beside_model_folder(monkeypatch):
    monkeypatch.delenv(STORE_ENV, raising=False)
    assert encoder_store_root("/models/repos/X") == Path(
        "/models/repos/.cache/stability-toys/zimage-text-encoder"
    )


# --- Load or build ---------------------------------------------------------


def test_miss_builds_and_stores(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    result = load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert seams.bf16 == 1
    assert seams.quant == 1
    assert result["device"] == "cuda"
    entry = store / _key(model)
    assert sorted(p.name for p in entry.iterdir()) == sorted(ENTRY_FILES)
    assert _tmp_dirs(store) == []


def test_store_manifest_records_key_inputs(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    stored = json.loads((store / _key(model) / "store_manifest.json").read_text())
    assert stored["key_inputs"] == compute_store_manifest(model / "text_encoder", QV)["key_inputs"]


def test_hit_does_not_load_bf16_encoder(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    seams.bf16 = 0
    result = load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert seams.bf16 == 0
    assert seams.load == 1
    assert result["loaded"] == str(store / _key(model))


def test_interrupted_write_leaves_no_entry(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    seams.save_error = OSError("disk full")
    with pytest.raises((ZImageEncoderStoreError, OSError)):
        load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert not (store / _key(model)).exists()
    assert _tmp_dirs(store) == []


def test_simultaneous_misses_quantize_once(tmp_path, monkeypatch, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    real_flock = zte.fcntl.flock
    second_waiter = threading.Event()
    exclusive_calls = []
    calls_lock = threading.Lock()

    def flock(fd, op):
        if op & zte.fcntl.LOCK_EX:
            with calls_lock:
                exclusive_calls.append(fd)
                if len(exclusive_calls) == 2:
                    second_waiter.set()
        return real_flock(fd, op)

    monkeypatch.setattr(zte.fcntl, "flock", flock)
    waited = []
    _install_fake_seams(seams, quant_hook=lambda: waited.append(second_waiter.wait(5)))

    results, errors = [], []

    def worker():
        try:
            results.append(
                load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
            )
        except Exception as exc:  # pragma: no cover - reported by the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert errors == []
    assert waited == [True], "the second loader did not wait on the lock during the build"
    assert seams.bf16 == 1
    assert len(results) == 2
    assert [p.name for p in store.iterdir() if p.is_dir() and not p.name.startswith(".")] == [_key(model)]
    assert _tmp_dirs(store) == []


def _child_build(model_path, store_root, log_path):
    import time

    import backends.zimage_text_encoder as child_zte

    state = _new_state()

    def record_build():
        with open(log_path, "a") as fh:
            fh.write(f"{os.getpid()}\n")
        time.sleep(0.5)

    # The fakes replace the module seams in this child process only.
    _install_fake_seams(state, quant_hook=record_build)
    child_zte.load_int8_text_encoder(model_path, device="cuda", dtype="bf16", store_root=Path(store_root))


def test_simultaneous_misses_in_two_processes_quantize_once(tmp_path):
    model = _model(tmp_path)
    store = tmp_path / "store"
    log_path = tmp_path / "builds.log"
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=_child_build, args=(str(model), str(store), str(log_path)))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert [p.exitcode for p in procs] == [0, 0]
    assert len(log_path.read_text().splitlines()) == 1


def test_temp_dirs_are_unique_per_writer(tmp_path, monkeypatch, seams):
    real_mkdtemp = zte.tempfile.mkdtemp
    recorded = []

    def mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        recorded.append(path)
        return path

    monkeypatch.setattr(zte.tempfile, "mkdtemp", mkdtemp)
    store = tmp_path / "store"
    a = _model(tmp_path / "a")
    b = _model(tmp_path / "b", shards={SHARD_1: b"Q" * 16})
    load_int8_text_encoder(str(a), device="cuda", dtype="bf16", store_root=store)
    load_int8_text_encoder(str(b), device="cuda", dtype="bf16", store_root=store)
    assert len(recorded) == 2
    assert recorded[0] != recorded[1]
    assert not any(p.endswith(str(os.getpid())) for p in recorded)


def test_stale_temp_dir_from_dead_writer_is_removed(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    stale = store / f".tmp-{_key(model)}-dead"
    stale.mkdir(parents=True)
    (stale / "model.safetensors").write_text("partial")
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert not stale.exists()
    entry = store / _key(model)
    assert sorted(p.name for p in entry.iterdir()) == sorted(ENTRY_FILES)


def test_stale_cleanup_leaves_other_keys_alone(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    other = store / ".tmp-otherkey-live"
    other.mkdir(parents=True)
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert other.exists()


def test_lock_file_is_kept_after_build_and_cleanup(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    key = _key(model)
    (store / f".tmp-{key}-dead").mkdir(parents=True)
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert (store / f"{key}.lock").is_file()


def test_each_call_opens_its_own_lock_descriptor(tmp_path, monkeypatch, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    real_open, real_close = zte.os.open, zte.os.close
    opened, closed = [], []

    def os_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if str(path).endswith(".lock"):
            opened.append((str(path), fd))
        return fd

    def os_close(fd):
        if any(fd == lock_fd for _, lock_fd in opened):
            closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(zte.os, "open", os_open)
    monkeypatch.setattr(zte.os, "close", os_close)
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert len(opened) == 2
    assert len(closed) == 2
    assert opened[0][0] == opened[1][0] == str(store / f"{_key(model)}.lock")


def test_lock_is_released_after_failure(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    seams.save_error = OSError("disk full")
    with pytest.raises((ZImageEncoderStoreError, OSError)):
        load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    seams.save_error = None

    done = []
    t = threading.Thread(
        target=lambda: done.append(
            load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
        )
    )
    t.start()
    t.join(5)
    assert not t.is_alive(), "second call blocked on a lock that was not released"
    assert len(done) == 1


def test_manifest_mismatch_fails_load(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    manifest_path = store / _key(model) / "store_manifest.json"
    data = json.loads(manifest_path.read_text())
    data["key_inputs"]["quanto_version"] = "9.9.9"
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ZImageEncoderStoreError, match="manifest"):
        load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)


def test_corrupt_entry_is_an_error_not_a_rebuild(tmp_path, seams):
    model = _model(tmp_path)
    store = tmp_path / "store"
    load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    seams.bf16 = 0
    seams.load_error = ValueError("truncated safetensors header")
    with pytest.raises(ZImageEncoderStoreError, match="truncated"):
        load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert seams.bf16 == 0


def test_unwritable_store_root_raises(tmp_path, seams):
    model = _model(tmp_path)
    blocker = tmp_path / "regular-file"
    blocker.write_text("not a folder")
    store = blocker / "store"
    with pytest.raises(ZImageEncoderStoreError, match=str(store).replace("\\", "\\\\")):
        load_int8_text_encoder(str(model), device="cuda", dtype="bf16", store_root=store)
    assert seams.bf16 == 0


def test_module_imports_without_torch():
    code = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'optimum', 'optimum.quanto', 'safetensors', 'accelerate'):\n"
        "    sys.modules[name] = None\n"
        "import backends.zimage_text_encoder\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
