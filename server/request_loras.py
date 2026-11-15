"""Validate the LoRA list of a generate request (STABL-fvwsvjvn).

The routes call resolve_request_loras before admission. Each entry resolves
to one weight file inside lora_root. Workers receive that file, never a
directory. A failure raises RequestLoraError, and no switch is queued.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from backends.hunyuandit_lora import ADAPTER_CONFIG
from backends.mode_loras import LoraSpec, RequestLoraError

ADAPTER_MODEL = "adapter_model.safetensors"


def ensure_request_loras_supported(req: Any) -> None:
    """Refuse request LoRAs on a runtime without the mode system.

    Only the mode system can load LoRAs. A dropped list would run the job on a
    LoRA set other than the one the request named.
    """
    if getattr(req, "loras", None):
        raise RequestLoraError(
            "request LoRAs need the mode system (CUDA or MPS backend with modes); "
            "this runtime cannot load them"
        )


def admit_generation_request(runtime: Any, req: Any, lora_root: Callable[[], str]) -> Any:
    """Check the request LoRAs, then admit the request. Return the authority.

    HTTP and WS share this. Each transport formats the errors: RequestLoraError
    (a rule failure), KeyError (an unknown mode), and others. lora_root is read
    only when the request has LoRAs. A request without LoRAs keeps the one-argument
    admit_generation call.
    """
    mode = getattr(req, "mode", None)
    if not getattr(req, "loras", None):
        return runtime.admit_generation(mode)
    return runtime.admit_generation(mode, resolve_request_loras(req.loras, lora_root()))


def select_weight_file(path: Path) -> Path:
    """Return the weight file for a request path.

    A file returns itself. A directory returns adapter_model.safetensors when
    present, else its only *.safetensors file.
    """
    if path.is_file():
        return path
    if path.is_dir():
        preferred = path / ADAPTER_MODEL
        if preferred.is_file():
            return preferred
        files = sorted(p for p in path.glob("*.safetensors") if p.is_file())
        if len(files) != 1:
            raise RequestLoraError(
                f"LoRA directory {path} must hold {ADAPTER_MODEL} or exactly one "
                f".safetensors file; found {len(files)}"
            )
        return files[0]
    raise RequestLoraError(f"LoRA not found: {path}")


def _entry_fields(entry: Any) -> tuple[str, Any]:
    if isinstance(entry, dict):
        return entry["path"], entry.get("strength", 1.0)
    return entry.path, getattr(entry, "strength", 1.0)


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_request_loras(entries: Sequence[Any] | None, lora_root: str) -> tuple[LoraSpec, ...]:
    """Resolve and check request LoRA entries. Return specs in request order."""
    if not entries:
        return ()
    root = Path(lora_root).resolve()
    specs: list[LoraSpec] = []
    seen: dict[Path, str] = {}
    for entry in entries:
        raw_path, raw_strength = _entry_fields(entry)
        strength = float(raw_strength)
        if not math.isfinite(strength):
            raise RequestLoraError(f"LoRA {raw_path} strength must be finite; got {raw_strength}")
        weight = select_weight_file(root / raw_path).resolve()
        if not _inside(weight, root):
            raise RequestLoraError(f"LoRA {raw_path} resolves outside lora_root {root}")
        sidecar = weight.with_name(ADAPTER_CONFIG)
        if sidecar.exists() and not _inside(sidecar.resolve(), root):
            raise RequestLoraError(
                f"LoRA {raw_path}: {ADAPTER_CONFIG} resolves outside lora_root {root}"
            )
        if weight in seen:
            raise RequestLoraError(
                f"LoRA {raw_path} and {seen[weight]} resolve to the same file {weight}"
            )
        seen[weight] = str(raw_path)
        specs.append(LoraSpec(str(weight), strength))
    return tuple(specs)
