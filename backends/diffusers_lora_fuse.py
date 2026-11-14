"""Shared diffusers LoRA fuse for mode LoRAs (STABL-fvwsvjvn).

The Z-Image, SD1.5, and SDXL workers merge mode LoRAs the same way:
diffusers load_lora_weights, a fit check, set_adapters, fuse_lora, then
unload_lora_weights. The fused weights stay in the base weights. Strength is
fixed at mode load. A failure raises, and the handle fails the mode load.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from backends.mode_loras import LoraSpec

logger = logging.getLogger(__name__)

_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora.down.weight")
_TEXT_ENCODER_KEY_PREFIXES = ("text_encoder", "lora_te")


class DiffusersLoraError(ValueError):
    """A mode LoRA cannot be applied to the diffusers pipeline."""


def resolve_lora_file(path: str | Path, error_cls: type[DiffusersLoraError] = DiffusersLoraError) -> Path:
    """Return the LoRA file. A directory must hold exactly one .safetensors file."""

    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.safetensors"))
        if len(files) != 1:
            raise error_cls(
                f"LoRA directory {p} must hold exactly one .safetensors file; found {len(files)}"
            )
        return files[0]
    if not p.is_file():
        raise error_cls(f"LoRA file not found: {p}")
    return p


def count_lora_down_tensors(state_dict: Mapping[str, Any], *, include_text_encoder: bool) -> int:
    """Count the LoRA down (A) tensors in a LoRA file.

    Text encoder keys count only when the fuse covers a text encoder.
    """

    return sum(
        1
        for key in state_dict
        if key.endswith(_DOWN_SUFFIXES)
        and (include_text_encoder or not key.startswith(_TEXT_ENCODER_KEY_PREFIXES))
    )


def count_adapter_modules(module_root: "torch.nn.Module", adapter_name: str) -> int:
    """Count the PEFT LoRA layers under module_root that hold the adapter."""

    from peft.tuners.tuners_utils import BaseTunerLayer

    count = 0
    for module in module_root.modules():
        if isinstance(module, BaseTunerLayer):
            lora_a = getattr(module, "lora_A", None)
            if lora_a is not None and adapter_name in lora_a:
                count += 1
    return count


def fuse_diffusers_loras(
    pipe: Any,
    specs: Sequence["LoraSpec"],
    *,
    components: Sequence[str],
    log_tag: str,
    worker_id: Any,
    error_cls: type[DiffusersLoraError] = DiffusersLoraError,
    target: str = "model",
    log: logging.Logger | None = None,
) -> None:
    """Load, check, fuse, and unload the LoRAs on the named pipe components."""

    log = log or logger
    specs = list(specs)
    if not specs:
        return
    for spec in specs:
        if not math.isfinite(float(spec.strength)):
            raise error_cls(f"LoRA {spec.path} strength must be finite; got {spec.strength}")
    files = [resolve_lora_file(spec.path, error_cls) for spec in specs]
    components = list(components)
    include_te = any(c.startswith("text_encoder") for c in components)

    from safetensors.torch import load_file

    names: list[str] = []
    counts: list[int] = []
    for i, lora_file in enumerate(files):
        name = f"mode_lora_{i}"
        state_dict = load_file(str(lora_file))
        expected = count_lora_down_tensors(state_dict, include_text_encoder=include_te)
        try:
            pipe.load_lora_weights(state_dict, adapter_name=name)
        except ValueError as exc:
            raise error_cls(f"LoRA {lora_file} does not fit the {target}: {exc}") from exc
        names.append(name)
        count = sum(
            count_adapter_modules(getattr(pipe, c), name)
            for c in components
            if getattr(pipe, c, None) is not None
        )
        if count == 0:
            raise error_cls(f"LoRA {lora_file} targets no {target} module")
        if count < expected:
            raise error_cls(
                f"LoRA {lora_file} fits only {count} of {expected} modules of the {target}"
            )
        counts.append(count)

    strengths = [float(spec.strength) for spec in specs]
    pipe.set_adapters(names, adapter_weights=strengths)
    pipe.fuse_lora(components=components, adapter_names=names, lora_scale=1.0)
    pipe.unload_lora_weights()

    for lora_file, strength, count in zip(files, strengths, counts):
        log.info(
            "[%s] worker %s merged LoRA %s strength=%s modules=%d",
            log_tag, worker_id, lora_file, strength, count,
        )
