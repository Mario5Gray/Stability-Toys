"""Mode LoRA specs and the configure_loras worker hook (STABL-ykowysrg).

A mode `loras:` list in conf/modes.yml becomes a list of LoraSpec values. Both
worker handles call apply_mode_loras() right after configure_conditioning, so
the worker merges the LoRAs at mode load. A mode that lists LoRAs on a worker
without the hook fails the mode load. It never loads without them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoraSpec:
    """One mode LoRA: an absolute file or directory path and a fixed strength."""

    path: str
    strength: float


def lora_specs_for_mode(mode: Any) -> list[LoraSpec]:
    """Pair mode.lora_paths with mode.loras strengths.

    ModeConfig always carries lists. Anything else (an absent attribute, a
    Mock test stub) means no LoRAs.
    """
    loras = getattr(mode, "loras", None)
    if not isinstance(loras, (list, tuple)) or not loras:
        return []
    paths = getattr(mode, "lora_paths", None)
    if not isinstance(paths, (list, tuple)) or len(paths) != len(loras):
        raise ValueError(
            f"mode lists {len(loras)} LoRA(s) but {len(paths or [])} resolved LoRA path(s)"
        )
    return [LoraSpec(str(p), float(l.strength)) for p, l in zip(paths, loras)]


def apply_mode_loras(worker: Any, mode: Any) -> None:
    """Pass the mode LoRA specs to worker.configure_loras, or refuse."""
    specs = lora_specs_for_mode(mode)
    if not specs:
        return
    hook = getattr(worker, "configure_loras", None)
    if not callable(hook):
        raise RuntimeError(
            f"mode lists {len(specs)} LoRA(s) but worker {type(worker).__name__} "
            "does not support LoRAs"
        )
    hook(specs)
