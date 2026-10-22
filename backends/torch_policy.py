"""Torch execution policy contract for Diffusers workers (STABL-xkiwjvrm).

Shared worker code owns family behavior: schedulers, styles, conditioning,
ControlNet, VAE tiling, and attention slicing. A policy owns the device
actions: settings, runtime quantization, attention backend, placement, and
allocator cache release.

This module must stay import-clean. It never imports Torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TorchExecutionSettings:
    """Resolved device settings for one worker."""

    device: str
    dtype: Any
    dtype_str: str
    enable_xformers: bool
    attention_slicing: bool
    quantize: str
    offload: str


@runtime_checkable
class TorchExecutionPolicy(Protocol):
    backend_id: str

    def resolve_settings(self, model_info: Any) -> TorchExecutionSettings:
        """Resolve settings from the environment and ``model_info.runtime_*``."""
        ...

    def quantize_modules(self, modules: tuple[Any, ...]) -> None:
        """Quantize the given modules to fp8 at runtime."""
        ...

    def enable_memory_efficient_attention(self, pipe: Any) -> None:
        """Enable the backend attention kernel. Raise when it is not available."""
        ...

    def place_pipeline(self, pipe: Any, *, device: str, offload: str) -> Any:
        """Place the pipeline on the device or register offload hooks."""
        ...

    def release_cache(self) -> None:
        """Release cached allocator blocks after a job."""
        ...
