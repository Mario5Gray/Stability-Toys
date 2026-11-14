"""Mode LoRAs for the SD1.5 and SDXL UNet workers (STABL-fvwsvjvn).

The mixin gives DiffusersCudaWorker, DiffusersSDXLCudaWorker, and
DiffusersMpsWorker the configure_loras hook. It fuses the LoRAs into the UNet
and each text encoder that the pipeline has, through the shared diffusers fuse.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from backends.diffusers_lora_fuse import DiffusersLoraError, fuse_diffusers_loras

if TYPE_CHECKING:
    from backends.mode_loras import LoraSpec

logger = logging.getLogger(__name__)

_TEXT_ENCODERS = ("text_encoder", "text_encoder_2")


class UNetLoraError(DiffusersLoraError):
    """A mode LoRA cannot be applied to the SD or SDXL pipeline."""


class UNetModeLoraMixin:
    """configure_loras for the SD1.5 and SDXL workers."""

    def configure_loras(self, specs: Sequence[LoraSpec]) -> None:
        """Refuse unsupported runtime states, then fuse the LoRAs. The fused weights stay."""

        specs = list(specs)
        if not specs:
            return
        if getattr(self, "_quantize", None) == "fp8" or getattr(self, "_checkpoint_precision", None) == "fp8":
            raise UNetLoraError("cannot merge LoRAs into a quantized (fp8) UNet")
        if getattr(self, "_offload", None) == "sequential":
            raise UNetLoraError("cannot merge LoRAs with sequential CPU offload")
        style_loaded = getattr(self, "_style_loaded", None) or {}
        if any(style_loaded.values()):
            raise UNetLoraError(
                "cannot merge mode LoRAs while style LoRA adapters are loaded"
            )

        pipe = self.pipe  # type: ignore[attr-defined]
        components = ["unet"] + [c for c in _TEXT_ENCODERS if getattr(pipe, c, None) is not None]
        fuse_diffusers_loras(
            pipe,
            specs,
            components=components,
            log_tag=getattr(self, "log_tag", "cuda"),
            worker_id=getattr(self, "worker_id", None),
            error_cls=UNetLoraError,
            target="UNet pipeline",
            log=logger,
        )
