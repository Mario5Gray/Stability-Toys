"""CUDA Diffusers worker for the Z-Image family (Turbo and base).

Slice 1 scope: txt2img and plain img2img. ControlNet is slice 2.

The transformer and VAE run in bfloat16. The family sets the dtype, so
``CUDA_DTYPE`` does not change it. The Qwen text encoder is int8 by default.
It comes from the content-keyed store in ``backends.zimage_text_encoder``.

Admitted runtime settings: offload ``none`` and runtime quantize ``none``.
The worker refuses other values before a weight file loads.

The worker accepts a Diffusers folder only. It never downloads a model file.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from backends.cuda_worker import CudaWorkerBase, _import_attr
from backends.family_profiles import ZIMAGE_PROFILE, FamilyProfile
from backends.zimage_text_encoder import load_int8_text_encoder

logger = logging.getLogger(__name__)

_ADMITTED_LOADER_FORMAT = ("diffusers_dir", "unknown")
_ADMITTED_OFFLOAD = ("none",)
_ADMITTED_QUANTIZE = ("none",)
_ADMITTED_TEXT_ENCODER_QUANTIZE = ("int8", "none")


class ZImageDependencyError(RuntimeError):
    """The installed Diffusers stack lacks a Z-Image pipeline class."""


class ZImageUnsupportedRuntimeError(RuntimeError):
    """A runtime setting or model format is not admitted for Z-Image."""


# The resolvers keep the Z-Image pipeline classes out of module import.
# They touch diffusers only when the worker is constructed.


def _zimage_pipeline_cls() -> Any:
    return _import_attr("diffusers", "ZImagePipeline")


def _zimage_img2img_pipeline_cls() -> Any:
    return _import_attr("diffusers", "ZImageImg2ImgPipeline")


def _zimage_dependency_versions() -> dict[str, str]:
    from importlib import metadata

    def _v(package: str) -> str:
        try:
            return metadata.version(package)
        except Exception:
            return "not installed"

    return {"diffusers": _v("diffusers"), "transformers": _v("transformers")}


def _zimage_dependency_preflight() -> None:
    """Resolve both Z-Image pipeline classes before any weight load."""

    try:
        _zimage_pipeline_cls()
        _zimage_img2img_pipeline_cls()
    except Exception as exc:
        versions = _zimage_dependency_versions()
        raise ZImageDependencyError(
            "Z-Image runtime dependencies are unavailable "
            f"(diffusers={versions['diffusers']}, "
            f"transformers={versions['transformers']}): {exc}"
        ) from exc


def _refuse(setting: str, value: Any, admitted: tuple[str, ...]) -> ZImageUnsupportedRuntimeError:
    return ZImageUnsupportedRuntimeError(
        f"Z-Image does not support {setting}={value}. Admitted: {', '.join(admitted)}."
    )


class DiffusersZImageCudaWorker(CudaWorkerBase):
    """CUDA worker for Z-Image-Turbo and Z-Image base Diffusers folders."""

    family_profile: FamilyProfile = ZIMAGE_PROFILE

    # The Z-Image transformer keeps its native attention processor. The
    # sliced and xformers processors are not validated for this family.
    supports_attention_processor_swap: bool = False

    log_tag: str = "zimage-cuda"

    def __init__(
        self,
        worker_id: int,
        model_path: str,
        model_info: Any | None = None,
        family_profile: FamilyProfile | None = None,
    ) -> None:
        super().__init__(worker_id, model_info=model_info, family_profile=family_profile)

        _zimage_dependency_preflight()
        encoder_mode = self._admit(model_path)

        # The family sets the dtype. CUDA_DTYPE does not change it.
        requested_dtype = os.environ.get("CUDA_DTYPE")
        if requested_dtype and requested_dtype.strip().lower() not in ("bf16", "bfloat16"):
            logger.warning(
                "[%s] worker %s: CUDA_DTYPE=%s is ignored. Z-Image runs in bf16.",
                self.log_tag, self.worker_id, requested_dtype,
            )
        self.dtype = torch.bfloat16
        self.dtype_str = "bf16"

        encoder_kw: dict[str, Any] = {}
        if encoder_mode == "int8":
            encoder_kw["text_encoder"] = load_int8_text_encoder(
                model_path, device=self.device, dtype=self.dtype
            )

        pipe = _zimage_pipeline_cls().from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            local_files_only=True,
            **encoder_kw,
        )
        pipe = self._setup_pipe_memory_opts(pipe)

        self.pipe = pipe
        self._capture_baseline_scheduler(self.pipe)
        # Z-Image does not load SD or SDXL style LoRAs.
        self._style_api = "none"

        logger.info(
            "[%s] worker %s loaded: %s on %s dtype=%s text_encoder=%s",
            self.log_tag, self.worker_id, os.path.basename(model_path.rstrip("/")),
            self.device, self.dtype_str, encoder_mode,
        )

    def _admit(self, model_path: str) -> str:
        """Refuse unadmitted settings before any weight load. Return the encoder mode."""

        # Format policy: the hint may be absent, but it must not name another format.
        format_hint = self._loader_format or "unknown"
        if format_hint not in _ADMITTED_LOADER_FORMAT:
            raise _refuse("loader_format", format_hint, _ADMITTED_LOADER_FORMAT)
        # The path itself must be a Diffusers folder. A hint is never enough.
        if not os.path.isdir(model_path):
            raise ZImageUnsupportedRuntimeError(
                f"Z-Image requires a Diffusers folder; {model_path!r} is not a directory"
            )
        if not os.path.isfile(os.path.join(model_path, "model_index.json")):
            raise ZImageUnsupportedRuntimeError(
                f"Z-Image requires a Diffusers folder; {model_path!r} has no model_index.json file"
            )
        if self._offload not in _ADMITTED_OFFLOAD:
            raise _refuse("offload", self._offload, _ADMITTED_OFFLOAD)
        if self._quantize not in _ADMITTED_QUANTIZE:
            raise _refuse("quantize", self._quantize, _ADMITTED_QUANTIZE)

        text_encoder_quantize = getattr(self.model_info, "text_encoder_quantize", None)
        if text_encoder_quantize is None:
            return "int8"
        if text_encoder_quantize not in _ADMITTED_TEXT_ENCODER_QUANTIZE:
            raise _refuse(
                "text_encoder_quantize", text_encoder_quantize, _ADMITTED_TEXT_ENCODER_QUANTIZE
            )
        return text_encoder_quantize

    def _quantization_targets(self, pipe: Any) -> tuple[Any, ...]:
        # Admission refuses runtime quantize, so this path is a defect guard.
        raise ZImageUnsupportedRuntimeError(
            "Z-Image does not support runtime transformer quantization"
        )
