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

import io
import json
import logging
import os
from typing import Any

import torch
from PIL import Image, PngImagePlugin

from backends.conditioning.artifacts import DelegatedConditioning
from backends.cuda_worker import CudaWorkerBase, _import_attr
from backends.family_profiles import ZIMAGE_PROFILE, FamilyProfile
from backends.step_progress import inject_step_progress
from backends.zimage_text_encoder import load_int8_text_encoder

logger = logging.getLogger(__name__)

_ADMITTED_LOADER_FORMAT = ("diffusers_dir", "unknown")
_ADMITTED_OFFLOAD = ("none",)
_ADMITTED_QUANTIZE = ("none",)
_ADMITTED_TEXT_ENCODER_QUANTIZE = ("int8", "none")

_DEFAULT_IMG2IMG_STRENGTH = 0.75
_GIB = 2**30


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

    # --- Peak memory --------------------------------------------------------

    def _reset_peak_memory(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def _read_peak_memory(self) -> tuple[int, int]:
        """Return peak (allocated, reserved) bytes for the worker device."""

        torch.cuda.synchronize(self.device)
        return (
            int(torch.cuda.max_memory_allocated(self.device)),
            int(torch.cuda.max_memory_reserved(self.device)),
        )

    def _try_reset_peak_memory(self) -> None:
        try:
            self._reset_peak_memory()
        except Exception as exc:
            logger.warning("[%s] peak memory reset failed: %r", self.log_tag, exc)

    def _try_read_peak_memory(self) -> tuple[int, int] | None:
        try:
            return self._read_peak_memory()
        except Exception as exc:
            logger.warning("[%s] peak memory read failed: %r", self.log_tag, exc)
            return None

    def _log_job(
        self, job_id: Any, op: str, width: int, height: int, steps: int,
        peak: tuple[int, int] | None,
    ) -> None:
        if peak is None:
            allocated = reserved = allocated_gib = reserved_gib = "unavailable"
        else:
            allocated, reserved = peak
            allocated_gib = f"{allocated / _GIB:.2f}"
            reserved_gib = f"{reserved / _GIB:.2f}"
        logger.info(
            "[%s] job=%s op=%s size=%sx%s steps=%s peak_allocated_bytes=%s "
            "peak_reserved_bytes=%s peak_allocated_gib=%s peak_reserved_gib=%s",
            self.log_tag, job_id, op, width, height, steps,
            allocated, reserved, allocated_gib, reserved_gib,
        )

    # --- Request rules ------------------------------------------------------

    def _mode_permits_negative_prompt(self) -> bool:
        info = self.model_info
        return bool(
            getattr(info, "allow_custom_negative_prompt", False)
            or getattr(info, "negative_prompt_templates", None)
        )

    @staticmethod
    def _img2img_strength(req: Any, steps: int) -> tuple[float, int]:
        """Return (strength, effective steps). Refuse a value the pipe cannot run."""

        raw = getattr(req, "denoise_strength", None)
        strength = _DEFAULT_IMG2IMG_STRENGTH if raw is None else float(raw)
        if not (0 < strength <= 1):
            raise ValueError(
                f"Z-Image img2img needs 0 < denoise_strength <= 1; got {strength}"
            )
        effective = min(int(steps * strength), steps)
        if effective < 1:
            raise ValueError(
                f"Z-Image img2img has no effective step: steps={steps}, "
                f"denoise_strength={strength}, effective_steps={effective}"
            )
        return strength, effective

    def _ensure_img2img_pipe(self) -> Any:
        # Build one time. from_pipe shares the base components and the scheduler.
        # torch_dtype=None keeps the bf16 weights as they are.
        if self._img2img_pipe is None:
            self._img2img_pipe = _zimage_img2img_pipeline_cls().from_pipe(
                self.pipe, torch_dtype=None
            )
        return self._img2img_pipe

    # --- Execution ----------------------------------------------------------

    def run_job(self, job: Any, progress: Any = None, should_cancel: Any = None) -> tuple[bytes, int]:
        req = job.req
        init_image = getattr(job, "init_image", None)

        if getattr(job, "controlnet_bindings", None) or getattr(req, "controlnets", None):
            raise RuntimeError("Z-Image does not support ControlNet")

        try:
            w_str, h_str = str(req.size).lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            raise RuntimeError(f"Invalid size '{req.size}', expected 'WIDTHxHEIGHT'")

        steps = int(req.num_inference_steps)
        strength: float | None = None
        effective_steps = steps
        if init_image is not None:
            strength, effective_steps = self._img2img_strength(req, steps)

        seed = int(req.seed) if req.seed is not None else int(torch.randint(0, 100_000_000, (1,)).item())
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        scheduler_id = self._apply_request_scheduler(req)

        job_id = getattr(job, "job_id", "unknown")
        out = None
        artifact = None
        pipe_kwargs: dict[str, Any] | None = None
        init_pil = None
        try:
            artifact = self._conditioning_artifact_for_request(req)
            if not isinstance(artifact, DelegatedConditioning):
                raise ValueError("Z-Image supports native conditioning only")

            pipe_kwargs = {"prompt": artifact.prompt}
            negative = artifact.negative_prompt
            if negative and negative.strip() and self._mode_permits_negative_prompt():
                pipe_kwargs["negative_prompt"] = negative
            pipe_kwargs.update(
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=float(req.guidance_scale),
                generator=gen,
            )

            if init_image is not None:
                op = "img2img"
                try:
                    init_pil = Image.open(io.BytesIO(init_image)).convert("RGB")
                except Exception as e:
                    raise RuntimeError(f"invalid init image bytes: {e!r}")
                pipe_kwargs["image"] = init_pil.resize((width, height))
                pipe_kwargs["strength"] = strength
                pipe = self._ensure_img2img_pipe()
                logger.info(
                    "[%s] img2img job=%s denoise_strength=%s effective_steps=%s requested_steps=%s",
                    self.log_tag, job_id, strength, effective_steps, steps,
                )
            else:
                op = "txt2img"
                pipe = self.pipe

            self._try_reset_peak_memory()
            try:
                with torch.inference_mode():
                    # img2img runs the effective steps only, so progress counts those.
                    inject_step_progress(
                        pipe, pipe_kwargs, progress, effective_steps,
                        should_cancel=should_cancel,
                    )
                    out = pipe(**pipe_kwargs)
            finally:
                self._log_job(
                    job_id, op, width, height, effective_steps, self._try_read_peak_memory()
                )

            img: Image.Image = out.images[0]  # type: ignore[union-attr]

            meta: dict[str, Any] = {
                "prompt": req.prompt,
                "seed": seed,
                "size": req.size,
                "steps": steps,
                "cfg": float(req.guidance_scale),
                # Record the value the pipe received. A dropped prompt reads as None.
                "negative_prompt": pipe_kwargs.get("negative_prompt"),
                "scheduler_id": scheduler_id,
            }
            if strength is not None:
                meta["denoise_strength"] = strength
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("lcm", json.dumps(meta))
            buf = io.BytesIO()
            img.save(buf, format="PNG", pnginfo=pnginfo)
            return buf.getvalue(), seed
        finally:
            out = None
            artifact = None
            pipe_kwargs = None
            init_pil = None
            self.execution_policy.release_cache()
