"""MPS SD1.5 LCM worker (STABL-hvvogszy).

`DiffusersMpsWorker` subclasses `CudaWorkerBase` and swaps only the execution
policy, which is the documented non-CUDA subclass contract. The base supplies
conditioning, scheduler, and style helpers. This module owns the SD1.5 load, the
text-to-image and img2img jobs, and PNG encoding, because the base has no PNG
helper.

The pipeline-class helpers below are module globals on purpose. The worker calls
them unqualified, so a test patches them here, where the call happens. A
qualified call through `backends.cuda_worker` would miss the patch and load real
weights.

Scope: LCM text-to-image, LCM img2img, and one Canny ControlNet attachment,
including combined img2img+ControlNet. The worker rejects attachments outside
the platform scope and any scheduler other than LCM.
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Optional

import torch
from PIL import Image, PngImagePlugin

from backends.cuda_worker import (
    CudaWorkerBase,
    _lcm_scheduler_cls,
    _sd_controlnet_img2img_pipeline_cls,
    _sd_controlnet_pipeline_cls,
    _sd_img2img_pipeline_cls,
    _sd_pipeline_cls,
    _validate_control_image_aspect_ratio,
)
from backends.family_profiles import FamilyProfile
from backends.platforms.mps_policy import MPS_TORCH_POLICY
from backends.step_progress import inject_step_progress

logger = logging.getLogger(__name__)

__all__ = ["DiffusersMpsWorker"]


class DiffusersMpsWorker(CudaWorkerBase):
    """Apple MPS Diffusers worker for SD1.5 LCM text-to-image and img2img.

    Supports two formats:
      - Single file: .safetensors or .ckpt checkpoint
      - Diffusers layout: directory with model_index.json

    The MPS policy fixes device `mps` and dtype float16. It rejects offload,
    quantization, and xformers. See `backends/platforms/mps_policy.py`.
    """

    execution_policy = MPS_TORCH_POLICY

    def __init__(
        self,
        worker_id: int,
        model_path: str,
        model_info: Optional[Any] = None,
        family_profile: FamilyProfile | None = None,
    ):
        super().__init__(worker_id, model_info=model_info, family_profile=family_profile)

        ckpt_path = model_path
        logger.info("[mps] ckpt_path=%s", ckpt_path)

        format_hint = self._loader_format
        is_diffusers_dir = format_hint == "diffusers_dir" or (
            format_hint == "unknown"
            and os.path.isdir(ckpt_path)
            and os.path.exists(os.path.join(ckpt_path, "model_index.json"))
        )

        if is_diffusers_dir:
            logger.info("[mps] loading diffusers")
            pipe = _sd_pipeline_cls().from_pretrained(
                ckpt_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            format_name = "diffusers"
        else:
            logger.info("[mps] loading safetensors")
            pipe = _sd_pipeline_cls().from_single_file(
                ckpt_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            format_name = "single-file"

        # Unconditional, unlike CUDA's `scheduler_profile != "native"` branch.
        # This slice accepts the LCM scheduler only, so a native profile has
        # nothing to select. The divergence is deliberate.
        pipe.scheduler = _lcm_scheduler_cls().from_config(pipe.scheduler.config)
        pipe = self._setup_pipe_memory_opts(pipe)

        self.pipe = pipe
        self._capture_baseline_scheduler(self.pipe)

        # No style-LoRA load loop here. STYLE_REGISTRY is empty, and the base
        # `_apply_style(None, 0)` cleanup path is safe with the base defaults.
        # The CUDA style-API detection sits inside that loop, so state the
        # result directly, as the HunyuanDiT worker does for the same reason.
        self._style_api = "none"

        logger.info(
            "[mps] worker %s loaded: %s (%s) on %s dtype=%s style_api=%s",
            self.worker_id, os.path.basename(ckpt_path), format_name, self.device,
            self.dtype_str, self._style_api,
        )

    # ---------------------------
    # Job execution
    # ---------------------------
    def _check_controlnet_scope(self, bindings: list[Any]) -> None:
        """Repeat the platform ControlNet scope the cell carries.

        One source states the LIMIT: the platform-family binding, which
        `admit_generation_operation` reads through the same field. The checking
        code is not shared, so the two messages can drift even though the values
        cannot. Admission runs first on every server path, so the message a
        client sees is admission's.
        """
        from backends.platform_registry import get_backend_provider

        cell = get_backend_provider().family_binding(self.family_profile.family_id)
        scope = getattr(cell, "controlnet_scope", None)
        if scope is None:
            return
        if scope.max_attachments is not None and len(bindings) > scope.max_attachments:
            raise ValueError(
                f"MPS SD1.5 admits at most {scope.max_attachments} controlnet "
                f"attachment(s), got {len(bindings)}"
            )
        if scope.allowed_control_types is None:
            return
        allowed = ", ".join(scope.allowed_control_types)
        for binding in bindings:
            if binding.control_type not in scope.allowed_control_types:
                raise ValueError(
                    f"MPS SD1.5 does not support control_type "
                    f"'{binding.control_type}'; allowed: {allowed}"
                )

    def _build_controlnet_pipe(self, controlnet_obj: Any) -> Any:
        # The helper is called unqualified, so a test patches it on this module.
        # Reaching into Diffusers here would load real weights under test.
        return _sd_controlnet_pipeline_cls().from_pipe(
            self.pipe,
            controlnet=controlnet_obj,
            torch_dtype=None,
        )

    def _build_controlnet_img2img_pipe(self, controlnet_obj: Any) -> Any:
        return _sd_controlnet_img2img_pipeline_cls().from_pipe(
            self.pipe,
            controlnet=controlnet_obj,
            torch_dtype=None,
        )

    def run_job(self, job, progress=None, should_cancel=None) -> tuple[bytes, int]:
        """Render one LCM job and return PNG bytes with the seed.

        An init image selects the img2img pipeline. Both paths share the loaded
        weights.

        `run_job_with_latents` is absent on purpose. No production caller needs
        it, and HunyuanDiT already uses this narrower contract.
        """
        init_image = getattr(job, "init_image", None)
        bindings = getattr(job, "controlnet_bindings", None) or []

        # Defense in depth. Admission applies the same platform scope before
        # preprocessing. Jobs without bindings never read the provider.
        if bindings:
            self._check_controlnet_scope(bindings)

        req = job.req
        if self._resolve_scheduler_id(req) != "lcm":
            raise ValueError("MPS SD1.5 supports LCM scheduler only")

        try:
            w_str, h_str = str(req.size).lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            raise RuntimeError(f"Invalid size '{req.size}', expected 'WIDTHxHEIGHT'")

        seed = int(req.seed) if req.seed is not None else int(torch.randint(0, 100_000_000, (1,)).item())

        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        sl = getattr(req, "style_lora", None)
        style_id = getattr(sl, "style", None) if sl else None
        level = int(getattr(sl, "level", 0)) if sl else 0

        self._apply_style(style_id, level)
        scheduler_id = self._apply_request_scheduler(req)

        out = None
        conditioning_artifact = None
        pipe_kwargs = None
        loaded_ids: list[str] = []
        controlnet_kwargs: dict[str, Any] = {}
        try:
            conditioning_artifact = self._conditioning_artifact_for_request(req)
            if init_image is not None:
                if bindings:
                    _validate_control_image_aspect_ratio(init_image, bindings)
                    controlnet_kwargs = self._build_controlnet_kwargs(
                        bindings, (width, height), loaded_ids,
                        image_kwarg="control_image",
                    )
                    controlnet_obj = controlnet_kwargs.pop("controlnet")
                    pipe = self._build_controlnet_img2img_pipe(controlnet_obj)
                else:
                    if self._img2img_pipe is None:
                        self._img2img_pipe = _sd_img2img_pipeline_cls()(**self.pipe.components)
                    pipe = self._img2img_pipe
                # Both img2img pipes read size from the init image; neither
                # takes width or height. The combined pipe uses `control_image`
                # for the map, leaving `image` for this init image.
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize(
                    (width, height)
                )
                # A prior run can upcast the shared VAE. Align it, or the next
                # init-image encode fails on a dtype mismatch.
                self._normalize_img2img_modules()
                pipe_kwargs = {
                    "image": init_pil,
                    "strength": float(getattr(req, "denoise_strength", 0.75)),
                    "num_inference_steps": int(req.num_inference_steps),
                    "guidance_scale": float(req.guidance_scale),
                    "generator": gen,
                    **controlnet_kwargs,
                }
            else:
                if bindings:
                    # The ControlNet weights load at self.dtype through the
                    # inherited loader. MPS policy fixes that at float16, and
                    # this path adds no dtype branch of its own.
                    controlnet_kwargs = self._build_controlnet_kwargs(
                        bindings, (width, height), loaded_ids
                    )
                pipe = self.pipe
                pipe_kwargs = {
                    "width": width,
                    "height": height,
                    "num_inference_steps": int(req.num_inference_steps),
                    "guidance_scale": float(req.guidance_scale),
                    "generator": gen,
                    **controlnet_kwargs,
                }
                controlnet_obj = pipe_kwargs.pop("controlnet", None)
                if controlnet_obj is not None:
                    # from_pipe shares the loaded components, so the ControlNet
                    # render costs no second copy of the weights.
                    pipe = self._build_controlnet_pipe(controlnet_obj)
            conditioning_kwargs = self._accept_conditioning_artifact(
                pipe, conditioning_artifact
            )
            with torch.inference_mode():
                inject_step_progress(pipe, pipe_kwargs, progress,
                                     req.num_inference_steps,
                                     should_cancel=should_cancel)
                out = pipe(**{**conditioning_kwargs, **pipe_kwargs})

            img: Image.Image = out.images[0]
            out = None  # release tensor reference before PNG encoding

            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("lcm", json.dumps({
                "prompt": req.prompt,
                "seed": seed,
                "size": req.size,
                "steps": int(req.num_inference_steps),
                "cfg": float(req.guidance_scale),
                "negative_prompt": getattr(req, "negative_prompt", None),
                "scheduler_id": scheduler_id,
            }))
            if bindings:
                pnginfo.add_text("controlnet", json.dumps(self._controlnet_metadata(bindings)))
            buf = io.BytesIO()
            img.save(buf, format="PNG", pnginfo=pnginfo)
            return buf.getvalue(), seed
        finally:
            out = None  # release on failure; no-op on success
            conditioning_artifact = None
            pipe_kwargs = None
            self._apply_style(None, 0)
            if loaded_ids:
                from backends.controlnet_cache import get_controlnet_cache

                cache = get_controlnet_cache()
                for model_id in loaded_ids:
                    cache.release(model_id)
            self.execution_policy.release_cache()
