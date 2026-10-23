"""MPS SD1.5 LCM worker (STABL-hvvogszy).

`DiffusersMpsWorker` subclasses `CudaWorkerBase` and swaps only the execution
policy, which is the documented non-CUDA subclass contract. The base supplies
conditioning, scheduler, and style helpers. This module owns the SD1.5 load, the
text-to-image job, and PNG encoding, because the base has no PNG helper.

The pipeline-class helpers below are module globals on purpose. The worker calls
them unqualified, so a test patches them here, where the call happens. A
qualified call through `backends.cuda_worker` would miss the patch and load real
weights.

Scope: LCM text-to-image only. The worker rejects img2img, ControlNet, and any
scheduler other than LCM.
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
    _sd_pipeline_cls,
)
from backends.family_profiles import FamilyProfile
from backends.platforms.mps_policy import MPS_TORCH_POLICY
from backends.step_progress import inject_step_progress

logger = logging.getLogger(__name__)

__all__ = ["DiffusersMpsWorker"]


class DiffusersMpsWorker(CudaWorkerBase):
    """Apple MPS Diffusers worker for SD1.5 LCM text-to-image.

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

        logger.info(
            "[mps] worker %s loaded: %s (%s) on %s dtype=%s style_api=%s",
            self.worker_id, os.path.basename(ckpt_path), format_name, self.device,
            self.dtype_str, self._style_api,
        )

    # ---------------------------
    # Job execution
    # ---------------------------
    def run_job(self, job, progress=None, should_cancel=None) -> tuple[bytes, int]:
        """Render one LCM text-to-image job and return PNG bytes with the seed.

        `run_job_with_latents` is absent on purpose. No production caller needs
        it, and HunyuanDiT already uses this narrower contract.
        """
        # Defense in depth. `admit_generation_operation` rejects both operations
        # earlier, from `ExecutionCapabilities(False, False, False)`.
        if getattr(job, "init_image", None) is not None:
            raise ValueError("MPS SD1.5 does not support img2img")
        if getattr(job, "controlnet_bindings", None):
            raise NotImplementedError("MPS SD1.5 does not support ControlNet")

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
        try:
            conditioning_artifact = self._conditioning_artifact_for_request(req)
            pipe_kwargs = {
                "width": width,
                "height": height,
                "num_inference_steps": int(req.num_inference_steps),
                "guidance_scale": float(req.guidance_scale),
                "generator": gen,
            }
            conditioning_kwargs = self._accept_conditioning_artifact(
                self.pipe, conditioning_artifact
            )
            with torch.inference_mode():
                inject_step_progress(self.pipe, pipe_kwargs, progress,
                                     req.num_inference_steps,
                                     should_cancel=should_cancel)
                out = self.pipe(**{**conditioning_kwargs, **pipe_kwargs})

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
            buf = io.BytesIO()
            img.save(buf, format="PNG", pnginfo=pnginfo)
            return buf.getvalue(), seed
        finally:
            out = None  # release on failure; no-op on success
            conditioning_artifact = None
            pipe_kwargs = None
            self._apply_style(None, 0)
            self.execution_policy.release_cache()
