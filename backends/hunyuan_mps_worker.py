"""MPS HunyuanDiT worker (STABL-appbjuws).

The CUDA HunyuanDiT worker already sends every device action through
``execution_policy``. The subclass swaps in the MPS policy and narrows the job
contract to text-to-image and img2img.

Scope: HunyuanDiT text-to-image and img2img. ControlNet and combined
img2img+ControlNet are dependent slices (STABL-xvnjfuae, STABL-wucttopg,
STABL-ziorpomq). Admission rejects them first. The worker repeats the rejection
for a direct ``run_job`` call that bypasses admission.

The img2img path is the repo-owned ``backends/hunyuandit_img2img.py`` loop,
because Diffusers has no HunyuanDiT img2img pipeline.
"""

from __future__ import annotations

from typing import Any

from backends.cuda_worker import DiffusersHunyuanDiTCudaWorker
from backends.platforms.mps_policy import MPS_TORCH_POLICY

__all__ = ["DiffusersHunyuanDiTMpsWorker"]


class DiffusersHunyuanDiTMpsWorker(DiffusersHunyuanDiTCudaWorker):
    """Apple MPS HunyuanDiT worker for text-to-image and img2img."""

    execution_policy = MPS_TORCH_POLICY
    log_tag = "hunyuandit-mps"

    def run_job(self, job: Any, progress: Any = None, should_cancel: Any = None) -> tuple[bytes, int]:
        family_id = self.family_profile.family_id
        if getattr(job, "controlnet_bindings", None):
            raise NotImplementedError(
                f"family '{family_id}' does not support the controlnet operation"
            )
        return super().run_job(job, progress=progress, should_cancel=should_cancel)
