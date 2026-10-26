"""MPS HunyuanDiT worker (STABL-appbjuws, STABL-xvnjfuae, STABL-wucttopg, STABL-ziorpomq).

The CUDA HunyuanDiT worker already sends every device action through
``execution_policy``. The subclass swaps in the MPS policy and narrows the job
contract to text-to-image, img2img, and one Canny, Depth or Pose ControlNet
attachment.

Scope: HunyuanDiT text-to-image, img2img, and plain text-to-image ControlNet
with one Canny, Depth or Pose attachment. Combined img2img+ControlNet stays
unbound on MPS (STABL-xvnjfuae). Admission rejects it first. The worker repeats
the rejection for a direct ``run_job`` call that bypasses admission.

The plain ControlNet path runs the shared ``HunyuanDiTControlNetPipeline``
branch of the CUDA worker. The img2img path is the repo-owned
``backends/hunyuandit_img2img.py`` loop, because Diffusers has no HunyuanDiT
img2img pipeline.

``HunyuanDiTControlNetPipeline`` accepts no per-step guidance window. The
shared kwarg builder drops ``control_guidance_start`` and
``control_guidance_end`` for this family, so a nondefault window would be
silently ignored. Shared admission refuses such a request before preprocessing,
and the shared HunyuanDiT worker repeats the refusal for a direct ``run_job``
call on both backends (STABL-iwmhonnn).
"""

from __future__ import annotations

from typing import Any

from backends.cuda_worker import DiffusersHunyuanDiTCudaWorker
from backends.platforms.mps_policy import MPS_TORCH_POLICY

__all__ = ["DiffusersHunyuanDiTMpsWorker"]


class DiffusersHunyuanDiTMpsWorker(DiffusersHunyuanDiTCudaWorker):
    """Apple MPS HunyuanDiT worker for text-to-image, img2img, Canny, Depth and Pose."""

    execution_policy = MPS_TORCH_POLICY
    log_tag = "hunyuandit-mps"

    def _check_platform_operation(
        self, *, has_init_image: bool, bindings: list[Any]
    ) -> None:
        """Repeat the cell's operation capability and ControlNet scope."""
        from backends.platforms.operation_guard import check_platform_operation

        check_platform_operation(
            family_id=self.family_profile.family_id,
            has_init_image=has_init_image,
            bindings=bindings,
            label="MPS HunyuanDiT",
        )

    def run_job(self, job: Any, progress: Any = None, should_cancel: Any = None) -> tuple[bytes, int]:
        init_image = getattr(job, "init_image", None)
        bindings = getattr(job, "controlnet_bindings", None) or []
        # Defense in depth. Admission checks the same cell before preprocessing.
        if init_image is not None or bindings:
            self._check_platform_operation(
                has_init_image=init_image is not None, bindings=bindings
            )
            # The shared HunyuanDiT worker repeats the timing-window rule in
            # run_job (STABL-iwmhonnn), so this path does not repeat it again.
        return super().run_job(job, progress=progress, should_cancel=should_cancel)
