"""MPS worker for Z-Image Turbo and Base (STABL-lcfvhjog).

The subclass changes device policy and job memory logging only. The shared
Z-Image constructor, generation loop, scheduler, conditioning, img2img and
LoRA fusion are inherited unchanged.

MPS has no allocator peak counters. The job line reports current and driver
allocation, and reports peaks as unavailable. A failed read must not fail a
completed generation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from backends.platforms.zimage_mps_policy import ZIMAGE_MPS_TORCH_POLICY
from backends.zimage_cuda_worker import DiffusersZImageCudaWorker

logger = logging.getLogger(__name__)


class DiffusersZImageMpsWorker(DiffusersZImageCudaWorker):
    """MPS worker for Z-Image-Turbo and Z-Image base Diffusers folders."""

    execution_policy = ZIMAGE_MPS_TORCH_POLICY
    log_tag = "zimage-mps"

    def _try_reset_peak_memory(self) -> bool:
        return False

    def _try_read_peak_memory(self) -> None:
        return None

    def _log_job(
        self, job_id: Any, op: str, width: int, height: int, steps: int,
        peak: tuple[int, int] | None,
    ) -> None:
        allocated: int | str = "unavailable"
        driver: int | str = "unavailable"
        try:
            torch.mps.synchronize()
            allocated, driver = self.execution_policy.memory_stats()
        except Exception as exc:
            logger.warning("[%s] memory read failed: %r", self.log_tag, exc)
        logger.info(
            "[%s] job=%s op=%s size=%sx%s steps=%s "
            "current_allocated_bytes=%s driver_allocated_bytes=%s "
            "peak_allocated_bytes=unavailable peak_reserved_bytes=unavailable",
            self.log_tag, job_id, op, width, height, steps, allocated, driver,
        )
