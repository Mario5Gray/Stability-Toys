"""Z-Image bf16 MPS execution policy (STABL-lcfvhjog).

Z-Image runs in bf16 on MPS. CUDA_DTYPE does not change that, and the worker
logs an ignored override. This policy changes only dtype resolution. It
refuses CPU fallback and proves a small bf16 MPS operation before any weight
loads.
"""

from __future__ import annotations

import os
from typing import Any

from backends.platforms.mps_policy import MpsTorchPolicy


def _check_bf16_available() -> None:
    """Refuse bf16 on an MPS runtime that cannot run it."""

    import torch

    try:
        with torch.inference_mode():
            value = torch.ones((2, 2), device="mps", dtype=torch.bfloat16)
            result = value @ value
            torch.mps.synchronize()
            if not bool(torch.isfinite(result).all().item()):
                raise RuntimeError("bf16 MPS probe returned non-finite values")
    except Exception as exc:
        raise RuntimeError("Z-Image requires working bf16 MPS operations") from exc


class ZImageMpsTorchPolicy(MpsTorchPolicy):
    def _resolve_dtype(self) -> tuple[Any, str]:
        import torch

        fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").lower()
        if fallback in ("1", "true", "yes", "on"):
            raise ValueError("Z-Image MPS does not support CPU fallback")
        _check_bf16_available()
        return torch.bfloat16, "bf16"


ZIMAGE_MPS_TORCH_POLICY = ZImageMpsTorchPolicy()
