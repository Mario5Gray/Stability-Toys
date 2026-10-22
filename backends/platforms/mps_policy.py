"""Strict MPS Torch execution policy (STABL-lhpndzru).

Require built, available MPS. Use float16 and MPS only. Reject CUDA-only
offload, quantization, and xformers settings, including env values masked by
safe model overrides. Ignore CUDA_DEVICE because device is always MPS.
"""

from __future__ import annotations

import os
from typing import Any

from backends.torch_policy import TorchExecutionSettings


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


class MpsTorchPolicy:
    backend_id: str = "mps"

    def memory_stats(self) -> tuple[int, int]:
        import torch

        return int(torch.mps.current_allocated_memory()), int(torch.mps.driver_allocated_memory())

    def resolve_settings(self, model_info: Any) -> TorchExecutionSettings:
        import torch

        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("MPS runtime must be built and available")

        dtype_str = os.environ.get("CUDA_DTYPE", "fp16").lower().strip()
        if dtype_str != "fp16":
            raise ValueError("MPS requires CUDA_DTYPE=fp16")

        env_offload = os.environ.get("CUDA_OFFLOAD", "none").lower().strip()
        env_quantize = os.environ.get("CUDA_QUANTIZE", "none").lower().strip()
        env_xformers = _bool_env("CUDA_ENABLE_XFORMERS")
        runtime_offload = getattr(model_info, "runtime_offload", None)
        runtime_quantize = getattr(model_info, "runtime_quantize", None)
        runtime_xformers = getattr(model_info, "runtime_enable_xformers", None)

        for offload in (env_offload, runtime_offload):
            if offload is not None and offload.lower().strip() != "none":
                raise ValueError("MPS does not support offload")
        for quantize in (env_quantize, runtime_quantize):
            if quantize is not None and quantize.lower().strip() != "none":
                raise ValueError("MPS does not support quantize")
        if env_xformers or runtime_xformers:
            raise ValueError("MPS does not support xformers")

        env_attention_slicing = _bool_env("CUDA_ATTENTION_SLICING")
        runtime_attention_slicing = getattr(
            model_info, "runtime_attention_slicing", None
        )
        return TorchExecutionSettings(
            device="mps",
            dtype=torch.float16,
            dtype_str="fp16",
            enable_xformers=False,
            attention_slicing=(
                env_attention_slicing
                if runtime_attention_slicing is None
                else bool(runtime_attention_slicing)
            ),
            quantize="none",
            offload="none",
        )

    def quantize_modules(self, modules: tuple[Any, ...]) -> None:
        raise RuntimeError("MPS does not support fp8 quantization")

    def enable_memory_efficient_attention(self, pipe: Any) -> None:
        raise RuntimeError("MPS does not support xformers")

    def place_pipeline(self, pipe: Any, *, device: str, offload: str) -> Any:
        if device != "mps" or offload != "none":
            raise ValueError("MPS placement requires device mps and no offload")
        return pipe.to("mps")

    def release_cache(self) -> None:
        import torch

        torch.mps.empty_cache()


MPS_TORCH_POLICY = MpsTorchPolicy()
