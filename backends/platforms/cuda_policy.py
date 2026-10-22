"""CUDA Torch execution policy (STABL-xkiwjvrm).

Owns every CUDA-only device action of the Diffusers workers. Torch and
optimum.quanto are imported at call time, so importing this module stays
import-clean on hosts without CUDA.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from backends.torch_policy import TorchExecutionSettings


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


class CudaTorchPolicy:
    backend_id: str = "cuda"

    def memory_stats(self) -> tuple[int, int]:
        import torch

        return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())

    def resolve_settings(self, model_info: Any) -> TorchExecutionSettings:
        """Parse all CUDA_* env vars. Non-None ``model_info.runtime_*`` values win."""
        import torch

        device = os.environ.get("CUDA_DEVICE", "cuda:0").strip()
        dtype_str = os.environ.get("CUDA_DTYPE", "fp16").lower().strip()
        if dtype_str == "bf16":
            dtype = torch.bfloat16
        elif dtype_str == "fp32":
            dtype = torch.float32
        else:
            dtype = torch.float16
        env_enable_xformers = _bool_env("CUDA_ENABLE_XFORMERS", "0")
        env_attention_slicing = _bool_env("CUDA_ATTENTION_SLICING", "0")
        env_quantize = os.environ.get("CUDA_QUANTIZE", "none").lower().strip()
        env_offload = os.environ.get("CUDA_OFFLOAD", "none").lower().strip()

        runtime_enable_xformers = getattr(model_info, "runtime_enable_xformers", None)
        runtime_attention_slicing = getattr(model_info, "runtime_attention_slicing", None)
        runtime_quantize = getattr(model_info, "runtime_quantize", None)
        runtime_offload = getattr(model_info, "runtime_offload", None)

        return TorchExecutionSettings(
            device=device,
            dtype=dtype,
            dtype_str=dtype_str,
            enable_xformers=(
                env_enable_xformers if runtime_enable_xformers is None else bool(runtime_enable_xformers)
            ),
            attention_slicing=(
                env_attention_slicing if runtime_attention_slicing is None else bool(runtime_attention_slicing)
            ),
            quantize=(runtime_quantize or env_quantize).lower().strip(),
            offload=(runtime_offload or env_offload).lower().strip(),
        )

    @staticmethod
    def device_index(device: str) -> int:
        """Parse the integer device index from a device string (e.g. 'cuda:1' → 1)."""
        if ":" in device:
            try:
                return int(device.split(":")[-1])
            except ValueError:
                return 0
        return 0

    def quantize_modules(self, modules: tuple[Any, ...]) -> None:
        quanto = import_module("optimum.quanto")
        for target in modules:
            quanto.quantize(target, weights=quanto.qfloat8)
            quanto.freeze(target)

    def enable_memory_efficient_attention(self, pipe: Any) -> None:
        pipe.enable_xformers_memory_efficient_attention()

    def place_pipeline(self, pipe: Any, *, device: str, offload: str) -> Any:
        """Register offload hooks on the configured GPU, or move the pipe to it."""
        gpu_id = self.device_index(device)
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        elif offload == "model":
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipe = pipe.to(device)
        return pipe

    def release_cache(self) -> None:
        import torch

        torch.cuda.empty_cache()


CUDA_TORCH_POLICY = CudaTorchPolicy()
