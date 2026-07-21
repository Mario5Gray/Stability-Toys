"""The single CUDA family-platform binding table.

Import-clean by contract: dotted worker-ref strings and booleans only. It must
not import Torch, Diffusers, or the CUDA worker module — `create_cuda_worker`
resolves `worker_ref` with importlib lazily.
"""

from __future__ import annotations

from backends.platforms.base import ExecutionCapabilities, FamilyPlatformBinding

CUDA_FAMILY_BINDINGS: dict[str, FamilyPlatformBinding] = {
    "sd15": FamilyPlatformBinding(
        "backends.cuda_worker.DiffusersCudaWorker",
        ExecutionCapabilities(True, True, True),
    ),
    "sdxl": FamilyPlatformBinding(
        "backends.cuda_worker.DiffusersSDXLCudaWorker",
        ExecutionCapabilities(True, True, True),
    ),
    # HunyuanDiT first delivery: ControlNet txt2img only — no img2img, no
    # combined img2img+ControlNet.
    "hunyuandit": FamilyPlatformBinding(
        "backends.cuda_worker.DiffusersHunyuanDiTCudaWorker",
        ExecutionCapabilities(False, True, False),
    ),
}
