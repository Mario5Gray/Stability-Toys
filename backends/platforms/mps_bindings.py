"""The single MPS family-platform binding table.

Import-clean by contract: dotted worker-ref strings and booleans only. It must
not import Torch, Diffusers, or the worker module — `create_mps_worker` resolves
`worker_ref` with importlib lazily.
"""

from __future__ import annotations

from backends.platforms.base import ExecutionCapabilities, FamilyPlatformBinding

MPS_FAMILY_BINDINGS: dict[str, FamilyPlatformBinding] = {
    # SD1.5 LCM text-to-image only. All three optional operations stay false, so
    # `admit_generation_operation` rejects img2img, ControlNet, and the combined
    # operation before any preprocessing.
    "sd15": FamilyPlatformBinding(
        "backends.mps_worker.DiffusersMpsWorker",
        ExecutionCapabilities(False, False, False),
    ),
}
