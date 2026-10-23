"""The single MPS family-platform binding table.

Import-clean by contract: dotted worker-ref strings and booleans only. It must
not import Torch, Diffusers, or the worker module — `create_mps_worker` resolves
`worker_ref` with importlib lazily.
"""

from __future__ import annotations

from backends.platforms.base import (
    ControlNetScope,
    ExecutionCapabilities,
    FamilyPlatformBinding,
)

MPS_FAMILY_BINDINGS: dict[str, FamilyPlatformBinding] = {
    # SD1.5 LCM text-to-image and img2img. ControlNet and the combined
    # operation stay false, so `admit_generation_operation` rejects them before
    # any preprocessing. The cell flips to admit ControlNet in Task 4 of
    # docs/superpowers/plans/2026-10-23-mps-sd15-controlnet.md, after admission
    # and execution exist.
    #
    # The scope narrows ControlNet to one Canny attachment on this platform.
    # conf/modes.yml permits canny, depth, pose, and two attachments, so the
    # narrowing cannot come from the mode policy.
    "sd15": FamilyPlatformBinding(
        "backends.mps_worker.DiffusersMpsWorker",
        ExecutionCapabilities(True, False, False),
        ControlNetScope(allowed_control_types=("canny",), max_attachments=1),
    ),
}
