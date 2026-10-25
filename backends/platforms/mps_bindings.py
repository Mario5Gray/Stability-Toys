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
    # SD1.5 LCM text-to-image, img2img, ControlNet, and combined img2img+ControlNet.
    #
    # The scope narrows ControlNet to one Canny attachment on this platform.
    # conf/modes.yml permits canny, depth, pose, and two attachments, so the
    # narrowing cannot come from the mode policy. Admission enforces the scope,
    # and the worker repeats it.
    "sd15": FamilyPlatformBinding(
        "backends.mps_worker.DiffusersMpsWorker",
        ExecutionCapabilities(True, True, True),
        ControlNetScope(allowed_control_types=("canny",), max_attachments=1),
    ),
    # HunyuanDiT text-to-image, img2img, and one ControlNet attachment
    # (STABL-appbjuws, STABL-sulxavyo, STABL-xvnjfuae, STABL-wucttopg). The
    # combined img2img+ControlNet operation stays unbound on MPS: the third flag
    # is false, so admission rejects it. The scope narrows ControlNet to one
    # Canny or Depth attachment, while conf/modes.yml permits canny, depth,
    # pose, and one attachment. Admission enforces the scope, and the worker
    # repeats it.
    "hunyuandit": FamilyPlatformBinding(
        "backends.hunyuan_mps_worker.DiffusersHunyuanDiTMpsWorker",
        ExecutionCapabilities(True, True, False),
        ControlNetScope(allowed_control_types=("canny", "depth"), max_attachments=1),
    ),
}
