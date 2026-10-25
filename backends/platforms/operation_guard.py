"""Worker-side repeat of the platform operation gate and ControlNet scope.

Admission runs first on every server path. This guard protects a worker called
directly, or through a future dispatch path that skips admission. It reads the
same platform-family cell that admission reads.

Torch-free at module level: the provider import stays inside the function.
"""

from __future__ import annotations

from typing import Any


def check_platform_operation(
    *,
    family_id: str,
    has_init_image: bool,
    bindings: list[Any],
    label: str,
) -> None:
    """Repeat the cell's operation capability and ControlNet scope.

    `label` names the platform and family in the ControlNet messages, for
    example "MPS SD1.5". The operation messages keep the shared admission text.
    """
    from backends.platform_registry import get_backend_provider

    cell = get_backend_provider().family_binding(family_id)
    caps = cell.execution_capabilities
    if has_init_image and bindings:
        if not caps.supports_img2img_and_controlnet:
            raise ValueError(
                f"family '{family_id}' does not support the img2img+controlnet operation"
            )
    elif bindings:
        if not caps.supports_controlnet:
            raise NotImplementedError(
                f"family '{family_id}' does not support the controlnet operation"
            )
    elif has_init_image and not caps.supports_img2img:
        raise ValueError(f"family '{family_id}' does not support the img2img operation")
    if not bindings:
        return
    scope = getattr(cell, "controlnet_scope", None)
    if scope is None:
        return
    if scope.max_attachments is not None and len(bindings) > scope.max_attachments:
        raise ValueError(
            f"{label} admits at most {scope.max_attachments} controlnet "
            f"attachment(s), got {len(bindings)}"
        )
    if scope.allowed_control_types is None:
        return
    allowed = ", ".join(scope.allowed_control_types)
    for binding in bindings:
        if binding.control_type not in scope.allowed_control_types:
            raise ValueError(
                f"{label} does not support control_type "
                f"'{binding.control_type}'; allowed: {allowed}"
            )
