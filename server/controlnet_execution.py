from __future__ import annotations

from dataclasses import dataclass

from server.asset_store import AssetStore
from server.controlnet_registry import get_controlnet_registry


def active_model_family_from_variant(variant: str) -> str:
    if variant.startswith("sdxl"):
        return "sdxl"
    if variant.startswith("sd1") or variant.startswith("sd2"):
        return "sd15"
    raise ValueError(f"unsupported active model family for ControlNet: {variant}")


def admit_generation_operation(req, *, snapshot, provider, has_init_image: bool) -> str:
    """Apply the per-family execution-capability matrix before preprocessing.

    Reads only the eager execution booleans on the platform binding for the
    snapshot's family — never detects or reads ambient mode state. Returns the
    admitted operation name. Preserves the existing user-facing error *types*
    (ValueError for img2img/combined, NotImplementedError for ControlNet) while
    naming the stable family and operation.
    """
    from backends.platforms.base import UnsupportedFamilyError

    family_id = snapshot.resolved.profile.family_id
    binding = provider.family_binding(family_id) if provider is not None else None
    if binding is None:
        raise UnsupportedFamilyError(
            f"family '{family_id}' has no platform binding for generation"
        )
    caps = binding.execution_capabilities
    has_controlnet = bool(getattr(req, "controlnets", None))

    if has_init_image and has_controlnet:
        if not caps.supports_img2img_and_controlnet:
            raise ValueError(
                f"family '{family_id}' does not support the img2img+controlnet operation"
            )
        return "img2img+controlnet"
    if has_controlnet:
        if not caps.supports_controlnet:
            raise NotImplementedError(
                f"family '{family_id}' does not support the controlnet operation"
            )
        return "controlnet"
    if has_init_image:
        if not caps.supports_img2img:
            raise ValueError(
                f"family '{family_id}' does not support the img2img operation"
            )
        return "img2img"
    return "txt2img"


@dataclass(frozen=True)
class ControlNetBinding:
    attachment_id: str
    control_type: str
    model_id: str
    model_path: str
    control_image_bytes: bytes
    strength: float
    start_percent: float
    end_percent: float


def resolve_controlnet_bindings(req, *, mode, store: AssetStore, active_family: str) -> list[ControlNetBinding]:
    del mode

    attachments = getattr(req, "controlnets", None) or []
    if not attachments:
        return []

    registry = get_controlnet_registry()
    bindings: list[ControlNetBinding] = []
    for attachment in attachments:
        if not attachment.model_id:
            raise ValueError("controlnet attachment missing model_id")
        if attachment.map_asset_ref is None:
            raise ValueError("controlnet attachment missing map_asset_ref")

        spec = registry.get_required(attachment.model_id)
        if attachment.control_type not in spec.control_types:
            raise ValueError(
                f"model_id '{attachment.model_id}' does not support control_type '{attachment.control_type}'"
            )
        if active_family not in spec.compatible_with:
            raise ValueError(
                f"model_id '{attachment.model_id}' is incompatible with active mode family '{active_family}'"
            )

        entry = store.resolve(attachment.map_asset_ref)
        bindings.append(
            ControlNetBinding(
                attachment_id=attachment.attachment_id,
                control_type=attachment.control_type,
                model_id=attachment.model_id,
                model_path=spec.path,
                control_image_bytes=entry.data,
                strength=1.0 if attachment.strength is None else float(attachment.strength),
                start_percent=float(attachment.start_percent),
                end_percent=float(attachment.end_percent),
            )
        )
    return bindings
