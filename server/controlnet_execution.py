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
