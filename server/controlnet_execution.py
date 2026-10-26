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


def _admit_controlnet_scope(req, binding, provider, family_id: str) -> None:
    """Apply the PLATFORM narrowing of the ControlNet operation.

    The mode policy decides which control types a mode allows; this is the
    narrower limit a platform adds on top, and it lives on the platform-family
    binding. The decision reads `binding.controlnet_scope` only. `backend_id` is
    read to name the platform in the message, never to decide — a backend test
    here would reject Depth on CUDA, which runs the same family.

    Through a real provider this is reachable once the platform cell admits
    the requested ControlNet operation. Plain ControlNet uses
    `supports_controlnet`; combined img2img+ControlNet uses
    `supports_img2img_and_controlnet`. The scope applies to both paths.
    """
    scope = getattr(binding, "controlnet_scope", None)
    if scope is None:
        return
    attachments = req.controlnets
    backend_id = getattr(provider, "backend_id", "this platform")
    if scope.max_attachments is not None and len(attachments) > scope.max_attachments:
        raise ValueError(
            f"backend '{backend_id}' admits at most {scope.max_attachments} "
            f"controlnet attachment(s) for family '{family_id}', "
            f"got {len(attachments)}"
        )
    if scope.allowed_control_types is None:
        return
    allowed = ", ".join(scope.allowed_control_types)
    for attachment in attachments:
        if attachment.control_type not in scope.allowed_control_types:
            raise ValueError(
                f"backend '{backend_id}' does not support control_type "
                f"'{attachment.control_type}' for family '{family_id}'; "
                f"allowed: {allowed}"
            )


def _reject_hunyuan_timing_windows(attachments, family_id: str) -> None:
    """Refuse a nondefault control timing window for HunyuanDiT.

    Neither backend's HunyuanDiT ControlNet pipeline can apply a per-step
    guidance window, so accepting the request would render while ignoring it
    (STABL-iwmhonnn). The default [0, 1] window stays accepted.
    """
    for attachment in attachments:
        if attachment.start_percent != 0.0 or attachment.end_percent != 1.0:
            raise ValueError(
                f"family '{family_id}' does not support control timing windows"
            )


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
        _admit_controlnet_scope(req, binding, provider, family_id)
        if family_id == "hunyuandit":
            for attachment in req.controlnets:
                if attachment.control_type != "canny":
                    raise ValueError("HunyuanDiT combined img2img+controlnet supports Canny only")
            _reject_hunyuan_timing_windows(req.controlnets, family_id)
        return "img2img+controlnet"
    if has_controlnet:
        if not caps.supports_controlnet:
            raise NotImplementedError(
                f"family '{family_id}' does not support the controlnet operation"
            )
        _admit_controlnet_scope(req, binding, provider, family_id)
        if family_id == "hunyuandit":
            _reject_hunyuan_timing_windows(req.controlnets, family_id)
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
