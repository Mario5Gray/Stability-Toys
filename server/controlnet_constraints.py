from typing import Any


def enforce_controlnet_policy(req: Any, mode: Any) -> None:
    attachments = getattr(req, "controlnets", None)
    if not attachments:
        return

    policy = getattr(mode, "controlnet_policy", None)
    if policy is None or not policy.enabled:
        raise ValueError(f"mode '{mode.name}' does not enable ControlNet")

    if len(attachments) > policy.max_attachments:
        raise ValueError(
            f"request has {len(attachments)} ControlNet attachments; "
            f"mode '{mode.name}' allows max_attachments={policy.max_attachments}"
        )

    seen_ids: set[str] = set()
    for attachment in attachments:
        if attachment.attachment_id in seen_ids:
            raise ValueError(f"duplicate attachment_id '{attachment.attachment_id}' in request")
        seen_ids.add(attachment.attachment_id)

        type_policy = policy.allowed_control_types.get(attachment.control_type)
        if type_policy is None:
            raise ValueError(
                f"control_type '{attachment.control_type}' not allowed for mode '{mode.name}'"
            )

        if attachment.preprocess is not None and not type_policy.allow_preprocess:
            raise ValueError(
                f"preprocessing not allowed for control_type '{attachment.control_type}' "
                f"in mode '{mode.name}'"
            )

        if attachment.model_id is None:
            if type_policy.default_model_id is None:
                raise ValueError(
                    f"model_id required for control_type '{attachment.control_type}' "
                    f"in mode '{mode.name}' (no default configured)"
                )
            attachment.model_id = type_policy.default_model_id
        elif attachment.model_id not in type_policy.allowed_model_ids:
            raise ValueError(
                f"model_id '{attachment.model_id}' not allowed for control_type "
                f"'{attachment.control_type}' in mode '{mode.name}'"
            )

        if not (type_policy.min_strength <= attachment.strength <= type_policy.max_strength):
            raise ValueError(
                f"strength {attachment.strength} outside policy range "
                f"[{type_policy.min_strength}, {type_policy.max_strength}] for "
                f"control_type '{attachment.control_type}' in mode '{mode.name}'"
            )


def ensure_controlnet_dispatch_supported(req: Any) -> None:
    attachments = getattr(req, "controlnets", None)
    if attachments:
        raise NotImplementedError(
            "ControlNet provider not yet implemented on this backend "
            "(Track 3 delivers execution)"
        )
