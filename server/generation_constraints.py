from typing import Any


def finalize_mode_generate_request(
    req: Any,
    mode: Any,
    *,
    env_default_size: str,
    env_default_steps: int,
    env_default_guidance: float,
) -> None:
    if req.size == env_default_size:
        req.size = mode.default_size
    if getattr(req, "num_inference_steps", None) == env_default_steps:
        req.num_inference_steps = mode.default_steps
    if getattr(req, "guidance_scale", None) == env_default_guidance:
        req.guidance_scale = mode.default_guidance

    allowed_sizes = {
        str(entry["size"])
        for entry in (mode.resolution_options or [])
    }
    if req.size not in allowed_sizes:
        raise ValueError(f"size '{req.size}' is not allowed for mode '{mode.name}'")
