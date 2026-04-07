from typing import Any


def finalize_mode_generate_request(req: Any, mode: Any, *, env_default_size: str) -> None:
    if req.size == env_default_size:
        req.size = mode.default_size

    allowed_sizes = {
        str(entry["size"])
        for entry in (mode.resolution_options or [])
    }
    if req.size not in allowed_sizes:
        raise ValueError(f"size '{req.size}' is not allowed for mode '{mode.name}'")
