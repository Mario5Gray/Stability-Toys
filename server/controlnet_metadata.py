"""Read ControlNet provenance metadata from control-map PNG bytes."""

from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image

CHUNK_KEY = "controlnet_map"


def read_control_map_metadata(png_bytes: bytes) -> dict[str, Any] | None:
    """Return the parsed controlnet_map payload, or None if absent/unreadable."""
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            raw = getattr(img, "text", {}).get(CHUNK_KEY)
    except Exception:
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
