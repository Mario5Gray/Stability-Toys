"""Embed ControlNet provenance metadata into emitted control-map PNGs.

Shared by the standalone map tools (canny_map, depth_map, pose_map). Metadata
is written as a single PNG tEXt chunk keyed "controlnet_map" holding a JSON
payload describing the tool, its parameters, and the source dimensions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin

CHUNK_KEY = "controlnet_map"
SCHEMA_VERSION = 1


def build_map_metadata(
    *,
    tool: str,
    control_type: str,
    source_size: tuple[int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the controlnet_map payload: common fields plus tool params."""
    width, height = source_size
    payload: dict[str, Any] = {
        "tool": tool,
        "version": SCHEMA_VERSION,
        "control_type": control_type,
        "source_width": int(width),
        "source_height": int(height),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(params)
    return payload


def save_with_metadata(
    image: Image.Image, destination: Path, payload: dict[str, Any]
) -> None:
    """Save a PIL image to PNG with the controlnet_map metadata chunk."""
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text(CHUNK_KEY, json.dumps(payload))
    image.save(destination, format="PNG", pnginfo=pnginfo)
