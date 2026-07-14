"""OpenAI-compatible VLM caption provider — the first real DescribeProvider.

Layering: this module never imports server.*. The caller supplies plain
connection params and an asset_resolver callable; server/analysis_routes.py
adapts config objects and the asset store.
"""
from __future__ import annotations

import base64
from typing import Any, Callable, Dict, List, Mapping, Tuple

from .contracts import DescribeTarget
from .providers import ProviderRun

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_SYSTEM_PROMPT = (
    "You are an image captioning assistant. "
    "Describe the image concisely and factually."
)

AssetResolver = Callable[[str], Tuple[bytes, str]]


def build_image_part(target: DescribeTarget, asset_resolver: AssetResolver) -> Dict[str, Any]:
    """Build the image_url content part for one target.

    URL targets pass through verbatim; the VLM host fetches them — the
    server never fetches remote URLs itself. asset_ref targets resolve to
    bytes and embed as a base64 data-URI.
    """
    if target.url:
        return {"type": "image_url", "image_url": {"url": target.url}}
    data, media_type = asset_resolver(target.asset_ref or "")
    payload = base64.b64encode(data).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{payload}"},
    }


def build_caption_messages(
    provider_run: ProviderRun,
    options: Mapping[str, Any],
    asset_resolver: AssetResolver,
) -> List[Dict[str, Any]]:
    """Assemble the chat messages: system instruction + user(image[, prompt])."""
    content: List[Dict[str, Any]] = [build_image_part(provider_run.target, asset_resolver)]
    caption = provider_run.task.caption
    if caption is not None and caption.prompt:
        content.append({"type": "text", "text": caption.prompt})
    return [
        {
            "role": "system",
            "content": options.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        },
        {"role": "user", "content": content},
    ]
