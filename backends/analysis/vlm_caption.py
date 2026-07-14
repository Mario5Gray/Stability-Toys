"""OpenAI-compatible VLM caption provider — the first real DescribeProvider.

Layering: this module never imports server.*. The caller supplies plain
connection params and an asset_resolver callable; server/analysis_routes.py
adapts config objects and the asset store.
"""
from __future__ import annotations

import base64
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import httpx

from .contracts import DescribeObservation, DescribeTarget, DescribeTask, TaskKind, TextObservation
from .providers import ProviderResult, ProviderRun
from .vlm_client import VLMChatClient

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


class OpenAIVLMCaptionProvider:
    """Caption tasks via an OpenAI-compatible multimodal endpoint.

    Raises on every failure (HTTP, timeout, malformed response, resolver
    error) — the orchestrator's per-run isolation maps a raise to a failed
    run with analysis_run_failed. No retries in v1.
    """

    def __init__(
        self,
        endpoint: str,
        api_key_env: str,
        model: str,
        options: Mapping[str, Any],
        asset_resolver: AssetResolver,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self._model = model
        self._options = dict(options)
        self._asset_resolver = asset_resolver
        self._client = VLMChatClient(
            endpoint=endpoint,
            api_key_env=api_key_env,
            timeout_s=float(self._options.get("timeout_s", DEFAULT_TIMEOUT_S)),
            transport=transport,
        )

    def supports(self, task: DescribeTask) -> bool:
        return task.kind == TaskKind.CAPTION

    async def run(self, provider_run: ProviderRun) -> ProviderResult:
        response = await self._client.complete(
            model=self._model,
            messages=build_caption_messages(
                provider_run, self._options, self._asset_resolver,
            ),
            max_tokens=int(self._options.get("max_tokens", DEFAULT_MAX_TOKENS)),
            temperature=float(self._options.get("temperature", DEFAULT_TEMPERATURE)),
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "VLM response missing choices[0].message.content"
            ) from exc
        if not content:
            raise ValueError("VLM response content is empty")
        observation = DescribeObservation(
            task_id=provider_run.plan.task_id,
            target_id=provider_run.plan.target_id,
            kind="text",
            text=TextObservation(content=content),
        )
        # raw_output carries the full completion response verbatim.
        return ProviderResult(observations=(observation,), raw_output=response)
