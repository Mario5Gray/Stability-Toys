"""Minimal multimodal OpenAI-compatible chat/completions client.

Deliberately separate from backends/chat_client.py: that client is typed
for text-only messages and returns only the content string; this one
accepts image_url content parts and returns the parsed full response dict
(the describe contract preserves raw provider output verbatim).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from backends.openai_compat import bearer_headers, chat_completions_url, ensure_success


class VLMChatClient:
    def __init__(
        self,
        endpoint: str,
        api_key_env: str,
        timeout_s: float,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self._endpoint = endpoint
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._transport = transport  # tests inject httpx.MockTransport

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(transport=self._transport) as client:
            resp = await client.post(
                chat_completions_url(self._endpoint),
                json=payload,
                headers=bearer_headers(self._api_key_env),
                timeout=self._timeout_s,
            )
        ensure_success(resp, endpoint=self._endpoint, api_key_env=self._api_key_env)
        return resp.json()
