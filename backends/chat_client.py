"""
Async OpenAI-compatible chat completions client.
"""

from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional
import json
import logging
import os

import httpx

from backends.openai_compat import bearer_headers, chat_completions_url, ensure_success


logger = logging.getLogger(__name__)

ChatMessage = Dict[str, str]


@dataclass
class ChatConfig:
    endpoint: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    timeout_s: float = 60.0


class ChatCompletionsClient:
    """Small client for OpenAI-compatible /chat/completions endpoints."""

    def __init__(self, config: ChatConfig):
        self.config = config

    def _request_payload(
        self,
        messages: List[ChatMessage],
        *,
        stream: bool,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> Dict[str, object]:
        return {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": stream,
        }

    def _message_kind(self, content: object) -> str:
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    json.loads(stripped)
                    return "json"
                except json.JSONDecodeError:
                    pass
            return "text"
        return type(content).__name__

    def _message_summary(self, messages: object) -> List[Dict[str, object]]:
        summary: List[Dict[str, object]] = []
        if not isinstance(messages, list):
            return summary
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            chars = len(content) if isinstance(content, str) else None
            summary.append(
                {
                    "role": message.get("role"),
                    "kind": self._message_kind(content),
                    "chars": chars,
                }
            )
        return summary

    def _log_outbound_request(self, payload: Dict[str, object]) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        messages = payload.get("messages")
        message_summary = self._message_summary(messages)
        debug_record: Dict[str, object] = {
            "url": chat_completions_url(self.config.endpoint),
            "model": payload.get("model"),
            "stream": payload.get("stream"),
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "message_count": len(message_summary),
            "message_summary": message_summary,
        }
        if os.environ.get("DEBUG_FULL_PAYLOAD") == "1":
            debug_record["payload"] = payload

        logger.debug("chat outbound request %s", debug_record)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        payload = self._request_payload(
            messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._log_outbound_request(payload)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                chat_completions_url(self.config.endpoint),
                json=payload,
                headers=bearer_headers(self.config.api_key_env),
                timeout=self.config.timeout_s,
            )
        ensure_success(
            resp,
            endpoint=self.config.endpoint,
            api_key_env=self.config.api_key_env,
        )
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError("chat completion response missing choices[0].message.content") from e

    async def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        payload = self._request_payload(
            messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._log_outbound_request(payload)
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                chat_completions_url(self.config.endpoint),
                json=payload,
                headers=bearer_headers(self.config.api_key_env),
                timeout=self.config.timeout_s,
            ) as resp:
                ensure_success(
                    resp,
                    endpoint=self.config.endpoint,
                    api_key_env=self.config.api_key_env,
                )
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_line = line[len("data:") :].strip()
                    if data_line == "[DONE]":
                        break
                    try:
                        event = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield str(delta)
