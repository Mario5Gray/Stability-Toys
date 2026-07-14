"""Tests for the multimodal chat/completions client."""
import json

import httpx
import pytest

from backends.analysis.vlm_client import VLMChatClient

RESPONSE = {
    "id": "cmpl-1",
    "choices": [{"message": {"role": "assistant", "content": "a red bicycle"}}],
    "usage": {"total_tokens": 42},
}


def _transport(capture):
    def handler(request: httpx.Request) -> httpx.Response:
        capture["url"] = str(request.url)
        capture["headers"] = dict(request.headers)
        capture["payload"] = json.loads(request.content)
        return httpx.Response(200, json=RESPONSE)
    return httpx.MockTransport(handler)


async def test_complete_posts_payload_and_returns_full_response(monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "sekrit")
    capture = {}
    client = VLMChatClient(
        endpoint="http://vlm.lan:8080/v1/",
        api_key_env="TEST_VLM_KEY",
        timeout_s=30,
        transport=_transport(capture),
    )
    messages = [
        {"role": "system", "content": "caption things"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
        ]},
    ]
    resp = await client.complete(
        model="qwen2.5-vl", messages=messages, max_tokens=256, temperature=0.0,
    )
    # Full response dict, not just the content string.
    assert resp == RESPONSE
    # Trailing slash trimmed, path joined.
    assert capture["url"] == "http://vlm.lan:8080/v1/chat/completions"
    assert capture["headers"]["authorization"] == "Bearer sekrit"
    assert capture["payload"] == {
        "model": "qwen2.5-vl",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.0,
    }


async def test_complete_omits_auth_header_when_env_unset(monkeypatch):
    monkeypatch.delenv("TEST_VLM_KEY", raising=False)
    capture = {}
    client = VLMChatClient(
        endpoint="http://vlm.lan:8080/v1",
        api_key_env="TEST_VLM_KEY",
        timeout_s=30,
        transport=_transport(capture),
    )
    await client.complete(model="m", messages=[], max_tokens=1, temperature=0.0)
    assert "authorization" not in capture["headers"]


async def test_complete_raises_on_non_2xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")
    client = VLMChatClient(
        endpoint="http://vlm.lan:8080/v1",
        api_key_env="TEST_VLM_KEY",
        timeout_s=30,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.complete(model="m", messages=[], max_tokens=1, temperature=0.0)
