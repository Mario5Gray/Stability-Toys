import os
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_chat_client_complete_posts_openai_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hello from model"}}]}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr("backends.chat_client.httpx.AsyncClient", FakeAsyncClient)

    from backends.chat_client import ChatCompletionsClient, ChatConfig

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        client = ChatCompletionsClient(
            ChatConfig(endpoint="http://localhost:11434/v1", model="llama3.2")
        )
        text = await client.complete([{"role": "user", "content": "hi"}])

    assert text == "hello from model"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["json"]["model"] == "llama3.2"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["json"]["stream"] is False
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_chat_client_stream_yields_token_deltas(monkeypatch):
    class FakeStreamResponse:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"hel"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"lo"}}]}'
            yield "data: [DONE]"

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json, headers, timeout):
            assert method == "POST"
            assert url == "http://localhost:11434/v1/chat/completions"
            assert json["stream"] is True
            return FakeStreamContext()

    monkeypatch.setattr("backends.chat_client.httpx.AsyncClient", FakeAsyncClient)

    from backends.chat_client import ChatCompletionsClient, ChatConfig

    client = ChatCompletionsClient(
        ChatConfig(endpoint="http://localhost:11434/v1", model="llama3.2")
    )
    chunks = []
    async for token in client.stream([{"role": "user", "content": "hello"}]):
        chunks.append(token)

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_chat_client_stream_ignores_empty_choices_events(monkeypatch):
    class FakeStreamResponse:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[]}'
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json, headers, timeout):
            return FakeStreamContext()

    monkeypatch.setattr("backends.chat_client.httpx.AsyncClient", FakeAsyncClient)

    from backends.chat_client import ChatCompletionsClient, ChatConfig

    client = ChatCompletionsClient(
        ChatConfig(endpoint="http://localhost:11434/v1", model="llama3.2")
    )
    chunks = []
    async for token in client.stream([{"role": "user", "content": "hello"}]):
        chunks.append(token)

    assert chunks == ["ok"]
