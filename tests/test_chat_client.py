import logging
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


@pytest.mark.asyncio
async def test_chat_client_complete_logs_metadata_only_by_default(monkeypatch, caplog):
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
            return FakeResponse()

    monkeypatch.setattr("backends.chat_client.httpx.AsyncClient", FakeAsyncClient)

    from backends.chat_client import ChatCompletionsClient, ChatConfig

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        client = ChatCompletionsClient(
            ChatConfig(endpoint="http://localhost:11434/v1", model="llama3.2")
        )
        with caplog.at_level(logging.DEBUG, logger="backends.chat_client"):
            await client.complete(
                [
                    {"role": "system", "content": '{"mode":"sdxl"}'},
                    {"role": "user", "content": "hi there"},
                ]
            )

    messages = [record.message for record in caplog.records if "chat outbound request" in record.message]
    assert len(messages) == 1
    message = messages[0]
    assert "llama3.2" in message
    assert "message_count" in message
    assert "json" in message
    assert "text" in message
    assert "payload" not in message
    assert "hi there" not in message
    assert "sk-test" not in message


@pytest.mark.asyncio
async def test_chat_client_stream_logs_full_payload_when_flag_enabled(monkeypatch, caplog):
    class FakeStreamResponse:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
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

    with patch.dict(os.environ, {"DEBUG_FULL_PAYLOAD": "1"}, clear=False):
        client = ChatCompletionsClient(
            ChatConfig(endpoint="http://localhost:11434/v1", model="llama3.2")
        )
        with caplog.at_level(logging.DEBUG, logger="backends.chat_client"):
            async for _ in client.stream([{"role": "user", "content": "hi there"}]):
                pass

    messages = [record.message for record in caplog.records if "chat outbound request" in record.message]
    assert len(messages) == 1
    message = messages[0]
    assert "payload" in message
    assert "hi there" in message
    assert "stream" in message
