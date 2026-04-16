from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def test_build_evidence_fingerprint_is_stable():
    from server.advisor_service import build_evidence_fingerprint

    evidence = {
        "version": 1,
        "gallery_id": "gal_1",
        "items": [{"cache_key": "a", "prompt": "cat", "steps": 10}],
    }

    assert build_evidence_fingerprint(evidence) == build_evidence_fingerprint(evidence)


@pytest.mark.asyncio
async def test_generate_digest_uses_resolved_chat_config_and_clamps_length_limit():
    from server.advisor_service import AdvisorDigestRequest, generate_digest

    chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="llama3.2",
        api_key_env="OPENAI_API_KEY",
        max_tokens=512,
        temperature=0.6,
        system_prompt="You are an advisor.",
    )
    mode = SimpleNamespace(maximum_len=120)
    config = SimpleNamespace(
        get_default_mode=lambda: "sdxl-general",
        get_mode=lambda name: mode,
        resolve_chat_config=lambda name, overrides=None: chat_cfg,
    )
    client_inst = SimpleNamespace(complete=AsyncMock(return_value="digest text"))

    req = AdvisorDigestRequest(
        gallery_id="gal_1",
        evidence={
            "version": 1,
            "gallery_id": "gal_1",
            "items": [{"cache_key": "a", "prompt": "cat"}],
        },
        temperature=0.2,
        length_limit=400,
    )

    with patch("server.advisor_service.get_mode_config", return_value=config), \
            patch("server.advisor_service.ChatCompletionsClient", return_value=client_inst):
        result = await generate_digest(req)

    assert result["gallery_id"] == "gal_1"
    assert result["digest_text"] == "digest text"
    assert result["mode"] == "sdxl-general"
    assert result["length_limit"] == 120
    messages, kwargs = client_inst.complete.await_args.args[0], client_inst.complete.await_args.kwargs
    assert messages[0] == {"role": "system", "content": "You are an advisor."}
    assert kwargs["max_tokens"] == 120
    assert kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_generate_digest_appends_request_system_prompt_after_mode_prompt():
    from server.advisor_service import AdvisorDigestRequest, generate_digest

    chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="llama3.2",
        api_key_env="OPENAI_API_KEY",
        max_tokens=256,
        temperature=0.6,
        system_prompt="Mode prompt\n\nRequest prompt",
    )
    mode = SimpleNamespace(maximum_len=None)
    config = SimpleNamespace(
        get_default_mode=lambda: "sdxl-general",
        get_mode=lambda name: mode,
        resolve_chat_config=lambda name, overrides=None: chat_cfg,
    )
    client_inst = SimpleNamespace(complete=AsyncMock(return_value="digest text"))

    req = AdvisorDigestRequest(
        gallery_id="gal_1",
        evidence={"version": 1, "gallery_id": "gal_1", "items": []},
        system_prompt="Request prompt",
    )

    with patch("server.advisor_service.get_mode_config", return_value=config), \
            patch("server.advisor_service.ChatCompletionsClient", return_value=client_inst):
        await generate_digest(req)

    messages = client_inst.complete.await_args.args[0]
    assert messages[0] == {"role": "system", "content": "Mode prompt"}
    assert messages[1] == {"role": "system", "content": "Request prompt"}


@pytest.mark.asyncio
async def test_generate_digest_treats_empty_string_system_prompt_override_as_missing():
    from server.advisor_service import AdvisorDigestRequest, generate_digest

    chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="llama3.2",
        api_key_env="OPENAI_API_KEY",
        max_tokens=256,
        temperature=0.6,
        system_prompt="Mode prompt",
    )
    mode = SimpleNamespace(maximum_len=None)
    config = SimpleNamespace(
        get_default_mode=lambda: "sdxl-general",
        get_mode=lambda name: mode,
        resolve_chat_config=lambda name, overrides=None: chat_cfg,
    )
    client_inst = SimpleNamespace(complete=AsyncMock(return_value="digest text"))

    req = AdvisorDigestRequest(
        gallery_id="gal_1",
        evidence={"version": 1, "gallery_id": "gal_1", "items": []},
        system_prompt="",
    )

    with patch("server.advisor_service.get_mode_config", return_value=config), \
            patch("server.advisor_service.ChatCompletionsClient", return_value=client_inst):
        await generate_digest(req)

    messages = client_inst.complete.await_args.args[0]
    assert [m for m in messages if m["role"] == "system"] == [{"role": "system", "content": "Mode prompt"}]


@pytest.mark.asyncio
async def test_generate_digest_requires_mode_chat_config():
    from server.advisor_service import AdvisorDigestRequest, generate_digest

    mode = SimpleNamespace(maximum_len=None)
    config = SimpleNamespace(
        get_default_mode=lambda: "sdxl-general",
        get_mode=lambda name: mode,
        resolve_chat_config=lambda name, overrides=None: None,
    )

    req = AdvisorDigestRequest(
        gallery_id="gal_1",
        evidence={"version": 1, "gallery_id": "gal_1", "items": []},
    )

    with patch("server.advisor_service.get_mode_config", return_value=config):
        with pytest.raises(ValueError, match="advisor digest requires chat configuration"):
            await generate_digest(req)
