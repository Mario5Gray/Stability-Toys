"""Tests for VLM caption message assembly and provider."""
import base64
import json

import httpx
import pytest

from backends.analysis import (
    CaptionParams,
    DescribeTarget,
    DescribeTask,
    DetectParams,
    TaskKind,
)
from backends.analysis.orchestrator import RunPlan
from backends.analysis.providers import ProviderRun
from backends.analysis.vlm_caption import (
    DEFAULT_SYSTEM_PROMPT,
    OpenAIVLMCaptionProvider,
    build_caption_messages,
    build_image_part,
)


def _resolver(ref):
    return b"png-bytes", "image/png"


def _run(target, prompt=None):
    task = DescribeTask(
        id="caption", kind=TaskKind.CAPTION,
        caption=CaptionParams(prompt=prompt),
    )
    return ProviderRun(
        plan=RunPlan(task_id="caption", target_id=target.id, delegate="vlm_caption"),
        task=task, target=target,
    )


def _url_target():
    return DescribeTarget(id="t1", url="http://images/a.png")


def _ref_target():
    return DescribeTarget(id="t1", asset_ref="Rabc123")


def test_image_part_url_target_passes_url_through():
    part = build_image_part(_url_target(), _resolver)
    assert part == {"type": "image_url", "image_url": {"url": "http://images/a.png"}}


def test_image_part_asset_ref_embeds_base64_data_uri():
    part = build_image_part(_ref_target(), _resolver)
    expected = "data:image/png;base64," + base64.b64encode(b"png-bytes").decode()
    assert part == {"type": "image_url", "image_url": {"url": expected}}


def test_image_part_resolver_failure_propagates():
    def failing_resolver(ref):
        raise KeyError(f"no such ref {ref}")
    with pytest.raises(KeyError):
        build_image_part(_ref_target(), failing_resolver)


def test_messages_shape_system_then_user_with_image():
    messages = build_caption_messages(_run(_url_target()), {}, _resolver)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert messages[1]["content"] == [
        {"type": "image_url", "image_url": {"url": "http://images/a.png"}},
    ]


def test_messages_include_caller_prompt_as_text_part_only_when_set():
    messages = build_caption_messages(
        _run(_url_target(), prompt="focus on lighting"), {}, _resolver,
    )
    assert {"type": "text", "text": "focus on lighting"} in messages[1]["content"]

    messages = build_caption_messages(_run(_url_target()), {}, _resolver)
    assert all(p["type"] != "text" for p in messages[1]["content"])


def test_messages_system_prompt_overridable_via_options():
    messages = build_caption_messages(
        _run(_url_target()), {"system_prompt": "catalog style"}, _resolver,
    )
    assert messages[0]["content"] == "catalog style"


RESPONSE = {
    "id": "cmpl-1",
    "choices": [{"message": {"role": "assistant", "content": "a red bicycle"}}],
    "usage": {"total_tokens": 42},
}


def _provider(capture=None, response=None, status=200, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["payload"] = json.loads(request.content)
        return httpx.Response(status, json=response if response is not None else RESPONSE)
    defaults = dict(
        endpoint="http://vlm.lan:8080/v1",
        api_key_env="TEST_VLM_KEY",
        model="qwen2.5-vl",
        options={},
        asset_resolver=_resolver,
        transport=httpx.MockTransport(handler),
    )
    defaults.update(kwargs)
    return OpenAIVLMCaptionProvider(**defaults)


async def test_run_maps_response_to_text_observation_and_raw_output():
    capture = {}
    result = await _provider(capture).run(_run(_url_target()))
    obs = result.observations
    assert len(obs) == 1 and obs[0].kind == "text"
    assert obs[0].text.content == "a red bicycle"
    assert obs[0].task_id == "caption" and obs[0].target_id == "t1"
    assert result.raw_output == RESPONSE
    # The wire payload uses the Task 3 assembly output verbatim.
    assert capture["payload"]["messages"] == build_caption_messages(
        _run(_url_target()), {}, _resolver,
    )


async def test_run_applies_default_and_overridden_options():
    capture = {}
    await _provider(capture).run(_run(_url_target()))
    assert capture["payload"]["max_tokens"] == 512
    assert capture["payload"]["temperature"] == 0.2
    assert capture["payload"]["model"] == "qwen2.5-vl"

    await _provider(
        capture, options={"max_tokens": 64, "temperature": 0.0},
    ).run(_run(_url_target()))
    assert capture["payload"]["max_tokens"] == 64
    assert capture["payload"]["temperature"] == 0.0


async def test_run_raises_on_http_error():
    with pytest.raises(httpx.HTTPStatusError):
        await _provider(status=500, response={}).run(_run(_url_target()))


@pytest.mark.parametrize("bad_response", [
    {},                                            # no choices
    {"choices": []},                               # empty choices
    {"choices": [{"message": {"content": ""}}]},   # empty content
    {"choices": [{"message": {}}]},                # missing content
])
async def test_run_raises_on_missing_or_empty_content(bad_response):
    with pytest.raises(ValueError):
        await _provider(response=bad_response).run(_run(_url_target()))


async def test_run_raises_when_asset_resolver_fails():
    def failing_resolver(ref):
        raise KeyError(f"no such ref {ref}")
    with pytest.raises(KeyError):
        await _provider(asset_resolver=failing_resolver).run(_run(_ref_target()))


def test_supports_caption_only():
    provider = _provider()
    assert provider.supports(DescribeTask(id="c", kind=TaskKind.CAPTION, caption=CaptionParams()))
    assert not provider.supports(DescribeTask(id="d", kind=TaskKind.DETECT, detect=DetectParams()))
