"""Tests for VLM caption message assembly and provider."""
import base64

import pytest

from backends.analysis import (
    CaptionParams,
    DescribeTarget,
    DescribeTask,
    TaskKind,
)
from backends.analysis.orchestrator import RunPlan
from backends.analysis.providers import ProviderRun
from backends.analysis.vlm_caption import (
    DEFAULT_SYSTEM_PROMPT,
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
