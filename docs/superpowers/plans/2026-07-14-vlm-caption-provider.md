# VLM Caption Provider (`openai_vlm`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagent-driven development is **forbidden** in this repo (AGENTS.md). Steps use checkbox (`- [ ]`) syntax for tracking.

**FP issue:** STABL-rylcqort
**Spec (authority):** `docs/superpowers/specs/2026-07-14-vlm-caption-provider-design.md`

**Goal:** First real `DescribeProvider` — caption tasks executed by a VLM behind an OpenAI-compatible `chat/completions` endpoint, opt-in per delegate via `provider: openai_vlm`.

**Architecture:** Config gains a delegate-level `provider` field (closed set, default `stub`) and an optional `options` mapping, both surviving every write path. A dedicated multimodal httpx client returns the full response dict; the provider maps it to one `text` observation + `raw_output` and raises on every failure (existing per-run isolation handles the rest). `build_providers` switches on `delegate.provider`.

**Tech Stack:** Python (httpx, FastAPI TestClient), existing `backends/analysis` package, `server/mode_config.py`.

## Global Constraints

- **Opt-in default:** `provider` omitted means `stub` — every existing config/test keeps today's behavior (spec: `conditioning.service: compel` discipline).
- **Round-trip persistence (spec, required):** `provider` + `options` survive `to_dict()`, `save_config()`, and bulk `PUT /api/modes`; defaults may be omitted on export (`provider` when `stub`, `options` when empty).
- **Layering:** `backends/analysis` never imports `server.*` — the provider takes plain constructor params and an injected asset-resolver callable; `server/analysis_routes.py` adapts config objects and the store.
- **Raw output verbatim:** `raw_output` = full completion response JSON, never restructured.
- **No retries, no new error codes:** provider raises; orchestrator maps to `analysis_run_failed` per run.
- **Server never fetches remote URLs:** `url` targets pass through verbatim to the VLM.
- **Unpatched-seam test (spec, required):** ≥1 endpoint integration case exercises the real `build_providers` switch with mocking at the httpx level only.
- Python env: `source /Users/darkbit1001/miniforge3/bin/activate base`, then `python -m pytest`.
- Commits reference STABL-rylcqort and state the next step.

---

### Task 1: Config — `provider` + `options` parsing, validation, round-trip persistence

**Files:**
- Modify: `server/mode_config.py` (`AnalysisDelegateConfig` ~line 75, `_parse_analysis_delegate_config` ~line 642, `save_config` delegate emission ~line 888, `to_dict()` delegate emission ~line 1007)
- Modify: `tests/test_analysis_mode_config.py`
- Modify: `tests/test_model_routes.py` (bulk-PUT regression)

**Interfaces:**
- Consumes: existing `AnalysisDelegateConfig`, `_parse_analysis_delegate_config`, `BASE_YAML`/`load()` fixtures in `tests/test_analysis_mode_config.py`.
- Produces (Task 5 relies on these): `AnalysisDelegateConfig.provider: str` (`"stub"` default), `AnalysisDelegateConfig.options: Dict[str, Any]` (empty default, keys validated against `max_tokens`/`temperature`/`timeout_s`/`system_prompt`); module constants `ANALYSIS_PROVIDERS = ("stub", "openai_vlm")`.

- [ ] **Step 1: Write failing parse/validation tests**

Append to `tests/test_analysis_mode_config.py` (reuse the existing `BASE_YAML` and `load()` helpers; `PROVIDER_YAML` below derives from `BASE_YAML`):

```python
PROVIDER_YAML = BASE_YAML.replace(
    "        model: qwen2.5-vl\n",
    "        model: qwen2.5-vl\n"
    "        provider: openai_vlm\n"
    "        options:\n"
    "          max_tokens: 256\n"
    "          temperature: 0.0\n"
    "          timeout_s: 90\n"
    "          system_prompt: \"Describe for a catalog.\"\n",
)


def test_delegate_provider_defaults_to_stub(tmp_path):
    cfg = load(tmp_path, BASE_YAML).config
    assert cfg.analysis_delegates["vlm_caption"].provider == "stub"
    assert cfg.analysis_delegates["vlm_caption"].options == {}


def test_delegate_provider_and_options_parse(tmp_path):
    cfg = load(tmp_path, PROVIDER_YAML).config
    d = cfg.analysis_delegates["vlm_caption"]
    assert d.provider == "openai_vlm"
    assert d.options == {
        "max_tokens": 256,
        "temperature": 0.0,
        "timeout_s": 90,
        "system_prompt": "Describe for a catalog.",
    }


def test_unknown_provider_fails_load(tmp_path):
    bad = BASE_YAML.replace(
        "        model: qwen2.5-vl\n",
        "        model: qwen2.5-vl\n        provider: nonsense\n",
    )
    with pytest.raises(ValueError, match="provider"):
        load(tmp_path, bad)


def test_openai_vlm_on_non_caption_kind_fails_load(tmp_path):
    bad = BASE_YAML.replace(
        "        model: yolo11x\n",
        "        model: yolo11x\n        provider: openai_vlm\n",
    )
    with pytest.raises(ValueError, match="openai_vlm"):
        load(tmp_path, bad)


@pytest.mark.parametrize("options_yaml, match", [
    ("          bogus_key: 1\n", "bogus_key"),
    ("          max_tokens: 0\n", "max_tokens"),
    ("          max_tokens: not-a-number\n", "max_tokens"),
    ("          temperature: -1\n", "temperature"),
    ("          timeout_s: 0\n", "timeout_s"),
    ("          system_prompt: \"\"\n", "system_prompt"),
])
def test_bad_options_fail_load(tmp_path, options_yaml, match):
    bad = BASE_YAML.replace(
        "        model: qwen2.5-vl\n",
        "        model: qwen2.5-vl\n        options:\n" + options_yaml,
    )
    with pytest.raises(ValueError, match=match):
        load(tmp_path, bad)


def test_options_accepted_without_provider_field(tmp_path):
    # options is a delegate-tuning surface, not an openai_vlm exclusive.
    ok = BASE_YAML.replace(
        "        model: qwen2.5-vl\n",
        "        model: qwen2.5-vl\n        options:\n          max_tokens: 128\n",
    )
    cfg = load(tmp_path, ok).config
    assert cfg.analysis_delegates["vlm_caption"].options == {"max_tokens": 128}
    assert cfg.analysis_delegates["vlm_caption"].provider == "stub"


def test_provider_and_options_survive_export_save_reload(tmp_path):
    # Spec: round-trip persistence is definition-of-done.
    mgr = load(tmp_path, PROVIDER_YAML)
    exported = mgr.to_dict()
    d = exported["analysis_delegates"]["vlm_caption"]
    assert d["provider"] == "openai_vlm"
    assert d["options"]["max_tokens"] == 256
    # Default-valued delegates omit the fields (clean exports).
    assert "provider" not in exported["analysis_delegates"]["yolo_detect"]
    assert "options" not in exported["analysis_delegates"]["yolo_detect"]

    mgr.save_config(exported)
    reloaded = mgr.config.analysis_delegates["vlm_caption"]
    assert reloaded.provider == "openai_vlm"
    assert reloaded.options["system_prompt"] == "Describe for a catalog."
```

Add `import pytest` to the file's imports if not present.

- [ ] **Step 2: Run tests, verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_mode_config.py -k "provider or options" -v`
Expected: FAIL — `AttributeError`/`AssertionError` (no `provider` attribute; unknown keys silently ignored)

- [ ] **Step 3: Implement config support**

In `server/mode_config.py`:

Extend the dataclass (~line 75):

```python
ANALYSIS_PROVIDERS = ("stub", "openai_vlm")

# Known analysis delegate option keys -> validator returning the coerced
# value or raising ValueError.
_ANALYSIS_OPTION_KEYS = ("max_tokens", "temperature", "timeout_s", "system_prompt")


@dataclass
class AnalysisDelegateConfig:
    """Named analyzer backend: connection + kind capability + model."""
    name: str
    connection: str  # key into analysis_connections
    kind: str        # closed TaskKind value; capability declaration
    model: str
    provider: str = "stub"  # closed ANALYSIS_PROVIDERS value
    options: Dict[str, Any] = field(default_factory=dict)
```

In `_parse_analysis_delegate_config` (~line 642), before the final `return`:

```python
        provider = (raw.get("provider") or "stub").strip()
        if provider not in ANALYSIS_PROVIDERS:
            raise ValueError(
                f"Analysis delegate '{name}' has invalid provider '{provider}' "
                f"(expected one of {sorted(ANALYSIS_PROVIDERS)})"
            )
        if provider == "openai_vlm" and kind != "caption":
            raise ValueError(
                f"Analysis delegate '{name}' sets provider 'openai_vlm' but kind "
                f"'{kind}' — openai_vlm supports kind 'caption' only"
            )
        options = self._parse_analysis_options(name, raw.get("options"))
        return AnalysisDelegateConfig(
            name=name, connection=connection, kind=kind, model=model,
            provider=provider, options=options,
        )
```

(Replace the existing `return AnalysisDelegateConfig(...)` line.)

Add the option parser as a method beside it:

```python
    def _parse_analysis_options(self, name: str, raw: Any) -> Dict[str, Any]:
        """Validate a delegate options mapping. Unknown keys fail load."""
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"Analysis delegate '{name}' options must be a mapping")
        options: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in _ANALYSIS_OPTION_KEYS:
                raise ValueError(
                    f"Analysis delegate '{name}' has unknown option '{key}' "
                    f"(expected one of {sorted(_ANALYSIS_OPTION_KEYS)})"
                )
            if key == "max_tokens":
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"Analysis delegate '{name}' option max_tokens must be an int > 0")
            elif key == "temperature":
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"Analysis delegate '{name}' option temperature must be a number >= 0")
            elif key == "timeout_s":
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"Analysis delegate '{name}' option timeout_s must be a number > 0")
            elif key == "system_prompt":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Analysis delegate '{name}' option system_prompt must be a non-empty string")
            options[key] = value
        return options
```

Extend **both** delegate serializers with the same emission helper. Add once:

```python
def _analysis_delegate_to_dict(delegate: AnalysisDelegateConfig) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "connection": delegate.connection,
        "kind": delegate.kind,
        "model": delegate.model,
    }
    if delegate.provider != "stub":
        d["provider"] = delegate.provider
    if delegate.options:
        d["options"] = dict(delegate.options)
    return d
```

Then in `save_config` (~line 888) replace the inline dict with
`yaml_data["analysis_delegates"][name] = _analysis_delegate_to_dict(d)`, and
in `to_dict()` (~line 1007) replace the delegate comprehension value with
`_analysis_delegate_to_dict(delegate)`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_analysis_mode_config.py -v`
Expected: all PASS (new + pre-existing)

- [ ] **Step 5: Write failing bulk-PUT regression, then verify it passes**

The bulk path backfills omitted analysis sections from `to_dict()` and
re-saves through `save_config` (`server/model_routes.py:447-449`), so it
inherits the fix — but the spec names this regression as definition-of-done.
Append to `tests/test_model_routes.py`, following the existing
`test_save_all_modes_preserves_analysis_sections_when_omitted` pattern
(read it first and mirror its mocking of `get_mode_config`/`save_config`):

```python
async def test_save_all_modes_preserves_delegate_provider_and_options_when_omitted():
    # Spec round-trip requirement: a bulk PUT that omits analysis sections
    # must not strip provider/options from existing delegates.
    existing = {
        "analysis_connections": {"local_vlm": {"endpoint": "http://x/v1", "api_key_env": "OPENAI_API_KEY"}},
        "analysis_delegates": {
            "vlm_caption": {
                "connection": "local_vlm", "kind": "caption", "model": "qwen2.5-vl",
                "provider": "openai_vlm", "options": {"max_tokens": 256},
            }
        },
        "analysis_profiles": {"default": {"task_routes": {"caption": "vlm_caption"}}},
    }
    config = Mock()
    config.to_dict.return_value = {**existing, "modes": {}, "resolution_sets": {}}
    saved = {}
    config.save_config.side_effect = lambda data: saved.update(data)

    request = ModesBulkSaveRequest(modes={}, default_mode=None)

    with patch("server.model_routes.get_mode_config", return_value=config):
        await model_routes.save_all_modes(request)

    delegate = saved["analysis_delegates"]["vlm_caption"]
    assert delegate["provider"] == "openai_vlm"
    assert delegate["options"] == {"max_tokens": 256}
```

Adjust the `ModesBulkSaveRequest` construction and `save_all_modes` call to
match the existing test's exact invocation (fields/awaiting may differ —
mirror `test_save_all_modes_preserves_analysis_sections_when_omitted`
line-for-line, changing only the delegate payload and assertions).

Run: `python -m pytest tests/test_model_routes.py -k "provider_and_options" -v`
Expected: PASS (the backfill carries whatever `to_dict()` emits). If it
FAILS, the bulk path strips fields — fix in `save_all_modes`, not the test.

- [ ] **Step 6: Full config + routes suites, then commit**

Run: `python -m pytest tests/test_analysis_mode_config.py tests/test_model_routes.py tests/test_analysis_routes.py -v`
Expected: all PASS

```bash
git add server/mode_config.py tests/test_analysis_mode_config.py tests/test_model_routes.py
git commit -m "feat(analysis): delegate provider/options config with round-trip persistence (STABL-rylcqort) — next: vlm_client"
```

---

### Task 2: Multimodal VLM client

**Files:**
- Create: `backends/analysis/vlm_client.py`
- Create: `tests/test_analysis_vlm_client.py`

**Interfaces:**
- Consumes: nothing project-internal (httpx only).
- Produces (Task 4 relies on this): `VLMChatClient(endpoint: str, api_key_env: str, timeout_s: float, transport: Optional[httpx.BaseTransport] = None)` with `async def complete(self, *, model: str, messages: list, max_tokens: int, temperature: float) -> Dict[str, Any]` returning the parsed full response dict.

- [ ] **Step 1: Write failing client tests**

`tests/test_analysis_vlm_client.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_analysis_vlm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.analysis.vlm_client'`

- [ ] **Step 3: Implement the client**

`backends/analysis/vlm_client.py`:

```python
"""Minimal multimodal OpenAI-compatible chat/completions client.

Deliberately separate from backends/chat_client.py: that client is typed
for text-only messages and returns only the content string; this one
accepts image_url content parts and returns the parsed full response dict
(the describe contract preserves raw provider output verbatim).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


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

    def _url(self) -> str:
        return f"{self._endpoint.rstrip('/')}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self._api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

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
                self._url(), json=payload, headers=self._headers(),
                timeout=self._timeout_s,
            )
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_analysis_vlm_client.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add backends/analysis/vlm_client.py tests/test_analysis_vlm_client.py
git commit -m "feat(analysis): multimodal VLM chat client returning full response (STABL-rylcqort) — next: caption provider"
```

---

### Task 3: Caption message assembly (pure functions)

**Files:**
- Create: `backends/analysis/vlm_caption.py` (assembly half only — no provider class yet)
- Create: `tests/test_analysis_vlm_caption.py`

**Interfaces:**
- Consumes: existing `ProviderRun`/`DescribeTarget`/`CaptionParams` from the package. No HTTP, no async — this task is pure.
- Produces (Task 4 relies on these): module constants `DEFAULT_MAX_TOKENS = 512`, `DEFAULT_TEMPERATURE = 0.2`, `DEFAULT_TIMEOUT_S = 60.0`, `DEFAULT_SYSTEM_PROMPT`; `AssetResolver = Callable[[str], Tuple[bytes, str]]`; `build_image_part(target: DescribeTarget, asset_resolver: AssetResolver) -> Dict[str, Any]`; `build_caption_messages(provider_run: ProviderRun, options: Mapping[str, Any], asset_resolver: AssetResolver) -> List[Dict[str, Any]]`.

- [ ] **Step 1: Write failing assembly tests**

`tests/test_analysis_vlm_caption.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_analysis_vlm_caption.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.analysis.vlm_caption'`

- [ ] **Step 3: Implement the assembly functions**

`backends/analysis/vlm_caption.py`:

```python
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
```

Check the exact `DescribeTarget`/`CaptionParams` field names against
`backends/analysis/contracts.py` before writing — `url`/`asset_ref`/`prompt`
in the code above must match the dataclasses, not be invented.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_analysis_vlm_caption.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add backends/analysis/vlm_caption.py tests/test_analysis_vlm_caption.py
git commit -m "feat(analysis): caption message assembly — image parts + prompting (STABL-rylcqort) — next: provider execution"
```

---

### Task 4: `OpenAIVLMCaptionProvider` execution

**Files:**
- Modify: `backends/analysis/vlm_caption.py` (add the provider class)
- Modify: `tests/test_analysis_vlm_caption.py` (append provider tests)
- Modify: `backends/analysis/__init__.py` (export `OpenAIVLMCaptionProvider`)

**Interfaces:**
- Consumes: Task 2's `VLMChatClient`, Task 3's assembly functions/constants; existing `ProviderResult`/`TextObservation`/`DescribeObservation`/`TaskKind`.
- Produces (Task 5 relies on this): `OpenAIVLMCaptionProvider(endpoint: str, api_key_env: str, model: str, options: Mapping[str, Any], asset_resolver: AssetResolver, transport: Optional[httpx.BaseTransport] = None)` implementing the `DescribeProvider` protocol.

- [ ] **Step 1: Write failing provider-execution tests**

Append to `tests/test_analysis_vlm_caption.py` (extend the import block with `json`, `httpx`, `DetectParams`, and `OpenAIVLMCaptionProvider`):

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_analysis_vlm_caption.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'OpenAIVLMCaptionProvider'`; Task 3's assembly tests still PASS

- [ ] **Step 3: Implement the provider class**

Append to `backends/analysis/vlm_caption.py` (extend the import block with `Optional`, `httpx`, `from .contracts import DescribeObservation, DescribeTask, TaskKind, TextObservation`, `from .providers import ProviderResult`, `from .vlm_client import VLMChatClient`):

```python
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
```

Export it: in `backends/analysis/__init__.py`, add
`from .vlm_caption import OpenAIVLMCaptionProvider` beside the providers
import and `"OpenAIVLMCaptionProvider"` to `__all__`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_analysis_vlm_caption.py tests/test_analysis_contracts.py tests/test_analysis_orchestrator.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add backends/analysis/vlm_caption.py backends/analysis/__init__.py tests/test_analysis_vlm_caption.py
git commit -m "feat(analysis): OpenAIVLMCaptionProvider execution — first real describe provider (STABL-rylcqort) — next: build_providers switch"
```

---

### Task 5: `build_providers` switch + endpoint integration

**Files:**
- Modify: `server/analysis_routes.py` (`build_providers` ~line 28 and its call site ~line 89)
- Modify: `tests/test_analysis_routes.py`

**Interfaces:**
- Consumes: Task 1's `AnalysisDelegateConfig.provider/.options`, Task 4's `OpenAIVLMCaptionProvider`, existing `get_store()` from `server.asset_store`.
- Produces: `build_providers(profile, delegates, connections)` — the completed real-provider seam.

- [ ] **Step 1: Write failing integration tests**

Append to `tests/test_analysis_routes.py` (reuse `BASE_YAML`, `_manager`, `_app`, `_request_body`; note the mock is injected by monkeypatching `httpx.AsyncClient` **in the vlm_client module namespace** — the chat_client test precedent — so `build_providers` itself runs unpatched, per the spec):

```python
VLM_PROVIDER_YAML = BASE_YAML.replace(
    "        model: qwen2.5-vl\n",
    "        model: qwen2.5-vl\n        provider: openai_vlm\n",
)

VLM_RESPONSE = {
    "id": "cmpl-1",
    "choices": [{"message": {"role": "assistant", "content": "a real caption"}}],
}


class _FakeVLMClient:
    """Stands in for httpx.AsyncClient inside backends.analysis.vlm_client.

    fail_on: 1-based call indices that return HTTP 500.
    """
    calls = 0
    fail_on: set = set()

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None, timeout=None):
        type(self).calls += 1
        import httpx
        request = httpx.Request("POST", url)
        if type(self).calls in type(self).fail_on:
            return httpx.Response(500, text="vlm down", request=request)
        return httpx.Response(200, json=VLM_RESPONSE, request=request)


@pytest.fixture
def fake_vlm(monkeypatch):
    _FakeVLMClient.calls = 0
    _FakeVLMClient.fail_on = set()
    monkeypatch.setattr("backends.analysis.vlm_client.httpx.AsyncClient", _FakeVLMClient)
    return _FakeVLMClient


def test_describe_openai_vlm_returns_real_caption_via_unpatched_factory(tmp_path, fake_vlm):
    # build_providers is NOT patched: the delegate-config -> provider-class
    # selection path is what's under test (spec requirement).
    body = {
        "targets": [{"id": "t1", "url": "http://example.com/a.png"}],
        "tasks": [{"id": "caption", "kind": "caption", "caption": {}}],
    }
    res = _post(_manager(tmp_path, VLM_PROVIDER_YAML, "vlm"), body)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["observations"][0]["text"]["content"] == "a real caption"
    run = data["runs"][0]
    assert run["delegate"] == "vlm_caption" and run["status"] == "succeeded"
    assert run["raw_output"] == VLM_RESPONSE


def test_describe_openai_vlm_one_failed_call_yields_partial(tmp_path, fake_vlm):
    fake_vlm.fail_on = {2}
    body = {
        "targets": [
            {"id": "t1", "url": "http://example.com/a.png"},
            {"id": "t2", "url": "http://example.com/b.png"},
        ],
        "tasks": [{"id": "caption", "kind": "caption", "caption": {}}],
    }
    res = _post(_manager(tmp_path, VLM_PROVIDER_YAML, "vlmpartial"), body)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "partial"
    statuses = {r["target_id"]: r["status"] for r in data["runs"]}
    assert sorted(statuses.values()) == ["failed", "succeeded"]
    failed = [r for r in data["runs"] if r["status"] == "failed"][0]
    assert failed["error"]["code"] == "analysis_run_failed"


def test_describe_stub_default_unchanged_with_vlm_available(tmp_path):
    # provider omitted -> stub, even with the VLM code importable: the
    # back-compat guarantee. (All pre-existing stub tests also still run.)
    res = _post(_manager(tmp_path, BASE_YAML, "stubdefault"), _request_body())
    assert res.status_code == 200
    assert res.json()["observations"][0]["text"]["content"].startswith("stub:")
```

Add `import pytest` to the test file imports if missing. Note the two-call
partial test relies on `asyncio.gather` over sequential fake-call counting —
if run-order/call-index coupling proves flaky, key the failure on the target
URL in `post()` instead (inspect `json["messages"]`).

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_analysis_routes.py -k "vlm or stub_default" -v`
Expected: FAIL — captions come back `stub:caption` (factory ignores `provider`)

- [ ] **Step 3: Implement the switch**

In `server/analysis_routes.py`, replace `build_providers` and its call:

```python
from backends.analysis import (
    AnalysisOrchestrator,
    AnalysisValidationError,
    DescribeProvider,
    OpenAIVLMCaptionProvider,
    StubProvider,
    parse_describe_request,
    response_to_dict,
)
from server.asset_store import get_store
from server.mode_config import (
    AnalysisConnectionConfig,
    AnalysisDelegateConfig,
    AnalysisProfileConfig,
    ModeConfigManager,
    get_mode_config,
)


def _resolve_asset(ref: str):
    """Adapt the asset store to the provider's (bytes, media_type) contract."""
    entry = get_store().resolve(ref)
    return entry.data, entry.metadata.get("media_type", "image/png")


def build_providers(
    profile: AnalysisProfileConfig,
    delegates: Mapping[str, AnalysisDelegateConfig],
    connections: Mapping[str, AnalysisConnectionConfig],
) -> Dict[str, DescribeProvider]:
    """Provider factory: selects the implementation per delegate.provider."""
    providers: Dict[str, DescribeProvider] = {}
    for delegate_name in profile.task_routes.values():
        delegate = delegates[delegate_name]
        if delegate.provider == "openai_vlm":
            connection = connections[delegate.connection]
            providers[delegate_name] = OpenAIVLMCaptionProvider(
                endpoint=connection.endpoint,
                api_key_env=connection.api_key_env,
                model=delegate.model,
                options=delegate.options,
                asset_resolver=_resolve_asset,
            )
        else:
            providers[delegate_name] = StubProvider(kind=delegate.kind)
    return providers
```

And the call site inside `describe()`:

```python
        orchestrator = AnalysisOrchestrator(
            profile.task_routes,
            build_providers(
                profile,
                manager.config.analysis_delegates,
                manager.config.analysis_connections,
            ),
        )
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_analysis_routes.py -v`
Expected: all PASS — new VLM cases and every pre-existing stub test

- [ ] **Step 5: Commit**

```bash
git add server/analysis_routes.py tests/test_analysis_routes.py
git commit -m "feat(analysis): build_providers selects openai_vlm per delegate config (STABL-rylcqort) — next: full-stack verification"
```

---

## Final Verification

- [ ] Python: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_routes.py tests/test_analysis_contracts.py tests/test_analysis_orchestrator.py tests/test_analysis_mode_config.py tests/test_analysis_vlm_client.py tests/test_analysis_vlm_caption.py tests/test_model_routes.py -v` — all green.
- [ ] `python -m py_compile server/mode_config.py server/analysis_routes.py backends/analysis/vlm_client.py backends/analysis/vlm_caption.py` — clean.
- [ ] `drift check` — report (do not relink) staleness on `server/mode_config.py` / `server/model_routes.py` bound docs; pre-existing deferral applies.
- [ ] FP comment on STABL-rylcqort per stopping-point policy; note that live `node2.lan` verification is deferred to the human (spec).
