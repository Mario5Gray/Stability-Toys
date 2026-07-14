# Describe Transport + `st describe` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagent-driven development is **forbidden** in this repo (AGENTS.md). Steps use checkbox (`- [ ]`) syntax for tracking.

**FP issue:** STABL-ucomsfel
**Spec (authority):** `docs/superpowers/specs/2026-07-13-describe-transport-cli-design.md`

**Goal:** Wire the merged describe/analysis contracts end-to-end: `POST /v1/describe` on the server, `stclient.Describe()`, and an `st describe` CLI verb — all against `StubProvider`.

**Architecture:** The endpoint builds the orchestrator per request from live `get_mode_config()` state (no lifespan snapshot — analysis policy follows the same reload discipline as the rest of the mode system). A provider factory keyed by delegate config is the seam real providers plug into later. The Go side adds one typed HTTP call and a CLI verb with frozen ordering, output, and exit-code contracts.

**Tech Stack:** FastAPI (`server/analysis_routes.py`), existing `backends/analysis` package, Go `pkg/stclient` + cobra `cmd/st`.

## Global Constraints

- **Ordering is contract** (spec "Ordering Determinism"): CLI positional args → `targets` in arg order with IDs `t1..tN` (1-based); task flags → canonical `TaskKind` order `caption, detect, ocr, pose, embed` with task id = kind string; run expansion tasks-major; response arrays in run order. Ordered collections only on these paths.
- **Frozen on merge:** `st describe --json` emits the wire `DescribeResponse` verbatim (indented, single terminal object); exit codes 0 = `ok`, 1 = transport/usage/validation error, 2 = `failed`, 3 = `partial`.
- **Failure rendering required:** `partial`/`failed` render every non-`succeeded` run's `task_id`/`target_id`/`delegate`/`status`/error code/message to stderr. Server non-2xx renders `code + message`; pure transport failures render a message only (no synthetic code).
- **No CLI concerns in `stclient`** (flags, stderr, cobra) — it must also serve the future MCP server.
- Error bodies on non-2xx: `{"error": {"code": "analysis_*", "message": "..."}}`; new additive codes `analysis_mode_not_found`, `analysis_internal`.
- Python env: `source /Users/darkbit1001/miniforge3/bin/activate base`, then `python -m pytest`.
- Go: run from `cli/go` (`go test ./...`).
- Commits reference STABL-ucomsfel and state the next step.

---

### Task 1: `POST /v1/describe` endpoint

**Files:**
- Create: `server/analysis_routes.py`
- Create: `tests/test_analysis_routes.py`
- Modify: `server/lcm_sr_server.py` (mount router, near the `include_router` block at ~line 877)

**Interfaces:**
- Consumes: `backends.analysis` exports (`parse_describe_request`, `response_to_dict`, `AnalysisValidationError`, `AnalysisOrchestrator`, `StubProvider`, `DescribeProvider`), `server.mode_config.get_mode_config()` (returns `ModeConfigManager`; `.config.modes[name].analysis_profile`, `.config.analysis_profiles[name].task_routes`, `.config.analysis_delegates[name].kind`, `.get_default_mode()`).
- Produces: `POST /v1/describe` JSON endpoint; `build_providers(profile, delegates) -> Dict[str, DescribeProvider]` (the real-provider seam); `router` for mounting.

- [ ] **Step 1: Write failing endpoint tests**

`tests/test_analysis_routes.py`:

```python
"""Tests for POST /v1/describe."""
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.analysis_routes import router as analysis_router
from server.mode_config import ModeConfigManager

BASE_YAML = textwrap.dedent("""\
    model_root: /tmp/models
    lora_root: /tmp/loras
    default_mode: SDXL
    resolution_sets:
      default:
        - size: 512x512
          aspect_ratio: "1:1"
    analysis_connections:
      local_vlm:
        endpoint: "http://node2.lan:8080/v1"
      local_detector:
        endpoint: "http://node2.lan:8090"
    analysis_delegates:
      vlm_caption:
        connection: local_vlm
        kind: caption
        model: qwen2.5-vl
      yolo_detect:
        connection: local_detector
        kind: detect
        model: yolo11x
    analysis_profiles:
      default:
        task_routes:
          caption: vlm_caption
          detect: yolo_detect
    modes:
      SDXL:
        model: sdxl/model.safetensors
        analysis_profile: default
""")

# Same base config, but the profile only routes caption — detect tasks
# become skipped runs (partial responses).
CAPTION_ONLY_YAML = BASE_YAML.replace(
    "        task_routes:\n          caption: vlm_caption\n          detect: yolo_detect\n",
    "        task_routes:\n          caption: vlm_caption\n",
)

# No analysis_profile on the mode at all.
NO_PROFILE_YAML = BASE_YAML.replace("        analysis_profile: default\n", "")


def _manager(tmp_path, yaml_text, subdir):
    d = tmp_path / subdir
    d.mkdir()
    (d / "modes.yml").write_text(yaml_text)
    return ModeConfigManager(str(d))


def _app():
    app = FastAPI()
    app.include_router(analysis_router)
    app.state.generation_runtime = SimpleNamespace(get_current_mode=lambda: "SDXL")
    return app


def _request_body(tasks=None):
    return {
        "targets": [
            {"id": "t1", "url": "http://example.com/a.png"},
            {"id": "t2", "url": "http://example.com/b.png"},
        ],
        "tasks": tasks or [
            {"id": "caption", "kind": "caption", "caption": {}},
            {"id": "detect", "kind": "detect", "detect": {}},
        ],
    }


def _post(mgr, body):
    with patch("server.analysis_routes.get_mode_config", return_value=mgr):
        with TestClient(_app()) as client:
            return client.post("/v1/describe", json=body)


def test_describe_happy_path_pins_run_order(tmp_path):
    res = _post(_manager(tmp_path, BASE_YAML, "base"), _request_body())
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    # Tasks-major expansion order is contract: caption t1,t2 then detect t1,t2.
    assert [(r["task_id"], r["target_id"]) for r in data["runs"]] == [
        ("caption", "t1"), ("caption", "t2"),
        ("detect", "t1"), ("detect", "t2"),
    ]
    assert all(r["status"] == "succeeded" for r in data["runs"])
    # StubProvider emits one text observation per run, in run order.
    assert [(o["task_id"], o["target_id"]) for o in data["observations"]] == [
        ("caption", "t1"), ("caption", "t2"),
        ("detect", "t1"), ("detect", "t2"),
    ]


def test_describe_partial_when_kind_unrouted(tmp_path):
    res = _post(_manager(tmp_path, CAPTION_ONLY_YAML, "cap"), _request_body())
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "partial"
    detect_runs = [r for r in data["runs"] if r["task_id"] == "detect"]
    assert all(r["status"] == "skipped" for r in detect_runs)
    assert all(r["error"]["code"] == "analysis_no_supported_delegate" for r in detect_runs)


def test_describe_malformed_json_body_is_analysis_invalid_request(tmp_path):
    mgr = _manager(tmp_path, BASE_YAML, "base")
    with patch("server.analysis_routes.get_mode_config", return_value=mgr):
        with TestClient(_app()) as client:
            res = client.post(
                "/v1/describe", content=b"not json",
                headers={"content-type": "application/json"},
            )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_invalid_request"


def test_describe_parse_error_returns_code(tmp_path):
    res = _post(_manager(tmp_path, BASE_YAML, "base"), {"targets": [], "tasks": []})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_invalid_request"


def test_describe_binding_error_returns_code(tmp_path):
    body = _request_body(tasks=[
        {"id": "caption", "kind": "caption", "caption": {}, "target_ids": ["nope"]},
    ])
    res = _post(_manager(tmp_path, BASE_YAML, "base"), body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_target_binding_invalid"


def test_describe_unknown_request_mode_is_mode_not_found(tmp_path):
    body = _request_body()
    body["mode"] = "NOPE"
    res = _post(_manager(tmp_path, BASE_YAML, "base"), body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_mode_not_found"


def test_describe_mode_without_profile_is_profile_not_found(tmp_path):
    res = _post(_manager(tmp_path, NO_PROFILE_YAML, "noprof"), _request_body())
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_profile_not_found"


def test_describe_reflects_reloaded_config_without_restart(tmp_path):
    # Request-time construction is contract: swapping the manager between
    # requests (as SIGHUP/file-watcher/POST /api/modes/reload do) must be
    # visible on the very next request.
    managers = {"current": _manager(tmp_path, BASE_YAML, "base")}
    with patch("server.analysis_routes.get_mode_config",
               side_effect=lambda: managers["current"]):
        with TestClient(_app()) as client:
            first = client.post("/v1/describe", json=_request_body())
            managers["current"] = _manager(tmp_path, CAPTION_ONLY_YAML, "cap")
            second = client.post("/v1/describe", json=_request_body())
    assert first.json()["status"] == "ok"
    assert second.json()["status"] == "partial"


def test_describe_unexpected_exception_is_analysis_internal(tmp_path):
    mgr = _manager(tmp_path, BASE_YAML, "base")
    with patch("server.analysis_routes.get_mode_config", return_value=mgr), \
            patch("server.analysis_routes.build_providers",
                  side_effect=RuntimeError("boom")):
        with TestClient(_app(), raise_server_exceptions=False) as client:
            res = client.post("/v1/describe", json=_request_body())
    assert res.status_code == 500
    assert res.json()["error"]["code"] == "analysis_internal"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.analysis_routes'`

- [ ] **Step 3: Implement `server/analysis_routes.py`**

```python
"""
Describe/analysis HTTP routes.

The orchestrator is built per request from live mode-config state — never a
lifespan snapshot — so SIGHUP / file-watcher / POST /api/modes/reload changes
to analysis policy take effect on the next request (spec: Ordering
Determinism + lifecycle sections of
docs/superpowers/specs/2026-07-13-describe-transport-cli-design.md).
"""
import logging
from typing import Dict, Mapping

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backends.analysis import (
    AnalysisOrchestrator,
    AnalysisValidationError,
    DescribeProvider,
    StubProvider,
    parse_describe_request,
    response_to_dict,
)
from server.mode_config import (
    AnalysisDelegateConfig,
    AnalysisProfileConfig,
    ModeConfigManager,
    get_mode_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


def build_providers(
    profile: AnalysisProfileConfig,
    delegates: Mapping[str, AnalysisDelegateConfig],
) -> Dict[str, DescribeProvider]:
    """Provider factory: the seam real providers plug into later.

    v1 yields a StubProvider per routed delegate; stubs are stateless so
    per-request construction is free. Real providers may cache connections
    keyed on config generation behind this same function.
    """
    return {
        delegate_name: StubProvider(kind=delegates[delegate_name].kind)
        for delegate_name in profile.task_routes.values()
    }


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _active_mode_name(request: Request, manager: ModeConfigManager) -> str:
    runtime = getattr(request.app.state, "generation_runtime", None)
    if runtime is not None:
        current = runtime.get_current_mode()
        if current:
            return current
    return manager.get_default_mode()


@router.post("/v1/describe")
async def describe(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return _error(400, "analysis_invalid_request", "request body is not valid JSON")

    try:
        describe_request = parse_describe_request(payload)
    except AnalysisValidationError as exc:
        return _error(400, exc.code, exc.message)

    manager = get_mode_config()
    mode_name = describe_request.mode or _active_mode_name(request, manager)
    mode = manager.config.modes.get(mode_name)
    if mode is None:
        return _error(400, "analysis_mode_not_found", f"unknown mode '{mode_name}'")
    if not mode.analysis_profile:
        return _error(
            400, "analysis_profile_not_found",
            f"mode '{mode_name}' has no analysis_profile configured",
        )
    profile = manager.config.analysis_profiles.get(mode.analysis_profile)
    if profile is None:
        # Load-time validation makes this unreachable in practice; stay defensive.
        return _error(
            400, "analysis_profile_not_found",
            f"analysis_profile '{mode.analysis_profile}' is not defined",
        )

    try:
        orchestrator = AnalysisOrchestrator(
            profile.task_routes,
            build_providers(profile, manager.config.analysis_delegates),
        )
        response = await orchestrator.describe(describe_request)
    except AnalysisValidationError as exc:
        return _error(400, exc.code, exc.message)
    except Exception:
        logger.exception("[analysis] describe failed unexpectedly")
        return _error(500, "analysis_internal", "unexpected server error")

    return response_to_dict(response)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_analysis_routes.py -v`
Expected: all PASS

- [ ] **Step 5: Mount the router in `server/lcm_sr_server.py`**

Add to the imports near the other route modules:

```python
from server.analysis_routes import router as analysis_router
```

Add alongside the existing `include_router` block (~line 877):

```python
app.include_router(analysis_router)
```

- [ ] **Step 6: Run the analysis + server-adjacent suites**

Run: `python -m pytest tests/test_analysis_routes.py tests/test_analysis_contracts.py tests/test_analysis_orchestrator.py tests/test_analysis_mode_config.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add server/analysis_routes.py tests/test_analysis_routes.py server/lcm_sr_server.py
git commit -m "feat(analysis): POST /v1/describe with request-time config resolution (STABL-ucomsfel) — next: capabilities.supports_describe"
```

---

### Task 2: `capabilities.supports_describe` in `GET /api/models/status`

**Files:**
- Modify: `server/model_routes.py` (`get_models_status`, capabilities dict at ~line 126)
- Modify: `tests/test_model_routes.py` (add two tests)

**Interfaces:**
- Consumes: `get_mode_config().get_mode(name).analysis_profile` (raises `KeyError` for unknown modes).
- Produces: `capabilities.supports_describe: bool` in the status payload — true iff the active mode has an `analysis_profile`. Consumed by nothing typed (status stays an untyped map in `stclient.Models()`), so purely additive.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_model_routes.py`, following the existing `test_models_status_*` mock pattern (runtime/registry/provider mocks + `_status_request()`):

```python
async def test_models_status_supports_describe_true_when_mode_has_profile():
    runtime = Mock()
    runtime.get_current_mode.return_value = "SDXL"
    runtime.is_model_loaded.return_value = True
    runtime.get_queue_size.return_value = 0
    registry = Mock()
    registry.get_vram_stats.return_value = {}
    provider = Mock()
    provider.backend_id = "cuda"
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=True, supports_modes=True, supports_superres=True,
        supports_model_registry_stats=True, supports_img2img=True,
    )
    mode_cfg = SimpleNamespace(analysis_profile="default")
    manager = Mock()
    manager.get_mode.return_value = mode_cfg

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry), \
            patch("server.model_routes.get_mode_config", return_value=manager):
        data = await model_routes.get_models_status(_status_request())

    assert data["capabilities"]["supports_describe"] is True


async def test_models_status_supports_describe_false_without_profile_or_mode():
    runtime = Mock()
    runtime.get_current_mode.return_value = None  # no active mode at all
    runtime.is_model_loaded.return_value = False
    runtime.get_queue_size.return_value = 0
    registry = Mock()
    registry.get_vram_stats.return_value = {}
    provider = Mock()
    provider.backend_id = "cpu"
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=True, supports_modes=False, supports_superres=False,
        supports_model_registry_stats=False, supports_img2img=False,
    )

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry):
        data = await model_routes.get_models_status(_status_request())

    assert data["capabilities"]["supports_describe"] is False
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_model_routes.py -k supports_describe -v`
Expected: FAIL — `KeyError: 'supports_describe'`

- [ ] **Step 3: Implement**

In `get_models_status` (`server/model_routes.py`), before the `return`, compute:

```python
    supports_describe = False
    current_mode = runtime.get_current_mode()
    if current_mode:
        try:
            supports_describe = bool(get_mode_config().get_mode(current_mode).analysis_profile)
        except Exception:
            supports_describe = False  # no config / unknown mode -> capability off
```

and add to the `"capabilities"` dict, after the existing `supports_*` keys:

```python
            "supports_describe": supports_describe,
```

(`get_mode_config` is already imported in `model_routes.py`.)

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_model_routes.py -v`
Expected: all PASS (existing status tests must stay green — they don't patch `get_mode_config`, which is why the implementation catches all exceptions and defaults to `False`)

- [ ] **Step 5: Commit**

```bash
git add server/model_routes.py tests/test_model_routes.py
git commit -m "feat(analysis): capabilities.supports_describe in /api/models/status (STABL-ucomsfel) — next: stclient Describe()"
```

---

### Task 3: `stclient.Describe()` + `APIError`

**Files:**
- Create: `cli/go/pkg/stclient/describe.go`
- Create: `cli/go/pkg/stclient/describe_test.go`

**Interfaces:**
- Consumes: `DescribeRequest.Validate()` (exists in `describe_types.go`), `Client{baseURL, http}` from `client.go`.
- Produces: `func (c *Client) Describe(ctx context.Context, req DescribeRequest) (*DescribeResponse, error)` and `type APIError struct{ Code, Message string }` with `func (e *APIError) Error() string` returning `"<code>: <message>"`. Task 5 relies on both, and on `errors.As(err, &apiErr)` working.

- [ ] **Step 1: Write failing tests**

`cli/go/pkg/stclient/describe_test.go`:

```go
package stclient

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func describeReq() DescribeRequest {
	u1, u2 := "http://x/a.png", "http://x/b.png"
	return DescribeRequest{
		Targets: []DescribeTarget{
			{ID: "t1", URL: &u1},
			{ID: "t2", URL: &u2},
		},
		Tasks: []DescribeTask{
			{ID: "caption", Kind: "caption", Caption: &CaptionParams{}},
		},
	}
}

func TestDescribeSendsWireShapeAndDecodes(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/describe" {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"status": "ok",
			"observations": [
				{"task_id": "caption", "target_id": "t1", "kind": "text",
				 "text": {"content": "stub:caption"}}
			],
			"runs": [
				{"task_id": "caption", "target_id": "t1",
				 "delegate": "vlm_caption", "status": "succeeded"},
				{"task_id": "caption", "target_id": "t2",
				 "delegate": "vlm_caption", "status": "succeeded"}
			]
		}`))
	}))
	defer srv.Close()

	resp, err := New(srv.URL).Describe(context.Background(), describeReq())
	if err != nil {
		t.Fatal(err)
	}
	// Wire-shape pin: request targets serialized in declaration order.
	targets := got["targets"].([]any)
	if len(targets) != 2 || targets[0].(map[string]any)["id"] != "t1" || targets[1].(map[string]any)["id"] != "t2" {
		t.Fatalf("target order not preserved: %v", targets)
	}
	if resp.Status != "ok" || len(resp.Runs) != 2 || resp.Runs[0].Delegate != "vlm_caption" {
		t.Fatalf("bad decode: %+v", resp)
	}
	if resp.Observations[0].Text.Content != "stub:caption" {
		t.Fatalf("bad observation decode: %+v", resp.Observations[0])
	}
}

func TestDescribeMapsAnalysisErrorToAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"code": "analysis_mode_not_found", "message": "unknown mode 'NOPE'"}}`))
	}))
	defer srv.Close()

	_, err := New(srv.URL).Describe(context.Background(), describeReq())
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if apiErr.Code != "analysis_mode_not_found" {
		t.Fatalf("bad code: %q", apiErr.Code)
	}
	if apiErr.Error() != "analysis_mode_not_found: unknown mode 'NOPE'" {
		t.Fatalf("bad Error(): %q", apiErr.Error())
	}
}

func TestDescribeNonJSONErrorBodyFallsBackToPlainError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("upstream exploded"))
	}))
	defer srv.Close()

	_, err := New(srv.URL).Describe(context.Background(), describeReq())
	if err == nil {
		t.Fatal("want error")
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		t.Fatalf("plain error expected for untyped body, got APIError %v", apiErr)
	}
}

func TestDescribeValidatesBeforeSending(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer srv.Close()

	_, err := New(srv.URL).Describe(context.Background(), DescribeRequest{})
	if err == nil {
		t.Fatal("want validation error")
	}
	if called {
		t.Fatal("invalid request must never leave the client")
	}
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd cli/go && go test ./pkg/stclient/ -run TestDescribe -v`
Expected: compile FAIL — `c.Describe undefined` / `undefined: APIError`

- [ ] **Step 3: Implement `cli/go/pkg/stclient/describe.go`**

```go
package stclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// APIError is a typed non-2xx server error carrying the operator-facing
// analysis_* (or other) error code from the {"error":{code,message}} body.
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *APIError) Error() string { return e.Code + ": " + e.Message }

// Describe POSTs a validated DescribeRequest to /v1/describe and decodes
// the typed DescribeResponse. Invalid requests never leave the client.
// Non-2xx responses with a typed error body return *APIError; anything
// else returns a plain error.
func (c *Client) Describe(ctx context.Context, req DescribeRequest) (*DescribeResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/describe", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		var envelope struct {
			Error APIError `json:"error"`
		}
		if json.Unmarshal(raw, &envelope) == nil && envelope.Error.Code != "" {
			return nil, &envelope.Error
		}
		return nil, fmt.Errorf("POST /v1/describe -> %s: %s", resp.Status, bytes.TrimSpace(raw))
	}
	var out DescribeResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd cli/go && go test ./pkg/stclient/ -v`
Expected: all PASS (including the pre-existing describe_types tests)

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient/describe.go cli/go/pkg/stclient/describe_test.go
git commit -m "feat(stclient): Describe() with typed APIError mapping (STABL-ucomsfel) — next: st describe request construction"
```

---

### Task 4: `st describe` — request construction (ordering contract)

**Files:**
- Create: `cli/go/cmd/st/describe.go`
- Create: `cli/go/cmd/st/describe_test.go`

**Interfaces:**
- Consumes: `stclient.DescribeRequest/DescribeTarget/DescribeTask/CaptionParams/DetectParams`.
- Produces (Task 5 uses all of these):
  - `type describeOptions struct { caption bool; prompt string; detect bool; labels []string; minConfidence float64; minConfidenceSet bool }`
  - `func validateDescribeFlags(o describeOptions) error` — usage errors
  - `type targetSpec struct { arg string; id string; isURL bool }`
  - `func classifyTargets(args []string) []targetSpec` — positional IDs `t1..tN`
  - `func buildDescribeTasks(o describeOptions) []stclient.DescribeTask` — canonical kind order

This task deliberately contains no network or upload code — construction is pure and fully testable.

- [ ] **Step 1: Write failing tests**

`cli/go/cmd/st/describe_test.go`:

```go
package main

import (
	"reflect"
	"testing"
)

func TestClassifyTargetsAssignsPositionalIDsInArgOrder(t *testing.T) {
	specs := classifyTargets([]string{"./b.png", "https://x/a.png", "./c.png"})
	ids := []string{specs[0].id, specs[1].id, specs[2].id}
	if !reflect.DeepEqual(ids, []string{"t1", "t2", "t3"}) {
		t.Fatalf("positional IDs broken: %v", ids)
	}
	if specs[0].isURL || !specs[1].isURL || specs[2].isURL {
		t.Fatalf("URL classification broken: %+v", specs)
	}
	// Order is contract: arg order, never sorted.
	if specs[0].arg != "./b.png" || specs[2].arg != "./c.png" {
		t.Fatalf("arg order not preserved: %+v", specs)
	}
}

func TestBuildDescribeTasksCanonicalOrderRegardlessOfFlagOrder(t *testing.T) {
	tasks := buildDescribeTasks(describeOptions{detect: true, caption: true})
	if len(tasks) != 2 {
		t.Fatalf("want 2 tasks, got %d", len(tasks))
	}
	// Canonical TaskKind order: caption before detect, task id = kind string.
	if tasks[0].ID != "caption" || string(tasks[0].Kind) != "caption" || tasks[0].Caption == nil {
		t.Fatalf("task 0 not caption: %+v", tasks[0])
	}
	if tasks[1].ID != "detect" || string(tasks[1].Kind) != "detect" || tasks[1].Detect == nil {
		t.Fatalf("task 1 not detect: %+v", tasks[1])
	}
}

func TestBuildDescribeTasksCarriesParams(t *testing.T) {
	tasks := buildDescribeTasks(describeOptions{
		caption: true, prompt: "focus on lighting",
		detect: true, labels: []string{"person", "car"},
		minConfidence: 0.4, minConfidenceSet: true,
	})
	if tasks[0].Caption.Prompt == nil || *tasks[0].Caption.Prompt != "focus on lighting" {
		t.Fatalf("prompt not carried: %+v", tasks[0].Caption)
	}
	if !reflect.DeepEqual(tasks[1].Detect.Labels, []string{"person", "car"}) {
		t.Fatalf("labels not carried: %+v", tasks[1].Detect)
	}
	if tasks[1].Detect.MinConfidence == nil || *tasks[1].Detect.MinConfidence != 0.4 {
		t.Fatalf("min confidence not carried: %+v", tasks[1].Detect)
	}
}

func TestValidateDescribeFlagsUsageErrors(t *testing.T) {
	cases := []struct {
		name string
		o    describeOptions
	}{
		{"no task flags", describeOptions{}},
		{"prompt without caption", describeOptions{detect: true, prompt: "x"}},
		{"labels without detect", describeOptions{caption: true, labels: []string{"a"}}},
		{"min-confidence without detect", describeOptions{caption: true, minConfidenceSet: true}},
	}
	for _, tc := range cases {
		if err := validateDescribeFlags(tc.o); err == nil {
			t.Fatalf("%s: want usage error", tc.name)
		}
	}
	if err := validateDescribeFlags(describeOptions{caption: true}); err != nil {
		t.Fatalf("valid flags rejected: %v", err)
	}
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd cli/go && go test ./cmd/st/ -run 'TestClassifyTargets|TestBuildDescribeTasks|TestValidateDescribeFlags' -v`
Expected: compile FAIL — `undefined: classifyTargets` etc.

- [ ] **Step 3: Implement construction in `cli/go/cmd/st/describe.go`**

Construction half only (the cobra command and runDescribe come in Task 5, in this same file):

```go
package main

import (
	"fmt"
	"strings"

	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

type describeOptions struct {
	caption          bool
	prompt           string
	detect           bool
	labels           []string
	minConfidence    float64
	minConfidenceSet bool
}

type targetSpec struct {
	arg   string
	id    string
	isURL bool
}

// classifyTargets maps positional args to targets in exact arg order with
// positional IDs t1..tN (1-based). Ordering is contract (spec: Ordering
// Determinism) — never sort, never use a map.
func classifyTargets(args []string) []targetSpec {
	specs := make([]targetSpec, len(args))
	for i, arg := range args {
		specs[i] = targetSpec{
			arg:   arg,
			id:    fmt.Sprintf("t%d", i+1),
			isURL: strings.HasPrefix(arg, "http://") || strings.HasPrefix(arg, "https://"),
		}
	}
	return specs
}

// buildDescribeTasks emits tasks in canonical TaskKind order (caption,
// detect, ocr, pose, embed) regardless of flag order; task id = kind string.
func buildDescribeTasks(o describeOptions) []stclient.DescribeTask {
	var tasks []stclient.DescribeTask
	if o.caption {
		params := &stclient.CaptionParams{}
		if o.prompt != "" {
			p := o.prompt
			params.Prompt = &p
		}
		tasks = append(tasks, stclient.DescribeTask{ID: "caption", Kind: "caption", Caption: params})
	}
	if o.detect {
		params := &stclient.DetectParams{}
		if len(o.labels) > 0 {
			params.Labels = o.labels
		}
		if o.minConfidenceSet {
			mc := o.minConfidence
			params.MinConfidence = &mc
		}
		tasks = append(tasks, stclient.DescribeTask{ID: "detect", Kind: "detect", Detect: params})
	}
	return tasks
}

func validateDescribeFlags(o describeOptions) error {
	if !o.caption && !o.detect {
		return fmt.Errorf("at least one task flag required (--caption, --detect)")
	}
	if o.prompt != "" && !o.caption {
		return fmt.Errorf("--prompt requires --caption")
	}
	if len(o.labels) > 0 && !o.detect {
		return fmt.Errorf("--labels requires --detect")
	}
	if o.minConfidenceSet && !o.detect {
		return fmt.Errorf("--min-confidence requires --detect")
	}
	return nil
}
```

Check the `CaptionParams`/`DetectParams` field names in
`cli/go/pkg/stclient/describe_types.go` (lines 59–66) and use those — the
field names above (`Prompt`, `Labels`, `MinConfidence`) must match the
existing types, not be invented.

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd cli/go && go test ./cmd/st/ -run 'TestClassifyTargets|TestBuildDescribeTasks|TestValidateDescribeFlags' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cli/go/cmd/st/describe.go cli/go/cmd/st/describe_test.go
git commit -m "feat(st): describe request construction with ordering contract (STABL-ucomsfel) — next: st describe execution + rendering"
```

---

### Task 5: `st describe` — execution, rendering, exit codes

**Files:**
- Modify: `cli/go/cmd/st/describe.go` (add command + run + rendering)
- Modify: `cli/go/cmd/st/describe_test.go` (add rendering/exit/wiring tests)
- Modify: `cli/go/USAGE.md` (document the verb)

**Interfaces:**
- Consumes: Task 3's `Describe()`/`APIError`, Task 4's helpers, `newClient()` (`main.go:113`), `Client.Upload(ctx, filename, data, bucket)` (`http.go:189`), `emitJSON` (`util.go:11`), `exitError{code, err}` + `exitCodeOf` (`history_runtime.go:37,50`), global `flagJSON`.
- Produces: the `st describe` verb; `renderDescribeHuman(w io.Writer, resp *stclient.DescribeResponse)`; `renderDescribeFailures(w io.Writer, resp *stclient.DescribeResponse)`; `describeStatusErr(resp *stclient.DescribeResponse) error` (nil / exit 2 / exit 3).

- [ ] **Step 1: Write failing rendering + exit-code tests**

Append to `cli/go/cmd/st/describe_test.go`:

```go
func sampleResponse() *stclient.DescribeResponse {
	conf := 0.92
	return &stclient.DescribeResponse{
		Status: "partial",
		Observations: []stclient.DescribeObservation{
			{TaskID: "caption", TargetID: "t1", Kind: "text",
				Text: &stclient.TextObservation{Content: "a red bicycle"}},
			{TaskID: "detect", TargetID: "t1", Kind: "detection",
				Detection: &stclient.DetectionObservation{
					Label: "bicycle", Confidence: conf,
					Box: stclient.Box{X: 0.1, Y: 0.2, W: 0.5, H: 0.4},
				}},
		},
		Runs: []stclient.DescribeRun{
			{TaskID: "caption", TargetID: "t1", Delegate: "vlm_caption", Status: "succeeded"},
			{TaskID: "detect", TargetID: "t1", Delegate: "yolo_detect", Status: "succeeded"},
			{TaskID: "detect", TargetID: "t2", Delegate: "yolo_detect", Status: "failed",
				Error: &stclient.RunError{Code: "analysis_run_failed", Message: "conn refused"}},
		},
	}
}

func TestRenderDescribeHumanShowsCaptionAndDetections(t *testing.T) {
	var buf strings.Builder
	renderDescribeHuman(&buf, sampleResponse())
	out := buf.String()
	for _, want := range []string{"caption (t1): a red bicycle", "bicycle", "0.92"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestRenderDescribeFailuresListsEveryNonSucceededRun(t *testing.T) {
	var buf strings.Builder
	renderDescribeFailures(&buf, sampleResponse())
	out := buf.String()
	// Frozen content requirement: task, target, delegate, status, code, message.
	for _, want := range []string{"detect", "t2", "yolo_detect", "failed", "analysis_run_failed", "conn refused"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "t1") {
		t.Fatalf("succeeded runs must not be rendered as failures:\n%s", out)
	}
}

func TestDescribeStatusErrExitCodes(t *testing.T) {
	ok := &stclient.DescribeResponse{Status: "ok"}
	if err := describeStatusErr(ok); err != nil {
		t.Fatalf("ok must be exit 0, got %v", err)
	}
	partial := &stclient.DescribeResponse{Status: "partial"}
	if code := exitCodeOf(describeStatusErr(partial)); code != 3 {
		t.Fatalf("partial must exit 3, got %d", code)
	}
	failed := &stclient.DescribeResponse{Status: "failed"}
	if code := exitCodeOf(describeStatusErr(failed)); code != 2 {
		t.Fatalf("failed must exit 2, got %d", code)
	}
}
```

Add the needed imports (`strings`, and the stclient path) to the test file's import block.

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd cli/go && go test ./cmd/st/ -run 'TestRenderDescribe|TestDescribeStatusErr' -v`
Expected: compile FAIL — `undefined: renderDescribeHuman` etc.

- [ ] **Step 3: Implement command, run function, and rendering**

Add to `cli/go/cmd/st/describe.go` (same file as Task 4; extend the import block with `context`, `errors`, `io`, `os`, `path/filepath`, `text/tabwriter`, `github.com/spf13/cobra`):

```go
var describeOpts describeOptions

var describeCmd = &cobra.Command{
	Use:   "describe <file-or-url> [more...]",
	Short: "Run analysis tasks (caption, detect) against images",
	Long: `Describe images through the server's analysis capability.

Positional arguments are targets, in order: local files are uploaded
first (target IDs t1..tN follow argument order); http(s):// arguments
pass through as URL targets. Task flags select what runs:

  st describe ./photo.png --caption
  st describe ./photo.png --detect --labels person,car
  st describe ./a.png ./b.png --caption --detect

Exit codes: 0 ok, 1 transport/usage/validation error, 2 failed, 3 partial.`,
	Args: cobra.MinimumNArgs(1),
	RunE: runDescribe,
}

func init() {
	f := describeCmd.Flags()
	f.BoolVar(&describeOpts.caption, "caption", false, "add a caption task")
	f.StringVar(&describeOpts.prompt, "prompt", "", "caption guidance prompt (requires --caption)")
	f.BoolVar(&describeOpts.detect, "detect", false, "add a detection task")
	f.StringSliceVar(&describeOpts.labels, "labels", nil, "detection label filter (requires --detect)")
	f.Float64Var(&describeOpts.minConfidence, "min-confidence", 0, "minimum detection confidence (requires --detect)")
	rootCmd.AddCommand(describeCmd)
}

func runDescribe(cmd *cobra.Command, args []string) error {
	describeOpts.minConfidenceSet = cmd.Flags().Changed("min-confidence")
	if err := validateDescribeFlags(describeOpts); err != nil {
		return err // usage error: exit 1 via main's default
	}

	client := newClient()
	ctx := cmd.Context()
	specs := classifyTargets(args)
	targets := make([]stclient.DescribeTarget, len(specs))
	for i, spec := range specs {
		if spec.isURL {
			u := spec.arg
			targets[i] = stclient.DescribeTarget{ID: spec.id, URL: &u}
			continue
		}
		data, err := os.ReadFile(spec.arg)
		if err != nil {
			return err
		}
		ref, err := client.Upload(ctx, filepath.Base(spec.arg), data, "upload")
		if err != nil {
			return fmt.Errorf("upload %s: %w", spec.arg, err)
		}
		targets[i] = stclient.DescribeTarget{ID: spec.id, AssetRef: &ref}
	}

	req := stclient.DescribeRequest{Targets: targets, Tasks: buildDescribeTasks(describeOpts)}
	resp, err := client.Describe(ctx, req)
	if err != nil {
		// *APIError prints "code: message", transport errors print their
		// message — both via main's "error:" stderr line. Exit 1.
		return err
	}

	if flagJSON {
		if err := emitJSON(cmd, resp); err != nil {
			return err
		}
	} else {
		renderDescribeHuman(cmd.OutOrStdout(), resp)
	}
	renderDescribeFailures(cmd.ErrOrStderr(), resp)
	return describeStatusErr(resp)
}

// renderDescribeHuman writes the human default rendering: caption text
// lines and a detection table. Scripts must not parse this — the frozen
// machine surface is --json.
func renderDescribeHuman(w io.Writer, resp *stclient.DescribeResponse) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	for _, obs := range resp.Observations {
		switch {
		case obs.Text != nil:
			fmt.Fprintf(tw, "%s (%s): %s\n", obs.TaskID, obs.TargetID, obs.Text.Content)
		case obs.Detection != nil:
			d := obs.Detection
			fmt.Fprintf(tw, "%s (%s):\t%s\t%.2f\tbox(%.3f, %.3f, %.3f, %.3f)\n",
				obs.TaskID, obs.TargetID, d.Label, d.Confidence, d.Box.X, d.Box.Y, d.Box.W, d.Box.H)
		}
	}
	tw.Flush()
}

// renderDescribeFailures writes every non-succeeded run to w (stderr in
// practice). Presence and content are frozen by the spec: task, target,
// delegate, status, error code, error message.
func renderDescribeFailures(w io.Writer, resp *stclient.DescribeResponse) {
	for _, run := range resp.Runs {
		if run.Status == "succeeded" {
			continue
		}
		code, msg := "", ""
		if run.Error != nil {
			code, msg = run.Error.Code, run.Error.Message
		}
		fmt.Fprintf(w, "run %s/%s (%s) %s: %s: %s\n",
			run.TaskID, run.TargetID, run.Delegate, run.Status, code, msg)
	}
}

// describeStatusErr maps the frozen exit-code contract: ok=0, failed=2,
// partial=3 (transport/usage/validation errors exit 1 elsewhere).
func describeStatusErr(resp *stclient.DescribeResponse) error {
	switch resp.Status {
	case "failed":
		return exitError{code: 2, err: errors.New("describe failed: no run succeeded")}
	case "partial":
		return exitError{code: 3, err: errors.New("describe partial: some runs did not succeed")}
	default:
		return nil
	}
}
```

Check `exitError` construction against `history_runtime.go:37` (unexported fields `code`, `err` — same package, so literal construction works) and match `DescribeRun.Status`/`RunStatus` comparison style to how `describe_types.go` declares it (string-typed constants — compare with a conversion if needed).

Also extend `resetCLIFlagState()` in `gen_test.go` to zero the new
package-global flag state — its whole purpose is that no command's globals
leak between CLI tests:

```go
	describeOpts = describeOptions{}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd cli/go && go test ./cmd/st/ -run 'TestRenderDescribe|TestDescribeStatusErr' -v`
Expected: PASS

- [ ] **Step 5: Write failing end-to-end wiring test (httptest + executeCLI)**

Append to `cli/go/cmd/st/describe_test.go` (follow the existing `executeCLI` test pattern in `validate_track3_test.go` / `main_test.go` for how the CLI is invoked with `--server`):

```go
func TestDescribeEndToEndUploadsAndExitsPartial(t *testing.T) {
	dir := t.TempDir()
	img := filepath.Join(dir, "a.png")
	if err := os.WriteFile(img, []byte("fakepng"), 0o644); err != nil {
		t.Fatal(err)
	}

	var describeBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/upload":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"fileRef": "ref-123"}`))
		case "/v1/describe":
			_ = json.NewDecoder(r.Body).Decode(&describeBody)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"status": "partial",
				"observations": [
					{"task_id": "caption", "target_id": "t1", "kind": "text",
					 "text": {"content": "stub"}}
				],
				"runs": [
					{"task_id": "caption", "target_id": "t1", "delegate": "vlm_caption", "status": "succeeded"},
					{"task_id": "detect", "target_id": "t1", "delegate": "yolo_detect", "status": "failed",
					 "error": {"code": "analysis_run_failed", "message": "boom"}}
				]
			}`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	// runCmdCaptureWithStateRoot (gen_test.go:192) resets global flag state
	// and overrides the history state root — required for any test that
	// drives the CLI, or globals leak across the ./cmd/st suite and history
	// writes hit the real user state dir.
	stdout, stderr, err := runCmdCaptureWithStateRoot(t, t.TempDir(),
		"describe", img, "--caption", "--detect", "--server", srv.URL)
	if code := exitCodeOf(err); code != 3 {
		t.Fatalf("partial must exit 3, got %d (%v)", code, err)
	}
	// The uploaded ref must land as target t1's asset_ref.
	target := describeBody["targets"].([]any)[0].(map[string]any)
	if target["id"] != "t1" || target["asset_ref"] != "ref-123" {
		t.Fatalf("upload not wired into target: %v", target)
	}
	if !strings.Contains(stdout, "stub") {
		t.Fatalf("caption not rendered to stdout:\n%s", stdout)
	}
	// Frozen failure-rendering contract, asserted on captured stderr:
	// runDescribe writes run failures through cmd.ErrOrStderr().
	for _, want := range []string{"detect", "t1", "yolo_detect", "failed", "analysis_run_failed", "boom"} {
		if !strings.Contains(stderr, want) {
			t.Fatalf("failure line missing %q on stderr:\n%s", want, stderr)
		}
	}
}
```

Also add the transport-failure case (reviewer-flagged gap: pure transport
failures guarantee a message and exit 1, with no synthetic code):

```go
func TestDescribeTransportFailureExitsOneWithMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close() // connection refused from here on

	_, _, err := runCmdCaptureWithStateRoot(t, t.TempDir(),
		"describe", "http://x/a.png", "--caption", "--server", srv.URL)
	if err == nil {
		t.Fatal("want transport error")
	}
	if code := exitCodeOf(err); code != 1 {
		t.Fatalf("transport failure must exit 1, got %d", code)
	}
	var apiErr *stclient.APIError
	if errors.As(err, &apiErr) {
		t.Fatalf("transport failure must not carry a synthetic code: %v", apiErr)
	}
	if err.Error() == "" {
		t.Fatal("transport failure must carry a human-readable message")
	}
}
```

Stderr note for this case: `rootCmd` sets `SilenceErrors: true`
(`main.go:34`), so cobra never writes RunE errors to the captured `Err`
writer — `main()` itself prints `error: <message>` to `os.Stderr`
(`main.go:122-125`). The captured-stderr assertion therefore lives in the
partial-response test above (run-failure lines, which `runDescribe` writes
through `cmd.ErrOrStderr()`); for transport failures the in-process
boundary is the returned error, and main's existing print is the frozen
stderr path — assert on `err.Error()` here, don't fight the harness.

Before finalizing, check the actual `/v1/upload` response shape `Client.Upload` expects (`http.go:189` region) — adjust the handler body to match reality, not this sketch.

- [ ] **Step 6: Run wiring test, verify it fails, then fix wiring until green**

Run: `cd cli/go && go test ./cmd/st/ -run TestDescribeEndToEnd -v`
Expected: FAIL first (before the command is registered correctly), then PASS after wiring fixes.

- [ ] **Step 7: Run the full Go suite**

Run: `cd cli/go && go test ./...`
Expected: all PASS

- [ ] **Step 8: Document the verb in `cli/go/USAGE.md`**

Add an `st describe` section next to the other verbs: synopsis, the two task flags and their param flags, target semantics (positional order → `t1..tN`, local files auto-upload, URLs pass through), the frozen `--json` shape, and the 0/1/2/3 exit-code table, matching the spec's "Output contract" and "Exit codes" sections verbatim.

- [ ] **Step 9: Commit**

```bash
git add cli/go/cmd/st/describe.go cli/go/cmd/st/describe_test.go cli/go/USAGE.md
git commit -m "feat(st): describe verb — auto-upload targets, human/json output, frozen exit codes (STABL-ucomsfel) — next: full-suite verification + review"
```

---

## Final Verification

- [ ] Python: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_routes.py tests/test_analysis_contracts.py tests/test_analysis_orchestrator.py tests/test_analysis_mode_config.py tests/test_model_routes.py -v` — all green.
- [ ] Go: `cd cli/go && go test ./...` — all green.
- [ ] `drift check` — report (do not relink) any stale anchors touched by `model_routes.py`/`lcm_sr_server.py` edits.
- [ ] FP comment on STABL-ucomsfel per stopping-point policy; report ready for review.
