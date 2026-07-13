# Describe/Analysis v1 Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagent-driven development is forbidden in this repo (AGENTS.md). Steps use checkbox (`- [ ]`) syntax for tracking.

**FP issue:** STABL-tlklfaxz
**Spec:** `docs/superpowers/specs/2026-07-11-describe-analysis-interface-design.md`

**Goal:** Implement spec items 1–3: the typed describe contract in `pkg/stclient`, `analysis_*` mode-config parsing/validation, and the async orchestration core with one stub provider and contract tests.

**Architecture:** Go contract types live in `cli/go/pkg/stclient` (client boundary validation, JSON shape pinned by tests). Server-side lives in a new `backends/analysis/` package mirroring the `backends/conditioning/` layout: `contracts.py` (wire dataclasses + request parsing/validation), `providers.py` (async `DescribeProvider` protocol + stub), `orchestrator.py` (async chain: validate → expand runs → dispatch → normalize → assemble). Config policy extends `server/mode_config.py` following the existing `chat_connections`/`chat_delegates` helper pattern.

**Tech Stack:** Go (`testing` stdlib), Python 3 dataclasses + asyncio + pytest.

## Global Constraints

Copied from the spec; every task inherits these.

- All operator-facing error codes are `analysis_*` prefixed, even though the CLI verb is `describe`.
- `DescribeTaskKind` closed enum: `caption | detect | ocr | pose | embed`.
- `ObservationKind` closed enum: `text | detection | attribute | keypoints`. `mask` is scoped out of v1.
- `ArtifactKind` closed enum: `embedding_ref` only.
- `RunStatus` closed enum: `succeeded | failed | skipped`. `error` required for `failed`/`skipped`, absent for `succeeded`.
- `DescribeStatus` closed enum: `ok | partial | failed`.
- Exactly one typed params block per task, matching `kind`. No `map[string]any` in the public library contract.
- `DescribeTarget` is exactly-one-of `asset_ref` / `url`.
- Omitted/empty `Role` means `primary`; only `primary` has v1 semantics.
- Zero-run binding is a request validation error (`analysis_target_binding_invalid`); every `DescribeResponse` has non-empty `runs`.
- Box/keypoint coordinates normalized to `[0,1]`, box origin top-left.
- Config load fails fast: unknown connection/delegate/profile references, and `task_routes` key ≠ delegate `kind` (`analysis_delegate_kind_mismatch`).
- `raw_output` is an opaque provider payload on both sides: `json.RawMessage` in Go, untyped passthrough (`Any`) in Python. Neither side types or restructures it.
- `analysis_run_failed` is the blessed error code for a single run whose delegate was invoked and raised; it is part of the extensible `analysis_*` vocabulary (spec Error Vocabulary section).
- Out of scope for this plan: transport (WS vs HTTP), `st describe`, real providers, `Summary` population, frontend.
- Python commands run under Miniforge base: `source /Users/darkbit1001/miniforge3/bin/activate base` then `python -m pytest ...`.
- Go commands run from `cli/go/`.

## File Structure

- Create: `cli/go/pkg/stclient/describe_types.go` — contract types + client-boundary validation
- Create: `cli/go/pkg/stclient/describe_types_test.go`
- Create: `backends/analysis/__init__.py` — package exports
- Create: `backends/analysis/contracts.py` — wire dataclasses, enums, error type, request parser/validator
- Create: `backends/analysis/providers.py` — `DescribeProvider` protocol + `StubProvider`
- Create: `backends/analysis/orchestrator.py` — run expansion + async dispatch + response assembly
- Create: `tests/test_analysis_contracts.py`
- Create: `tests/test_analysis_mode_config.py`
- Create: `tests/test_analysis_orchestrator.py`
- Modify: `server/mode_config.py` — `Analysis*Config` dataclasses, parse helpers, `ModesYAML` + `ModeConfig` fields

---

### Task 1: Go contract types in `pkg/stclient`

**Files:**
- Create: `cli/go/pkg/stclient/describe_types.go`
- Test: `cli/go/pkg/stclient/describe_types_test.go`

**Interfaces:**
- Consumes: nothing (leaf task).
- Produces: `DescribeRequest`, `DescribeTarget`, `DescribeTask`, `DescribeTaskKind` (+ consts `TaskKindCaption` etc.), params structs (`CaptionParams`, `DetectParams`, `OcrParams`, `PoseParams`, `EmbedParams`), `DescribeResponse`, `DescribeObservation`, `ObservationKind`, `TextObservation`, `DetectionObservation`, `AttributeObservation`, `KeypointsObservation`, `Keypoint`, `Box`, `DescribeArtifact`, `DescribeRun`, `RunStatus`, `DescribeStatus`, `RunError`, and `(*DescribeRequest).Validate() error`. Future transport/CLI tasks (out of this plan) consume these.

- [x] **Step 1: Write the failing tests**

`cli/go/pkg/stclient/describe_types_test.go`:

```go
package stclient

import (
	"encoding/json"
	"strings"
	"testing"
)

func strp(s string) *string { return &s }

func validDescribeRequest() DescribeRequest {
	return DescribeRequest{
		Targets: []DescribeTarget{{ID: "t1", AssetRef: strp("asset-1")}},
		Tasks:   []DescribeTask{{ID: "cap1", Kind: TaskKindCaption, Caption: &CaptionParams{}}},
	}
}

// Pins the wire shape of a full request per the spec's Request Contract.
func TestDescribeRequestWireShape(t *testing.T) {
	req := validDescribeRequest()
	req.Mode = strp("SDXL")
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if m["mode"] != "SDXL" {
		t.Fatalf("mode: %s", b)
	}
	tg := m["targets"].([]any)[0].(map[string]any)
	if tg["id"] != "t1" || tg["asset_ref"] != "asset-1" {
		t.Fatalf("target: %s", b)
	}
	if _, ok := tg["url"]; ok {
		t.Fatalf("nil url must be omitted: %s", b)
	}
	tk := m["tasks"].([]any)[0].(map[string]any)
	if tk["id"] != "cap1" || tk["kind"] != "caption" {
		t.Fatalf("task: %s", b)
	}
	if _, ok := tk["detect"]; ok {
		t.Fatalf("unset params blocks must be omitted: %s", b)
	}
}

// Pins the inbound response shape: status/runs/observations decode with
// task_id/target_id correlation and normalized payload blocks.
func TestDescribeResponseDecodes(t *testing.T) {
	raw := `{
		"status": "partial",
		"observations": [
			{"task_id":"cap1","target_id":"t1","kind":"text","text":{"content":"an owl"}},
			{"task_id":"det1","target_id":"t1","kind":"detection",
			 "detection":{"label":"owl","confidence":0.93,"box":{"x":0.1,"y":0.2,"w":0.3,"h":0.4}}}
		],
		"artifacts": [
			{"task_id":"emb1","target_id":"t1","kind":"embedding_ref","ref":"asset-9","dims":768}
		],
		"runs": [
			{"task_id":"cap1","target_id":"t1","delegate":"vlm_caption","status":"succeeded"},
			{"task_id":"det1","target_id":"t1","delegate":"yolo_detect","status":"succeeded"},
			{"task_id":"ocr1","target_id":"t1","delegate":"","status":"skipped",
			 "error":{"code":"analysis_no_supported_delegate","message":"no route for kind ocr"}}
		]
	}`
	var resp DescribeResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Status != StatusPartial {
		t.Fatalf("status: %+v", resp.Status)
	}
	if resp.Summary != nil {
		t.Fatalf("summary must be nil when absent")
	}
	if len(resp.Observations) != 2 || len(resp.Runs) != 3 {
		t.Fatalf("counts: %+v", resp)
	}
	o0 := resp.Observations[0]
	if o0.TaskID != "cap1" || o0.TargetID != "t1" || o0.Kind != ObsKindText || o0.Text.Content != "an owl" {
		t.Fatalf("text obs: %+v", o0)
	}
	o1 := resp.Observations[1]
	if o1.Detection == nil || o1.Detection.Box.W != 0.3 || o1.Detection.Confidence != 0.93 {
		t.Fatalf("detection obs: %+v", o1)
	}
	a0 := resp.Artifacts[0]
	if a0.Kind != ArtifactKindEmbeddingRef || a0.Ref != "asset-9" || a0.Dims == nil || *a0.Dims != 768 {
		t.Fatalf("artifact: %+v", a0)
	}
	r2 := resp.Runs[2]
	if r2.Status != RunSkipped || r2.Error == nil || r2.Error.Code != "analysis_no_supported_delegate" {
		t.Fatalf("skipped run: %+v", r2)
	}
}

func TestDescribeRequestValidate(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*DescribeRequest)
		wantErr string // substring; "" means valid
	}{
		{"valid", func(r *DescribeRequest) {}, ""},
		{"no targets", func(r *DescribeRequest) { r.Targets = nil }, "analysis_invalid_request"},
		{"no tasks", func(r *DescribeRequest) { r.Tasks = nil }, "analysis_invalid_request"},
		{"target both sources", func(r *DescribeRequest) {
			r.Targets[0].URL = strp("http://x/i.png")
		}, "analysis_invalid_request"},
		{"target neither source", func(r *DescribeRequest) {
			r.Targets[0].AssetRef = nil
		}, "analysis_invalid_request"},
		{"duplicate target id", func(r *DescribeRequest) {
			r.Targets = append(r.Targets, DescribeTarget{ID: "t1", URL: strp("http://x/i.png")})
		}, "analysis_invalid_request"},
		{"unknown kind", func(r *DescribeRequest) {
			r.Tasks[0].Kind = "segment"
		}, "analysis_invalid_request"},
		{"params kind mismatch", func(r *DescribeRequest) {
			r.Tasks[0].Caption = nil
			r.Tasks[0].Detect = &DetectParams{}
		}, "analysis_invalid_request"},
		{"two params blocks", func(r *DescribeRequest) {
			r.Tasks[0].Detect = &DetectParams{}
		}, "analysis_invalid_request"},
		{"unknown target id", func(r *DescribeRequest) {
			r.Tasks[0].TargetIDs = []string{"nope"}
		}, "analysis_target_binding_invalid"},
		{"zero-run: no primary targets", func(r *DescribeRequest) {
			r.Targets[0].Role = "reference"
		}, "analysis_target_binding_invalid"},
		{"explicit binding to non-primary is allowed", func(r *DescribeRequest) {
			r.Targets[0].Role = "reference"
			r.Tasks[0].TargetIDs = []string{"t1"}
		}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := validDescribeRequest()
			tc.mutate(&req)
			err := req.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected valid, got %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("want %q in error, got %v", tc.wantErr, err)
			}
		})
	}
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd cli/go && go test ./pkg/stclient/ -run 'TestDescribe' -v`
Expected: compile FAILURE — `undefined: DescribeRequest` etc.

- [x] **Step 3: Write the implementation**

`cli/go/pkg/stclient/describe_types.go`:

```go
package stclient

import (
	"encoding/json"
	"fmt"
)

// Describe contract (spec: docs/superpowers/specs/2026-07-11-describe-analysis-
// interface-design.md). Closed enums; exactly one typed params block per task;
// server-side policy (mode -> analysis_profile -> delegates) is never expressed
// in the request.

type DescribeTaskKind string

const (
	TaskKindCaption DescribeTaskKind = "caption"
	TaskKindDetect  DescribeTaskKind = "detect"
	TaskKindOcr     DescribeTaskKind = "ocr"
	TaskKindPose    DescribeTaskKind = "pose"
	TaskKindEmbed   DescribeTaskKind = "embed"
)

// RolePrimary is the only target role with defined v1 semantics; omitted or
// empty Role means primary. Other role strings pass through as opaque labels.
const RolePrimary = "primary"

type DescribeRequest struct {
	Mode    *string          `json:"mode,omitempty"`
	Targets []DescribeTarget `json:"targets"`
	Tasks   []DescribeTask   `json:"tasks"`
}

type DescribeTarget struct {
	ID       string  `json:"id"`
	AssetRef *string `json:"asset_ref,omitempty"`
	URL      *string `json:"url,omitempty"`
	Role     string  `json:"role,omitempty"`
}

func (t DescribeTarget) effectiveRole() string {
	if t.Role == "" {
		return RolePrimary
	}
	return t.Role
}

type DescribeTask struct {
	ID        string           `json:"id"`
	Kind      DescribeTaskKind `json:"kind"`
	TargetIDs []string         `json:"target_ids,omitempty"`
	Caption   *CaptionParams   `json:"caption,omitempty"`
	Detect    *DetectParams    `json:"detect,omitempty"`
	Ocr       *OcrParams       `json:"ocr,omitempty"`
	Pose      *PoseParams      `json:"pose,omitempty"`
	Embed     *EmbedParams     `json:"embed,omitempty"`
}

// v1-minimal params; fields are additive later.
type CaptionParams struct {
	Prompt *string `json:"prompt,omitempty"`
}

type DetectParams struct {
	Labels        []string `json:"labels,omitempty"`
	MinConfidence *float64 `json:"min_confidence,omitempty"`
}

type OcrParams struct{}
type PoseParams struct{}
type EmbedParams struct{}

type DescribeStatus string

const (
	StatusOK      DescribeStatus = "ok"
	StatusPartial DescribeStatus = "partial"
	StatusFailed  DescribeStatus = "failed"
)

type DescribeResponse struct {
	Status       DescribeStatus        `json:"status"`
	Summary      *string               `json:"summary,omitempty"`
	Observations []DescribeObservation `json:"observations"`
	Artifacts    []DescribeArtifact    `json:"artifacts,omitempty"`
	Runs         []DescribeRun         `json:"runs"`
}

type ObservationKind string

const (
	ObsKindText      ObservationKind = "text"
	ObsKindDetection ObservationKind = "detection"
	ObsKindAttribute ObservationKind = "attribute"
	ObsKindKeypoints ObservationKind = "keypoints"
)

type DescribeObservation struct {
	TaskID   string          `json:"task_id"`
	TargetID string          `json:"target_id"`
	Kind     ObservationKind `json:"kind"`

	Text      *TextObservation      `json:"text,omitempty"`
	Detection *DetectionObservation `json:"detection,omitempty"`
	Attribute *AttributeObservation `json:"attribute,omitempty"`
	Keypoints *KeypointsObservation `json:"keypoints,omitempty"`
}

type TextObservation struct {
	Content string `json:"content"`
}

type DetectionObservation struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"` // [0,1]
	Box        Box     `json:"box"`
}

// Box coordinates are normalized to [0,1] relative to the target image;
// x,y is the top-left corner.
type Box struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	W float64 `json:"w"`
	H float64 `json:"h"`
}

type AttributeObservation struct {
	Name       string   `json:"name"`
	Value      string   `json:"value"`
	Confidence *float64 `json:"confidence,omitempty"` // [0,1]
	Box        *Box     `json:"box,omitempty"`
}

type KeypointsObservation struct {
	Skeleton string     `json:"skeleton,omitempty"`
	Points   []Keypoint `json:"points"`
}

type Keypoint struct {
	Name       string   `json:"name,omitempty"`
	X          float64  `json:"x"` // normalized [0,1]
	Y          float64  `json:"y"` // normalized [0,1]
	Confidence *float64 `json:"confidence,omitempty"`
}

type ArtifactKind string

const ArtifactKindEmbeddingRef ArtifactKind = "embedding_ref"

type DescribeArtifact struct {
	TaskID   string       `json:"task_id"`
	TargetID string       `json:"target_id"`
	Kind     ArtifactKind `json:"kind"`
	Ref      string       `json:"ref"`
	Dims     *int         `json:"dims,omitempty"`
}

type RunStatus string

const (
	RunSucceeded RunStatus = "succeeded"
	RunFailed    RunStatus = "failed"
	RunSkipped   RunStatus = "skipped"
)

type RunError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type DescribeRun struct {
	TaskID   string    `json:"task_id"`
	TargetID string    `json:"target_id"`
	Delegate string    `json:"delegate"`
	Status   RunStatus `json:"status"`
	Error    *RunError `json:"error,omitempty"`
	// RawOutput is the opaque provider payload; the contract deliberately
	// does not type it (spec: raw provider outputs preserved verbatim).
	RawOutput json.RawMessage `json:"raw_output,omitempty"`
}

func validationErr(code, format string, args ...any) error {
	return fmt.Errorf("%s: %s", code, fmt.Sprintf(format, args...))
}

// Validate applies the client-boundary contract rules. The server enforces
// the same rules; this exists so misuse fails before a request is sent.
func (r *DescribeRequest) Validate() error {
	if len(r.Targets) == 0 || len(r.Tasks) == 0 {
		return validationErr("analysis_invalid_request", "targets and tasks must be non-empty")
	}
	targetRoles := make(map[string]string, len(r.Targets))
	primaryCount := 0
	for _, tg := range r.Targets {
		if tg.ID == "" {
			return validationErr("analysis_invalid_request", "target id must be set")
		}
		if _, dup := targetRoles[tg.ID]; dup {
			return validationErr("analysis_invalid_request", "duplicate target id %q", tg.ID)
		}
		hasRef := tg.AssetRef != nil && *tg.AssetRef != ""
		hasURL := tg.URL != nil && *tg.URL != ""
		if hasRef == hasURL {
			return validationErr("analysis_invalid_request",
				"target %q must set exactly one of asset_ref or url", tg.ID)
		}
		targetRoles[tg.ID] = tg.effectiveRole()
		if tg.effectiveRole() == RolePrimary {
			primaryCount++
		}
	}
	taskIDs := make(map[string]bool, len(r.Tasks))
	for _, tk := range r.Tasks {
		if tk.ID == "" {
			return validationErr("analysis_invalid_request", "task id must be set")
		}
		if taskIDs[tk.ID] {
			return validationErr("analysis_invalid_request", "duplicate task id %q", tk.ID)
		}
		taskIDs[tk.ID] = true
		if err := tk.validateParams(); err != nil {
			return err
		}
		for _, id := range tk.TargetIDs {
			if _, ok := targetRoles[id]; !ok {
				return validationErr("analysis_target_binding_invalid",
					"task %q references unknown target %q", tk.ID, id)
			}
		}
		// Zero-run binding is a validation error: omitted target_ids requires
		// at least one effective-primary target.
		if len(tk.TargetIDs) == 0 && primaryCount == 0 {
			return validationErr("analysis_target_binding_invalid",
				"task %q binds to zero targets: no primary targets declared", tk.ID)
		}
	}
	return nil
}

func (t DescribeTask) validateParams() error {
	blocks := 0
	var matched bool
	if t.Caption != nil {
		blocks++
		matched = matched || t.Kind == TaskKindCaption
	}
	if t.Detect != nil {
		blocks++
		matched = matched || t.Kind == TaskKindDetect
	}
	if t.Ocr != nil {
		blocks++
		matched = matched || t.Kind == TaskKindOcr
	}
	if t.Pose != nil {
		blocks++
		matched = matched || t.Kind == TaskKindPose
	}
	if t.Embed != nil {
		blocks++
		matched = matched || t.Kind == TaskKindEmbed
	}
	switch t.Kind {
	case TaskKindCaption, TaskKindDetect, TaskKindOcr, TaskKindPose, TaskKindEmbed:
	default:
		return validationErr("analysis_invalid_request", "task %q has unknown kind %q", t.ID, t.Kind)
	}
	if blocks != 1 || !matched {
		return validationErr("analysis_invalid_request",
			"task %q must set exactly one params block matching kind %q", t.ID, t.Kind)
	}
	return nil
}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd cli/go && go test ./pkg/stclient/ -v`
Expected: all PASS (including pre-existing stclient tests — no regressions).

- [x] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient/describe_types.go cli/go/pkg/stclient/describe_types_test.go
git commit -m "feat(stclient): typed describe contract with client-boundary validation (STABL-tlklfaxz) — next: Task 2 python contracts"
```

---

### Task 2: Python wire contracts — `backends/analysis/contracts.py`

**Files:**
- Create: `backends/analysis/__init__.py`
- Create: `backends/analysis/contracts.py`
- Test: `tests/test_analysis_contracts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TaskKind`, `ObservationKind`, `RunStatus`, `DescribeStatus` (all `str`-enums); typed param dataclasses `CaptionParams`, `DetectParams`, `OcrParams`, `PoseParams`, `EmbedParams` (mirroring the Go contract — no untyped params anywhere); frozen dataclasses `DescribeTarget`, `DescribeTask` (with exactly-one optional typed params block), `DescribeRequest`, `TextObservation`, `DetectionObservation`, `AttributeObservation`, `KeypointsObservation`, `Box`, `Keypoint`, `DescribeObservation`, `DescribeArtifact`, `RunError`, `DescribeRun`, `DescribeResponse`; `AnalysisValidationError(code, message)`; `parse_describe_request(payload: dict) -> DescribeRequest`; `effective_role(target) -> str`; `PRIMARY_ROLE = "primary"`; `response_to_dict(resp) -> dict`. Tasks 4–5 consume these.

- [x] **Step 1: Write the failing tests**

`tests/test_analysis_contracts.py`:

```python
import pytest

from backends.analysis import (
    AnalysisValidationError,
    DescribeStatus,
    RunStatus,
    TaskKind,
    parse_describe_request,
    response_to_dict,
)
from backends.analysis.contracts import (
    Box,
    DescribeObservation,
    DescribeResponse,
    DescribeRun,
    DetectionObservation,
    RunError,
    TextObservation,
)


def valid_payload():
    return {
        "targets": [{"id": "t1", "asset_ref": "asset-1"}],
        "tasks": [{"id": "cap1", "kind": "caption", "caption": {}}],
    }


def test_parse_valid_request():
    req = parse_describe_request(valid_payload())
    assert req.mode is None
    assert req.targets[0].id == "t1"
    assert req.targets[0].asset_ref == "asset-1"
    assert req.tasks[0].kind == TaskKind.CAPTION
    # params materialize as the typed block matching kind, never a raw dict
    from backends.analysis import CaptionParams
    assert req.tasks[0].caption == CaptionParams()
    assert req.tasks[0].detect is None


def test_parse_typed_detect_params():
    payload = valid_payload()
    payload["tasks"] = [{
        "id": "det1", "kind": "detect",
        "detect": {"labels": ["owl"], "min_confidence": 0.5},
    }]
    from backends.analysis import DetectParams
    req = parse_describe_request(payload)
    assert req.tasks[0].detect == DetectParams(labels=("owl",), min_confidence=0.5)


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda p: p.update(targets=[]), "analysis_invalid_request"),
        (lambda p: p.update(tasks=[]), "analysis_invalid_request"),
        (lambda p: p["targets"][0].update(url="http://x/i.png"), "analysis_invalid_request"),
        (lambda p: p["targets"][0].pop("asset_ref"), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(kind="segment"), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(detect={}), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(target_ids=["nope"]), "analysis_target_binding_invalid"),
        (lambda p: p["targets"][0].update(role="reference"), "analysis_target_binding_invalid"),
    ],
)
def test_parse_rejects_invalid(mutate, code):
    payload = valid_payload()
    mutate(payload)
    with pytest.raises(AnalysisValidationError) as exc:
        parse_describe_request(payload)
    assert exc.value.code == code


def test_explicit_binding_to_non_primary_is_allowed():
    payload = valid_payload()
    payload["targets"][0]["role"] = "reference"
    payload["tasks"][0]["target_ids"] = ["t1"]
    req = parse_describe_request(payload)
    assert req.tasks[0].target_ids == ("t1",)


def test_response_to_dict_wire_shape():
    resp = DescribeResponse(
        status=DescribeStatus.PARTIAL,
        summary=None,
        observations=(
            DescribeObservation(
                task_id="cap1", target_id="t1", kind="text",
                text=TextObservation(content="an owl"),
            ),
            DescribeObservation(
                task_id="det1", target_id="t1", kind="detection",
                detection=DetectionObservation(
                    label="owl", confidence=0.93, box=Box(x=0.1, y=0.2, w=0.3, h=0.4),
                ),
            ),
        ),
        artifacts=(),
        runs=(
            DescribeRun(task_id="cap1", target_id="t1", delegate="vlm_caption",
                        status=RunStatus.SUCCEEDED),
            DescribeRun(task_id="det1", target_id="t1", delegate="",
                        status=RunStatus.SKIPPED,
                        error=RunError(code="analysis_no_supported_delegate",
                                       message="no route for kind detect")),
        ),
    )
    d = response_to_dict(resp)
    assert d["status"] == "partial"
    assert "summary" not in d
    obs0 = d["observations"][0]
    assert obs0 == {"task_id": "cap1", "target_id": "t1", "kind": "text",
                    "text": {"content": "an owl"}}
    obs1 = d["observations"][1]
    assert obs1["detection"]["box"] == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    run1 = d["runs"][1]
    assert run1["status"] == "skipped"
    assert run1["error"]["code"] == "analysis_no_supported_delegate"
    run0 = d["runs"][0]
    assert "error" not in run0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_contracts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.analysis'`

- [x] **Step 3: Write the implementation**

`backends/analysis/contracts.py`:

```python
"""Wire contracts for the describe/analysis capability.

Spec: docs/superpowers/specs/2026-07-11-describe-analysis-interface-design.md
Closed enums; exactly one typed params block per task; zero-run binding is a
request validation error so every DescribeResponse carries non-empty runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

PRIMARY_ROLE = "primary"


class AnalysisValidationError(ValueError):
    """Request/config validation failure with an operator-facing analysis_* code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class TaskKind(str, Enum):
    CAPTION = "caption"
    DETECT = "detect"
    OCR = "ocr"
    POSE = "pose"
    EMBED = "embed"


class ObservationKind(str, Enum):
    TEXT = "text"
    DETECTION = "detection"
    ATTRIBUTE = "attribute"
    KEYPOINTS = "keypoints"


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class DescribeStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class DescribeTarget:
    id: str
    asset_ref: Optional[str] = None
    url: Optional[str] = None
    role: str = ""


def effective_role(target: DescribeTarget) -> str:
    return target.role or PRIMARY_ROLE


# v1-minimal typed params, mirroring the Go contract field-for-field.
@dataclass(frozen=True)
class CaptionParams:
    prompt: Optional[str] = None


@dataclass(frozen=True)
class DetectParams:
    labels: Tuple[str, ...] = ()
    min_confidence: Optional[float] = None


@dataclass(frozen=True)
class OcrParams:
    pass


@dataclass(frozen=True)
class PoseParams:
    pass


@dataclass(frozen=True)
class EmbedParams:
    pass


@dataclass(frozen=True)
class DescribeTask:
    id: str
    kind: TaskKind
    target_ids: Tuple[str, ...] = ()
    # Exactly one typed params block is set, matching `kind`; parse enforces it.
    caption: Optional[CaptionParams] = None
    detect: Optional[DetectParams] = None
    ocr: Optional[OcrParams] = None
    pose: Optional[PoseParams] = None
    embed: Optional[EmbedParams] = None


@dataclass(frozen=True)
class DescribeRequest:
    targets: Tuple[DescribeTarget, ...]
    tasks: Tuple[DescribeTask, ...]
    mode: Optional[str] = None


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class TextObservation:
    content: str


@dataclass(frozen=True)
class DetectionObservation:
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class AttributeObservation:
    name: str
    value: str
    confidence: Optional[float] = None
    box: Optional[Box] = None


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    name: str = ""
    confidence: Optional[float] = None


@dataclass(frozen=True)
class KeypointsObservation:
    points: Tuple[Keypoint, ...]
    skeleton: str = ""


@dataclass(frozen=True)
class DescribeObservation:
    task_id: str
    target_id: str
    kind: str  # ObservationKind value
    text: Optional[TextObservation] = None
    detection: Optional[DetectionObservation] = None
    attribute: Optional[AttributeObservation] = None
    keypoints: Optional[KeypointsObservation] = None


@dataclass(frozen=True)
class DescribeArtifact:
    task_id: str
    target_id: str
    kind: str  # "embedding_ref" only in v1
    ref: str
    dims: Optional[int] = None


@dataclass(frozen=True)
class RunError:
    code: str
    message: str


@dataclass(frozen=True)
class DescribeRun:
    task_id: str
    target_id: str
    delegate: str
    status: RunStatus
    error: Optional[RunError] = None
    # Opaque provider payload; the contract deliberately does not type it.
    # Must be JSON-serializable; serialized verbatim, never restructured.
    raw_output: Optional[Any] = None


@dataclass(frozen=True)
class DescribeResponse:
    status: DescribeStatus
    observations: Tuple[DescribeObservation, ...]
    runs: Tuple[DescribeRun, ...]
    artifacts: Tuple[DescribeArtifact, ...] = ()
    summary: Optional[str] = None


_PARAM_KEYS = {
    TaskKind.CAPTION: "caption",
    TaskKind.DETECT: "detect",
    TaskKind.OCR: "ocr",
    TaskKind.POSE: "pose",
    TaskKind.EMBED: "embed",
}
_ALL_PARAM_KEYS = set(_PARAM_KEYS.values())


def _parse_params(kind: TaskKind, raw: Mapping[str, Any]):
    if kind == TaskKind.CAPTION:
        return CaptionParams(prompt=raw.get("prompt"))
    if kind == TaskKind.DETECT:
        return DetectParams(
            labels=tuple(raw.get("labels") or ()),
            min_confidence=raw.get("min_confidence"),
        )
    if kind == TaskKind.OCR:
        return OcrParams()
    if kind == TaskKind.POSE:
        return PoseParams()
    return EmbedParams()


def _invalid(message: str) -> AnalysisValidationError:
    return AnalysisValidationError("analysis_invalid_request", message)


def _binding_invalid(message: str) -> AnalysisValidationError:
    return AnalysisValidationError("analysis_target_binding_invalid", message)


def parse_describe_request(payload: Mapping[str, Any]) -> DescribeRequest:
    if not isinstance(payload, Mapping):
        raise _invalid("request body must be an object")
    raw_targets = payload.get("targets") or []
    raw_tasks = payload.get("tasks") or []
    if not raw_targets or not raw_tasks:
        raise _invalid("targets and tasks must be non-empty")

    targets = []
    roles: Dict[str, str] = {}
    primary_count = 0
    for raw in raw_targets:
        if not isinstance(raw, Mapping):
            raise _invalid("each target must be an object")
        target_id = (raw.get("id") or "").strip()
        if not target_id:
            raise _invalid("target id must be set")
        if target_id in roles:
            raise _invalid(f"duplicate target id '{target_id}'")
        asset_ref = raw.get("asset_ref")
        url = raw.get("url")
        if bool(asset_ref) == bool(url):
            raise _invalid(f"target '{target_id}' must set exactly one of asset_ref or url")
        target = DescribeTarget(
            id=target_id,
            asset_ref=asset_ref,
            url=url,
            role=(raw.get("role") or "").strip(),
        )
        roles[target_id] = effective_role(target)
        if roles[target_id] == PRIMARY_ROLE:
            primary_count += 1
        targets.append(target)

    tasks = []
    seen_task_ids = set()
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise _invalid("each task must be an object")
        task_id = (raw.get("id") or "").strip()
        if not task_id:
            raise _invalid("task id must be set")
        if task_id in seen_task_ids:
            raise _invalid(f"duplicate task id '{task_id}'")
        seen_task_ids.add(task_id)
        try:
            kind = TaskKind(raw.get("kind"))
        except ValueError:
            raise _invalid(f"task '{task_id}' has unknown kind '{raw.get('kind')}'")
        set_blocks = [k for k in _ALL_PARAM_KEYS if raw.get(k) is not None]
        if set_blocks != [_PARAM_KEYS[kind]]:
            raise _invalid(
                f"task '{task_id}' must set exactly one params block matching kind '{kind.value}'"
            )
        target_ids = tuple(raw.get("target_ids") or ())
        for tid in target_ids:
            if tid not in roles:
                raise _binding_invalid(f"task '{task_id}' references unknown target '{tid}'")
        if not target_ids and primary_count == 0:
            raise _binding_invalid(
                f"task '{task_id}' binds to zero targets: no primary targets declared"
            )
        tasks.append(
            DescribeTask(
                id=task_id,
                kind=kind,
                target_ids=target_ids,
                **{_PARAM_KEYS[kind]: _parse_params(kind, raw.get(_PARAM_KEYS[kind]) or {})},
            )
        )

    mode = payload.get("mode")
    return DescribeRequest(targets=tuple(targets), tasks=tuple(tasks), mode=mode)


def _drop_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def response_to_dict(resp: DescribeResponse) -> Dict[str, Any]:
    """Serialize to the wire shape pinned by the stclient contract tests."""

    def obs_dict(o: DescribeObservation) -> Dict[str, Any]:
        d: Dict[str, Any] = {"task_id": o.task_id, "target_id": o.target_id, "kind": o.kind}
        if o.text is not None:
            d["text"] = {"content": o.text.content}
        if o.detection is not None:
            d["detection"] = {
                "label": o.detection.label,
                "confidence": o.detection.confidence,
                "box": vars(o.detection.box).copy(),
            }
        if o.attribute is not None:
            d["attribute"] = _drop_nones({
                "name": o.attribute.name,
                "value": o.attribute.value,
                "confidence": o.attribute.confidence,
                "box": vars(o.attribute.box).copy() if o.attribute.box else None,
            })
        if o.keypoints is not None:
            d["keypoints"] = _drop_nones({
                "skeleton": o.keypoints.skeleton or None,
                "points": [
                    _drop_nones({"name": p.name or None, "x": p.x, "y": p.y,
                                 "confidence": p.confidence})
                    for p in o.keypoints.points
                ],
            })
        return d

    def run_dict(r: DescribeRun) -> Dict[str, Any]:
        return _drop_nones({
            "task_id": r.task_id,
            "target_id": r.target_id,
            "delegate": r.delegate,
            "status": r.status.value,
            "error": {"code": r.error.code, "message": r.error.message} if r.error else None,
            "raw_output": r.raw_output,  # opaque passthrough, never restructured
        })

    out: Dict[str, Any] = {
        "status": resp.status.value,
        "observations": [obs_dict(o) for o in resp.observations],
        "runs": [run_dict(r) for r in resp.runs],
    }
    if resp.summary is not None:
        out["summary"] = resp.summary
    if resp.artifacts:
        out["artifacts"] = [
            _drop_nones({
                "task_id": a.task_id, "target_id": a.target_id,
                "kind": a.kind, "ref": a.ref, "dims": a.dims,
            })
            for a in resp.artifacts
        ]
    return out
```

`backends/analysis/__init__.py`:

```python
from .contracts import (
    PRIMARY_ROLE,
    AnalysisValidationError,
    AttributeObservation,
    Box,
    CaptionParams,
    DescribeArtifact,
    DescribeObservation,
    DescribeRequest,
    DescribeResponse,
    DescribeRun,
    DescribeStatus,
    DescribeTarget,
    DescribeTask,
    DetectParams,
    DetectionObservation,
    EmbedParams,
    Keypoint,
    KeypointsObservation,
    ObservationKind,
    OcrParams,
    PoseParams,
    RunError,
    RunStatus,
    TaskKind,
    TextObservation,
    effective_role,
    parse_describe_request,
    response_to_dict,
)

__all__ = [
    "PRIMARY_ROLE",
    "AnalysisValidationError",
    "AttributeObservation",
    "Box",
    "CaptionParams",
    "DetectParams",
    "EmbedParams",
    "OcrParams",
    "PoseParams",
    "DescribeArtifact",
    "DescribeObservation",
    "DescribeRequest",
    "DescribeResponse",
    "DescribeRun",
    "DescribeStatus",
    "DescribeTarget",
    "DescribeTask",
    "DetectionObservation",
    "Keypoint",
    "KeypointsObservation",
    "ObservationKind",
    "RunError",
    "RunStatus",
    "TaskKind",
    "TextObservation",
    "effective_role",
    "parse_describe_request",
    "response_to_dict",
]
```

- [x] **Step 4: Run tests to verify they pass**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_contracts.py -v`
Expected: all PASS.

- [x] **Step 5: Commit**

```bash
git add backends/analysis/__init__.py backends/analysis/contracts.py tests/test_analysis_contracts.py
git commit -m "feat(analysis): python wire contracts + request validation (STABL-tlklfaxz) — next: Task 3 mode-config parsing"
```

---

### Task 3: Mode-config `analysis_*` sections

**Files:**
- Modify: `server/mode_config.py` (dataclasses near `ChatDelegateConfig` ~line 57; parse calls in the loader near the `chat_delegates` block ~line 240; parse helpers near `_parse_chat_delegate_config` ~line 540; `ModesYAML` ~line 129; `ModeConfig` ~line 93)
- Test: `tests/test_analysis_mode_config.py`

**Interfaces:**
- Consumes: `TaskKind` from `backends.analysis` (closed kind set for delegate validation).
- Produces: `AnalysisConnectionConfig(endpoint, api_key_env)`, `AnalysisDelegateConfig(name, connection, kind, model)`, `AnalysisProfileConfig(name, task_routes: Dict[str, str])`; `ModesYAML.analysis_connections/analysis_delegates/analysis_profiles` dicts; `ModeConfig.analysis_profile: Optional[str]`. Task 5's orchestrator wiring consumes `AnalysisProfileConfig` and `AnalysisDelegateConfig`.

- [x] **Step 1: Write the failing tests**

`tests/test_analysis_mode_config.py`. Follow the existing mode-config test pattern: write a temp `modes.yml`, load via `ModeConfigManager`, assert. Base fixture:

```python
import textwrap

import pytest

from server.mode_config import ModeConfigManager

BASE_YAML = textwrap.dedent("""\
    model_root: /tmp/models
    lora_root: /tmp/loras
    default_mode: SDXL
    resolution_sets:
      default:
        - size: 1024x1024
          aspect_ratio: "1:1"
    analysis_connections:
      local_vlm:
        endpoint: "http://node2.lan:8080/v1"
        api_key_env: "OPENAI_API_KEY"
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


def load(tmp_path, yaml_text):
    # ModeConfigManager takes the config *directory* and appends modes.yml
    # itself (server/mode_config.py:164) — pass tmp_path, not the file.
    (tmp_path / "modes.yml").write_text(yaml_text)
    return ModeConfigManager(str(tmp_path))


def test_parses_analysis_sections(tmp_path):
    mgr = load(tmp_path, BASE_YAML)
    cfg = mgr.config
    assert cfg.analysis_connections["local_vlm"].endpoint == "http://node2.lan:8080/v1"
    assert cfg.analysis_delegates["vlm_caption"].kind == "caption"
    assert cfg.analysis_delegates["vlm_caption"].connection == "local_vlm"
    assert cfg.analysis_profiles["default"].task_routes == {
        "caption": "vlm_caption", "detect": "yolo_detect",
    }
    assert cfg.modes["SDXL"].analysis_profile == "default"


def test_sections_default_empty(tmp_path):
    # Drop every analysis_* section and the mode's analysis_profile line.
    yaml_text = BASE_YAML[: BASE_YAML.index("analysis_connections:")] + BASE_YAML[BASE_YAML.index("modes:"):]
    yaml_text = yaml_text.replace("        analysis_profile: default\n", "")
    cfg = load(tmp_path, yaml_text).config
    assert cfg.analysis_connections == {}
    assert cfg.analysis_profiles == {}
    assert cfg.modes["SDXL"].analysis_profile is None


@pytest.mark.parametrize(
    "needle,replacement,err_fragment",
    [
        # delegate references unknown connection
        ("connection: local_vlm", "connection: nope", "unknown connection"),
        # delegate kind outside closed enum
        ("kind: caption", "kind: segment", "kind"),
        # profile routes to unknown delegate
        ("caption: vlm_caption", "caption: nope", "unknown delegate"),
        # route key != delegate kind -> analysis_delegate_kind_mismatch
        ("caption: vlm_caption", "detect: vlm_caption", "analysis_delegate_kind_mismatch"),
        # mode references unknown profile
        ("analysis_profile: default", "analysis_profile: nope", "unknown analysis_profile"),
    ],
)
def test_fail_fast_validation(tmp_path, needle, replacement, err_fragment):
    bad = BASE_YAML.replace(needle, replacement, 1)
    with pytest.raises(ValueError, match=err_fragment):
        load(tmp_path, bad)
```

Note: the duplicate-route case (`detect: vlm_caption` replacing `caption: vlm_caption`) leaves `detect` mapped twice in YAML; YAML takes the last value, so the surviving route is `detect: vlm_caption`, whose delegate kind is `caption` — the mismatch fires. Adjust the fixture strings if the loaded YAML behaves differently; the assertion that matters is `analysis_delegate_kind_mismatch`.

- [x] **Step 2: Run tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_mode_config.py -v`
Expected: FAIL — `AttributeError` (no `analysis_connections` on config) or `TypeError` on `ModesYAML`.

- [x] **Step 3: Implement in `server/mode_config.py`**

Dataclasses (place after `ChatDelegateConfig`):

```python
@dataclass
class AnalysisConnectionConfig:
    """Reusable transport settings for analysis backends."""
    endpoint: str
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class AnalysisDelegateConfig:
    """Named analyzer backend: connection + kind capability + model."""
    name: str
    connection: str  # key into analysis_connections
    kind: str        # closed TaskKind value; capability declaration
    model: str


@dataclass
class AnalysisProfileConfig:
    """Maps task kinds to delegate names; modes select one profile."""
    name: str
    task_routes: Dict[str, str] = field(default_factory=dict)
```

Fields: add to `ModesYAML` (after `chat_delegates`):

```python
    analysis_connections: Dict[str, AnalysisConnectionConfig]
    analysis_delegates: Dict[str, AnalysisDelegateConfig]
    analysis_profiles: Dict[str, AnalysisProfileConfig]
```

Add to `ModeConfig` (after `chat_delegate`):

```python
    analysis_profile: Optional[str] = None
```

Parse helpers (place after `_parse_chat_delegate_config`). The closed kind set comes from `backends.analysis.TaskKind`:

```python
    def _parse_analysis_connection_config(self, name: str, raw: Dict[str, Any]) -> AnalysisConnectionConfig:
        if not isinstance(raw, dict):
            raise ValueError(f"Analysis connection '{name}' must be a mapping")
        endpoint = (raw.get("endpoint") or "").strip()
        if not endpoint:
            raise ValueError(f"Analysis connection '{name}' missing required field: endpoint")
        return AnalysisConnectionConfig(
            endpoint=endpoint,
            api_key_env=(raw.get("api_key_env") or "OPENAI_API_KEY").strip(),
        )

    def _parse_analysis_delegate_config(
        self,
        name: str,
        raw: Dict[str, Any],
        connections: Dict[str, AnalysisConnectionConfig],
    ) -> AnalysisDelegateConfig:
        from backends.analysis import TaskKind

        if not isinstance(raw, dict):
            raise ValueError(f"Analysis delegate '{name}' must be a mapping")
        connection = (raw.get("connection") or "").strip()
        if not connection:
            raise ValueError(f"Analysis delegate '{name}' missing required field: connection")
        if connection not in connections:
            raise ValueError(f"Analysis delegate '{name}' references unknown connection '{connection}'")
        kind = (raw.get("kind") or "").strip()
        valid_kinds = {k.value for k in TaskKind}
        if kind not in valid_kinds:
            raise ValueError(
                f"Analysis delegate '{name}' has invalid kind '{kind}' (expected one of {sorted(valid_kinds)})"
            )
        model = (raw.get("model") or "").strip()
        if not model:
            raise ValueError(f"Analysis delegate '{name}' missing required field: model")
        return AnalysisDelegateConfig(name=name, connection=connection, kind=kind, model=model)

    def _parse_analysis_profile_config(
        self,
        name: str,
        raw: Dict[str, Any],
        delegates: Dict[str, AnalysisDelegateConfig],
    ) -> AnalysisProfileConfig:
        if not isinstance(raw, dict):
            raise ValueError(f"Analysis profile '{name}' must be a mapping")
        raw_routes = raw.get("task_routes")
        if not isinstance(raw_routes, dict) or not raw_routes:
            raise ValueError(f"Analysis profile '{name}' missing required mapping: task_routes")
        task_routes: Dict[str, str] = {}
        for route_kind, delegate_name in raw_routes.items():
            delegate_name = (str(delegate_name) or "").strip()
            if delegate_name not in delegates:
                raise ValueError(
                    f"Analysis profile '{name}' route '{route_kind}' references unknown delegate '{delegate_name}'"
                )
            delegate_kind = delegates[delegate_name].kind
            if route_kind != delegate_kind:
                raise ValueError(
                    f"analysis_delegate_kind_mismatch: profile '{name}' routes kind "
                    f"'{route_kind}' to delegate '{delegate_name}' of kind '{delegate_kind}'"
                )
            task_routes[str(route_kind)] = delegate_name
        return AnalysisProfileConfig(name=name, task_routes=task_routes)
```

Loader wiring — after the `chat_delegates` block (~line 250), mirroring its shape exactly:

```python
        raw_analysis_connections = data.get("analysis_connections") or {}
        if not isinstance(raw_analysis_connections, dict):
            raise ValueError("modes.yml field 'analysis_connections' must be a mapping")
        analysis_connections = {
            name: self._parse_analysis_connection_config(name, raw)
            for name, raw in raw_analysis_connections.items()
        }

        raw_analysis_delegates = data.get("analysis_delegates") or {}
        if not isinstance(raw_analysis_delegates, dict):
            raise ValueError("modes.yml field 'analysis_delegates' must be a mapping")
        analysis_delegates = {
            name: self._parse_analysis_delegate_config(name, raw, analysis_connections)
            for name, raw in raw_analysis_delegates.items()
        }

        raw_analysis_profiles = data.get("analysis_profiles") or {}
        if not isinstance(raw_analysis_profiles, dict):
            raise ValueError("modes.yml field 'analysis_profiles' must be a mapping")
        analysis_profiles = {
            name: self._parse_analysis_profile_config(name, raw, analysis_delegates)
            for name, raw in raw_analysis_profiles.items()
        }
```

In the per-mode loop (next to the `chat_delegate` handling, ~line 278):

```python
            analysis_profile = self._normalize_optional_string(mode_data.get("analysis_profile"))
            if analysis_profile and analysis_profile not in analysis_profiles:
                raise ValueError(
                    f"Mode '{mode_name}' references unknown analysis_profile '{analysis_profile}'"
                )
```

Pass `analysis_profile=analysis_profile` in the `ModeConfig(...)` construction (next to `chat_delegate=chat_delegate`), and `analysis_connections=analysis_connections, analysis_delegates=analysis_delegates, analysis_profiles=analysis_profiles` in the `ModesYAML(...)` construction.

- [x] **Step 4: Run tests — new file and the existing mode-config suite**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_mode_config.py tests/test_mode_config*.py -v`
Expected: all PASS (existing mode-config tests construct `ModesYAML` via the loader, but if any construct it directly they will fail on the three new required fields — fix those call sites by adding empty dicts, or give the three `ModesYAML` fields `field(default_factory=dict)` defaults; prefer defaults since the chat fields are positional-required only for historical reasons).

- [x] **Step 5: Commit**

```bash
git add server/mode_config.py tests/test_analysis_mode_config.py
git commit -m "config(analysis): analysis_connections/delegates/profiles with kind-route invariant (STABL-tlklfaxz) — next: Task 4 run expansion"
```

---

### Task 4: Run expansion — `backends/analysis/orchestrator.py` (part 1)

**Files:**
- Create: `backends/analysis/orchestrator.py`
- Test: `tests/test_analysis_orchestrator.py`

**Interfaces:**
- Consumes: `DescribeRequest`, `DescribeTask`, `effective_role`, `PRIMARY_ROLE`, `AnalysisValidationError` from `backends.analysis`; `AnalysisProfileConfig` shape from Task 3 (only `task_routes: Mapping[str, str]` is used — accept any mapping to avoid a server import in the backends package).
- Produces: `RunPlan(task_id, target_id, delegate: Optional[str], skip_error: Optional[RunError])` frozen dataclass and `expand_runs(request, task_routes: Mapping[str, str]) -> tuple[RunPlan, ...]`. Task 5 dispatches these.

- [x] **Step 1: Write the failing tests**

Append to a new `tests/test_analysis_orchestrator.py`:

```python
import pytest

from backends.analysis import parse_describe_request
from backends.analysis.orchestrator import RunPlan, expand_runs

ROUTES = {"caption": "vlm_caption", "detect": "yolo_detect"}


def two_target_payload():
    return {
        "targets": [
            {"id": "t1", "asset_ref": "asset-1"},
            {"id": "t2", "asset_ref": "asset-2", "role": "reference"},
        ],
        "tasks": [
            {"id": "cap1", "kind": "caption", "caption": {}},
            {"id": "det1", "kind": "detect", "target_ids": ["t2"], "detect": {}},
        ],
    }


def test_expand_binds_omitted_target_ids_to_primary_only():
    req = parse_describe_request(two_target_payload())
    runs = expand_runs(req, ROUTES)
    assert (
        RunPlan(task_id="cap1", target_id="t1", delegate="vlm_caption", skip_error=None)
        in runs
    )
    assert not any(r.task_id == "cap1" and r.target_id == "t2" for r in runs)


def test_expand_explicit_target_ids_bind_verbatim():
    req = parse_describe_request(two_target_payload())
    runs = expand_runs(req, ROUTES)
    det = [r for r in runs if r.task_id == "det1"]
    assert det == [RunPlan(task_id="det1", target_id="t2", delegate="yolo_detect", skip_error=None)]


def test_expand_unrouted_kind_produces_skip_plan():
    payload = two_target_payload()
    payload["tasks"].append({"id": "ocr1", "kind": "ocr", "ocr": {}})
    req = parse_describe_request(payload)
    runs = expand_runs(req, ROUTES)
    ocr = [r for r in runs if r.task_id == "ocr1"]
    assert len(ocr) == 1
    assert ocr[0].delegate is None
    assert ocr[0].skip_error.code == "analysis_no_supported_delegate"


def test_expand_is_deterministic_task_major_order():
    req = parse_describe_request(two_target_payload())
    assert expand_runs(req, ROUTES) == expand_runs(req, ROUTES)
    assert [r.task_id for r in expand_runs(req, ROUTES)] == ["cap1", "det1"]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.analysis.orchestrator'`

- [x] **Step 3: Write the implementation**

`backends/analysis/orchestrator.py` (part 1 — Task 5 extends this file):

```python
"""Async orchestration for the describe/analysis capability.

Chain shape mirrors backends/conditioning: validate -> resolve profile ->
expand runs -> dispatch -> normalize -> assemble. Providers stay simple; the
composition lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .contracts import (
    PRIMARY_ROLE,
    DescribeRequest,
    RunError,
    effective_role,
)


@dataclass(frozen=True)
class RunPlan:
    """One concrete (task, target) execution unit produced by expansion.

    delegate is None only for skip plans, which always carry skip_error.
    """
    task_id: str
    target_id: str
    delegate: Optional[str]
    skip_error: Optional[RunError] = None


def expand_runs(request: DescribeRequest, task_routes: Mapping[str, str]) -> Tuple[RunPlan, ...]:
    """Expand tasks x bound targets into RunPlans, task-major order.

    Requests reaching here already passed parse_describe_request, so every
    task binds to >=1 target; an unrouted kind yields one skip plan per
    bound target rather than a validation error (spec: RunStatus skipped).
    """
    primary_ids = [t.id for t in request.targets if effective_role(t) == PRIMARY_ROLE]
    plans = []
    for task in request.tasks:
        bound = list(task.target_ids) if task.target_ids else primary_ids
        delegate = task_routes.get(task.kind.value)
        for target_id in bound:
            if delegate is None:
                plans.append(RunPlan(
                    task_id=task.id,
                    target_id=target_id,
                    delegate=None,
                    skip_error=RunError(
                        code="analysis_no_supported_delegate",
                        message=f"no route for kind {task.kind.value}",
                    ),
                ))
            else:
                plans.append(RunPlan(task_id=task.id, target_id=target_id, delegate=delegate))
    return tuple(plans)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_orchestrator.py tests/test_analysis_contracts.py -v`
Expected: all PASS.

- [x] **Step 5: Commit**

```bash
git add backends/analysis/orchestrator.py tests/test_analysis_orchestrator.py
git commit -m "feat(analysis): deterministic run expansion with skip plans (STABL-tlklfaxz) — next: Task 5 dispatch + stub provider"
```

---

### Task 5: Async dispatch, stub provider, response assembly

**Files:**
- Create: `backends/analysis/providers.py`
- Modify: `backends/analysis/orchestrator.py` (append)
- Modify: `backends/analysis/__init__.py` (add exports)
- Test: `tests/test_analysis_orchestrator.py` (append)

**Interfaces:**
- Consumes: `RunPlan`, `expand_runs` (Task 4); contracts (Task 2).
- Produces: `ProviderRun(plan: RunPlan, task: DescribeTask, target: DescribeTarget)`; `ProviderResult(observations, artifacts, raw_output)`; `DescribeProvider` protocol (`supports(task) -> bool`, `async run(provider_run) -> ProviderResult`); `StubProvider(kind, observation_factory=None)`; `AnalysisOrchestrator(task_routes, providers: Mapping[str, DescribeProvider]).describe(request) -> DescribeResponse`. Future transport wiring (out of this plan) consumes `AnalysisOrchestrator`.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_analysis_orchestrator.py`:

```python
import asyncio

from backends.analysis import (
    DescribeObservation,
    DescribeStatus,
    RunStatus,
    TextObservation,
)
from backends.analysis.orchestrator import AnalysisOrchestrator
from backends.analysis.providers import ProviderResult, StubProvider


class ExplodingProvider:
    def supports(self, task):
        return True

    async def run(self, provider_run):
        raise RuntimeError("backend unreachable")


def run_describe(orchestrator, payload):
    req = parse_describe_request(payload)
    return asyncio.run(orchestrator.describe(req))


def single_caption_payload():
    return {
        "targets": [{"id": "t1", "asset_ref": "asset-1"}],
        "tasks": [{"id": "cap1", "kind": "caption", "caption": {}}],
    }


def test_all_success_is_ok_with_correlated_observations():
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption"},
        providers={"vlm_caption": StubProvider(kind="caption")},
    )
    resp = run_describe(orch, single_caption_payload())
    assert resp.status == DescribeStatus.OK
    assert resp.summary is None  # orchestrator-owned; unset in v1
    assert len(resp.runs) == 1
    run = resp.runs[0]
    assert (run.task_id, run.target_id, run.delegate, run.status) == (
        "cap1", "t1", "vlm_caption", RunStatus.SUCCEEDED,
    )
    obs = resp.observations[0]
    assert (obs.task_id, obs.target_id, obs.kind) == ("cap1", "t1", "text")
    assert obs.text.content  # stub emits non-empty text


def test_provider_exception_isolates_to_failed_run_partial_status():
    payload = single_caption_payload()
    payload["tasks"].append({"id": "det1", "kind": "detect", "detect": {}})
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption", "detect": "yolo_detect"},
        providers={
            "vlm_caption": StubProvider(kind="caption"),
            "yolo_detect": ExplodingProvider(),
        },
    )
    resp = run_describe(orch, payload)
    assert resp.status == DescribeStatus.PARTIAL
    by_task = {r.task_id: r for r in resp.runs}
    assert by_task["cap1"].status == RunStatus.SUCCEEDED
    failed = by_task["det1"]
    assert failed.status == RunStatus.FAILED
    assert failed.error is not None
    assert failed.error.code == "analysis_run_failed"
    assert "backend unreachable" in failed.error.message
    # sibling isolation: the caption observation still landed
    assert any(o.task_id == "cap1" for o in resp.observations)


def test_all_failed_status_failed():
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption"},
        providers={"vlm_caption": ExplodingProvider()},
    )
    resp = run_describe(orch, single_caption_payload())
    assert resp.status == DescribeStatus.FAILED
    assert resp.observations == ()


def test_unrouted_kind_yields_skipped_run_and_partial():
    payload = single_caption_payload()
    payload["tasks"].append({"id": "ocr1", "kind": "ocr", "ocr": {}})
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption"},
        providers={"vlm_caption": StubProvider(kind="caption")},
    )
    resp = run_describe(orch, payload)
    assert resp.status == DescribeStatus.PARTIAL
    skipped = [r for r in resp.runs if r.status == RunStatus.SKIPPED]
    assert len(skipped) == 1
    assert skipped[0].error.code == "analysis_no_supported_delegate"
```

The failed-run code is pinned deliberately: `analysis_run_failed` is the
blessed contract code for any single run whose delegate was invoked and
raised (see Global Constraints and the spec's Error Vocabulary). The
provider-specific detail lives in `error.message`; `analysis_all_runs_failed`
remains reserved for the aggregate request-level case.

- [x] **Step 2: Run tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.analysis.providers'`

- [x] **Step 3: Write the implementation**

`backends/analysis/providers.py`:

```python
"""Provider protocol and the v1 stub provider."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

from .contracts import (
    DescribeArtifact,
    DescribeObservation,
    DescribeTarget,
    DescribeTask,
    TextObservation,
)
from .orchestrator import RunPlan


@dataclass(frozen=True)
class ProviderRun:
    """Everything a provider needs for one concrete run."""
    plan: RunPlan
    task: DescribeTask
    target: DescribeTarget


@dataclass(frozen=True)
class ProviderResult:
    observations: Tuple[DescribeObservation, ...] = ()
    artifacts: Tuple[DescribeArtifact, ...] = ()
    # Opaque, JSON-serializable provider payload; passed through verbatim.
    raw_output: Optional[Any] = None


class DescribeProvider(Protocol):
    def supports(self, task: DescribeTask) -> bool: ...
    async def run(self, provider_run: ProviderRun) -> ProviderResult: ...


ObservationFactory = Callable[[ProviderRun], Tuple[DescribeObservation, ...]]


def _default_text_observation(provider_run: ProviderRun) -> Tuple[DescribeObservation, ...]:
    return (
        DescribeObservation(
            task_id=provider_run.plan.task_id,
            target_id=provider_run.plan.target_id,
            kind="text",
            text=TextObservation(content=f"stub:{provider_run.task.kind.value}"),
        ),
    )


@dataclass(frozen=True)
class StubProvider:
    """Deterministic in-process provider for contract tests."""
    kind: str
    observation_factory: ObservationFactory = field(default=_default_text_observation)

    def supports(self, task: DescribeTask) -> bool:
        return task.kind.value == self.kind

    async def run(self, provider_run: ProviderRun) -> ProviderResult:
        return ProviderResult(
            observations=self.observation_factory(provider_run),
            raw_output={"stub": True, "kind": self.kind},
        )
```

Append to `backends/analysis/orchestrator.py`:

```python
import asyncio

from .contracts import (
    DescribeArtifact,
    DescribeObservation,
    DescribeResponse,
    DescribeRun,
    DescribeStatus,
    RunStatus,
)


class AnalysisOrchestrator:
    """Owns validation, routing, dispatch, normalization, and assembly.

    Providers are keyed by delegate name. Runs against distinct delegates
    execute concurrently; per-run failure is isolated and degrades the
    response status rather than aborting siblings.
    """

    def __init__(self, task_routes, providers):
        self._task_routes = dict(task_routes)
        self._providers = dict(providers)

    async def describe(self, request: DescribeRequest) -> DescribeResponse:
        from .providers import ProviderRun  # local import: providers imports RunPlan from here

        plans = expand_runs(request, self._task_routes)
        tasks_by_id = {t.id: t for t in request.tasks}
        targets_by_id = {t.id: t for t in request.targets}

        async def execute(plan: RunPlan):
            if plan.delegate is None:
                return plan, None, plan.skip_error
            provider = self._providers.get(plan.delegate)
            if provider is None:
                return plan, None, RunError(
                    code="analysis_delegate_not_found",
                    message=f"no provider registered for delegate '{plan.delegate}'",
                )
            provider_run = ProviderRun(
                plan=plan,
                task=tasks_by_id[plan.task_id],
                target=targets_by_id[plan.target_id],
            )
            try:
                result = await provider.run(provider_run)
                return plan, result, None
            except Exception as exc:  # per-run isolation is the contract
                return plan, None, RunError(
                    code="analysis_run_failed",
                    message=f"{type(exc).__name__}: {exc}",
                )

        outcomes = await asyncio.gather(*(execute(p) for p in plans))

        observations: list[DescribeObservation] = []
        artifacts: list[DescribeArtifact] = []
        runs: list[DescribeRun] = []
        for plan, result, error in outcomes:
            if plan.delegate is None:
                runs.append(DescribeRun(
                    task_id=plan.task_id, target_id=plan.target_id, delegate="",
                    status=RunStatus.SKIPPED, error=error,
                ))
            elif error is not None:
                runs.append(DescribeRun(
                    task_id=plan.task_id, target_id=plan.target_id, delegate=plan.delegate,
                    status=RunStatus.FAILED, error=error,
                ))
            else:
                observations.extend(result.observations)
                artifacts.extend(result.artifacts)
                runs.append(DescribeRun(
                    task_id=plan.task_id, target_id=plan.target_id, delegate=plan.delegate,
                    status=RunStatus.SUCCEEDED, raw_output=result.raw_output,
                ))

        succeeded = sum(1 for r in runs if r.status == RunStatus.SUCCEEDED)
        if succeeded == len(runs):
            status = DescribeStatus.OK
        elif succeeded > 0:
            status = DescribeStatus.PARTIAL
        else:
            status = DescribeStatus.FAILED

        return DescribeResponse(
            status=status,
            observations=tuple(observations),
            artifacts=tuple(artifacts),
            runs=tuple(runs),
            summary=None,  # orchestrator-owned; deliberately unset in v1
        )
```

Add to `backends/analysis/__init__.py` imports/`__all__`: `RunPlan`, `expand_runs`, `AnalysisOrchestrator` (from `.orchestrator`), `DescribeProvider`, `ProviderResult`, `ProviderRun`, `StubProvider` (from `.providers`).

- [x] **Step 4: Run the full analysis suite plus mode-config**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_analysis_orchestrator.py tests/test_analysis_contracts.py tests/test_analysis_mode_config.py -v`
Expected: all PASS.

- [x] **Step 5: Run the Go suite once more for cross-task regression**

Run: `cd cli/go && go test ./pkg/stclient/ -v`
Expected: all PASS.

- [x] **Step 6: Commit**

```bash
git add backends/analysis/providers.py backends/analysis/orchestrator.py backends/analysis/__init__.py tests/test_analysis_orchestrator.py
git commit -m "feat(analysis): async orchestrator + stub provider with per-run isolation (STABL-tlklfaxz) — next: review, then transport/CLI track"
```

---

## Out of Scope (deferred to a follow-on plan)

- Transport (WS vs HTTP) and the server route for `describe`.
- `st describe` CLI and the `stclient.Describe(...)` transport call.
- Real VLM/YOLO providers behind `analysis_connections`.
- `mask` observation/artifact kinds and `Summary` population.
