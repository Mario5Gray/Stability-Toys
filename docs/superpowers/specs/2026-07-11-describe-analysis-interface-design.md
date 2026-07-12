# Describe Capability Interface — v1 Contract Design

**FP issue:** STABL-tlklfaxz
**Source brainstorm:** `fp://brainstorm?id=rpoxcauqeltrgqplfejzlxdpyqkijlhn` (frozen at v3)
**Status:** Authority artifact. This spec supersedes the brainstorm; the brainstorm is frozen and its `Draft v1 / For Theta Review` labels are historical, not current state.

## Goal

Add a server-owned `describe` capability that fronts multiple analyzer backend
families — a visual LLM, a YOLO-style detector, and future analyzers (OCR,
pose, embeddings) — behind one typed contract, without forcing all backends
into a single lossy output shape.

This spec fixes the v1 request/response contract, the mode-config policy
shape, correlation and failure semantics, and the implementation-framework
direction. It does not choose a transport (WS vs HTTP) or implement any real
provider.

## Ownership Model

| Layer | Owns |
| --- | --- |
| Server | provider selection and fan-out, mode/profile policy, run expansion, result normalization, provenance and raw-output retention, backend integration |
| `cli/go/pkg/stclient` | typed request/response contracts, transport call surface, decode/validation at the client boundary |
| `st` CLI | argument parsing, stdin/stdout composition, file and pipe ergonomics — zero capability policy beyond selecting request fields |

Naming: the user-facing operation is `describe`; the internal capability
family and all config keys are `analysis_*`. **All operator-facing validation
and configuration errors use the `analysis_*` vocabulary**, even where the CLI
verb is `describe`.

## Request Contract

```go
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
```

Contract rules:

- `DescribeTaskKind` is a closed enum: `caption | detect | ocr | pose | embed`.
  Unknown kinds are a client-side validation error before any request is sent.
- Each task sets **exactly one** typed params block, and it must match `kind`.
  No `map[string]any` params exist anywhere in the public library contract.
- `DescribeTarget` is exactly-one-of `asset_ref` / `url`. Both set or neither
  set fails validation (`analysis_invalid_request`).
- **Target roles (followup 2 resolved):** `Role` omitted or empty means
  `primary`. `primary` is the only role with defined semantics in v1; other
  role strings are accepted, carried through as opaque labels, and never
  interpreted by routing. A task with `target_ids` omitted binds to all
  targets whose effective role is `primary`. Explicit `target_ids` must
  reference declared target IDs (`analysis_target_binding_invalid` otherwise).
- **Zero-run binding is a validation error:** after binding resolution, every
  task must bind to at least one target. A task that binds to zero targets —
  including the case where `target_ids` is omitted and no target has
  effective role `primary` — fails request validation with a non-2xx
  `analysis_target_binding_invalid` error naming the offending `task_id`. No
  `DescribeResponse` is produced, no synthetic skipped runs exist, and
  consequently every `DescribeResponse` carries a non-empty `runs` array.
- There is no request-level `profile`, provider, or delegate field in v1. The
  server resolves the analysis profile from the effective mode
  (`request.mode` when provided, otherwise the server's active mode).

## Response Contract

```go
type DescribeResponse struct {
    Status       DescribeStatus        `json:"status"`
    Summary      *string               `json:"summary,omitempty"`
    Observations []DescribeObservation `json:"observations"`
    Artifacts    []DescribeArtifact    `json:"artifacts,omitempty"`
    Runs         []DescribeRun         `json:"runs"`
}
```

- `DescribeStatus` is a closed enum: `ok | partial | failed`.
  - `ok`: all concrete runs succeeded.
  - `partial`: at least one run succeeded and at least one run failed or was
    skipped.
  - `failed`: no concrete run succeeded.
- `Runs` is always non-empty (zero-run requests fail validation; see Request
  Contract).

### Observation and Artifact Types

`DescribeObservation` follows the same pattern as `DescribeTask`: a closed
`kind` enum plus exactly one typed payload block matching `kind`.

```go
type DescribeObservation struct {
    TaskID   string          `json:"task_id"`
    TargetID string          `json:"target_id"`
    Kind     ObservationKind `json:"kind"` // text | detection | attribute | keypoints

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
    Box        *Box     `json:"box,omitempty"`         // set for localized OCR fields
}

type KeypointsObservation struct {
    Skeleton string     `json:"skeleton,omitempty"` // e.g. "coco17"; opaque label in v1
    Points   []Keypoint `json:"points"`
}

type Keypoint struct {
    Name       string   `json:"name,omitempty"`
    X          float64  `json:"x"` // normalized [0,1]
    Y          float64  `json:"y"` // normalized [0,1]
    Confidence *float64 `json:"confidence,omitempty"`
}
```

`DescribeArtifact` carries results too large or too binary to inline. In v1
the only artifact kind is `embedding_ref`; the embedding bytes live in the
asset store and the artifact points at them.

```go
type DescribeArtifact struct {
    TaskID   string       `json:"task_id"`
    TargetID string       `json:"target_id"`
    Kind     ArtifactKind `json:"kind"` // embedding_ref (only kind in v1)
    Ref      string       `json:"ref"`  // asset store ref
    Dims     *int         `json:"dims,omitempty"`
}
```

Normalization mapping:

| Backend output | Normalized shape |
| --- | --- |
| Caption / summary text | `observation.kind = text` |
| Bounding boxes | `observation.kind = detection` |
| OCR fields / labels / tags | `observation.kind = attribute` |
| Pose / landmarks | `observation.kind = keypoints` |
| Embeddings | `artifact.kind = embedding_ref` |
| Provider-specific extras | preserved in `runs[].raw_output` |

**Masks are scoped out of v1.** No v1 task kind produces mask output, and a
mask payload needs an asset-store storage decision (inline RLE vs stored
artifact) that should be made alongside a real segmentation provider. A
future `mask` observation or artifact kind is an additive, non-breaking
change to the closed enums.

- **`Summary` ownership (followup 4 resolved):** `Summary` is owned by the
  orchestrator only. Providers never populate it. In v1 the orchestrator
  leaves it unset; the field is reserved for future orchestrator-level
  synthesis across runs. Consumers must not treat `Summary == nil` as failure.

## Run Model and Correlation

The server expands `tasks` into concrete `(task_id, target_id)` runs. Each run
binds to exactly one delegate selected by the active analysis profile.

```go
type DescribeRun struct {
    TaskID    string          `json:"task_id"`
    TargetID  string          `json:"target_id"`
    Delegate  string          `json:"delegate"`
    Status    RunStatus       `json:"status"`
    Error     *RunError       `json:"error,omitempty"`
    RawOutput json.RawMessage `json:"raw_output,omitempty"`
}
```

- **`RunStatus` (followup 3 resolved):** closed enum
  `succeeded | failed | skipped`. `skipped` is real and occurs when a run is
  never dispatched: the active profile has no route for the task kind
  (`analysis_no_supported_delegate`), or the run's target failed resolution
  before dispatch. `failed` means the delegate was invoked and did not
  produce a usable result. `error` is required for `failed` and `skipped`,
  absent for `succeeded`.
- Request validation failures never produce a `DescribeResponse`; they return
  a non-2xx error with an `analysis_*` code.

## Config: Mode Policy

New mode-config sections, mirroring the existing `chat_connections` /
`chat_delegates` vocabulary in `server/mode_config.py`:

```yaml
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
    analysis_profile: default
```

Semantics:

- `analysis_connections` owns transport/auth settings.
- `analysis_delegates` names one concrete analyzer backend and declares its
  `kind` (capability declaration).
- `analysis_profiles.task_routes` maps task kinds to delegate names.
- Modes select one profile. Requests never choose delegates directly in v1.

Config-load validation (all fail-fast at parse time, matching the existing
chat config discipline):

- **Delegate-kind invariant (followup 1 resolved):** `kind` stays on the
  delegate as a capability declaration, and every `task_routes` entry must
  satisfy `route key == delegate.kind`. A mismatch fails config load with
  `analysis_delegate_kind_mismatch`. Rationale: the redundancy is intentional
  — the delegate declares what it can do, the profile declares how it is
  used, and the invariant catches wiring mistakes at load rather than at
  request time.
- `analysis_delegates[*].connection` must reference a declared connection.
- `analysis_profiles[*].task_routes` values must reference declared delegates.
- `modes.<name>.analysis_profile` must reference a declared profile.

## Error Vocabulary

Operator-facing codes (extensible, `analysis_` prefixed):

- `analysis_invalid_request`
- `analysis_target_binding_invalid`
- `analysis_profile_not_found`
- `analysis_delegate_not_found`
- `analysis_delegate_kind_mismatch`
- `analysis_no_supported_delegate`
- `analysis_all_runs_failed`

## Server Implementation Framework

Direction set by the human reviewer on the brainstorm: the core
implementation follows the **prompt-conditioning composition pattern** — an
async, composable chain, not an ad hoc dispatcher.

- The orchestration layer is a chain of async stages: validate → resolve
  profile → expand runs → dispatch → normalize → assemble response. Stages
  are composable units in the same spirit as the prompt-conditioning seam;
  providers and resolvers themselves are not required to be composited, but
  the framework around them is.
- Providers implement an async protocol:

```python
class DescribeProvider(Protocol):
    def supports(self, task: DescribeTask) -> bool: ...
    async def run(self, req: ProviderDescribeRequest) -> ProviderDescribeResult: ...
```

- Runs against distinct delegates may execute concurrently; per-run failure
  is isolated (one run failing never aborts sibling runs — it degrades the
  response `status` to `partial`).
- The orchestration layer owns validation, routing, normalization, and
  provenance/raw-output retention. Providers own model-specific preparation,
  inference calls, raw-output parsing, and capability declaration.

## Non-Goals for v1

- No attempt to unify all providers behind one identical output payload.
- No frontend/UI work before the CLI and library contract are stable.
- No transport commitment (WS vs HTTP) in this spec.
- No real provider implementation; contracts and a stub provider first.
- No request-level provider/delegate selection.
- No untyped task params in the public library contract.
- No population of `Response.Summary`.
- No `mask` observation/artifact kind (deferred with segmentation providers).

## Implementation Order (input to the plan)

1. Freeze the typed request/response contract in `pkg/stclient`.
2. Mode-config parsing and validation for `analysis_connections`,
   `analysis_delegates`, `analysis_profiles`, including the delegate-kind
   invariant.
3. One stub provider plus contract tests: target binding, role defaulting,
   run expansion, partial-failure and skipped-run cases.
4. `st describe` only after `pkg/stclient` exposes the typed surface.

## Followup Ledger (from brainstorm review)

All five spec-time followups from the theta v3 review are resolved above:

| # | Followup | Resolution |
| --- | --- | --- |
| 1 | delegate `kind` vs route-key duplication | kept both; load-time invariant `route key == delegate.kind` |
| 2 | target role default undefined | omitted `Role` = `primary`; only `primary` has v1 semantics |
| 3 | `DescribeRun.status` not enumerated | closed enum `succeeded / failed / skipped`; `skipped` = never dispatched |
| 4 | `Summary` provenance undefined | orchestrator-owned; unset in v1, reserved |
| 5 | stale brainstorm labels | this spec supersedes the brainstorm as authority; labels historical |

Spec-review round 2 (2026-07-12) resolved two further gaps: observation and
artifact wire types are now fully defined (with `mask` explicitly scoped out
of v1), and zero-run binding is a request validation error
(`analysis_target_binding_invalid`), guaranteeing `runs` is non-empty in every
`DescribeResponse`.
