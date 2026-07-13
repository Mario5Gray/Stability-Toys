# Describe Transport + `st describe` — Design

**FP issue:** STABL-ucomsfel
**Predecessor:** `2026-07-11-describe-analysis-interface-design.md` (STABL-tlklfaxz, merged at a7c70a2)
**Status:** Authority artifact for the transport + CLI track.

## Goal

Wire the merged describe/analysis contracts end-to-end: a server HTTP
endpoint, a typed `pkg/stclient` call surface, and an `st describe` CLI verb —
all running against `StubProvider`. Real providers (VLM caption, YOLO detect)
are a separate follow-on track; this track proves the full path and freezes
the operator-facing surfaces.

The v1 contract spec deferred the transport choice. This spec resolves it:
**HTTP `POST /v1/describe`**. `DescribeResponse` is a single terminal object
with no streaming or progress contract, so the generation WS hub buys nothing;
HTTP matches the `/v1/superres` request/response pattern and `stclient`'s
existing plumbing. A WS or job-queue path for long-running providers can be
added additively later without breaking this surface.

## Ownership Model (unchanged from predecessor)

| Layer | Owns |
| --- | --- |
| Server | endpoint, mode/profile resolution, orchestrator lifecycle, provider registry |
| `pkg/stclient` | typed `Describe()` call, client-boundary validation, wire decode, typed error mapping |
| `st` CLI | argument parsing, target auto-upload, task-flag construction, output rendering, exit codes |

## Ordering Determinism (normative)

Ordering is contract, not implementation detail. Implementors at every layer
MUST use ordered collections (Go slices, Python `list`/`tuple`) for the
sequences below; sets and unordered maps are forbidden anywhere order-bearing
data flows.

1. **CLI targets:** positional arguments map to `DescribeRequest.targets` in
   exact command-line order. Target IDs are generated positionally:
   `t1, t2, …, tN` (1-based, arg order). The Nth positional argument is
   always `tN`.
2. **CLI tasks:** task flags map to `DescribeRequest.tasks` in canonical kind
   order — `caption, detect, ocr, pose, embed` (the `TaskKind` enum
   declaration order) — regardless of the order flags appear on the command
   line (flag parse order is not reliably observable). Task IDs equal the
   kind string (`"caption"`, `"detect"`); the flag surface permits at most
   one task per kind, so this is unique and deterministic.
3. **Run expansion (server):** `runs` are expanded tasks-major: for each task
   in request order, one run per bound target in the task's effective target
   order (declared `target_ids` order when explicit; request `targets` order
   when defaulted to role `primary`). This pins the existing `expand_runs`
   behavior as contract.
4. **Response arrays:** `runs` appears in expansion order. `observations`
   and `artifacts` appear grouped by run, in run order; within one run they
   preserve the provider's emission order. Consumers may rely on these
   orderings; servers MUST NOT reorder.

A round-trip consequence worth stating plainly: for
`st describe a.png b.png --caption --detect`, the response `runs` order is
exactly `(caption,t1) (caption,t2) (detect,t1) (detect,t2)`.

## Server: `POST /v1/describe`

New module `server/analysis_routes.py` (pattern: `advisor_routes.py`),
mounted from `lcm_sr_server.py`.

Request flow:

1. Parse JSON body with the existing `parse_describe_request()` —
   all malformed-input handling stays in the parser (`analysis_invalid_request`
   and friends).
2. Resolve the effective mode: `request.mode` when set, otherwise the
   server's active mode.
3. Look up the mode's `analysis_profile` via `ModeConfigManager`. A mode with
   no `analysis_profile` fails with `analysis_profile_not_found`.
4. Dispatch to the `AnalysisOrchestrator` for that profile; `await describe()`.
5. Serialize with the existing `response_to_dict()`; return 200.

Orchestrator lifecycle: **request-time construction from live config.**
`modes.yml` is live-reloadable everywhere else in this server (SIGHUP
handler, file watcher, `POST /api/modes/reload`), so the analysis policy must
follow the same discipline — no lifespan-frozen snapshot. The endpoint reads
the current `ModeConfigManager` state (`get_mode_config()`) on every request
and builds the orchestrator and provider set from it; a reloaded
`analysis_profiles`/`analysis_delegates` section takes effect on the next
request with no process restart. A **provider registry** is a factory keyed
by delegate config and is the seam for real providers. In this track the
factory yields a `StubProvider` for every configured delegate — stateless and
cheap, so per-request construction costs nothing. When real providers arrive
they may need connection reuse; caching keyed on config generation is that
track's concern, behind the same factory seam, without touching the endpoint.

Error mapping (no `DescribeResponse` body on non-2xx, per predecessor spec):

| Condition | HTTP | Body |
| --- | --- | --- |
| `AnalysisValidationError` (parse or binding) | 400 | `{"error": {"code": "analysis_*", "message": "..."}}` |
| unknown mode named in `request.mode` | 400 | code `analysis_mode_not_found` |
| effective mode has no `analysis_profile` | 400 | code `analysis_profile_not_found` |
| unexpected server exception | 500 | code `analysis_internal` |

Two additive members join the error vocabulary:

- `analysis_mode_not_found` — `request.mode` names a mode the config layer
  does not know. Kept distinct from `analysis_profile_not_found` (a known
  mode that has no `analysis_profile` configured) so a mode-name typo stays
  diagnosable; the config layer already distinguishes the two cases.
- `analysis_internal` — operator-facing catch-all for faults that are not the
  client's request. Run failures never surface here — they are represented
  in-band as `failed`/`skipped` runs inside a 200 response.

Capability surfacing: `GET /api/models/status` gains
`capabilities.supports_describe: bool` — nested under the existing
`capabilities` object alongside `supports_generation` etc., true iff the
active mode has an `analysis_profile` configured. Additive field; existing
consumers (including the untyped `stclient.Models()` map) are unaffected.

## `pkg/stclient`: `Describe()`

```go
func (c *Client) Describe(ctx context.Context, req DescribeRequest) (*DescribeResponse, error)
```

- Calls the existing `req.Validate()` before any network I/O; invalid
  requests never leave the client.
- POSTs JSON to `/v1/describe`; decodes the typed `DescribeResponse`.
- Non-2xx maps to a typed error exposing the server's `analysis_*` code:

```go
type APIError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
}

func (e *APIError) Error() string
```

  (If a suitable typed error shape already exists in `stclient` at
  implementation time, reuse it; do not create a parallel one.)
- No CLI concerns (flags, stderr, cobra) enter `stclient` — frozen boundary
  rule. `Describe()` must serve both the CLI and the future MCP server.

## `st describe` CLI

### Targets (positional)

- Local file paths are auto-uploaded via the existing upload plumbing
  (`Client.Upload`, `upload` bucket); the returned ref becomes the target's
  `asset_ref`.
- Arguments beginning `http://` or `https://` pass through as `url` targets.
- IDs assigned positionally per the Ordering Determinism section.
- Roles are not settable from the CLI in v1; every target is effective-role
  `primary`.
- At least one target is required.

### Tasks (flags)

| Flag | Task | Params |
| --- | --- | --- |
| `--caption` | `caption` | `--prompt <string>` (optional) |
| `--detect` | `detect` | `--labels a,b,c` (optional), `--min-confidence <float>` (optional) |

- At least one task flag is required; zero task flags is a CLI usage error
  before any upload or network call.
- Param flags without their task flag (`--prompt` without `--caption`) are a
  CLI usage error.
- Tasks bind to all targets (`target_ids` omitted; server defaults to role
  `primary`). No per-task binding flags in v1.
- `ocr` / `pose` / `embed` flags are deferred until real providers exist;
  the contract enum already supports them and the flag surface extends
  additively.

### Output contract (frozen on merge)

- **Default (human):** caption text lines and a detection table
  (label / confidence / box) to stdout. Human rendering may evolve; scripts
  must not parse it.
- **Failure rendering (required):** a degraded exit must never be silent.
  When `status` is `partial` or `failed`, every non-`succeeded` run MUST be
  rendered to stderr with its `task_id`, `target_id`, `delegate`, run
  `status`, and the error `code` and `message` — e.g.
  `run detect/t2 (yolo_detect) failed: analysis_run_failed: <message>`.
  When the server returns a non-2xx error (or transport fails), the error
  `code` and `message` MUST be rendered to stderr —
  `error: analysis_mode_not_found: <message>`. The exact line format may
  evolve with the rest of the human rendering; the presence and content
  requirements are frozen.
- **`--json`:** the wire `DescribeResponse` verbatim — indented, a single
  terminal JSON object, not NDJSON, no added or removed fields. Same
  discipline as the frozen `st gen --json` contract.

### Exit codes (frozen on merge)

| Code | Meaning |
| --- | --- |
| 0 | `status == ok` |
| 1 | transport failure, CLI usage error, upload failure, or server-side `analysis_*` validation error |
| 2 | `status == failed` |
| 3 | `status == partial` |

Scripts branch on degraded runs via exit 3 without parsing output.

## Testing

- **Python (endpoint):** FastAPI `TestClient` against a stub-configured mode
  config — happy path, parse/binding 400s with correct `analysis_*` codes,
  no-profile 400 (`analysis_profile_not_found`), unknown-mode 400
  (`analysis_mode_not_found`), `partial` passthrough, run-order pin for the
  multi-task × multi-target expansion, and a reload-visibility case: swap the
  analysis policy via `reload_mode_config()` and prove the next request
  reflects it without restart.
- **Go (`stclient`):** `Describe()` against `httptest` — wire-shape pin
  (request serialization and response decode), `APIError` code mapping,
  pre-flight `Validate()` short-circuit.
- **Go (CLI):** arg→request construction unit tests including the positional
  ID assignment (`t1..tN`) and canonical task ordering; exit-code table;
  usage-error cases (no targets, no task flags, orphan param flags); failure
  rendering — `partial`/`failed` responses emit every non-`succeeded` run's
  identity and error to stderr, and non-2xx errors emit code + message.

## Non-Goals

- No real providers (VLM, YOLO) — separate track; the provider registry is
  the seam.
- No WS transport, job queue, or progress surface for describe.
- No `--file` full-fidelity JSON request input (additive later).
- No per-task `target_ids` or role flags in the CLI.
- No `ocr` / `pose` / `embed` CLI flags yet.
- No `Summary` rendering (field stays unset per predecessor spec).
- No frontend work (CLI-first policy).

## Implementation Order (input to the plan)

1. Server endpoint + orchestrator lifecycle + provider registry, with
   endpoint tests including the run-order pin.
2. `stclient.Describe()` + `APIError` mapping, with `httptest` coverage.
3. `st describe` verb: target auto-upload, task flags, output rendering,
   exit codes.
4. `supports_describe` in `/models/status` (small, can ride with step 1).
