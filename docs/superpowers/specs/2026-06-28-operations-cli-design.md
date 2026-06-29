# Operations CLI (Go) Design

## Summary

Stability Toys currently has no real user-facing CLI — the web front-end
(`lcm-sr-ui`) is the only client that drives the backend. This design adds a
Go command-line client, `st`, that exercises the **core operational path** of
the backend through its existing HTTP and WebSocket APIs. The CLI is the first
of potentially several language front-ends under `cli/` (a Zig TUI is
anticipated), so the backend operation contract — not any one front-end — is
the shared anchor.

The Go code is structured so a future MCP server is a thin wrapper: a single
`stclient` package owns every operation, and both the CLI and a later MCP
server are thin adapters over it.

This delivers the project's **CLI-first** principle: establish full user-facing
operability from the command line before building any new front-end.

## Goals

- A single Go CLI binary `st` that can drive the core operational path:
  generation (txt2img / img2img / ControlNet), asset upload, super-resolution,
  job cancel/priority, and model + mode status/switch.
- WS-first for jobs (live progress, cancel, priority; ControlNet artifacts via
  `job:complete`), HTTP for reads and simple synchronous calls.
- An `stclient` Go package that is the single operation surface, transport- and
  caller-agnostic, so a later MCP server reuses it with no logic duplication.
- Keep the Go request/response contract in sync with the Python (FastAPI)
  backend via generated HTTP types plus a small hand-written WS envelope.
- A `validate-track3` subcommand that runs the ControlNet Track 3 manual
  checklist end-to-end, proving the CUDA backend is operable from a user's seat.

## Non-Goals

- No web/TUI front-end work (the Zig TUI is future, separate).
- No full UI parity in v1: advisor digest, keymap, comfy/workflows, and
  client-side galleries (IndexedDB — not backend operations) are deferred.
- No MCP server implementation in v1 (designed-for, not built).
- No server lifecycle control: `st` is a black-box client against an
  already-running server. It never starts, stops, or mutates server process
  state outside normal request payloads.
- No installable distribution/packaging in v1 (run from the built binary).

## Current State

- Backend is FastAPI (`server/lcm_sr_server.py`), default `/openapi.json`
  enabled (HTTP contract is machine-readable).
- Generation primary path is the WebSocket job protocol at `/v1/ws`
  (`server/ws_routes.py`): client sends `job:submit`, server replies
  `job:ack` → `job:progress`* → `job:complete` | `job:error`. `job:cancel` and
  `job:priority` are **WS-only**.
- `POST /generate` is a synchronous HTTP fallback over the same
  `GenerateRequest` model; `POST /superres` and `POST /v1/upload` are HTTP;
  model/mode reads are HTTP under `/api/...`.
- The existing un-built ControlNet
  [client-validator design](2026-05-05-controlnet-client-validator-design.md)
  proposed a Python `scripts/controlnet_client.py`. **This design supersedes
  it**: that tool's flows become Go subcommands (`upload`, `generate`,
  `validate-track3`).

## Layout

```text
cli/
  go/                       # new go.mod — first Go in the repo
    pkg/stclient/           # THE operation surface (shared by CLI + future MCP)
       http.go              #   HTTP reads/calls (wraps generated client)
       ws.go                #   WS job client: Generate / Cancel / SetPriority
       types.go             #   hand-written WS envelope + job:* frames
    internal/openapi/       # oapi-codegen output (generated HTTP client/types)
    cmd/st/                 # Cobra CLI (this design)
    cmd/st-mcp/             # MCP server (future; thin wrapper over stclient)
    openapi.snapshot.json   # committed /openapi.json snapshot (codegen + drift guard)
    Makefile                # gen / build / test
  zig/                      # future Zig TUI front-end (separate; same contract)
```

## Design

### 1. `stclient` — the operation surface

A transport-aware but caller-agnostic package. It does not know whether a CLI,
an MCP server, or a test is calling it.

- **HTTP operations** (generated client wrapped for ergonomics):
  - `Upload(ctx, file) (FileRef, error)` → `POST /v1/upload`
  - `SuperRes(ctx, file, magnitude) (Image, error)` → `POST /superres`
  - `Models` group → `GET /api/models/{status,free-vram}`,
    `POST /api/models/reload`, `GET /api/inventory/{models,loras}`
  - `Modes` group → `GET /api/modes`, `POST /api/modes/switch`
- **WS job operations** (`/v1/ws`):
  - `Generate(ctx, params GenerateRequest) (Job, <-chan Progress, error)` —
    sends `{type:"job:submit", id, jobType:"generate", params}`; consumes
    `job:ack` (captures `jobId`), streams `job:progress`, resolves on
    `job:complete` (image + `controlnet_artifacts`) or `job:error`.
  - `Cancel(ctx, jobID) error` → `job:cancel`
  - `SetPriority(ctx, jobID, level) error` → `job:priority`

`GenerateRequest` is the **same model the backend uses for both** `POST
/generate` and the WS `params` payload (`_build_generate_request(params)` in
`server/ws_routes.py`). It is therefore covered by OpenAPI and comes from
codegen — img2img (`init_image_ref` + denoise) and ControlNet attachments are
fields on it. Only the WS *envelope* (`type`/`id`/`jobType`/`params`) and the
response frames (`job:ack|progress|complete|error`) are hand-written.

### 2. CLI surface (`cmd/st`, Cobra)

Core operational path only:

| Subcommand | Backend op |
|---|---|
| `st generate` | WS `job:submit` (generate); txt2img / `--init-image`+`--denoise` (img2img) / `--controlnet` (repeatable) |
| `st upload <file>` | `POST /v1/upload` → prints `fileRef` |
| `st superres <file> --magnitude N` | `POST /superres` |
| `st cancel <job_id>` | WS `job:cancel` |
| `st priority <job_id> --level N` | WS `job:priority` |
| `st models status\|free-vram\|reload\|inventory` | `/api/models/*`, `/api/inventory/*` |
| `st modes list\|switch <name>` | `/api/modes`, `/api/modes/switch` |
| `st validate-track3` | scripted end-to-end ControlNet checklist |

`st generate` flags: `-p/--prompt`, `--negative`, `--size`, `--steps`,
`--cfg`, `--seed`, `--scheduler`, `--sr-level`, `--mode`, `--init-image`,
`--denoise`, `--controlnet "model_id=…,image=…,scale=…,start=…,end=…"`
(repeatable), `-o/--output <dir>`. Progress streams to stderr; the result PNG
and artifacts are written to the output dir; a result JSON is printed to stdout
(`--json` for machine-only output).

Global flags: `--server` / `$ST_SERVER`, `--config <json>`, `-o/--output-dir`,
`--json`, `--timeout`.

### 3. Contract sync

- Commit `openapi.snapshot.json`, fetched from a running server's
  `/openapi.json`. `oapi-codegen` generates `internal/openapi` (types + HTTP
  client). `make gen` regenerates.
- A **gated contract test** compares a live server's `/openapi.json` to the
  committed snapshot and fails on divergence (drift guard). It is opt-in
  (requires a running server), not part of unit CI.
- The WS envelope + `job:*` frames are hand-written in `stclient/types.go` and
  guarded by a WS integration test against the documented shape in
  `server/ws_routes.py`.
- The snapshot is the shared contract anchor for the future Zig front-end too.

### 4. Config & output

JSON config file with flag/env overrides (server URL, defaults, output dir).
Per-run artifact capture under the output dir: result image(s), request +
response JSON, WS transcript, and a run log — the capture model is reused from
the client-validator design.

### 5. MCP future (designed-for, not built)

`cmd/st-mcp` will expose each `stclient` operation as an MCP tool via a Go MCP
SDK. Because `stclient` is independent of CLI concerns, the MCP server is a
thin adapter (tool schema ↔ `stclient` call). No operation logic is duplicated.
Out of scope for v1 implementation; the package boundary is the only thing v1
must get right to keep it cheap.

## Failure Handling

- HTTP errors map to non-zero exit codes with the server's error detail
  surfaced.
- `job:error` (including pre-submit policy failures, which may carry
  `controlnet_artifacts`) is reported with the error string and a non-zero exit.
- WS connect/timeout failures are explicit; `--timeout` bounds a job wait.
- ControlNet on the img2img path is rejected server-side
  (`NotImplementedError`, v1); the CLI surfaces that as a clear error rather
  than attempting a workaround.

## Testing Strategy

- `stclient` HTTP ops: `httptest` mock server, table-driven.
- `stclient` WS job client: a mock WS server replaying
  `job:ack/progress/complete/error`; assert progress streams and the call
  resolves; `Cancel` emits `job:cancel`.
- `cmd/st`: Cobra command tests over a fake `stclient`, asserting
  flag→param mapping and exit codes.
- `validate-track3`: integration smoke, gated on a running server.
- Contract drift test: gated (live `/openapi.json` vs snapshot).

All new Go code is built test-first (TDD): a failing Go test before each unit.

## Risks And Tradeoffs

- **Contract drift (Python ↔ Go).** Mitigated by OpenAPI codegen for HTTP and a
  gated drift test; residual risk is the hand-written WS envelope, kept minimal
  and integration-tested.
- **New language in the repo.** Go is isolated under `cli/go` with its own
  `go.mod` and CI target; it does not touch the Python or JS build.
- **WS-first complexity.** An async WS client is more work than HTTP-only, but
  it is required for cancel/priority and matches how the product actually runs.

## Scope And Decomposition

**In v1:** `cli/go` module scaffold + `oapi-codegen` wiring + snapshot;
`stclient` (HTTP ops + WS job client); `cmd/st` subcommands for the core path;
`validate-track3`; config + artifact capture; tests.

**Deferred:** `cmd/st-mcp` (MCP server); advisor/keymap/comfy/workflows
subcommands; `cli/zig` TUI; installable packaging.

## References

- Supersedes: [ControlNet Client Validator Design](2026-05-05-controlnet-client-validator-design.md)
- Backend WS protocol: `server/ws_routes.py` (`job:submit`/`job:*`)
- Backend HTTP: `server/lcm_sr_server.py` (`/generate`, `/superres`),
  `server/upload_routes.py`, `server/model_routes.py`, `server/mode_config.py`
- ControlNet backend (complete): [controlnet-design](2026-04-18-controlnet-design.md) §6
- `validate-track3` checklist source: `docs/TESTING_CONTROLNET_TRACK3.md`
- CLI-first principle and core-path scope: this session's brainstorm
