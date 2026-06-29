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
  job cancel/priority, and model + mode status/switch — plus local PNG
  metadata read (`st read`) and recipe re-creation (`gen --recreate`).
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
- No batch generation in v1: one image per invocation. The `out-####` scheme
  and config leave room for batch in v2.
- No client-defined or ephemeral modes in v1: `st` only *selects* existing
  server-side modes. Client-side "mode split" is v2.

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

Core operational path only. Cobra / POSIX double-dash flags. Generation
defaults come from the config file's `defaults.generation`; any flag overrides
them.

| Subcommand | Backend op |
|---|---|
| `st gen [flags] <prompt>` | WS `job:submit` (generate). txt2img / `--init-image`+`--denoise` (img2img) / `--controlnet` (repeatable) / `--recreate <png>` (reuse a PNG's baked recipe) |
| `st read <png>` | local only: print the baked `lcm` metadata as JSON (no server call) |
| `st upload <file>` | `POST /v1/upload` → prints `fileRef` |
| `st superres <file> --magnitude N` | `POST /superres` |
| `st cancel <job_id>` | WS `job:cancel` |
| `st priority <job_id> --level N` | WS `job:priority` |
| `st models status\|free-vram\|reload\|inventory` | `/api/models/*`, `/api/inventory/*` |
| `st modes list\|switch <name>` | `/api/modes`, `/api/modes/switch` |
| `st validate-track3` | scripted end-to-end ControlNet checklist |

`st gen` takes the prompt as a **positional** argument. Flags: `--negative`,
`--genres <WxH>` (image size; `--size` alias), `--steps`, `--cfg`,
`--seed <int|random>` (absolute — no relative deltas in v1), `--scheduler`,
`--sr-level`, `--mode <name>` (server-side mode; see §5), `--init-image
<path|fileref:ID>` + `--denoise` (img2img — a local path is uploaded via
`/v1/upload`; a `fileref:ID` reuses an existing server upload), `--controlnet
"model_id=…,image=…,scale=…,start=…,end=…"` (repeatable), `--recreate <png>`
(load the PNG's baked recipe as defaults; txt2img re-run, flags override; does
**not** feed the image as an init), `--outfile <path>` (override output path;
extension optional). Progress streams to stderr; the result image (+ ControlNet
artifacts) is written to the output dir; a result JSON is printed to stdout
(`--json` for machine-only output).

The `--init-image` / `--controlnet image=` help text shows both forms: a local
file path, or `fileref:ID` for an already-uploaded asset (the `fileref:` prefix
distinguishes a server reference from a path).

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

### 4. Config, output, and metadata

**Config file** (`--config`, JSON) — server URL, generation defaults, and
output/metadata settings; any CLI flag overrides the matching config value:

```json
{
  "config": {
    "defaults": {
      "generation": { "mode": "default", "cfg": 2.5, "steps": 10, "genres": "512x512", "seed": "random" },
      "output_format": "png",
      "output_directory": "/home/.../stability_toys",
      "include_meta": true,
      "meta": { "producer_name": "...", "include_date": true, "misc": [ { "k": "v" } ] }
    }
  }
}
```

**Discovery & bootstrap** — the config path resolves as `--config` →
`$ST_CONFIG` → default `$XDG_CONFIG_HOME/stability-toys/config.json` (fallback
`~/.config/stability-toys/config.json`). If no config exists at the resolved
path, `st` writes this template with placeholder values to that path and exits
non-zero, printing the path and edit directions — it never runs on silent
defaults.

**Output** — written under `output_directory` (or `-o`) using the default
filename scheme `out-####.<ext>`, where `####` is the next free cardinal index
and `<ext>` comes from `output_format`. `--outfile <path>` overrides the name
(extension optional — appended from `output_format` if absent).

**Metadata** — the server already bakes an `lcm` JSON chunk (prompt, seed,
size, steps, cfg, negative, scheduler) into every PNG. When `include_meta` is
true the CLI writes an additional client-side chunk from `meta`
(`producer_name`, `include_date`, `misc`) on top — no backend change.

**`st read <png>`** parses and prints the baked metadata chunk(s) as JSON.
**`--recreate <png>`** reads the `lcm` chunk to seed `gen`'s parameters
(overridable by flags). Both are purely client-side and need no server.

**Run artifacts** — per-run capture under the output dir (request + response
JSON, WS transcript, run log), reused from the client-validator design.

### 5. Data flow & precedence

Each generation parameter resolves by layering, lowest to highest:

1. **Config defaults** (`defaults.generation`, including `mode`) — the implicit
   base.
2. **Init-image / recipe baked params** — when `--init-image <path>` or
   `--recreate <png>` points at a *local* PNG, its baked `lcm` chunk (prompt,
   seed, cfg, steps, size, scheduler, …) overrides config. A bare `fileref:ID`
   supplies init pixels only — no local bytes to parse, so it contributes no
   param layer.
3. **Explicit CLI flags** — highest precedence; override everything.

The resolved request is sent to the server, which backfills any still-unset
fields from the active mode / env (`finalize_mode_generate_request`). Client
precedence decides what is *set*; the server only fills the gaps.

**Mode** resolves through the same chain (config `mode` < `--mode`). Modes are
**server-side only in v1**, so "selecting a mode" means the CLI ensures the
server's active mode matches before submitting: it issues `POST
/api/modes/switch` when the resolved mode differs from the server's current
mode, then submits the job (the WS path finalizes against the server's
*current* mode, so the switch must precede submit). v2 may add a client-defined
ephemeral "mode split" that does not mutate global server state.

### 6. MCP future (designed-for, not built)

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
- `st gen` / `read` / `--recreate` interaction style: [generation-cli.md](generation-cli.md)
- CLI-first principle and core-path scope: this session's brainstorm
