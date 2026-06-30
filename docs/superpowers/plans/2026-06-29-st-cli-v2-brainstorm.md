# st CLI v2 — Brainstorm

> FP: STABL-kczspmud
> Status: brainstorm — nothing here is committed
> Companion to: `2026-06-28-operations-cli.md` (v1 plan) and `2026-06-28-operations-cli-as-built.md`

---

## What v1 left on the table

Known gaps called out during implementation:

| Gap | Where noted |
| --- | --- |
| Progress streaming is silently dropped (`select { default: }`) | T5 deadlock fix |
| `st gen --json` does not expose `jobId` — cancel/priority have no target from gen output | T12 |
| Mode switch is soft — silent skip if `CurrentMode` unreachable | T12 |
| ControlNet JSON-per-flag is verbose; ergonomic shorthand belongs here | T15 |
| `SetPriority` is a server no-op stub | T6 |
| `pkg/stclient` was designed as the shared surface for a future MCP server | T12 README |

---

## Idea clusters

### A — Job observability

The WS protocol already streams `job:progress` frames. v1 drops them. Options:

- **A1. Live progress bar** — render delta text to stderr during `st gen`; `--quiet` to suppress
- **A2. `--jobid` output flag** — emit jobId on ack (before complete) so the user can `st cancel` a running job from another terminal
- **A3. `st jobs`** — poll or WS-watch for in-flight queue state (if the backend exposes it)
- **A4. Progress JSON events** — `--json` mode emits a stream of `{"event":"progress","delta":"..."}` lines, one per frame, then a final `{"event":"complete",...}` — composable with `jq`

### B — Batch and pipeline

- **B1. `st gen --batch <prompts.txt>`** — one generation per non-empty line; writes `out-####.png` per result; `--json` emits one JSON object per line (NDJSON)
- **B2. `st pipe`** — read image path on stdin, write output path on stdout; enables `echo sketch.png | st pipe gen --mode sdxl | st pipe superres`
- **B3. Parallel batch** — `--batch` with `--concurrency N` (N WS connections); useful for prompt sweeps
- **B4. `st gen --variations N`** — submit N jobs with same params but different seeds; writes `out-####-var{1..N}.png`

### C — ControlNet ergonomics

v1 requires raw JSON per `--controlnet`. T15 noted this is where shorthand belongs:

- **C1. Shorthand syntax** — `--controlnet canny:fileref:R1` expands to `{"attachment_id":"cn-1","control_type":"canny","map_asset_ref":"R1"}`
- **C2. `--controlnet-file <json>` flag** — read the attachment from a JSON file (avoids shell escaping)
- **C3. Auto-upload shorthand** — `--controlnet canny:./map.png` uploads and threads ref automatically
- **C4. Preset ControlNet profiles in config** — named attachments in `config.json`, referenced by `--controlnet @preset-name`

### D — Config management

v1 bootstraps a template but offers no editing surface:

- **D1. `st config get <key>`** — print a single config value by dot-path (`defaults.generation.genres`)
- **D2. `st config set <key> <value>`** — write a single config value
- **D3. `st config edit`** — open `$EDITOR` on the resolved config path
- **D4. `st config show`** — print fully-resolved config (with defaults filled in) as JSON
- **D5. Per-profile configs** — `--profile portrait` selects `~/.config/stability-toys/portrait.json`; gen flags default to that profile's generation block

### E — Mode management

Currently modes are only editable via the backend's `modes.yaml` or the REST API directly:

- **E1. `st modes switch <name>`** — explicit switch command (currently only happens implicitly in `st gen --mode`)
- **E2. `st modes show <name>`** — print full mode config as JSON
- **E3. `st modes reload`** — POST `/api/modes/reload` to hot-reload `modes.yaml` on the server
- **E4. `st modes set-default <name>`** — POST to set the backend default mode

### F — MCP server

`pkg/stclient` was explicitly designed as the shared surface:

- **F1. `st serve mcp`** — expose all stclient ops as an MCP (Model Context Protocol) tool server; LLM agents can call `gen`, `read`, `upload`, `modes`, etc.
- **F2. MCP tools map 1:1 to existing commands** — no new stclient logic; same precedence, same output scheme
- **F3. Streaming progress as MCP notifications** — use the WS progress channel for partial tool results
- **F4. Config gate in MCP** — same `requireConfig` pattern; MCP server refuses to start without a valid config

### G — Output and metadata

- **G1. `st compare <a.png> <b.png>`** — diff the `lcm` tEXt chunks; show what changed between two generations
- **G2. `--negative` in config defaults** — currently `negative` must be a flag; add to `generation` defaults block
- **G3. Prompt templates** — `st gen --template portrait` expands to a pre-configured prompt + negative + scheduler combo from config
- **G4. `st gallery`** — list `out-####.png` files in output_directory with their baked `lcm` fields; tabular or JSON
- **G5. `st gen --open`** — open the output file in the default image viewer after write (macOS: `open`, Linux: `xdg-open`)

### H — Developer / ops

- **H1. `st doctor`** — check: server reachable? config present and valid? output_directory writable? snapshot drifted?
- **H2. `make completions`** — generate shell completions (Cobra already supports `st completion bash/zsh/fish`) and install to `~/.local/share/...`
- **H3. `--dry-run`** — for `gen`, print the resolved params and mode without submitting; validates precedence without burning a generation slot
- **H4. `st snapshot refresh`** — fetch live `/openapi.json`, write to `openapi.snapshot.json`, run `make gen` — automates the drift-fix workflow
- **H5. Structured logging** — `--log-level debug` emits WS frame traces and HTTP request/response to stderr; useful when diagnosing backend issues

---

## Rough signal

Not a priority grid — just a first-pass read:

| Cluster | Why interesting |
| --- | --- |
| A — Job observability | Progress drop is a known wart; jobId exposure unblocks cancel/priority UX |
| F — MCP server | Was always the second consumer of `pkg/stclient`; low stclient delta |
| C — ControlNet ergonomics | Raw JSON is painful for interactive use; shorthand is mechanical to add |
| H1 doctor | Reduces onboarding friction; one command to confirm the setup works |
| D — Config management | `st config set` removes manual JSON editing for common tweaks |
| B — Batch | High value for prompt sweeps and automation |
| E — Mode management | Mostly thin wrappers; `reload` is the most-needed one |
| G — Output/metadata | `compare` and `gallery` are useful but non-urgent |

---

## Open questions for discussion

1. **MCP first or batch first?** MCP reuses stclient cleanly; batch requires a new concurrency model. Which delivers more day-to-day value?
2. **Progress display — stderr or TUI?** A raw delta-to-stderr approach is simple; a proper progress bar (e.g. `charmbracelet/bubbles`) is polished but adds a dep.
3. **ControlNet shorthand — syntax?** `canny:./map.png`, `canny:fileref:R1`, or something else? Does the shorthand need to support the full attachment schema or just the common case?
4. **Config profiles — separate files or named blocks in one file?** Separate files are simpler to discover; named blocks avoid proliferation.
5. **`st serve mcp` — transport?** stdio (standard for local MCP) vs HTTP SSE (for remote agents). stdio is the immediate need.
6. **`--dry-run` scope** — just print resolved params, or also simulate the WS handshake against a mock?

---

*Update this doc as the discussion narrows. When a cluster solidifies into a task plan, promote it to its own `YYYY-MM-DD-st-cli-v2-<cluster>.md`.*
