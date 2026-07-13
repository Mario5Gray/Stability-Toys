# st — Stability-Toys operations CLI

`st` is the user-facing command-line interface to a Stability-Toys backend:
image generation (WebSocket jobs), metadata reads, uploads, super-resolution,
and job control. Every operation goes through one package, [`pkg/stclient`](pkg/stclient),
which is the single op surface shared by this CLI today and an MCP server later.

> The backend is **remote**. Point `st` at one with `--server` or `$ST_SERVER`
> (e.g. `http://enigma.lan:4200`). There is no embedded server.

For practical examples — first run, img2img, recreate, ControlNet, job control — see **[USAGE.md](USAGE.md)**.

## Build, test, generate

This is a standalone Go module (`go 1.26.1`), isolated from the repo's Python/JS
builds.

```bash
cd cli/go
make build      # go build ./...
make test       # go test ./...
make gen        # regenerate internal/openapi from the snapshot (see below)
```

`go test ./...` and `go build ./...` work directly too; the `make` targets are
thin wrappers.

## Commands

| Command | Purpose |
| --- | --- |
| `st gen [prompt]` | Generate an image (WS); resolves config < baked PNG < flags |
| `st conflate ...` | Toggle or configure gen-only parameter inheritance from history |
| `st replay <id>` | Re-run one historical generation entry exactly |
| `st read <png>` | Print embedded PNG metadata (`lcm`, `controlnet`, `controlnet_map`) |
| `st upload <file>` | Upload a file, print its fileref |
| `st superres <file>` | Upscale via the backend super-resolution endpoint |
| `st cancel <jobId>` | Cancel a running job |
| `st priority <jobId> <level>` | Set a job's priority |
| `st models` | Show model/backend status |
| `st modes` | List available model modes |
| `st validate-track3` | Script the ControlNet Track 3 acceptance check (needs `--server`) |

Global flags: `--server`/`$ST_SERVER`, `--config`, `-o/--output-dir`, `--json`,
`--timeout`.

## Configuration

On first run, config-dependent commands write a template to the resolved path and
exit non-zero so you can edit it. Discovery order:

1. `--config <path>`
2. `$ST_CONFIG`
3. `$XDG_CONFIG_HOME/stability-toys/config.json` (or `~/.config/...`)

Generation parameters layer by precedence: **config defaults < baked PNG params
(`--recreate`/local `--init-image`) < explicit CLI flags**.

History/state lives under `$XDG_STATE_HOME/st/` (or `~/.local/state/st/`) in
`history.jsonl`, `conflate-policy.json`, `next-id`, and `state.lock`. History is
always written; conflation is opt-in.

## OpenAPI contract & drift guard

`openapi.snapshot.json` is the backend's OpenAPI document, captured **verbatim**
(OpenAPI **3.1.0**, the version FastAPI serves). It is the source of truth for the
generated client types in `internal/openapi`.

- **Codegen:** `oapi-codegen` only supports OpenAPI 3.0.x, so `make gen` runs
  `tools/downspec` to emit a *throwaway* 3.0 intermediate that codegen consumes.
  The downspec is a codegen-only concern — it must **never** mutate the snapshot.
- **Drift guard:** `internal/openapi/drift_test.go` fetches the live
  `/openapi.json` and diffs it (canonicalized — sorted keys, normalized
  whitespace) against the snapshot. It is **skipped unless `ST_SERVER` is set**,
  so it stays a 3.1-vs-3.1 comparison and never produces false drift from the
  downspec.

```bash
# Run the gated drift check against a live backend:
ST_SERVER=http://enigma.lan:4200 go test ./internal/openapi/...
```

When the backend contract changes: refresh `openapi.snapshot.json` from the live
`/openapi.json`, run `make gen`, and commit both the snapshot and the regenerated
`internal/openapi/openapi.gen.go`.

## CI

This module's CI contract is simply `go vet ./...` + `go test ./...` run from
`cli/go`. The gated OpenAPI drift test is skipped in CI (no `ST_SERVER`).

> Note: `.github/workflows/` is git-ignored in this repo, so a GitHub Actions job
> for `cli/go` lives only locally. The **authoritative** build/CI pipeline for the
> platform is the shared Concourse/Kaniko setup documented in `../continuous/docs`
> (Harbor layout, base-image and fingerprinting strategy included). Wire the
> `cli/go` `go test ./...` step into that pipeline rather than introducing a
> tracked workflow here.
