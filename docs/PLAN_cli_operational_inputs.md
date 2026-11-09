# PLAN: CLI operational inputs — runtime explanations

Status: implemented on this branch; see the resolution table in the review doc.

Input: `docs/REVIEW_cli_operational_inputs.md` (findings F1–F8).
Branch: `fix/cli-operational-inputs`.
Goal: every high-impact missing/invalid input produces an actionable line on
STDERR, and duplicated wire/auth helpers are consolidated before the edits.

## Execution order

1. C1–C2 (Go client plumbing) — everything else in `cmd/st` depends on
   `newClient` returning an error.
2. C3–C7 (Go command behaviour + output/state helpers).
3. C8–C9 (Python shared wire helpers + auth explanation).
4. C10 architecture: extract duplicated routines found in step 0.
5. Tests + docs.

## Changes

### C1 — URL validation and error surfacing (`cmd/st/main.go`)

Before:
```go
func resolveServerURL(serverFlag, configFlag string) string {
    if serverFlag != "" { return serverFlag }
    path, err := config.Resolve(configFlag); if err != nil { return "" }
    cfg, err := config.Load(path); if err != nil { return "" }
    return cfg.ServerURL
}
func newClient() *stclient.Client { ... return stclient.New(resolveServerURL(...), opts...) }
```
After:
- `resolveServerURL(...) (string, error)` — config resolution/load errors are
  returned, not swallowed; empty value names all three sources and shows an
  example.
- `normalizeServerURL(raw, source) (string, error)` — trims, strips trailing
  `/`, prepends `http://` when no scheme, rejects non-http(s) or host-less URLs.
- `newClient() (*stclient.Client, error)`; every `RunE` updates to
  `client, err := newClient()`.

Call sites (12): `models.go:19`, `cancel.go:22`, `describe.go:132`,
`superres.go:36`, `gen.go:676`, `validate_track3.go:43`, `upload.go:38`,
`modes.go:44,81,89,102`, `priority.go:27`.

### C2 — Timeout relationship (`cmd/st/main.go`, `cmd/st/describe.go`)

- `--timeout` help says it governs HTTP requests, not WS job duration.
- Describe uses a 30-minute default when `--timeout` is unset.
- On a client timeout, return the underlying error plus
  `hint: raise the client timeout with --timeout (currently <d>)`.

### C3 — Gen output preflight (`cmd/st/gen.go`)

New `preflightOutputDestination(cfg, genArgs) error`, called from
`runPatchedGen` before `buildConflatedParams`:
- reject empty `output_directory` with the config key and `-o` alternative;
- reject a path containing `REPLACE_ME` (untouched template);
- reject `output_format` other than `png` (the backend returns PNG);
- `os.MkdirAll` the destination (or the explicit outfile's parent) so a
  permission/ENOENT failure happens before the job is submitted.
After the job, gen and superres use the new `output.Save` helper (C10) — no
behaviour change beyond the earlier failure point.

### C4 — Usage hints (`cmd/st/main.go`)

- `usageError{cmd, err}` with `Unwrap`; `main` prints
  `hint: run 'st <path> --help'` when the error is a usage error.
- `rootCmd.SetFlagErrorFunc` wraps flag-parse errors.
- Positional-arg helpers `exactArgs`, `minArgs`, `maxArgs` wrap Cobra's
  messages in `usageError`; applied to all commands that declare `Args`.

`exitCodeOf` unwraps `usageError` so exit codes do not change.

### C5 — WS job error enrichment (`pkg/stclient/ws.go`)

- Empty `job:error` text becomes `server reported failure without detail`.
- `Generate` appends the requested mode when `params["mode"]` is a string.
- `controlFrame` uses the same empty-text fallback.

### C6 — State root explanation (`internal/history/fs.go`)

Wrap `MkdirAll`/`Chmod` failures:
`cannot initialise state directory <path> ($XDG_STATE_HOME / ~/.local/state): <err>`.

### C7 — Output helper (`internal/output/output.go`)

Add `Save(outfile, dir, format string, data []byte) (string, error)` = Resolve
+ Write. Replace the duplicated two-step blocks in `gen.go:743-748` and
`superres.go:44-49`.

### C8 — Shared OpenAI-compatible wire helpers (`backends/openai_compat.py`)

New module:
- `chat_completions_url(endpoint)`
- `bearer_headers(api_key_env)`
- `ensure_success(resp, *, endpoint, api_key_env)` — on 401/403 raises
  `AuthError` naming the endpoint and `api_key_env`; otherwise delegates to
  `resp.raise_for_status()`.
`VLMChatClient` and `ChatCompletionsClient` import these; their duplicated
`_url`/`_headers`/`raise_for_status` bodies are deleted.

### C9 — Auth explanation tests (`tests/`)

- `test_analysis_vlm_client.py`: 401 with env set/unset → message names the
  env var and endpoint.
- `test_chat_client.py`: fake response exposes `status_code`; 401 path asserted.

### C10 — Architecture round (duplicated routines)

Found and abstracted:
| Duplication | Where | Fix |
|---|---|---|
| URL + auth headers + status check (≈15 lines each) | `backends/analysis/vlm_client.py`, `backends/chat_client.py` | `backends/openai_compat.py` (C8) |
| Resolve-then-write output sequence (5 lines each) | `cmd/st/gen.go`, `cmd/st/superres.go` | `output.Save` (C7) |
| Dial/write/close WebSocket skeleton (≈8 lines each) | `pkg/stclient/ws.go` `Generate`, `controlFrame` | shared `dialWS` helper |

After these, no multi-line duplicated routines remain in the touched
packages. The architecture diagram (Go packages + Python client layer) is
attached to the review document as a class diagram because Go has no classes.

## Net impact

- Go: ~12 call sites mechanically changed; +~120 lines (validation/helpers),
  −~20 duplicated lines.
- Python: +~50-line module; −~15 duplicated lines.
- Tests: +2 cases, 1 fake extended.

## Risks

- `newClient` signature change is compile-time only; no public API outside
  `cmd/st`.
- URL normalization changes scheme-less hosts from hard failure to `http://`;
  an `https` host must be given its scheme (unchanged behaviour).
- `output_format` rejection could break a config using a non-png value; that
  config was already producing mislabeled files.
- Auth error type changes from `httpx.HTTPStatusError` to `AuthError` for
  401/403 only; 500 keeps the old type (existing test).
