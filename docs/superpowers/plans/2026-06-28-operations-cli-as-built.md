# Operations CLI — As-Built Differential (T1–T10)

> **Companion to:** `2026-06-28-operations-cli.md`
>
> Records deviations from the original plan — wrong contract assumptions, structural
> fixes, hardening beyond scope, and naming corrections. Code is omitted; only what
> diverged and why is documented here. Tasks not listed had no deviations.

---

## T1 — Module scaffold + OpenAPI codegen

**Plan assumed:** `oapi-codegen` could consume `openapi.snapshot.json` directly.

**What happened:** FastAPI serves OpenAPI **3.1.0**. `oapi-codegen` v2 rejects 3.1's
nullable form (`anyOf: [{schema}, {type: null}]`). Direct codegen fails.

**Resolution:** Added `cli/go/tools/downspec/` — a deterministic converter that emits
a throwaway 3.0.3 intermediate (`openapi.codegen.json`, gitignored). `make gen`
pipeline became:

```shell
downspec openapi.snapshot.json openapi.codegen.json
oapi-codegen -config ... openapi.codegen.json
rm openapi.codegen.json
```

The snapshot stays **verbatim 3.1.0** so the T16 drift guard diffs it against the live
3.1 backend (a mutated 3.0 snapshot would produce permanent false drift).

**`downspec` behaviour (conservative):**

- Strips `{type:null}` from `anyOf`/`oneOf`; marks parent `nullable: true`.
- Single remaining inline member: hoisted into parent (sibling keywords preserved).
- Single remaining `$ref` member: wrapped in `allOf` (a bare `$ref` sibling of
  `nullable` is silently ignored by strict 3.0 parsers).
- Multi-member non-null unions: left untouched.
- Pins top-level `openapi` to `3.0.3`.

**Tests added beyond plan:** 4 behavioural cases in `downspec_test.go` (version pin,
nullable scalar collapse with sibling-keyword preservation, nullable `$ref` → `allOf`
with negative assertion, genuine union untouched).

---

## T2 — stclient Client core + HTTP reads (models, modes)

**Plan assumed (wrong contract):**

| Endpoint | Plan | Actual (verified vs `server/model_routes.py`) |
| --- | --- | --- |
| `GET /api/modes` response shape | `{"modes":[{"name":"default"},…],"current":"default"}` — list + `current` key | `{"modes":{"default":{…},"cartoony":{…}},…}` — dict keyed by name, **no `current` key** |
| `CurrentMode` source | `/api/modes` → `.current` field | `/api/models/status` → `.current_mode` |
| `SwitchMode` transport | `POST /api/modes/switch?mode=<name>` (query param) | `POST /api/modes/switch` with JSON body `{"mode": name}` (`ModeSwitchRequest`) |

**Structural changes:**

- `Modes()` decodes `body.Modes` as `map[string]json.RawMessage`, extracts keys,
  sorts them (deterministic output despite non-deterministic map iteration).
- `CurrentMode()` calls `/api/models/status`, not `/api/modes`.
- `SwitchMode()` marshals `{"mode": name}` and sets `Content-Type: application/json`.

**Test added beyond plan:** `TestGetJSONReturnsErrorOnNon2xx` (error propagation on HTTP
5xx — the plan only had the happy-path tests).

---

## T3 — stclient Upload + SuperRes + storage fetch

**Bug in plan's Upload sketch:**

```go
// plan's code — always returns "":
return body.FileRef, json.NewDecoder(resp.Body).Decode(&body)
```

In Go, named return values in a `return` statement are evaluated before the statement
executes. `body.FileRef` is `""` when evaluated; `Decode` runs but its result is used
only for the `error` return. The call always returns an empty `fileRef`.

**Fix:** decode into `body` first, then return `body.FileRef` separately.

**Structural addition:** `multipartFile(filename, data, fields)` helper extracted. Both
`Upload` and `SuperRes` use it; the plan sketched the multipart construction inline in
each function.

**Contract verified:** `magnitude` form field name confirmed at `lcm_sr_server.py:685`
(`magnitude: int = Form(2)`). `fileRef` response field confirmed at
`upload_routes.py:32`.

**Test added beyond plan:** `TestFetchStorageErrorsOn404` (error path for missing
storage key — the plan's test sketch only covered the 200 path).

---

## T5 — WS Generate (submit → ack → complete)

**Deadlock in plan's implementation:**

The plan's `Generate` loop sent to the progress channel with a blocking send:

```go
case "job:progress":
    prog <- Progress{Delta: f.Delta}   // blocks when buffer is full
```

The backend (`ws_routes.py`, `_on_job_update`) emits `job:progress` on every job
mutation — unbounded. The progress channel is not read until `Generate` returns. With
the plan's 16-slot buffer, any job that emits more than 16 progress frames deadlocks
`Generate` permanently.

**Fix:** non-blocking send; excess frames are dropped:

```go
case "job:progress":
    select {
    case prog <- Progress{Delta: f.Delta}:
    default:
    }
```

**Other deviations from plan sketch:**

- Plan used static correlation ID `"c1"`; actual uses `corrID()` (6-byte
  `crypto/rand` hex) so concurrent calls don't collide.
- Plan closed connection and returned `nil, nil, err` on `job:error`; actual returns
  `prog` (closed), `nil`, `err` — the closed channel is harmless; callers check error
  first.

**Test added beyond plan:** `TestGenerateDoesNotDeadlockOnManyProgress` — in-process WS
server emits 50 `job:progress` frames (3× the buffer) before `job:complete`; test
asserts `Generate` resolves within 5 s.

---

## T6 — WS Cancel + SetPriority

**Priority field name confirmed:** plan left `priority` vs `level` as an open question
("confirm against `server/ws_routes.py:296`"). Actual field in the sent frame is
`priority`. The backend handler is a no-op stub that returns
`{"type":"job:priority:ack","detail":"priority not yet implemented"}` without reading
the field — documented in `SetPriority`'s godoc.

**`controlFrame` injects corrID:** plan's sketch did not add a correlation ID to the
control frames. Actual implementation injects `send["id"] = corrID()` before dialing,
matching the server's echo behaviour.

**Test added beyond plan:** `TestControlFrameReturnsErrorOnJobError` — verifies
`controlFrame` propagates a `job:error` reply as a Go error (tested via `Cancel`).
The plan only tested the happy-path ack.

---

## T7 — Config load + discovery + bootstrap

**No implementation deviations.** The config structs, `Resolve`, `Load`, and
`BootstrapTemplate` match the plan exactly.

**Test added beyond plan:** `TestResolveDiscoveryOrder` — pins the three-level
discovery chain (`--config` flag > `$ST_CONFIG` > XDG default) using `t.Setenv`
(isolated, no env leakage between subtests). The plan only had load and bootstrap
tests.

---

## T8 — Precedence resolver

**Name collision:** The plan named the resolver `config.Resolve(cfg, baked, flags)`,
which collides with the path-discovery `config.Resolve(flagPath)` already committed in
T7. Go has no overloading; the second declaration is a compile error.

**Fix:** renamed to `config.ResolveParams`. All T12+ references use `ResolveParams`.

**Tests added beyond plan:**

- `TestPrecedenceSeedAndSRFlags` — pins two flag-layer branches the plan's single test
  did not cover: a numeric seed is kept verbatim (not treated as "random"), and
  `--sr-level 9` maps to `superres=true, superres_magnitude=3` (clamped to max).
- `strp()` helper added alongside `intp()` for the seed test.

---

## T9 — pngmeta (lcm tEXt read/write + BakedParams)

**No implementation deviations.** Chunk surgery, CRC32-IEEE, keyword+NUL format, and
the `lcm`→`GenerateRequest` field renaming all match the plan's intent.

**Tests added beyond plan:**

- `TestWriteTextKeepsValidPNG` — after `WriteText`, passes the result through
  `png.Decode` (stdlib). This is the only reliable guard against CRC or length bugs;
  the plan did not include a structural-validity test.
- `TestBakedParamsMapsToRequestFields` — asserts that an unmapped key (`"unrelated"`)
  is **dropped** by `BakedParams` (whitelist approach). The plan described the mapping
  but did not test the drop.

---

## T10 — output (out-#### scheme + --outfile)

**Plan sketch carried a dead `strings` import** (artifact of a draft that used
`strings.TrimSuffix`). Removed.

**Test added beyond plan:** `TestResolveRelativeOutfileJoinsDir` — asserts that a
relative `--outfile` (with extension already present) is joined under the output
directory. The plan's test sketch only covered the extension-append case.

---

---

## T11 — Cobra root + global flags + config gate

**Config gate relocated (deviation from plan):** The plan placed the bootstrap gate in
`main()`, which would force every subcommand — including `modes`, `upload`, and future
peripheral commands — through a config requirement. Moved to `requireConfig()`, called
only by config-dependent commands (currently `gen`). The pure discovery seam
`resolveConfig()` is unchanged.

**Split: pure seam vs. impure boundary:**

| Function | Role |
| --- | --- |
| `resolveConfig(path)` | Returns `(cfg, message, bootstrapped)` — no side effects, testable |
| `requireConfig()` | Resolves path via `config.Resolve`, calls `resolveConfig`, exits on bootstrap |

Tests target `resolveConfig` directly, avoiding the untestable `os.Exit` in
`requireConfig`.

**`SilenceErrors: true, SilenceUsage: true`** set on root — correct for a CLI that
owns its own error display. Without these, Cobra double-prints errors and dumps usage
on any `RunE` failure.

**`newClient()` timeout:** `--timeout` is wired only when `> 0`; `0` leaves the
`stclient` default (120 s) in place.

---

## T12 — `st gen` spine

**`genArgsFromFlags` uses `flag.Changed()`:** The plan sketched bare variable reads.
`flag.Changed()` is the correct Cobra idiom for distinguishing "user explicitly set
`--cfg 0`" from "user left `--cfg` unset". Without it, a zero value for a numeric
flag would incorrectly override a higher-precedence baked layer.

**`localRecipePath` priority:** Returns `--recreate` path first, then `--init-image`
path. A `fileref:` prefix or a non-existent path is skipped. Ensures the recipe layer
always comes from the explicitly-named recreate file when both are provided. `--recreate`
never contributes an `init_image_ref`; `--init-image` does (uploaded or referenced).

**ControlNet shape verified and kept JSON:** The plan noted the controlnet field shape
as uncertain. Full `ControlNetAttachment` schema (`controlnet_models.py`):

| Field | Required | Notes |
| --- | --- | --- |
| `attachment_id` | ✓ | |
| `control_type` | ✓ | |
| `map_asset_ref` | one-of | incompatible with `preprocess` |
| `source_asset_ref` | one-of | requires `preprocess` |
| `model_id` | optional | |
| `preprocess` | optional | `ControlNetPreprocessRequest` |
| `strength` | optional | 0.0–2.0 |
| `start_percent` | optional | default 0.0 |
| `end_percent` | optional | default 1.0 |

A model_validator enforces the one-of constraint. JSON-per-`--controlnet` is the
correct wire format — any mini-format would have required speculative field mapping
against this validator. T15 (`validate-track3`) is where ergonomic shorthand belongs.

**Mode switch is conditional:** `SwitchMode` is called only when the resolved mode
differs from `CurrentMode`. If `CurrentMode` fails (network hiccup), the switch is
silently skipped. Acceptable for v1; a strict mode would return an error.

**Meta stamp is best-effort:** `pngmeta.WriteText` is skipped if the fetched bytes
are not a valid PNG. A non-PNG backend response (format change, SR edge case) never
blocks the write.

**`printGenResult` uses `cmd.OutOrStdout()`:** Allows test capture via
`rootCmd.SetOut(&buf)` without redirecting the real process stdout.

**`buildGenParams(nil, args)` in tests:** `nil` cfg is handled by substituting an
empty `&config.Config{}`, so the parameter-seam tests require no config file on disk.

---

## T13 — `st read` + `--recreate` characterization test

**No implementation deviations.** `read.go` is a pure local-file command: `ReadLCM` →
`json.MarshalIndent` → `cmd.OutOrStdout()`. No config gate, no network. Matches the
plan's intent exactly.

**Characterization test approach (deviation from TDD):** `TestRecreateSeedsParams` is
not a RED-first test — the `--recreate` baked layer was already wired in T12's
`buildGenParams`. The test characterizes the existing contract rather than driving new
code. This was the correct call: the real RED signal for T13 was the `st read` command
(`unknown command "read"` before implementation). The characterization test pins three
contracts the plan described but didn't test:

1. `{Recreate: path}` with no flags → baked `cfg` and `prompt` come through from the lcm chunk.
2. `--recreate` never sets `init_image_ref` (recipe-only).
3. Explicit `--cfg 1` flag overrides the baked `cfg: 9`.

**`pngWithLCM` test helper:** synthesizes a structurally valid 1×1 RGBA PNG via
`png.Encode` (stdlib), then stamps the lcm chunk via `pngmeta.WriteText`. Using a real
PNG prevents silent pass on CRC/length bugs in `WriteText`; a raw byte literal would
not catch structural corruption.

---

## T14 — Peripheral commands (upload, superres, cancel, priority, models, modes)

**No new stclient logic.** All six are thin Cobra wrappers over existing `pkg/stclient`
methods. Plan constraint held.

**`emitJSON` shared helper (deviation from plan):** Plan sketched inline JSON encoding
in each command. Extracted to `util.go` as `emitJSON(cmd *cobra.Command, v any) error`
— shared by upload, superres, cancel, priority, modes. DRY move; no logic change.

**`models` always emits JSON (deviation from plan):** `models.go` calls `emitJSON`
unconditionally — it does not branch on `--json`. A status map has no useful
human-readable rendering; always-JSON matches the command's semantics. Tests confirm
this: `TestModelsCmdPrints` passes no `--json` flag and still expects JSON-formatted
output.

**`superres` output dir defaults to `"."` (deviation from plan):** Plan was silent on
the default output directory for `superres` (which has no config gate). Defaults to
`"."` when neither `-o` nor config is available — writing to the working directory is
the least-surprise default for a file-transformation command.

**Contract notes:**

- `superres --magnitude` flag default is `2`, matching the server default
  (`magnitude: int = Form(2)` at `lcm_sr_server.py:685`).
- `priority <jobId> <level>`: `level` parsed via `strconv.Atoi` with clear error
  message ("level must be an integer: …"). Server handler is a no-op stub (T6 godoc);
  the command still sends the correct frame.
- `modes` without `--json` prints one name per line in sorted order (sort happens in
  `stclient.Modes()`, so the CLI layer doesn't need to re-sort).

**Tests — key assertions:**

- `TestCancelCmdAcks` captures the WS frame `type` field server-side, asserts
  `"job:cancel"`, and asserts the jobId appears in the output.
- `TestPriorityCmdAcks` captures `priority` from the WS frame as `float64` (JSON
  decode type), asserts `float64(5)` — the correct type assertion for JSON-decoded
  numbers in Go.
- `TestSuperresCmdWritesOutput` uses `-o` to specify a temp dir and asserts
  `out-0001.png` exists with expected bytes, exercising the full `output.Resolve` /
  `output.Write` path.

---

## T15 — `st validate-track3` acceptance script

**No structural deviations.** The upload→generate→artifact-assert sequence matches the
plan's checklist.

**ControlNet attachment shape:** hardcoded `attachment_id: "track3-1"` in the submit
params. Acceptable for an acceptance script with a fixed checklist; the value is
arbitrary (it is echoed back in `controlnet_artifacts` but not validated by the
backend beyond uniqueness). The three required fields present:

- `attachment_id` — required ✓
- `control_type` — required, taken from `--control-type` flag (default `"canny"`) ✓
- `map_asset_ref` — one-of (the uploaded fileRef) ✓

**`CNArtifacts` type:** `Result.CNArtifacts []json.RawMessage`, populated from
`inFrame.CNArts` (`controlnet_artifacts`). `len(res.CNArtifacts) == 0` is the correct
check — any non-empty slice means the backend returned artifacts.

**`execCmd` helper in tests (deviation from plan):** The plan's test sketches used
`runCmd` (which calls `t.Fatalf` on error). T15 needs to assert non-zero exit, so a
new `execCmd` helper was added that returns `(output, error)` without failing the
test. `runCmd` and `execCmd` both exist in the `cmd/st` test package; they differ only
in error handling.

**`track3Calls.uploadBeforeSub` ordering assertion:** the httptest handler captures
`calls.uploaded` at the moment the WS `job:submit` frame arrives. This is the correct
idiom for asserting cross-protocol ordering when HTTP and WS calls arrive on different
goroutines — capturing state at submit time rather than asserting at test end avoids a
race condition.

---

## T16 — OpenAPI drift guard + README

**`canonicalize` approach:** uses Go's `encoding/json` round-trip (unmarshal → remarshal).
`json.Marshal` sorts object keys alphabetically and drops insignificant whitespace
deterministically. No external dependencies; the two red-first unit tests
(`TestCanonicalizeIgnoresKeyOrderAndWhitespace`, `TestCanonicalizeDetectsRealDiff`)
confirm both the positive and negative cases before the gated live test.

**Snapshot path in the drift test:** `../../openapi.snapshot.json` resolves from
`cli/go/internal/openapi/` to `cli/go/openapi.snapshot.json`. Correct.

**Gated test stays 3.1-vs-3.1:** `TestOpenAPISnapshotMatchesLive` is skipped unless
`$ST_SERVER` is set. The snapshot is verbatim 3.1.0 (the form FastAPI serves); the
live endpoint also serves 3.1.0. No downspec artefacts appear in either side of the
diff. This was the explicit design goal: the downspec is a codegen-only concern and
must never corrupt the drift signal.

**CI deviation (correct call):** The plan said "modify CI config" but `.github/workflows/`
is git-ignored (`.gitignore:21`) and per AGENTS.md the authoritative pipeline is
Concourse in `../continuous/docs`. The GitHub Actions job was not committed. The README
documents the CI contract (`go vet ./...` + `go test ./...`) and tells the reader to
wire it into the shared Concourse pipeline rather than introducing a tracked workflow
here. This is the correct boundary per the project architecture.

**README content:** covers all 9 commands, config discovery chain and precedence
layering, OpenAPI contract/downspec split, drift guard invocation, and CI note. The
backend-is-remote note is the first callout — matches CLI-first delivery constraint.
