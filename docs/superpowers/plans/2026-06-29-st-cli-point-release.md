# st CLI v1.x Point Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Every implementer dispatch must open with — read both files before writing any code:**
>
> 1. [`.superpowers/sdd/project-context.md`](../../../.superpowers/sdd/project-context.md) — locked decisions, protocol facts, code patterns
> 2. This file (`docs/superpowers/plans/2026-06-29-st-cli-point-release.md`) — task requirements and global constraints
>
> The context doc carries: exact `inFrame` struct, what `job:ack` does and doesn't carry, why callbacks replace the channel, why `bucket` is a form field not a query param, the frozen `--json` contract with example output, all breaking-change callsites by name, what is NOT in scope, and the two test patterns (`fakeGenServer`, HTTP method template, Cobra subcommand template). The plan alone is not sufficient.

**Goal:** Ship job observability (`--stream`, `--quiet`, job_id on stderr), mode management subcommands (`switch`, `show`, `reload`), upload bucket intent declaration, `--controlnet-file`, and ControlNet config presets as a backward-compatible point release.

**Architecture:** Three independent tracks (E mode cmds, A observation, C ControlNet) that each commit cleanly. Task 3 is the only breaking change — it removes the unused progress channel from `Generate()` and replaces it with synchronous callbacks — all broken callsites are fixed in the same task. `--json` output contract is frozen; `--stream` is the new NDJSON flag.

**Tech Stack:** Go, Cobra, `github.com/coder/websocket`, `net/http/httptest`

## Global Constraints

- `--json` output shape is frozen: `{"output","seed","storage_key","storage_url"}` — do not add fields, do not reorder
- `--stream` emits compact (not indented) NDJSON: one JSON object per line, no trailing whitespace
- `Upload()` backward compat: `bucket=""` adds no form field (existing `gen.go` init-image call passes `""`)
- `st modes` (no subcommand) keeps its current list behavior — do not remove `RunE` from `modesCmd`
- Run tests from `cli/go/`: `go test ./...`
- `set-default` is NOT in scope — no `POST /api/modes/default` endpoint exists in the backend

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `cli/go/pkg/stclient/http.go` | Modify | Add `ReloadModes()`; add `bucket string` to `Upload()` |
| `cli/go/pkg/stclient/http_test.go` | Modify | Add `TestReloadModesPostsToEndpoint` |
| `cli/go/pkg/stclient/upload_test.go` | Modify | Update `TestUploadReturnsFileRef` call to pass `bucket=""` |
| `cli/go/pkg/stclient/ws.go` | Modify | New `Generate(ctx, params, onAck, onProgress)` signature — drop channel, expose jobID |
| `cli/go/pkg/stclient/types.go` | Modify | Remove `Progress` struct and `progressBuffer` constant |
| `cli/go/pkg/stclient/ws_test.go` | Modify | Update 3 callsites; replace deadlock test with callback coverage test; add jobID test |
| `cli/go/cmd/st/modes.go` | Modify | Add `switch`, `show`, `reload` cobra subcommands |
| `cli/go/cmd/st/gen.go` | Modify | Use new `Generate` sig; add `--stream`, `--quiet`, `--controlnet-file`; preset expansion; update init-image `Upload` call |
| `cli/go/cmd/st/upload.go` | Modify | Parse `type:path` arg; pass bucket to `Upload()` |
| `cli/go/cmd/st/gen_test.go` | Modify | Add test for `--controlnet-file`; add test for `@preset` expansion |
| `cli/go/internal/config/config.go` | Modify | Add `ControlnetPresets map[string]ControlnetPreset` to `Config` |

---

## Task 1: E — stclient: ReloadModes method

**Files:**
- Modify: `cli/go/pkg/stclient/http.go`
- Modify: `cli/go/pkg/stclient/http_test.go`

**Interfaces:**
- Produces: `func (c *Client) ReloadModes(ctx context.Context) error` — consumed by Task 2

- [ ] **Step 1: Write failing test** in `cli/go/pkg/stclient/http_test.go`

```go
func TestReloadModesPostsToEndpoint(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/modes/reload" || r.Method != http.MethodPost {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := New(srv.URL).ReloadModes(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("POST /api/modes/reload not called")
	}
}
```

- [ ] **Step 2: Run test to confirm it fails**

```
cd cli/go && go test ./pkg/stclient/... -run TestReloadModesPostsToEndpoint -v
```
Expected: `FAIL` — `New(srv.URL).ReloadModes` undefined

- [ ] **Step 3: Implement `ReloadModes` in `cli/go/pkg/stclient/http.go`**

Add after `SwitchMode`:

```go
// ReloadModes requests the server to hot-reload modes.yaml from disk via
// POST /api/modes/reload. The reload is applied after any pending jobs complete.
func (c *Client) ReloadModes(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/modes/reload", nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("modes reload -> %s", resp.Status)
	}
	return nil
}
```

- [ ] **Step 4: Run test to confirm it passes**

```
cd cli/go && go test ./pkg/stclient/... -run TestReloadModesPostsToEndpoint -v
```
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient/http.go cli/go/pkg/stclient/http_test.go
git commit -m "feat(cli): stclient.ReloadModes — POST /api/modes/reload (STABL-kczspmud) — next: modes subcommands"
```

---

## Task 2: E — modes switch/show/reload subcommands

**Files:**
- Modify: `cli/go/cmd/st/modes.go`
- Test: `cli/go/cmd/st/modes_test.go` (new)

**Interfaces:**
- Consumes: `client.SwitchMode(ctx, name)` (already in `http.go:122`), `client.Modes(ctx)` (already in `http.go:55`), `client.ReloadModes(ctx)` (Task 1)
- Produces: `st modes switch <name>`, `st modes show <name>`, `st modes reload`

- [ ] **Step 1: Write failing tests** — create `cli/go/cmd/st/modes_test.go`

```go
package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModesReloadCmdHitsEndpoint(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/modes/reload" && r.Method == http.MethodPost {
			called = true
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	cfgPath := writeTestConfig(t, t.TempDir())
	out := runCmd(t, "--server", srv.URL, "--config", cfgPath, "modes", "reload")
	if !called {
		t.Fatal("POST /api/modes/reload was not called")
	}
	if !strings.Contains(out, "reloaded") {
		t.Errorf("expected 'reloaded' in output, got: %q", out)
	}
}

func TestModesSwitchCmdHitsEndpoint(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/modes/switch" && r.Method == http.MethodPost {
			called = true
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	cfgPath := writeTestConfig(t, t.TempDir())
	out := runCmd(t, "--server", srv.URL, "--config", cfgPath, "modes", "switch", "fast")
	if !called {
		t.Fatal("POST /api/modes/switch was not called")
	}
	if !strings.Contains(out, "fast") {
		t.Errorf("expected mode name in output, got: %q", out)
	}
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cd cli/go && go test ./cmd/st/... -run "TestModesReloadCmd|TestModesSwitchCmd" -v
```
Expected: FAIL — `modes reload` and `modes switch` are unknown subcommands (Cobra returns error → `runCmd` fatals)

- [ ] **Step 3: Implement subcommands** — replace `cli/go/cmd/st/modes.go`

The existing `modesCmd` has `Args: cobra.NoArgs` and `RunE: runModes` — keep both. Cobra routes `st modes switch foo` to the child command; `st modes` alone still calls `runModes`.

Replace the entire content of `cli/go/cmd/st/modes.go` with:

```go
package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

var modesCmd = &cobra.Command{
	Use:   "modes",
	Short: "List or manage model modes",
	Args:  cobra.NoArgs,
	RunE:  runModes,
}

var modesSwitchCmd = &cobra.Command{
	Use:   "switch <name>",
	Short: "Switch to a model mode",
	Args:  cobra.ExactArgs(1),
	RunE:  runModesSwitch,
}

var modesShowCmd = &cobra.Command{
	Use:   "show <name>",
	Short: "Print configuration for a mode as JSON",
	Args:  cobra.ExactArgs(1),
	RunE:  runModesShow,
}

var modesReloadCmd = &cobra.Command{
	Use:   "reload",
	Short: "Hot-reload modes.yaml on the server",
	Args:  cobra.NoArgs,
	RunE:  runModesReload,
}

func init() {
	modesCmd.AddCommand(modesSwitchCmd, modesShowCmd, modesReloadCmd)
	rootCmd.AddCommand(modesCmd)
}

func runModes(cmd *cobra.Command, args []string) error {
	modes, err := newClient().Modes(cmd.Context())
	if err != nil {
		return err
	}
	if flagJSON {
		return emitJSON(cmd, modes)
	}
	for _, m := range modes {
		name := m.Name
		if m.IsDefault {
			name += " (default)"
		}
		extra := ""
		if m.ControlNetEnabled {
			extra += "  controlnet"
		}
		if m.ChatEnabled {
			extra += "  chat"
		}
		fmt.Fprintf(cmd.OutOrStdout(), "%s\n  model=%s  size=%s  steps=%d  cfg=%.1f%s\n",
			name, m.Model, m.DefaultSize, m.DefaultSteps, m.DefaultGuidance, extra)
	}
	return nil
}

func runModesSwitch(cmd *cobra.Command, args []string) error {
	if err := newClient().SwitchMode(cmd.Context(), args[0]); err != nil {
		return err
	}
	fmt.Fprintf(cmd.OutOrStdout(), "switched to %s\n", args[0])
	return nil
}

func runModesShow(cmd *cobra.Command, args []string) error {
	modes, err := newClient().Modes(cmd.Context())
	if err != nil {
		return err
	}
	for _, m := range modes {
		if m.Name == args[0] {
			return emitJSON(cmd, m)
		}
	}
	return fmt.Errorf("mode %q not found", args[0])
}

func runModesReload(cmd *cobra.Command, args []string) error {
	if err := newClient().ReloadModes(cmd.Context()); err != nil {
		return err
	}
	fmt.Fprintln(cmd.OutOrStdout(), "modes reloaded")
	return nil
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```
cd cli/go && go test ./cmd/st/... -run "TestModesReloadCmd|TestModesSwitchCmd" -v
```
Expected: PASS

- [ ] **Step 5: Verify compile and existing modes tests pass**

```
cd cli/go && go build ./... && go test ./cmd/st/... -run TestModes -v
```
Expected: build succeeds, all modes tests pass (check `peripherals_test.go` for any existing modes coverage)

- [ ] **Step 6: Manually verify subcommand routing**

```
cd cli/go && go run ./cmd/st modes --help
```
Expected: shows `switch`, `show`, `reload` as subcommands in the help output.

- [ ] **Step 7: Commit**

```bash
git add cli/go/cmd/st/modes.go cli/go/cmd/st/modes_test.go
git commit -m "feat(cli): st modes switch/show/reload subcommands (STABL-kczspmud) — next: Generate callbacks"
```

---

## Task 3: A — Generate callbacks, jobID, --stream, --quiet

This task makes a breaking change to `Generate()` — drops the unused progress channel, adds `onAck`/`onProgress` callbacks, returns `jobID`. All callsites are fixed in the same task.

**Files:**
- Modify: `cli/go/pkg/stclient/ws.go`
- Modify: `cli/go/pkg/stclient/types.go`
- Modify: `cli/go/pkg/stclient/ws_test.go`
- Modify: `cli/go/cmd/st/gen.go`

**Interfaces:**
- Produces: `func (c *Client) Generate(ctx context.Context, p GenParams, onAck func(string), onProgress func(string)) (string, *Result, error)`
  - `onAck` called once with `jobID` immediately after server ack
  - `onProgress` called once per `job:progress` frame with the delta text
  - either callback may be `nil`
  - returns `(jobID string, result *Result, err error)`

- [ ] **Step 1: Update `ws_test.go` to match the new signature**

The three existing tests call `Generate` with the old signature. Update all three, replace the deadlock test with a callback coverage test, and add a jobID test.

Replace the entire content of `cli/go/pkg/stclient/ws_test.go`:

```go
package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

// fakeGenServer spins up a test WS server that reads one job:submit and
// then sends the provided frames in order. Handlers that do not match
// /v1/ws fall through (e.g. /storage/ for FetchStorage in gen_test.go).
func fakeGenServer(t *testing.T, frames []any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/ws") {
			http.NotFound(w, r)
			return
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("WS accept: %v", err)
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		wsjson.Read(r.Context(), conn, &sub)
		for _, f := range frames {
			if err := wsjson.Write(r.Context(), conn, f); err != nil {
				return
			}
		}
	}))
}

func TestGenerateResolvesOnComplete(t *testing.T) {
	srv := fakeGenServer(t, []any{
		map[string]any{"type": "job:ack", "id": "corr1", "jobId": "J1"},
		map[string]any{
			"type":    "job:complete",
			"jobId":   "J1",
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta":    map[string]any{"seed": float64(777)},
		},
	})
	defer srv.Close()

	_, res, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": "owl"}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.StorageKey != "K1" || res.StorageURL != "/storage/K1" || res.Seed != 777 {
		t.Fatalf("got %+v", res)
	}
}

func TestGenerateReturnsJobID(t *testing.T) {
	srv := fakeGenServer(t, []any{
		map[string]any{"type": "job:ack", "id": "corr1", "jobId": "job-abc"},
		map[string]any{
			"type":    "job:complete",
			"jobId":   "job-abc",
			"outputs": []any{map[string]any{"url": "/storage/K2", "key": "K2"}},
			"meta":    map[string]any{"seed": float64(1)},
		},
	})
	defer srv.Close()

	var gotAck string
	jobID, _, err := New(srv.URL).Generate(context.Background(), GenParams{}, func(id string) { gotAck = id }, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gotAck != "job-abc" {
		t.Errorf("onAck got %q, want job-abc", gotAck)
	}
	if jobID != "job-abc" {
		t.Errorf("returned jobID = %q, want job-abc", jobID)
	}
}

func TestGenerateReturnsErrorOnJobError(t *testing.T) {
	srv := fakeGenServer(t, []any{
		map[string]any{"type": "job:ack", "id": "corr1", "jobId": "J1"},
		map[string]any{"type": "job:error", "jobId": "J1", "error": "Missing prompt"},
	})
	defer srv.Close()

	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": ""}, nil, nil)
	if err == nil {
		t.Fatal("expected error on job:error, got nil")
	}
	if !strings.Contains(err.Error(), "Missing prompt") {
		t.Fatalf("error should carry server message, got: %v", err)
	}
}

func TestGenerateCallsOnProgressForAllFrames(t *testing.T) {
	const n = 50
	frames := []any{
		map[string]any{"type": "job:ack", "jobId": "J1"},
	}
	for range n {
		frames = append(frames, map[string]any{"type": "job:progress", "jobId": "J1", "delta": "x"})
	}
	frames = append(frames, map[string]any{
		"type":    "job:complete",
		"jobId":   "J1",
		"outputs": []any{map[string]any{"url": "/storage/K9", "key": "K9"}},
		"meta":    map[string]any{},
	})
	srv := fakeGenServer(t, frames)
	defer srv.Close()

	count := 0
	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": "x"}, nil, func(string) { count++ })
	if err != nil {
		t.Fatal(err)
	}
	if count != n {
		t.Errorf("onProgress called %d times, want %d", count, n)
	}
}
```

- [ ] **Step 2: Run tests to confirm they fail (old signature)**

```
cd cli/go && go test ./pkg/stclient/... -run "TestGenerate" -v 2>&1 | head -30
```
Expected: compile errors — `Generate` called with wrong number of args

- [ ] **Step 3: Update `types.go` — remove `Progress` and `progressBuffer`**

In `cli/go/pkg/stclient/types.go`, remove:
- The line `const progressBuffer = 16` (it's in ws.go, not types.go — double-check with the read above; `progressBuffer` is in ws.go line 17)
- The `Progress` struct (types.go line 49: `type Progress struct{ Delta string }`)

The `Progress` struct removal: delete lines 48-49 from `types.go`:
```go
// Progress is a streamed generation update.
type Progress struct{ Delta string }
```

- [ ] **Step 4: Update `ws.go` — new `Generate` signature**

Replace the entire `Generate` function in `cli/go/pkg/stclient/ws.go`:

```go
// Generate dials /v1/ws, submits a generate job, and blocks until the job
// resolves. It returns the jobID assigned by the server (from job:ack), a
// Result on job:complete, or an error on job:error.
//
// onAck is called once with the jobID immediately after the server
// acknowledges the submission. onProgress is called synchronously for each
// job:progress frame in the order received. Either callback may be nil.
func (c *Client) Generate(ctx context.Context, p GenParams, onAck func(jobID string), onProgress func(delta string)) (string, *Result, error) {
	conn, _, err := websocket.Dial(ctx, c.wsURL(), &websocket.DialOptions{HTTPClient: c.http})
	if err != nil {
		return "", nil, err
	}
	if err := wsjson.Write(ctx, conn, newSubmitFrame(corrID(), p)); err != nil {
		conn.Close(websocket.StatusInternalError, "submit failed")
		return "", nil, err
	}
	var jobID string
	for {
		var f inFrame
		if err := wsjson.Read(ctx, conn, &f); err != nil {
			conn.Close(websocket.StatusInternalError, "read failed")
			return "", nil, err
		}
		switch f.Type {
		case "job:ack":
			jobID = f.JobID
			if onAck != nil {
				onAck(jobID)
			}
		case "job:progress":
			if onProgress != nil {
				onProgress(f.Delta)
			}
		case "job:error":
			conn.Close(websocket.StatusNormalClosure, "")
			return "", nil, fmt.Errorf("job error: %s", f.Error)
		case "job:complete":
			conn.Close(websocket.StatusNormalClosure, "")
			res := &Result{Meta: f.Meta, CNArtifacts: f.CNArts}
			if len(f.Outputs) > 0 {
				res.StorageKey = f.Outputs[0].Key
				res.StorageURL = f.Outputs[0].URL
			}
			if s, ok := f.Meta["seed"].(float64); ok {
				res.Seed = int64(s)
			}
			return jobID, res, nil
		}
	}
}
```

Also remove the `progressBuffer` constant at the top of `ws.go` (line 17: `const progressBuffer = 16`).

- [ ] **Step 5: Run stclient tests to confirm all pass**

```
cd cli/go && go test ./pkg/stclient/... -v
```
Expected: all tests pass

- [ ] **Step 6: Update `gen.go` — new callsite + `--stream` + `--quiet`**

In `cli/go/cmd/st/gen.go`, make these changes:

**6a.** Add flag variables after the existing `genOutfile` var block:

```go
var (
	genStream bool
	genQuiet  bool
)
```

**6b.** Add flags in `init()` after the `genOutfile` flag:

```go
f.BoolVar(&genStream, "stream", false, "stream progress as NDJSON to stdout (job_id, progress events, complete)")
f.BoolVar(&genQuiet, "quiet", false, "suppress progress and job_id output on stderr")
```

**6c.** Add `buildObservationCallbacks` helper (add before `runGen`):

```go
// buildObservationCallbacks returns onAck and onProgress callbacks for
// Generate based on the active output flags.
//   - quiet:  both nil (silent)
//   - stream: NDJSON to stdout — job_id line on ack, progress lines per frame
//   - default: job_id + progress delta to stderr
func buildObservationCallbacks(cmd *cobra.Command, quiet, stream bool) (func(string), func(string)) {
	if quiet {
		return nil, nil
	}
	if stream {
		onAck := func(id string) {
			b, _ := json.Marshal(map[string]any{"job_id": id})
			fmt.Fprintln(cmd.OutOrStdout(), string(b))
		}
		onProg := func(delta string) {
			b, _ := json.Marshal(map[string]any{"event": "progress", "delta": delta})
			fmt.Fprintln(cmd.OutOrStdout(), string(b))
		}
		return onAck, onProg
	}
	onAck := func(id string) { fmt.Fprintf(cmd.ErrOrStderr(), "job_id=%s\n", id) }
	onProg := func(delta string) { fmt.Fprint(cmd.ErrOrStderr(), delta) }
	return onAck, onProg
}
```

**6d.** In `runGen`, add the mutual-exclusion guard and wire callbacks. Replace the `Generate` callsite (line ~224):

```go
// before:
_, res, err := client.Generate(ctx, params)

// after (add just before this block):
if genStream && flagJSON {
    return fmt.Errorf("--stream and --json are mutually exclusive")
}
onAck, onProgress := buildObservationCallbacks(cmd, genQuiet, genStream)
jobID, res, err := client.Generate(ctx, params, onAck, onProgress)
_ = jobID // surfaced to caller via onAck; reserved for future st watch composition
```

**6e.** Replace `printGenResult` entirely (full function replacement, not a partial edit):

```go
func printGenResult(cmd *cobra.Command, path string, res *stclient.Result) error {
	if genStream {
		out := map[string]any{
			"event":       "complete",
			"output":      path,
			"seed":        res.Seed,
			"storage_key": res.StorageKey,
			"storage_url": res.StorageURL,
		}
		b, err := json.Marshal(out)
		if err != nil {
			return err
		}
		fmt.Fprintln(cmd.OutOrStdout(), string(b))
		return nil
	}
	if flagJSON {
		out := map[string]any{
			"output":      path,
			"storage_key": res.StorageKey,
			"storage_url": res.StorageURL,
			"seed":        res.Seed,
		}
		b, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			return err
		}
		fmt.Fprintln(cmd.OutOrStdout(), string(b))
		return nil
	}
	fmt.Fprintf(cmd.OutOrStdout(), "wrote %s (seed %d)\n", path, res.Seed)
	return nil
}
```

- [ ] **Step 7: Run all tests**

```
cd cli/go && go test ./... -v 2>&1 | tail -30
```
Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add cli/go/pkg/stclient/ws.go cli/go/pkg/stclient/types.go cli/go/pkg/stclient/ws_test.go cli/go/cmd/st/gen.go
git commit -m "feat(cli): Generate callbacks + jobID + --stream + --quiet (STABL-kczspmud) — next: --controlnet-file"
```

---

## Task 4: C2 — `--controlnet-file` flag

**Files:**
- Modify: `cli/go/cmd/st/gen.go`
- Modify: `cli/go/cmd/st/gen_test.go`

**Interfaces:**
- Consumes: existing `buildGenParams(cfg, genArgs)` — adds `ControlnetFile string` field to `genArgs`
- Produces: `--controlnet-file <path>` flag on `gen`

- [ ] **Step 1: Write failing test** in `cli/go/cmd/st/gen_test.go`

```go
func TestBuildGenParamsControlnetFile(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "cn-*.json")
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(`{"attachment_id":"a2","control_type":"depth","map_asset_ref":"fileref:D1"}`)
	f.Close()

	args := genArgs{Prompt: "x", ControlnetFile: f.Name()}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("controlnets: %+v", p["controlnets"])
	}
	entry, _ := list[0].(map[string]any)
	if entry["control_type"] != "depth" {
		t.Fatalf("control_type = %v, want depth", entry["control_type"])
	}
}

func TestBuildGenParamsControlnetFileMergesWithFlag(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "cn-*.json")
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(`{"attachment_id":"file-cn","control_type":"depth"}`)
	f.Close()

	cn := `{"attachment_id":"flag-cn","control_type":"canny"}`
	args := genArgs{Prompt: "x", Controlnets: []string{cn}, ControlnetFile: f.Name()}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 2 {
		t.Fatalf("expected 2 controlnets, got: %+v", p["controlnets"])
	}
}
```

Add `"os"` to the test file imports if not already present.

- [ ] **Step 2: Run test to confirm it fails**

```
cd cli/go && go test ./cmd/st/... -run "TestBuildGenParamsControlnetFile" -v
```
Expected: compile error — `ControlnetFile` not defined on `genArgs`

- [ ] **Step 3: Add `ControlnetFile` to `genArgs` struct** in `cli/go/cmd/st/gen.go`

```go
type genArgs struct {
	Prompt         string
	Negative       *string
	Genres         *string
	Steps          *int
	Cfg            *float64
	Seed           *string
	Scheduler      *string
	Mode           *string
	SR             *int
	InitImage      string
	Recreate       string
	Controlnets    []string
	ControlnetFile string   // add this line
	Outfile        string
}
```

- [ ] **Step 4: Add flag variable and register flag** in `gen.go`

Add to the var block:
```go
var genControlnetFile string
```

Add to `init()` after the `genControlnets` flag:
```go
f.StringVar(&genControlnetFile, "controlnet-file", "", "ControlNetAttachment JSON file (merged with --controlnet entries)")
```

Add to `genArgsFromFlags`, after the `genControlnets` line:
```go
a.ControlnetFile = genControlnetFile
```

- [ ] **Step 5: Add expansion in `buildGenParams`** after the existing `Controlnets` block

In `buildGenParams`, after:
```go
	if len(a.Controlnets) > 0 {
		...
		p["controlnets"] = cns
	}
```

Add:
```go
	if a.ControlnetFile != "" {
		data, err := os.ReadFile(a.ControlnetFile)
		if err != nil {
			return nil, fmt.Errorf("--controlnet-file %q: %w", a.ControlnetFile, err)
		}
		var cn map[string]any
		if err := json.Unmarshal(data, &cn); err != nil {
			return nil, fmt.Errorf("--controlnet-file %q: invalid JSON: %w", a.ControlnetFile, err)
		}
		cns, _ := p["controlnets"].([]any)
		p["controlnets"] = append(cns, cn)
	}
```

- [ ] **Step 6: Run tests**

```
cd cli/go && go test ./cmd/st/... -run "TestBuildGenParamsControlnetFile" -v
```
Expected: `PASS`

- [ ] **Step 7: Run full suite**

```
cd cli/go && go test ./...
```
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add cli/go/cmd/st/gen.go cli/go/cmd/st/gen_test.go
git commit -m "feat(cli): --controlnet-file flag merges JSON attachment into controlnets (STABL-kczspmud) — next: upload bucket"
```

---

## Task 5: C3 — Upload bucket argument

**Files:**
- Modify: `cli/go/pkg/stclient/http.go` (add `bucket` param to `Upload`)
- Modify: `cli/go/pkg/stclient/upload_test.go` (update existing test + add bucket tests)
- Modify: `cli/go/cmd/st/upload.go` (parse `type:path` arg)
- Modify: `cli/go/cmd/st/gen.go` (update init-image `Upload` call to pass `""`)

**Interfaces:**
- Produces: `func (c *Client) Upload(ctx context.Context, filename string, data []byte, bucket string) (string, error)`
  - `bucket=""` → no `type` form field (backward compat)
  - `bucket="canny"` → `type=canny` form field in multipart body

- [ ] **Step 1: Update `upload_test.go` to test the new `bucket` parameter**

Replace the content of `cli/go/pkg/stclient/upload_test.go`:

```go
package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestUploadReturnsFileRef(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/upload" || r.Method != http.MethodPost {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if _, _, err := r.FormFile("file"); err != nil {
			t.Fatalf("no file part: %v", err)
		}
		w.Write([]byte(`{"fileRef":"abc123"}`))
	}))
	defer srv.Close()

	ref, err := New(srv.URL).Upload(context.Background(), "x.png", []byte("PNGBYTES"), "")
	if err != nil {
		t.Fatal(err)
	}
	if ref != "abc123" {
		t.Fatalf("got %q", ref)
	}
}

func TestUploadSendsBucketFormField(t *testing.T) {
	var gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		gotType = r.FormValue("type")
		w.Write([]byte(`{"fileRef":"R-abc"}`))
	}))
	defer srv.Close()

	ref, err := New(srv.URL).Upload(context.Background(), "map.png", []byte("data"), "canny")
	if err != nil {
		t.Fatal(err)
	}
	if ref != "R-abc" {
		t.Errorf("ref = %q, want R-abc", ref)
	}
	if gotType != "canny" {
		t.Errorf("form type = %q, want canny", gotType)
	}
}

func TestUploadNoTypeFieldWhenBucketEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.ParseMultipartForm(1 << 20)
		if v := r.FormValue("type"); v != "" {
			t.Errorf("expected no type field when bucket is empty, got %q", v)
		}
		w.Write([]byte(`{"fileRef":"R-xyz"}`))
	}))
	defer srv.Close()

	New(srv.URL).Upload(context.Background(), "img.png", []byte("data"), "")
}

func TestSuperResSendsFileAndMagnitudeReturnsImage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/superres" || r.Method != http.MethodPost {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if _, _, err := r.FormFile("file"); err != nil {
			t.Fatalf("no file part: %v", err)
		}
		if got := r.FormValue("magnitude"); got != "3" {
			t.Fatalf("magnitude = %q, want 3", got)
		}
		w.Header().Set("Content-Type", "image/png")
		w.Write([]byte("SRIMAGE"))
	}))
	defer srv.Close()

	out, err := New(srv.URL).SuperRes(context.Background(), []byte("INPUT"), 3)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != "SRIMAGE" {
		t.Fatalf("got %q, want SRIMAGE", out)
	}
}

func TestFetchStorageReturnsBytes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/storage/out-key.png" || r.Method != http.MethodGet {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		w.Write([]byte("RESULTPNG"))
	}))
	defer srv.Close()

	out, err := New(srv.URL).FetchStorage(context.Background(), "out-key.png")
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != "RESULTPNG" {
		t.Fatalf("got %q", out)
	}
}

func TestFetchStorageErrorsOn404(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	if _, err := New(srv.URL).FetchStorage(context.Background(), "missing"); err == nil {
		t.Fatal("expected error on 404, got nil")
	}
}
```

- [ ] **Step 2: Run tests to confirm they fail (old `Upload` signature)**

```
cd cli/go && go test ./pkg/stclient/... -run "TestUpload" -v
```
Expected: compile error — `Upload` called with wrong number of args

- [ ] **Step 3: Update `Upload` in `cli/go/pkg/stclient/http.go`**

Replace the `Upload` function:

```go
// Upload posts data as a multipart "file" to POST /v1/upload and returns the
// fileRef the backend assigns. bucket is an optional intent label (e.g.
// "image", "canny") sent as a "type" form field; an empty bucket adds no
// extra field. The backend may use the type field for routing; v1.x treats
// it as client-side intent only.
func (c *Client) Upload(ctx context.Context, filename string, data []byte, bucket string) (string, error) {
	var fields map[string]string
	if bucket != "" {
		fields = map[string]string{"type": bucket}
	}
	buf, contentType, err := multipartFile(filename, data, fields)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/upload", buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", contentType)
	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("upload -> %s", resp.Status)
	}
	var body struct {
		FileRef string `json:"fileRef"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", err
	}
	return body.FileRef, nil
}
```

- [ ] **Step 4: Run stclient tests**

```
cd cli/go && go test ./pkg/stclient/... -v
```
Expected: all pass

- [ ] **Step 5: Update `upload.go` to parse `type:path` arg**

Replace the content of `cli/go/cmd/st/upload.go`:

```go
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var uploadCmd = &cobra.Command{
	Use:   "upload [type:]<file>",
	Short: "Upload a file and print its fileref",
	Long: `Upload a local file to the server and print the assigned fileref.

Optionally prefix the path with a type label to declare intent:

  st upload image:./owl.png      # general image upload
  st upload canny:./map.png      # declare as a canny control map

The type label is sent as a "type" form field. Without a prefix the file
is uploaded with no type declared.`,
	Args: cobra.ExactArgs(1),
	RunE: runUpload,
}

func init() {
	rootCmd.AddCommand(uploadCmd)
}

func runUpload(cmd *cobra.Command, args []string) error {
	bucket, filePath := parseUploadArg(args[0])
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}
	ref, err := newClient().Upload(cmd.Context(), filepath.Base(filePath), data, bucket)
	if err != nil {
		return err
	}
	if flagJSON {
		return emitJSON(cmd, map[string]any{"fileRef": ref, "bucket": bucket})
	}
	fmt.Fprintln(cmd.OutOrStdout(), ref)
	return nil
}

// parseUploadArg splits "type:path" into (type, path). If no colon is
// present, returns ("", arg) — no bucket, plain path.
func parseUploadArg(arg string) (bucket, path string) {
	if before, after, ok := strings.Cut(arg, ":"); ok {
		return before, after
	}
	return "", arg
}
```

- [ ] **Step 6: Fix init-image `Upload` call in `gen.go`** (line ~208)

In `runGen`, find:
```go
ref, err := client.Upload(ctx, filepath.Base(a.InitImage), data)
```

Replace with:
```go
ref, err := client.Upload(ctx, filepath.Base(a.InitImage), data, "")
```

- [ ] **Step 7: Run full test suite**

```
cd cli/go && go test ./...
```
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add cli/go/pkg/stclient/http.go cli/go/pkg/stclient/upload_test.go cli/go/cmd/st/upload.go cli/go/cmd/st/gen.go
git commit -m "feat(cli): upload bucket intent + st upload type:path syntax (STABL-kczspmud) — next: controlnet presets"
```

---

## Task 6: C4 — ControlNet config presets

**Files:**
- Modify: `cli/go/internal/config/config.go`
- Modify: `cli/go/cmd/st/gen.go`
- Modify: `cli/go/cmd/st/gen_test.go`

**Interfaces:**
- Consumes: `Config.ControlnetPresets map[string]ControlnetPreset` (new)
- Produces: `--controlnet @preset-name` expansion in `buildGenParams`

- [ ] **Step 1: Write failing test** in `gen_test.go`

```go
func TestBuildGenParamsControlnetPreset(t *testing.T) {
	cfg := &config.Config{}
	cfg.ControlnetPresets = map[string]config.ControlnetPreset{
		"owl-canny": {"attachment_id": "cn-1", "control_type": "canny", "map_asset_ref": "fileref:D1"},
	}
	args := genArgs{Prompt: "x", Controlnets: []string{"@owl-canny"}}
	p, err := buildGenParams(cfg, args)
	if err != nil {
		t.Fatal(err)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("controlnets: %+v", p["controlnets"])
	}
	entry, _ := list[0].(map[string]any)
	if entry["control_type"] != "canny" {
		t.Fatalf("control_type = %v, want canny", entry["control_type"])
	}
}

func TestBuildGenParamsControlnetPresetMissingErrors(t *testing.T) {
	args := genArgs{Prompt: "x", Controlnets: []string{"@unknown"}}
	_, err := buildGenParams(nil, args)
	if err == nil {
		t.Fatal("expected error for unknown preset, got nil")
	}
	if !strings.Contains(err.Error(), "@unknown") {
		t.Errorf("error should name the preset, got: %v", err)
	}
}
```

Add `"strings"` to gen_test.go imports if not already present.

- [ ] **Step 2: Run tests to confirm they fail**

```
cd cli/go && go test ./cmd/st/... -run "TestBuildGenParamsControlnetPreset" -v
```
Expected: compile error — `config.ControlnetPreset` undefined

- [ ] **Step 3: Add `ControlnetPreset` and `ControlnetPresets` to `config.go`**

In `cli/go/internal/config/config.go`, add after the `Meta` type:

```go
// ControlnetPreset is a named ControlNetAttachment stored in config and
// referenced by --controlnet @name. Values are passed verbatim to the
// backend as a controlnet attachment object.
type ControlnetPreset map[string]any
```

Update `Config`:

```go
// Config is the root document (unwrapped from the "config" key on disk).
type Config struct {
	Defaults          Defaults                    `json:"defaults"`
	ControlnetPresets map[string]ControlnetPreset `json:"controlnet_presets,omitempty"`
}
```

- [ ] **Step 4: Add preset expansion in `buildGenParams`** in `gen.go`

Replace the existing `if len(a.Controlnets) > 0` block with:

```go
	if len(a.Controlnets) > 0 {
		cns := make([]any, 0, len(a.Controlnets))
		for _, raw := range a.Controlnets {
			if presetName, ok := strings.CutPrefix(raw, "@"); ok {
				var preset config.ControlnetPreset
				if cfg != nil {
					preset = cfg.ControlnetPresets[presetName]
				}
				if preset == nil {
					return nil, fmt.Errorf("--controlnet @%s: preset not found in config", presetName)
				}
				cns = append(cns, map[string]any(preset))
			} else {
				var cn map[string]any
				if err := json.Unmarshal([]byte(raw), &cn); err != nil {
					return nil, fmt.Errorf("--controlnet %q: %w", raw, err)
				}
				cns = append(cns, cn)
			}
		}
		p["controlnets"] = cns
	}
```

Ensure `"strings"` is in the `gen.go` import block (it already is: `"strings"` is used for `strings.HasPrefix` and `strings.CutPrefix`).

- [ ] **Step 5: Run tests**

```
cd cli/go && go test ./cmd/st/... -run "TestBuildGenParamsControlnetPreset" -v
```
Expected: `PASS`

- [ ] **Step 6: Run full test suite**

```
cd cli/go && go test ./...
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add cli/go/internal/config/config.go cli/go/cmd/st/gen.go cli/go/cmd/st/gen_test.go
git commit -m "feat(cli): controlnet config presets — --controlnet @name (STABL-kczspmud) — point release complete"
```

---

## Self-Review

**Spec coverage:**
- A1 progress to stderr ✅ Task 3 `onProgress` default path
- A2 job_id on ack ✅ Task 3 `onAck` default path (stderr) + stream (NDJSON)
- A4 NDJSON `--stream` ✅ Task 3
- `--quiet` ✅ Task 3
- E reload ✅ Task 1 + Task 2
- E switch ✅ Task 2 (stclient already existed)
- E show ✅ Task 2
- C2 `--controlnet-file` ✅ Task 4
- C3 upload bucket ✅ Task 5
- C4 config presets ✅ Task 6
- `--json` contract frozen ✅ `printGenResult` unchanged for `flagJSON` path
- `set-default` omitted ✅ no backend endpoint

**Breaking changes handled in-task:**
- `Generate()` signature: `ws_test.go` updated in Task 3, `gen.go` callsite updated in Task 3
- `Upload()` signature: `upload_test.go` updated in Task 5, `gen.go` init-image call updated in Task 5

**`--stream` NDJSON shape:**
```
{"job_id":"<id>"}
{"event":"progress","delta":"<text>"}
{"event":"complete","output":"<path>","seed":<n>,"storage_key":"<key>","storage_url":"<url>"}
```
Compact (no indent) — correct for NDJSON piped to `jq`.
