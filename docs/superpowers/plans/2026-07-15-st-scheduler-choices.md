# Expose Mode Scheduler Choices in st CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Checkboxes (`- [ ]`) are inline execution-step markers only — they are not the tracking authority. Task/review state lives in FP + waveplan (review gates). (This repo forbids subagent-driven development — do not dispatch subagents.)

**Goal:** Make the scheduler IDs allowed by each generation mode discoverable from the `st` CLI, both in human `st modes` output and in `st modes show <name> --json`.

**Architecture:** `GET /api/modes` already returns `allowed_scheduler_ids` per mode, but `pkg/stclient.Mode` drops it during decode. Add the field to `Mode` (and the inner decode struct) so it flows through unchanged into `st modes show --json`, then render one `schedulers:` line per mode in the human `st modes` list. No backend change; `default_scheduler_id` decode already exists.

**Tech Stack:** Go 1.x, cobra, stdlib `net/http`/`net/http/httptest`, `encoding/json`.

**FP issue:** STABL-xyywague

## Global Constraints

- `st gen --json` output contract is frozen and unrelated to this work — do not touch it.
- `pkg/stclient` is a shared surface (CLI + future MCP). Do not add CLI concerns (flags, cobra, stderr) into it.
- Human `st modes` output must stay back-compatible: when the backend sends no `allowed_scheduler_ids`, no new line appears.
- Do not use `grep`/`find` for symbol lookup; use semantic tools. `grep` for raw text only.
- Run Go commands from the module dir: `cd /Users/darkbit1001/workspace/Stability-Toys/cli/go`.

---

### Task 1: Preserve `allowed_scheduler_ids` in `stclient.Mode`

**Files:**
- Modify: `cli/go/pkg/stclient/http.go:18-28` (`Mode` struct), `cli/go/pkg/stclient/http.go:70-93` (`Modes` decode + literal)
- Test: `cli/go/pkg/stclient/http_test.go`

**Interfaces:**
- Consumes: live `GET /api/modes` shape — each mode object may carry `"allowed_scheduler_ids": ["euler", ...]` and `"default_scheduler_id": "euler"`.
- Produces: `Mode.AllowedSchedulerIDs []string` (JSON tag `allowed_scheduler_ids,omitempty`), populated from the per-mode config; `nil` when the field is absent. Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

Add to `cli/go/pkg/stclient/http_test.go`. Extend the shared `realModesResponse` const so the `cartoony` mode carries an allowed list and the `default` mode omits it, then assert both:

```go
func TestModesPreservesAllowedSchedulerIDs(t *testing.T) {
	body := `{
	  "default_mode": "default",
	  "modes": {
	    "default":  {"model": "sdxl-base", "default_size": "1024x1024", "default_steps": 20, "default_guidance": 7.5,
	                 "default_scheduler_id": "euler", "controlnet_policy": {"enabled": true}, "chat_enabled": false},
	    "cartoony": {"model": "sdxl-cartoon", "default_size": "512x512", "default_steps": 8, "default_guidance": 2.5,
	                 "default_scheduler_id": "lcm", "allowed_scheduler_ids": ["lcm", "euler", "ddim"],
	                 "controlnet_policy": {"enabled": false}, "chat_enabled": false}
	  }
	}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	modes, err := New(srv.URL).Modes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// modes[0] == cartoony (sorted), modes[1] == default
	if got := modes[0].AllowedSchedulerIDs; len(got) != 3 || got[0] != "lcm" || got[2] != "ddim" {
		t.Fatalf("cartoony allowed_scheduler_ids not preserved: %+v", got)
	}
	if modes[0].DefaultSchedulerID != "lcm" {
		t.Fatalf("cartoony default_scheduler_id = %q, want lcm", modes[0].DefaultSchedulerID)
	}
	// Absent field must decode to nil, not panic or empty-marker.
	if modes[1].AllowedSchedulerIDs != nil {
		t.Fatalf("default mode should have nil AllowedSchedulerIDs, got %+v", modes[1].AllowedSchedulerIDs)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/darkbit1001/workspace/Stability-Toys/cli/go && go test ./pkg/stclient/ -run TestModesPreservesAllowedSchedulerIDs -v`
Expected: FAIL — compile error `modes[0].AllowedSchedulerIDs undefined (type Mode has no field ...)`.

- [ ] **Step 3: Add the field to `Mode`**

In `cli/go/pkg/stclient/http.go`, add to the `Mode` struct (after `DefaultSchedulerID`, line 25):

```go
	DefaultSchedulerID  string   `json:"default_scheduler_id,omitempty"`
	AllowedSchedulerIDs []string `json:"allowed_scheduler_ids,omitempty"`
```

- [ ] **Step 4: Decode and copy the field in `Modes`**

In the inner `cfg` struct (around line 70-80), add after `DefaultSchedulerID`:

```go
			DefaultSchedulerID  string   `json:"default_scheduler_id"`
			AllowedSchedulerIDs []string `json:"allowed_scheduler_ids"`
```

In the `Mode{...}` literal (around line 82-92), add after `DefaultSchedulerID: cfg.DefaultSchedulerID,`:

```go
			AllowedSchedulerIDs: cfg.AllowedSchedulerIDs,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/darkbit1001/workspace/Stability-Toys/cli/go && go test ./pkg/stclient/ -run TestModesPreservesAllowedSchedulerIDs -v`
Expected: PASS.

- [ ] **Step 6: Run the full stclient package to confirm no regression**

Run: `cd /Users/darkbit1001/workspace/Stability-Toys/cli/go && go test ./pkg/stclient/`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add cli/go/pkg/stclient/http.go cli/go/pkg/stclient/http_test.go
git commit -m "feat(stclient): preserve allowed_scheduler_ids on Mode decode (STABL-xyywague) — next: st modes human rendering + docs"
```

---

### Task 2: Render scheduler choices in `st modes` + docs

**Files:**
- Modify: `cli/go/cmd/st/modes.go:42-66` (`runModes` human list)
- Modify: `cli/go/USAGE.md` (Model modes section, around line 381-401)
- Test: `cli/go/cmd/st/modes_test.go`

**Interfaces:**
- Consumes: `Mode.AllowedSchedulerIDs []string` and `Mode.DefaultSchedulerID string` from Task 1.
- Produces: no new exported symbols; only human output and a JSON passthrough assertion.

- [ ] **Step 1: Write the failing human-output test**

Add to `cli/go/cmd/st/modes_test.go`. Give `fakeModeServer` a mode with an allowed list, then assert the `schedulers:` line and the default marker. Because `fakeModeServer` is shared, add a dedicated server inline rather than editing the shared helper:

```go
func TestModesListRendersSchedulers(t *testing.T) {
	body := `{"default_mode":"fast","modes":{
	  "fast":{"model":"sdxl-turbo","default_size":"512x512","default_steps":4,"default_guidance":0,
	          "default_scheduler_id":"lcm","allowed_scheduler_ids":["lcm","euler","ddim"],
	          "controlnet_policy":{"enabled":true}},
	  "plain":{"model":"sd15","default_size":"512x512","default_steps":20,"default_guidance":7.5,
	           "controlnet_policy":{"enabled":false}}
	}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	out, err := runCmdMayFail(t, "--server", srv.URL, "modes")
	if err != nil {
		t.Fatal(err)
	}
	// The allowed list renders with the default marked.
	if !strings.Contains(out, "schedulers: lcm (default), euler, ddim") {
		t.Errorf("expected rendered schedulers line, got:\n%s", out)
	}
	// A mode with no allowed list must not emit a schedulers line.
	if strings.Contains(out, "schedulers: \n") || strings.Count(out, "schedulers:") != 1 {
		t.Errorf("mode without allowed_scheduler_ids should omit the line, got:\n%s", out)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/darkbit1001/workspace/Stability-Toys/cli/go && go test ./cmd/st/ -run TestModesListRendersSchedulers -v`
Expected: FAIL — output has no `schedulers:` line.

- [ ] **Step 3: Render the schedulers line in `runModes`**

In `cli/go/cmd/st/modes.go`, inside the `for _, m := range modes` loop, after the existing `fmt.Fprintf(... extra)` call (line 62-63), add:

```go
		if len(m.AllowedSchedulerIDs) > 0 {
			labeled := make([]string, len(m.AllowedSchedulerIDs))
			for i, id := range m.AllowedSchedulerIDs {
				if id == m.DefaultSchedulerID {
					labeled[i] = id + " (default)"
				} else {
					labeled[i] = id
				}
			}
			fmt.Fprintf(cmd.OutOrStdout(), "  schedulers: %s\n", strings.Join(labeled, ", "))
		}
```

Add `"strings"` to the import block at the top of `modes.go`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/darkbit1001/workspace/Stability-Toys/cli/go && go test ./cmd/st/ -run TestModesListRendersSchedulers -v`
Expected: PASS.

- [ ] **Step 5: Add a JSON-passthrough assertion**

Extend `TestModesShowCmdJSON` (or add a sibling) to prove the field reaches `--json`. Update `fakeModeServer`'s body to include the allowed list on `fast`, then assert:

```go
func TestModesShowCmdJSONIncludesSchedulers(t *testing.T) {
	body := `{"default_mode":"fast","modes":{"fast":{"model":"sdxl-turbo","default_size":"512x512","default_steps":4,"default_guidance":0,"default_scheduler_id":"lcm","allowed_scheduler_ids":["lcm","euler"],"controlnet_policy":{"enabled":true}}}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	out, err := runCmdMayFail(t, "--server", srv.URL, "modes", "show", "fast")
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if jsonErr := json.Unmarshal([]byte(strings.TrimSpace(out)), &m); jsonErr != nil {
		t.Fatalf("output not valid JSON: %v\noutput: %q", jsonErr, out)
	}
	ids, ok := m["allowed_scheduler_ids"].([]any)
	if !ok || len(ids) != 2 || ids[0] != "lcm" {
		t.Errorf("allowed_scheduler_ids missing/wrong in JSON: %v", m["allowed_scheduler_ids"])
	}
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/darkbit1001/workspace/Stability-Toys/cli/go && go test ./cmd/st/ -run TestModesShowCmdJSONIncludesSchedulers -v`
Expected: PASS.

- [ ] **Step 7: Document mode-specific scheduler availability in USAGE.md**

In `cli/go/USAGE.md`, in the Model modes section (near the `st modes` / `st modes show` examples around line 384-396), add prose + example:

```markdown
Scheduler availability is mode-specific. `st modes` lists the allowed scheduler
IDs per mode (the mode default is marked), and `st modes show <name> --json`
exposes both `default_scheduler_id` and `allowed_scheduler_ids` for scripting:

# See allowed schedulers per mode (default marked):
st modes
# cartoony
#   model=sdxl-cartoon  size=512x512  steps=8  cfg=2.5
#   schedulers: lcm (default), euler, ddim

# Machine-readable scheduler choices for one mode:
st modes show cartoony --json | jq '.allowed_scheduler_ids'
```

- [ ] **Step 8: Run the full command package + build**

Run: `cd /Users/darkbit1001/workspace/Stability-Toys/cli/go && go test ./cmd/st/ ./pkg/stclient/ && go build ./...`
Expected: `ok` for both packages, clean build.

- [ ] **Step 9: Commit**

```bash
git add cli/go/cmd/st/modes.go cli/go/cmd/st/modes_test.go cli/go/USAGE.md
git commit -m "feat(st): render mode scheduler choices in st modes + USAGE docs (STABL-xyywague) — next: drift check + fp update"
```

---

## Post-implementation

- [ ] `drift check` on `cli/go/USAGE.md` and `cli/go/cmd/st/modes.go`; update prose before relinking if anything is stale.
- [ ] `fp comment STABL-xyywague` with STOP/NEXT, then report ready for review per the waveplan cycle. Do not self-advance past implementation.

## Self-Review Notes

- **Spec coverage:** preserve allowed list (Task 1), expose via modes output — human + JSON (Task 2 steps 1-6), document mode-specific availability (Task 2 step 7), client + command tests (both tasks). All Done-criteria covered.
- **Compatibility:** absent `allowed_scheduler_ids` → nil → no human line (Task 1 step 1 nil assertion, Task 2 step 1 omit assertion). `omitempty` keeps it out of JSON when empty.
- **Type consistency:** `AllowedSchedulerIDs []string` and `DefaultSchedulerID string` used identically across both tasks.
