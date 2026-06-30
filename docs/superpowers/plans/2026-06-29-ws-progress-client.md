# WS job:progress client-side hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Read before writing any code:**
>
> 1. [`.superpowers/sdd/project-context.md`](../../../.superpowers/sdd/project-context.md) — protocol facts, code patterns, test conventions
> 2. This file — task requirements and constraints

**Goal:** Harden the Go client's `job:progress` handling: skip the `onProgress` callback for frames with no/empty `delta`, assert the delta value in the existing progress test, and update USAGE.md to reflect the actual server progress format.

**Architecture:** Two tasks. Task 1 is a one-line guard in `ws.go` plus two new tests in `ws_test.go`. Task 2 is a prose-only USAGE.md edit — no code. Both are backward-compatible; the callback signature and Generate signature do not change.

**Tech Stack:** Go, `net/http/httptest`, `github.com/coder/websocket`

**FP issue:** STABL-ykcbssxk — tag every commit with it.

**Depends on:** STABL-dakbipff (done — backend now emits `delta` string in `job:progress` frames using format `"node {n}/{total} ({pct}%)"`)

## Global Constraints

- Do not change the `Generate` signature or the `onProgress func(delta string)` type
- Empty-delta frames must be silently skipped — no error, no nil callback invocation
- `--json` output shape is frozen — do not touch `printGenResult`'s `flagJSON` branch
- Run tests from `cli/go/`: `cd cli/go && go test ./...`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `cli/go/pkg/stclient/ws.go` | Modify | Add `f.Delta != ""` guard in `job:progress` case |
| `cli/go/pkg/stclient/ws_test.go` | Modify | Add `TestGenerateSkipsOnProgressForEmptyDelta`; add `TestGeneratePassesDeltaValueToCallback` |
| `cli/go/USAGE.md` | Modify | Update `--stream` example to show real `"node N/T (P%)"` progress lines |

---

## Task 1: Empty-delta guard + delta value tests

**Files:**
- Modify: `cli/go/pkg/stclient/ws.go`
- Modify: `cli/go/pkg/stclient/ws_test.go`

**Interfaces:**
- No signature changes. Guard is internal to `Generate`.

- [x] **Step 1: Write two failing tests** — append to `cli/go/pkg/stclient/ws_test.go`

```go
func TestGenerateSkipsOnProgressForEmptyDelta(t *testing.T) {
	srv := fakeGenServer(t, []any{
		map[string]any{"type": "job:ack", "jobId": "J1"},
		// frame with no delta field at all (legacy server / non-image path)
		map[string]any{"type": "job:progress", "jobId": "J1"},
		// frame with explicitly empty delta
		map[string]any{"type": "job:progress", "jobId": "J1", "delta": ""},
		// frame with a real delta — only this one should fire the callback
		map[string]any{"type": "job:progress", "jobId": "J1", "delta": "node 1/4 (25%)"},
		map[string]any{
			"type":    "job:complete",
			"jobId":   "J1",
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta":    map[string]any{"seed": float64(1)},
		},
	})
	defer srv.Close()

	var received []string
	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{}, nil, func(delta string) {
		received = append(received, delta)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(received) != 1 {
		t.Fatalf("expected 1 onProgress call (non-empty delta only), got %d: %v", len(received), received)
	}
	if received[0] != "node 1/4 (25%)" {
		t.Errorf("delta = %q, want %q", received[0], "node 1/4 (25%)")
	}
}

func TestGeneratePassesDeltaValueToCallback(t *testing.T) {
	deltas := []string{"node 1/6 (16%)", "node 2/6 (33%)", "node 3/6 (50%)"}
	frames := []any{
		map[string]any{"type": "job:ack", "jobId": "J1"},
	}
	for _, d := range deltas {
		frames = append(frames, map[string]any{"type": "job:progress", "jobId": "J1", "delta": d})
	}
	frames = append(frames, map[string]any{
		"type":    "job:complete",
		"jobId":   "J1",
		"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
		"meta":    map[string]any{"seed": float64(42)},
	})
	srv := fakeGenServer(t, frames)
	defer srv.Close()

	var received []string
	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{}, nil, func(delta string) {
		received = append(received, delta)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(received) != len(deltas) {
		t.Fatalf("got %d callbacks, want %d", len(received), len(deltas))
	}
	for i, want := range deltas {
		if received[i] != want {
			t.Errorf("callback[%d] = %q, want %q", i, received[i], want)
		}
	}
}
```

- [x] **Step 2: Run tests to confirm they fail**

```bash
cd cli/go && go test ./pkg/stclient/... -run "TestGenerateSkipsOnProgressForEmptyDelta|TestGeneratePassesDeltaValueToCallback" -v
```

Expected: `TestGenerateSkipsOnProgressForEmptyDelta` FAIL — `expected 1 onProgress call, got 3` (current code fires for all frames including empty delta). `TestGeneratePassesDeltaValueToCallback` PASS — it will pass already since the current code does pass delta through. That's fine; it hardens coverage.

- [x] **Step 3: Add empty-delta guard in `cli/go/pkg/stclient/ws.go`**

In the `job:progress` case (currently lines 55–58), replace:

```go
		case "job:progress":
			if onProgress != nil {
				onProgress(f.Delta)
			}
```

with:

```go
		case "job:progress":
			if onProgress != nil && f.Delta != "" {
				onProgress(f.Delta)
			}
```

- [x] **Step 4: Run tests to confirm both pass**

```bash
cd cli/go && go test ./pkg/stclient/... -run "TestGenerateSkipsOnProgressForEmptyDelta|TestGeneratePassesDeltaValueToCallback" -v
```

Expected: both `PASS`

- [x] **Step 5: Run full suite**

```bash
cd cli/go && go test ./...
```

Expected: all tests pass (no regressions — `TestGenerateCallsOnProgressForAllFrames` sends `"delta": "x"` frames, non-empty, so it is unaffected)

- [x] **Step 6: Commit**

```bash
git add cli/go/pkg/stclient/ws.go cli/go/pkg/stclient/ws_test.go
git commit -m "feat(cli): skip onProgress for empty delta; assert delta value in tests (STABL-ykcbssxk) — next: USAGE.md"
```

---

## Task 2: Update USAGE.md --stream example

**Files:**
- Modify: `cli/go/USAGE.md`

**Interfaces:** None — documentation only.

- [x] **Step 1: Update the `--stream` example in `cli/go/USAGE.md`**

Find the `--stream` block (around line 73). Replace:

```bash
# NDJSON stream — one object per line as events arrive:
st gen "an owl" --stream
# {"job_id":"J9a3b2c1"}
# {"event":"progress","delta":"step 1/8..."}
# {"event":"progress","delta":"step 2/8..."}
# ...
# {"event":"complete","output":"images/out-0001.png","seed":3847291,"storage_key":"...","storage_url":"..."}
```

with:

```bash
# NDJSON stream — one object per line as events arrive:
st gen "an owl" --stream
# {"job_id":"J9a3b2c1"}
# {"delta":"node 1/8 (12%)","event":"progress"}
# {"delta":"node 2/8 (25%)","event":"progress"}
# {"delta":"node 3/8 (37%)","event":"progress"}
# ...
# {"event":"complete","output":"images/out-0001.png","seed":3847291,"storage_key":"lcm_image:...","storage_url":"/storage/lcm_image:..."}
```

Note: keys are alphabetical (Go's `json.Marshal` sorts map keys). `delta` before `event`.

- [x] **Step 2: Commit**

```bash
git add cli/go/USAGE.md
git commit -m "docs(cli): update --stream example with real progress delta format (STABL-ykcbssxk)"
```

---

## Self-Review

**Spec coverage:**
- Empty-delta guard: frames with no `delta` field or `delta=""` skip `onProgress` ✅ Task 1 guard + `TestGenerateSkipsOnProgressForEmptyDelta`
- Delta value asserted in test ✅ `TestGeneratePassesDeltaValueToCallback`
- `TestGenerateCallsOnProgressForAllFrames` unaffected ✅ sends `"delta":"x"`, non-empty
- USAGE.md progress format matches backend output `"node N/T (P%)"` ✅ Task 2
- NDJSON key order in USAGE.md matches `json.Marshal` alphabetical sort (`delta` before `event`) ✅
- Generate signature unchanged ✅
- `--json` branch untouched ✅

**Placeholder scan:** None found.

**Type consistency:** `f.Delta` is `string`; guard `f.Delta != ""` is valid for the zero value of a missing JSON field.
