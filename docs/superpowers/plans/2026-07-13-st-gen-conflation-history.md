# st Gen Conflation and History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project policy forbids subagent-driven development in this repo.

**Goal:** Add always-on global CLI history, persistent `st conflate` policy, exact `st replay`, and gen-only conflation semantics without breaking existing `st gen` stdout contracts.

**Architecture:** Introduce a small `internal/history` package that owns immutable history entries, policy/state models, rendering, and filesystem-backed persistence under XDG state. Wrap Cobra execution in a new history-aware runtime in `cmd/st`, then thread resolved baselines into the existing `gen` path so `st gen`, root shorthand, and `st replay` all converge on one generation executor.

**Tech Stack:** Go 1.26.1, Cobra/pflag, JSON/JSONL, `os`/`filepath`, advisory file locking, existing `internal/config`, `internal/output`, `internal/pngmeta`, `pkg/stclient`, `go test`.

**Authority:** `docs/superpowers/specs/2026-07-12-st-gen-conflation-history-design.md`

---

## Global Constraints

- Work from `cli/go`; verification commands in this plan assume that module root.
- Preserve frozen stdout contracts for `st gen --json` and `st gen --stream`. New conflation/replay diagnostics go to stderr and must honor `--quiet`.
- Keep conflation eligibility fixed to `gen` in v1. `conflate` and `replay` are command-surface additions, not generic history search.
- Root shorthand is flag-only. Bare positional text at the `st` root must not become paid prompt patches.
- History must be append-only and immutable after append. Reserve IDs before execution; tolerate gaps after crashes.
- Hold `state.lock` only around one reserve, append, or policy-replacement operation; never across parsing, uploads, or a remote generation.
- Keep storage abstract behind interfaces even though v1 ships only the filesystem backend.
- Do not regress current config precedence for ordinary `st gen` runs when conflation is disabled.

## File Structure

- **Create `cli/go/internal/history/types.go`:** history entry, raw/effective command, lineage, policy snapshot, selector/filter models, and family constants.
- **Create `cli/go/internal/history/render.go`:** stable shell-escaped display rendering plus canonical effective `argv` rendering for resolved `gen` params.
- **Create `cli/go/internal/history/store.go`:** `HistoryStore`/`PolicyStore` interfaces and typed errors (`ErrNotFound`, `ErrNoEligibleEntry`, `ErrCorruptState`).
- **Create `cli/go/internal/history/fs.go`:** XDG state-path resolution, `state.lock`, `history.jsonl` append/read, `next-id`, `conflate-policy.json`, and lock-scoped atomic updates.
- **Create tests:** `cli/go/internal/history/render_test.go`, `cli/go/internal/history/fs_test.go`.
- **Create `cli/go/cmd/st/conflate.go`:** `st conflate`, policy toggle/idempotent forms, selector parsing, and status output.
- **Create `cli/go/cmd/st/replay.go`:** `st replay <id>` command surface and replay validation.
- **Create `cli/go/cmd/st/history_runtime.go`:** top-level history wrapper, root shorthand detection, baseline/replay resolution, and final append path.
- **Create `cli/go/cmd/st/history_runtime_test.go`, `cli/go/cmd/st/conflate_test.go`, `cli/go/cmd/st/replay_test.go`:** runtime/history command coverage.
- **Modify `cli/go/cmd/st/main.go`:** replace direct `rootCmd.Execute()` with the history-aware runner and expose injectable state/client helpers for tests.
- **Modify `cli/go/cmd/st/gen.go`:** split flag parsing from generation execution, preserve explicit-field intent, support inherited `effective.params`, emit stderr baseline diagnostics, and record concrete seeds.
- **Modify `cli/go/internal/config/precedence.go` and `precedence_test.go`:** add the explicit four-layer resolver used by conflated generations.
- **Modify `cli/go/cmd/st/gen_test.go`:** route the shared command harness through the history-aware runner and add conflation/replay helpers beside existing generation coverage.
- **Modify `cli/go/USAGE.md`, `cli/go/README.md`:** document `conflate`, `replay`, XDG state, and root shorthand restrictions.

---

### Task 1: Add history domain types, rendering, and state-path helpers

**Files:**
- Create: `cli/go/internal/history/types.go`
- Create: `cli/go/internal/history/render.go`
- Create: `cli/go/internal/history/render_test.go`

- [x] **Step 1: Write failing rendering and selector tests**

Create `cli/go/internal/history/render_test.go`:

```go
package history

import "fmt"

import (
	"reflect"
	"testing"
)

func TestRenderRawDisplayShellEscapesArgv(t *testing.T) {
	got := RenderArgv([]string{"st", "--prompt", "two horses drinking"})
	want := "st --prompt 'two horses drinking'"
	if got != want {
		t.Fatalf("display = %q, want %q", got, want)
	}
}

func TestCanonicalGenArgvNormalizesPromptAndStableOrder(t *testing.T) {
	params := map[string]any{
		"prompt":             "horse bartender",
		"guidance_scale":     4.5,
		"size":               "1024x1024",
		"num_inference_steps": 20,
	}
	got := CanonicalGenArgv(params)
	want := []string{"st", "gen", "--prompt", "horse bartender", "--size", "1024x1024", "--steps", "20", "--cfg", "4.5"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("argv = %#v, want %#v", got, want)
	}
}

func TestSelectorSnapshotForPinnedHistory(t *testing.T) {
	s := Selector{Kind: SelectorHistory, HistoryID: 12345}
	if s.Kind != SelectorHistory || s.HistoryID != 12345 {
		t.Fatalf("selector = %#v", s)
	}
}
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./internal/history -run 'TestRenderRawDisplayShellEscapesArgv|TestCanonicalGenArgvNormalizesPromptAndStableOrder|TestSelectorSnapshotForPinnedHistory' -count=1
```

Expected: FAIL with `stat .../internal/history: directory not found`.

- [x] **Step 3: Implement the core history models**

Create `cli/go/internal/history/types.go`:

```go
package history

type Family string

const (
	FamilyGen      Family = "gen"
	FamilyConflate Family = "conflate"
	FamilyUnknown  Family = "unknown"
)

type SelectorKind string

const (
	SelectorRecent  SelectorKind = "recent"
	SelectorHistory SelectorKind = "history"
)

type CommandView struct {
	Argv    []string       `json:"argv"`
	Display string         `json:"display"`
	Params  map[string]any `json:"params,omitempty"`
}

type Selector struct {
	Kind      SelectorKind `json:"kind"`
	Family    Family       `json:"family,omitempty"`
	ExitCodes []int        `json:"exit_codes,omitempty"`
	HistoryID int64        `json:"history_id,omitempty"`
}

type Policy struct {
	SchemaVersion int      `json:"schema_version"`
	Enabled       bool     `json:"enabled"`
	Selector      Selector `json:"selector"`
	UpdatedAt     string   `json:"updated_at"`
}

type PolicySnapshot struct {
	Selector  string `json:"selector"`
	HistoryID int64  `json:"history_id,omitempty"`
}

type Entry struct {
	SchemaVersion         int            `json:"schema_version"`
	ID                    int64          `json:"id"`
	StartedAt             string         `json:"started_at"`
	FinishedAt            string         `json:"finished_at"`
	Family                Family         `json:"family"`
	Raw                   CommandView    `json:"raw"`
	Effective             *CommandView   `json:"effective,omitempty"`
	ExitCode              int            `json:"exit_code"`
	DerivedFromHistoryID  *int64         `json:"derived_from_history_id,omitempty"`
	ReplayedFromHistoryID *int64         `json:"replayed_from_history_id,omitempty"`
	ConflatePolicy        *PolicySnapshot `json:"conflate_policy,omitempty"`
	Error                 *string        `json:"error"`
}

func DefaultPolicy() Policy {
	return Policy{
		SchemaVersion: 1,
		Enabled:       false,
		Selector: Selector{
			Kind:      SelectorRecent,
			Family:    FamilyGen,
			ExitCodes: []int{0},
		},
	}
}

func ValidatePolicy(policy Policy) error {
	if policy.SchemaVersion != 1 {
		return fmt.Errorf("unsupported policy schema_version %d", policy.SchemaVersion)
	}
	switch policy.Selector.Kind {
	case SelectorHistory:
		if policy.Selector.HistoryID < 1 {
			return fmt.Errorf("history selector requires a positive history_id")
		}
	case SelectorRecent:
		if policy.Selector.Family != FamilyGen || len(policy.Selector.ExitCodes) == 0 {
			return fmt.Errorf("recent selector requires family gen and at least one exit code")
		}
		for _, code := range policy.Selector.ExitCodes {
			if code < 0 || code > 255 {
				return fmt.Errorf("invalid exit code %d", code)
			}
		}
	default:
		return fmt.Errorf("unsupported selector kind %q", policy.Selector.Kind)
	}
	return nil
}
```

- [x] **Step 4: Implement rendering helpers**

Create `cli/go/internal/history/render.go`:

```go
package history

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

func RenderArgv(argv []string) string {
	parts := make([]string, 0, len(argv))
	for _, arg := range argv {
		if arg == "" || strings.ContainsAny(arg, " \t\n'\"") {
			parts = append(parts, "'"+strings.ReplaceAll(arg, "'", "'\\''")+"'")
			continue
		}
		parts = append(parts, arg)
	}
	return strings.Join(parts, " ")
}

func CanonicalGenArgv(params map[string]any) []string {
	argv := []string{"st", "gen"}
	appendStr := func(flag, key string) {
		if v, ok := params[key].(string); ok && v != "" {
			argv = append(argv, flag, v)
		}
	}
	appendNum := func(flag, key string) {
		switch v := params[key].(type) {
		case int:
			argv = append(argv, flag, strconv.Itoa(v))
		case int64:
			argv = append(argv, flag, strconv.FormatInt(v, 10))
		case float64:
			argv = append(argv, flag, strconv.FormatFloat(v, 'f', -1, 64))
		}
	}

	appendStr("--prompt", "prompt")
	appendStr("--negative", "negative_prompt")
	appendStr("--size", "size")
	appendNum("--steps", "num_inference_steps")
	appendNum("--skip-step", "skip_step")
	appendNum("--cfg", "guidance_scale")
	appendStr("--seed", "seed")
	appendStr("--scheduler", "scheduler_id")
	appendStr("--mode", "mode")
	return argv
}

func CanonicalGenDisplay(params map[string]any) string {
	return RenderArgv(CanonicalGenArgv(params))
}

func CloneParams(params map[string]any) map[string]any {
	out := make(map[string]any, len(params))
	for key, value := range params {
		out[key] = value
	}
	return out
}

func NormalizeExitCodes(codes []int) []int {
	seen := map[int]struct{}{}
	out := make([]int, 0, len(codes))
	for _, code := range codes {
		if _, ok := seen[code]; ok {
			continue
		}
		seen[code] = struct{}{}
		out = append(out, code)
	}
	slices.Sort(out)
	return out
}

func SnapshotSelector(sel Selector) *PolicySnapshot {
	switch sel.Kind {
	case SelectorHistory:
		return &PolicySnapshot{Selector: "history", HistoryID: sel.HistoryID}
	case SelectorRecent:
		return &PolicySnapshot{Selector: fmt.Sprintf("recent:%s", sel.Family)}
	default:
		return nil
	}
}
```

- [x] **Step 5: Run the tests and verify GREEN**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./internal/history -run 'TestRenderRawDisplayShellEscapesArgv|TestCanonicalGenArgvNormalizesPromptAndStableOrder|TestSelectorSnapshotForPinnedHistory' -count=1
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
cd /Users/darkbit1001/workspace/Stability-Toys
git add cli/go/internal/history/types.go cli/go/internal/history/render.go cli/go/internal/history/render_test.go
git commit -m "feat: add st history domain models"
```

---

### Task 2: Implement filesystem-backed history and policy persistence

**Files:**
- Create: `cli/go/internal/history/store.go`
- Create: `cli/go/internal/history/fs.go`
- Create: `cli/go/internal/history/fs_test.go`

- [x] **Step 1: Write failing storage, crash-tolerance, and concurrency tests**

Create `cli/go/internal/history/fs_test.go`:

```go
package history

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func TestFSStoreReserveAppendAndLatest(t *testing.T) {
	root := t.TempDir()
	store := NewFSStore(root)
	ctx := context.Background()

	id, err := store.ReserveID(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("id = %d, want 1", id)
	}

	entry := Entry{
		SchemaVersion: 1,
		ID:            id,
		Family:        FamilyGen,
		Raw:           CommandView{Argv: []string{"st", "gen"}, Display: "st gen"},
		Effective:     &CommandView{Params: map[string]any{"prompt": "owl"}},
		ExitCode:      0,
	}
	if err := store.Append(ctx, entry); err != nil {
		t.Fatal(err)
	}

	got, err := store.Latest(ctx, Filter{Family: FamilyGen, ExitCodes: []int{0}, RequireEffective: true})
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != id {
		t.Fatalf("latest id = %d, want %d", got.ID, id)
	}
}

func TestPolicyStoreRoundTripsDefaultPolicy(t *testing.T) {
	root := t.TempDir()
	store := NewFSStore(root)
	ctx := context.Background()

	policy := DefaultPolicy()
	policy.Enabled = true
	if err := store.SavePolicy(ctx, policy); err != nil {
		t.Fatal(err)
	}

	got, err := store.LoadPolicy(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Enabled || len(got.Selector.ExitCodes) != 1 || got.Selector.ExitCodes[0] != 0 {
		t.Fatalf("policy = %#v", got)
	}
}

func TestResolveStateRootUsesXDGStateHome(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	got, err := ResolveStateRoot()
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(got) != "st" {
		t.Fatalf("state root = %q", got)
	}
	if _, err := os.Stat(filepath.Dir(got)); err != nil {
		t.Fatalf("state parent missing: %v", err)
	}
}

func TestFSStoreIgnoresOneIncompleteTrailingLine(t *testing.T) {
	root := t.TempDir()
	store := NewFSStore(root)
	ctx := context.Background()
	if err := store.Append(ctx, Entry{SchemaVersion: 1, ID: 1, Family: FamilyGen, Raw: CommandView{Argv: []string{"st", "gen"}}, ExitCode: 0}); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(root, "history.jsonl"), os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.WriteString(`{"schema_version":1,"id":2`)
	_ = f.Close()
	got, err := store.Get(ctx, 1)
	if err != nil || got.ID != 1 {
		t.Fatalf("entry=%#v err=%v", got, err)
	}
}

func TestFSStoreRejectsMalformedInteriorLine(t *testing.T) {
	root := t.TempDir()
	data := "{\"schema_version\":1,\"id\":1,\"family\":\"gen\",\"raw\":{\"argv\":[\"st\",\"gen\"],\"display\":\"st gen\"},\"exit_code\":0,\"error\":null}\n{bad}\n{\"schema_version\":1,\"id\":2}\n"
	if err := os.WriteFile(filepath.Join(root, "history.jsonl"), []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := NewFSStore(root).Get(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "corrupt history state") {
		t.Fatalf("err = %v", err)
	}
}

func TestFSStoreConcurrentProcessReservationsAreUnique(t *testing.T) {
	root := t.TempDir()
	const workers = 12
	ids := make(chan int, workers)
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cmd := exec.Command(os.Args[0], "-test.run=^TestFSStoreReservationHelper$")
			cmd.Env = append(os.Environ(), "ST_HISTORY_HELPER=1", "ST_HISTORY_ROOT="+root)
			out, err := cmd.Output()
			if err != nil {
				errs <- err
				return
			}
			id, err := strconv.Atoi(strings.TrimSpace(string(out)))
			if err != nil {
				errs <- fmt.Errorf("parse helper id %q: %w", out, err)
				return
			}
			ids <- id
		}()
	}
	wg.Wait()
	close(ids)
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	got := make([]int, 0, workers)
	for id := range ids {
		got = append(got, id)
	}
	sort.Ints(got)
	for i, id := range got {
		if id != i+1 {
			t.Fatalf("ids = %v", got)
		}
	}
	entries, err := NewFSStore(root).readAll()
	if err != nil || len(entries) != workers {
		t.Fatalf("entries=%d err=%v", len(entries), err)
	}
}

func TestFSStoreReservationHelper(t *testing.T) {
	if os.Getenv("ST_HISTORY_HELPER") != "1" {
		return
	}
	store := NewFSStore(os.Getenv("ST_HISTORY_ROOT"))
	id, err := store.ReserveID(context.Background())
	if err != nil {
		panic(err)
	}
	entry := Entry{SchemaVersion: 1, ID: id, Family: FamilyUnknown, Raw: CommandView{Argv: []string{"st"}, Display: "st"}, ExitCode: 0}
	if err := store.Append(context.Background(), entry); err != nil {
		panic(err)
	}
	b, _ := json.Marshal(id)
	_, _ = os.Stdout.Write(b)
	os.Exit(0)
}
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./internal/history -run 'TestFSStoreReserveAppendAndLatest|TestPolicyStoreRoundTripsDefaultPolicy|TestResolveStateRootUsesXDGStateHome|TestFSStoreIgnoresOneIncompleteTrailingLine|TestFSStoreRejectsMalformedInteriorLine|TestFSStoreConcurrentProcessReservationsAreUnique' -count=1
```

Expected: FAIL with undefined `NewFSStore`, `Filter`, and `ResolveStateRoot`.

- [x] **Step 3: Implement store contracts and errors**

Create `cli/go/internal/history/store.go`:

```go
package history

import (
	"context"
	"errors"
)

var (
	ErrNotFound       = errors.New("history entry not found")
	ErrNoEligibleEntry = errors.New("no eligible history entry")
	ErrCorruptState   = errors.New("corrupt history state")
)

type Filter struct {
	Family           Family
	ExitCodes        []int
	RequireEffective bool
}

type HistoryStore interface {
	ReserveID(context.Context) (int64, error)
	Append(context.Context, Entry) error
	Get(context.Context, int64) (Entry, error)
	Latest(context.Context, Filter) (Entry, error)
}

type PolicyStore interface {
	LoadPolicy(context.Context) (Policy, error)
	SavePolicy(context.Context, Policy) error
}

type Store interface {
	HistoryStore
	PolicyStore
}
```

- [x] **Step 4: Implement the lock-scoped filesystem backend**

Create `cli/go/internal/history/fs.go`:

```go
package history

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

type FSStore struct {
	root string
}

func NewFSStore(root string) *FSStore { return &FSStore{root: root} }

func ResolveStateRoot() (string, error) {
	base := os.Getenv("XDG_STATE_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".local", "state")
	}
	root := filepath.Join(base, "st")
	if err := os.MkdirAll(root, 0o700); err != nil {
		return "", err
	}
	if err := os.Chmod(root, 0o700); err != nil {
		return "", err
	}
	return root, nil
}

func (s *FSStore) historyPath() string { return filepath.Join(s.root, "history.jsonl") }
func (s *FSStore) nextIDPath() string  { return filepath.Join(s.root, "next-id") }
func (s *FSStore) policyPath() string  { return filepath.Join(s.root, "conflate-policy.json") }
func (s *FSStore) lockPath() string    { return filepath.Join(s.root, "state.lock") }

func (s *FSStore) withLock(fn func() error) error {
	if err := os.MkdirAll(s.root, 0o700); err != nil {
		return fmt.Errorf("initialize state directory %s: %w", s.root, err)
	}
	if err := os.Chmod(s.root, 0o700); err != nil {
		return fmt.Errorf("protect state directory %s: %w", s.root, err)
	}
	f, err := os.OpenFile(s.lockPath(), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("open state lock %s: %w", s.lockPath(), err)
	}
	defer f.Close()
	if err := f.Chmod(0o600); err != nil {
		return fmt.Errorf("protect state lock %s: %w", s.lockPath(), err)
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		return fmt.Errorf("lock state %s: %w", s.lockPath(), err)
	}
	defer syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
	return fn()
}

func (s *FSStore) ReserveID(_ context.Context) (int64, error) {
	var reserved int64
	err := s.withLock(func() error {
		next := int64(1)
		data, err := os.ReadFile(s.nextIDPath())
		if err == nil && strings.TrimSpace(string(data)) != "" {
			next, err = strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
			if err != nil || next < 1 {
				return fmt.Errorf("%w: %s", ErrCorruptState, s.nextIDPath())
			}
		} else if err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("read %s: %w", s.nextIDPath(), err)
		}
		reserved = next
		return atomicReplace(s.nextIDPath(), []byte(strconv.FormatInt(next+1, 10)+"\n"))
	})
	return reserved, err
}

func (s *FSStore) Append(_ context.Context, entry Entry) error {
	b, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	return s.withLock(func() error {
		f, err := os.OpenFile(s.historyPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			return fmt.Errorf("open %s: %w", s.historyPath(), err)
		}
		defer f.Close()
		if err := f.Chmod(0o600); err != nil {
			return fmt.Errorf("protect %s: %w", s.historyPath(), err)
		}
		if _, err := f.Write(append(b, '\n')); err != nil {
			return fmt.Errorf("append %s: %w", s.historyPath(), err)
		}
		if err := f.Sync(); err != nil {
			return fmt.Errorf("fsync %s: %w", s.historyPath(), err)
		}
		return nil
	})
}

func (s *FSStore) Get(_ context.Context, id int64) (Entry, error) {
	entries, err := s.readAll()
	if err != nil {
		return Entry{}, err
	}
	for _, entry := range entries {
		if entry.ID == id {
			return entry, nil
		}
	}
	return Entry{}, ErrNotFound
}

func (s *FSStore) Latest(_ context.Context, filter Filter) (Entry, error) {
	entries, err := s.readAll()
	if err != nil {
		return Entry{}, err
	}
	var best *Entry
	for i := range entries {
		entry := entries[i]
		if filter.Family != "" && entry.Family != filter.Family {
			continue
		}
		if filter.RequireEffective && entry.Effective == nil {
			continue
		}
		if len(filter.ExitCodes) > 0 && !containsExit(filter.ExitCodes, entry.ExitCode) {
			continue
		}
		if best == nil || entry.ID > best.ID {
			best = &entry
		}
	}
	if best == nil {
		return Entry{}, ErrNoEligibleEntry
	}
	return *best, nil
}

func (s *FSStore) LoadPolicy(_ context.Context) (Policy, error) {
	data, err := os.ReadFile(s.policyPath())
	if os.IsNotExist(err) {
		return DefaultPolicy(), nil
	}
	if err != nil {
		return Policy{}, err
	}
	var policy Policy
	if err := json.Unmarshal(data, &policy); err != nil {
		return Policy{}, fmt.Errorf("%w: conflate-policy.json: %v", ErrCorruptState, err)
	}
	if err := ValidatePolicy(policy); err != nil {
		return Policy{}, fmt.Errorf("%w: %s: %v", ErrCorruptState, s.policyPath(), err)
	}
	return policy, nil
}

func (s *FSStore) SavePolicy(_ context.Context, policy Policy) error {
	b, err := json.MarshalIndent(policy, "", "  ")
	if err != nil {
		return err
	}
	return s.withLock(func() error {
		return atomicReplace(s.policyPath(), append(b, '\n'))
	})
}

func (s *FSStore) readAll() ([]Entry, error) {
	data, err := os.ReadFile(s.historyPath())
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []Entry
	lines := bytes.Split(data, []byte{'\n'})
	completeFinalLine := bytes.HasSuffix(data, []byte{'\n'})
	for i, raw := range lines {
		line := bytes.TrimSpace(raw)
		if len(line) == 0 {
			continue
		}
		var entry Entry
		if err := json.Unmarshal(line, &entry); err != nil {
			if i == len(lines)-1 && !completeFinalLine {
				break
			}
			return nil, fmt.Errorf("%w: history.jsonl: %v", ErrCorruptState, err)
		}
		if entry.SchemaVersion != 1 || entry.ID < 1 {
			return nil, fmt.Errorf("%w: history.jsonl entry has schema_version=%d id=%d", ErrCorruptState, entry.SchemaVersion, entry.ID)
		}
		out = append(out, entry)
	}
	return out, nil
}

func atomicReplace(path string, data []byte) error {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if err := f.Chmod(0o600); err != nil {
		f.Close()
		return err
	}
	if _, err := io.Copy(f, bytes.NewReader(data)); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

func containsExit(codes []int, code int) bool {
	for _, candidate := range codes {
		if candidate == code {
			return true
		}
	}
	return false
}
```

- [x] **Step 5: Run the tests and verify GREEN**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./internal/history -run 'TestFSStoreReserveAppendAndLatest|TestPolicyStoreRoundTripsDefaultPolicy|TestResolveStateRootUsesXDGStateHome|TestFSStoreIgnoresOneIncompleteTrailingLine|TestFSStoreRejectsMalformedInteriorLine|TestFSStoreConcurrentProcessReservationsAreUnique' -count=1
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
cd /Users/darkbit1001/workspace/Stability-Toys
git add cli/go/internal/history/store.go cli/go/internal/history/fs.go cli/go/internal/history/fs_test.go
git commit -m "feat: add st history filesystem store"
```

---

### Task 3: Add `st conflate` policy command and validation

**Files:**
- Create: `cli/go/cmd/st/conflate.go`
- Create: `cli/go/cmd/st/conflate_test.go`
- Modify: `cli/go/cmd/st/main.go`

- [x] **Step 1: Write failing conflate command, selector, and transactional-pin tests**

Create `cli/go/cmd/st/conflate_test.go`:

```go
package main

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

func TestConflateBareToggleEnablesDefaultSelector(t *testing.T) {
	root := t.TempDir()
	out := runCmdWithStateRoot(t, root, "conflate")
	if !strings.Contains(out, "Conflating recent successful gen runs.") {
		t.Fatalf("output = %q", out)
	}
	store := history.NewFSStore(root)
	policy, err := store.LoadPolicy(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !policy.Enabled || policy.Selector.Kind != history.SelectorRecent {
		t.Fatalf("policy = %#v", policy)
	}
}

func TestConflateOffRejectsSelectors(t *testing.T) {
	root := t.TempDir()
	_, err := runCmdMayFailWithStateRoot(t, root, "conflate", "off", "--with-exit", "1")
	if err == nil || !strings.Contains(err.Error(), "off") {
		t.Fatalf("err = %v", err)
	}
}

func TestConflateStatusDoesNotMutatePolicy(t *testing.T) {
	root := t.TempDir()
	runCmdWithStateRoot(t, root, "conflate", "on")
	before, _ := history.NewFSStore(root).LoadPolicy(context.Background())
	runCmdWithStateRoot(t, root, "conflate", "status")
	after, _ := history.NewFSStore(root).LoadPolicy(context.Background())
	if before.UpdatedAt != after.UpdatedAt || before.Enabled != after.Enabled {
		t.Fatalf("status mutated policy: before=%#v after=%#v", before, after)
	}
}

func TestConflatePinValidatesBeforeReplacingPolicy(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	ctx := context.Background()
	original := history.DefaultPolicy()
	original.Enabled = true
	original.UpdatedAt = "before"
	if err := store.SavePolicy(ctx, original); err != nil {
		t.Fatal(err)
	}
	_, err := runCmdMayFailWithStateRoot(t, root, "conflate", "history:99")
	if err == nil || !strings.Contains(err.Error(), "history:99") {
		t.Fatalf("err = %v", err)
	}
	after, err := store.LoadPolicy(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if after.UpdatedAt != original.UpdatedAt || after.Selector.Kind != original.Selector.Kind {
		t.Fatalf("failed pin changed policy: before=%#v after=%#v", original, after)
	}
}

func TestConflatePinRejectsNonGenAndMissingEffective(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	ctx := context.Background()
	for _, entry := range []history.Entry{
		{SchemaVersion: 1, ID: 1, Family: history.FamilyConflate, Raw: history.CommandView{Argv: []string{"st", "conflate"}}, ExitCode: 0},
		{SchemaVersion: 1, ID: 2, Family: history.FamilyGen, Raw: history.CommandView{Argv: []string{"st", "gen"}}, ExitCode: 1},
	} {
		if err := store.Append(ctx, entry); err != nil {
			t.Fatal(err)
		}
	}
	for _, id := range []string{"1", "2"} {
		_, err := runCmdMayFailWithStateRoot(t, root, "conflate", "history:"+id)
		if err == nil {
			t.Fatalf("history:%s unexpectedly accepted", id)
		}
	}
}

func TestConflateRejectsConflictingAndInvalidSelectors(t *testing.T) {
	root := t.TempDir()
	cases := [][]string{
		{"conflate", "history:5", "--with-exit", "1"},
		{"conflate", "status", "--inclusive", "gen"},
		{"conflate", "--inclusive", "replay"},
		{"conflate", "--with-exit", "-1"},
		{"conflate", "--with-exit", "256"},
	}
	for _, args := range cases {
		if _, err := runCmdMayFailWithStateRoot(t, root, args...); err == nil {
			t.Fatalf("%v unexpectedly accepted", args)
		}
	}
}

func TestConflateOnMaySetRecentSelector(t *testing.T) {
	root := t.TempDir()
	runCmdWithStateRoot(t, root, "conflate", "on", "--with-exit", "1", "--with-exit", "0")
	policy, err := history.NewFSStore(root).LoadPolicy(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !policy.Enabled || policy.Selector.Kind != history.SelectorRecent || fmt.Sprint(policy.Selector.ExitCodes) != "[0 1]" {
		t.Fatalf("policy = %#v", policy)
	}
}
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./cmd/st -run 'TestConflateBareToggleEnablesDefaultSelector|TestConflateOffRejectsSelectors|TestConflateStatusDoesNotMutatePolicy|TestConflatePinValidatesBeforeReplacingPolicy|TestConflatePinRejectsNonGenAndMissingEffective|TestConflateRejectsConflictingAndInvalidSelectors|TestConflateOnMaySetRecentSelector' -count=1
```

Expected: FAIL with undefined `runCmdWithStateRoot`, `runCmdMayFailWithStateRoot`, and missing `conflate` command.

- [x] **Step 3: Add injectable state-root resolution in `main.go`**

Modify `cli/go/cmd/st/main.go`:

```go
var (
	resolveStateRoot = history.ResolveStateRoot
	newHistoryStore  = func(root string) history.Store { return history.NewFSStore(root) }
)

func loadStateStore() (history.Store, error) {
	root, err := resolveStateRoot()
	if err != nil {
		return nil, err
	}
	return newHistoryStore(root), nil
}
```

- [x] **Step 4: Implement the conflate command**

Create `cli/go/cmd/st/conflate.go`:

```go
package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

var (
	conflateInclusive []string
	conflateExitCodes []int
)

var conflateCmd = &cobra.Command{
	Use:   "conflate [on|off|status|history:<id>]",
	Short: "Toggle or configure generation conflation policy",
	Args:  cobra.MaximumNArgs(1),
	RunE:  runConflate,
}

func init() {
	f := conflateCmd.Flags()
	f.StringArrayVar(&conflateInclusive, "inclusive", nil, "eligible family selector (v1: gen only)")
	f.IntArrayVar(&conflateExitCodes, "with-exit", nil, "eligible exit code (repeatable)")
	rootCmd.AddCommand(conflateCmd)
}

func runConflate(cmd *cobra.Command, args []string) error {
	store, err := loadStateStore()
	if err != nil {
		return err
	}
	ctx := cmd.Context()
	policy, err := store.LoadPolicy(ctx)
	if err != nil {
		return err
	}

	verb := ""
	if len(args) == 1 {
		verb = args[0]
	}
	hasSelectors := len(conflateInclusive) > 0 || len(conflateExitCodes) > 0
	if verb == "status" {
		if hasSelectors {
			return fmt.Errorf("conflate status does not accept selector arguments")
		}
		return writeConflateStatus(cmd, ctx, store, policy)
	}

	next, err := deriveConflatePolicy(ctx, store, policy, verb, conflateInclusive, conflateExitCodes)
	if err != nil {
		return err
	}
	next.UpdatedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if err := store.SavePolicy(ctx, next); err != nil {
		return err
	}
	return writeConflateStatus(cmd, ctx, store, next)
}

func deriveConflatePolicy(ctx context.Context, store history.HistoryStore, current history.Policy, verb string, inclusive []string, exits []int) (history.Policy, error) {
	next := current
	hasSelectors := len(inclusive) > 0 || len(exits) > 0
	if verb == "off" && hasSelectors {
		return history.Policy{}, fmt.Errorf("conflate off does not accept selector arguments")
	}
	if strings.HasPrefix(verb, "history:") && hasSelectors {
		return history.Policy{}, fmt.Errorf("history selector is mutually exclusive with --inclusive and --with-exit")
	}
	for _, family := range inclusive {
		if family != string(history.FamilyGen) {
			return history.Policy{}, fmt.Errorf("--inclusive accepts only gen in v1, got %q", family)
		}
	}
	for _, code := range exits {
		if code < 0 || code > 255 {
			return history.Policy{}, fmt.Errorf("--with-exit must be in 0..255, got %d", code)
		}
	}
	switch {
	case verb == "" && !hasSelectors:
		next.Enabled = !current.Enabled
	case verb == "on":
		next.Enabled = true
	case verb == "off":
		next.Enabled = false
	case strings.HasPrefix(verb, "history:"):
		id, err := strconv.ParseInt(strings.TrimPrefix(verb, "history:"), 10, 64)
		if err != nil || id < 1 {
			return history.Policy{}, fmt.Errorf("invalid history selector %q", verb)
		}
		entry, err := store.Get(ctx, id)
		if err != nil {
			return history.Policy{}, fmt.Errorf("history:%d: %w", id, err)
		}
		if entry.Family != history.FamilyGen || entry.Effective == nil || len(entry.Effective.Params) == 0 {
			return history.Policy{}, fmt.Errorf("history:%d is not an eligible gen baseline", id)
		}
		next.Enabled = true
		next.Selector = history.Selector{Kind: history.SelectorHistory, HistoryID: id}
	case verb == "" && hasSelectors:
		next.Enabled = true
	default:
		return history.Policy{}, fmt.Errorf("unknown conflate action %q", verb)
	}
	if hasSelectors {
		next.Enabled = true
		next.Selector = history.Selector{
			Kind:      history.SelectorRecent,
			Family:    history.FamilyGen,
			ExitCodes: history.NormalizeExitCodes(defaultExitCodes(exits)),
		}
	}
	return next, nil
}

func defaultExitCodes(exits []int) []int {
	if len(exits) == 0 {
		return []int{0}
	}
	return exits
}

func writeConflateStatus(cmd *cobra.Command, ctx context.Context, store history.HistoryStore, policy history.Policy) error {
	if !policy.Enabled {
		fmt.Fprintln(cmd.OutOrStdout(), "Conflation off.")
		return nil
	}
	if policy.Selector.Kind == history.SelectorHistory {
		entry, err := store.Get(ctx, policy.Selector.HistoryID)
		if err != nil {
			return err
		}
		fmt.Fprintln(cmd.OutOrStdout(), "Conflating only the selected history reference:")
		fmt.Fprintln(cmd.OutOrStdout(), entry.Effective.Display)
		return nil
	}
	if len(policy.Selector.ExitCodes) == 1 && policy.Selector.ExitCodes[0] == 0 {
		fmt.Fprintln(cmd.OutOrStdout(), "Conflating recent successful gen runs.")
		return nil
	}
	fmt.Fprintf(cmd.OutOrStdout(), "Conflating recent gen runs with exit code(s) %v.\n", policy.Selector.ExitCodes)
	return nil
}
```

- [x] **Step 5: Add state-root-aware test helpers and verify GREEN**

Modify the existing command helpers in `cli/go/cmd/st/gen_test.go` (they are defined there, not in `main_test.go`):

```go
func runCmdWithStateRoot(t *testing.T, stateRoot string, args ...string) string {
	t.Helper()
	resetCLIFlagState()
	old := resolveStateRoot
	resolveStateRoot = func() (string, error) { return stateRoot, nil }
	defer func() { resolveStateRoot = old }()
	return runCmd(t, args...)
}

func runCmdMayFailWithStateRoot(t *testing.T, stateRoot string, args ...string) (string, error) {
	t.Helper()
	resetCLIFlagState()
	old := resolveStateRoot
	resolveStateRoot = func() (string, error) { return stateRoot, nil }
	defer func() { resolveStateRoot = old }()
	return runCmdMayFail(t, args...)
}

func resetCLIFlagState() {
	flagServer = os.Getenv("ST_SERVER")
	flagConfig, flagOutputDir, genPrompt, genNegative, genSize = "", "", "", "", ""
	flagJSON, genStream, genQuiet = false, false, false
	flagTimeout = 0
	genSeed, genScheduler, genMode, genInitImage, genRecreate = "", "", "", "", ""
	genControlnetFile, genOutfile = "", ""
	genSteps, genSkipStep, genSR = 0, 0, 0
	genCfg, genControlStrength = 0, 0
	genControlnets, genControlImages = nil, nil
	conflateInclusive, conflateExitCodes = nil, nil
	var clearChanged func(*cobra.Command)
	clearChanged = func(cmd *cobra.Command) {
		cmd.Flags().VisitAll(func(flag *pflag.Flag) { flag.Changed = false })
		cmd.PersistentFlags().VisitAll(func(flag *pflag.Flag) { flag.Changed = false })
		for _, child := range cmd.Commands() { clearChanged(child) }
	}
	clearChanged(rootCmd)
}
```

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./cmd/st -run 'TestConflateBareToggleEnablesDefaultSelector|TestConflateOffRejectsSelectors|TestConflateStatusDoesNotMutatePolicy|TestConflatePinValidatesBeforeReplacingPolicy|TestConflatePinRejectsNonGenAndMissingEffective|TestConflateRejectsConflictingAndInvalidSelectors|TestConflateOnMaySetRecentSelector' -count=1
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
cd /Users/darkbit1001/workspace/Stability-Toys
git add cli/go/cmd/st/main.go cli/go/cmd/st/gen_test.go cli/go/cmd/st/conflate.go cli/go/cmd/st/conflate_test.go
git commit -m "feat: add st conflate policy command"
```

---

### Task 4: Add history-aware runtime and root shorthand parsing

**Files:**
- Create: `cli/go/cmd/st/history_runtime.go`
- Create: `cli/go/cmd/st/history_runtime_test.go`
- Modify: `cli/go/cmd/st/main.go`
- Modify: `cli/go/cmd/st/gen.go`

- [x] **Step 1: Write failing runtime tests**

Create `cli/go/cmd/st/history_runtime_test.go`:

```go
package main

import (
	"context"
	"strings"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

func TestRootShorthandRejectsPositionalPrompt(t *testing.T) {
	root := t.TempDir()
	runCmdWithStateRoot(t, root, "conflate", "on")
	_, err := runCmdMayFailWithStateRoot(t, root, "--cfg", "3", "owl")
	if err == nil || !strings.Contains(err.Error(), "--prompt") {
		t.Fatalf("err = %v", err)
	}
}

func TestParseFailureStillWritesHistoryEntry(t *testing.T) {
	root := t.TempDir()
	_, _ = runCmdMayFailWithStateRoot(t, root, "genrate", "--cfg", "3")
	entry, err := history.NewFSStore(root).Get(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if entry.Family != history.FamilyUnknown {
		t.Fatalf("family = %q, want unknown", entry.Family)
	}
	if entry.ExitCode == 0 {
		t.Fatalf("exit code = %d, want non-zero", entry.ExitCode)
	}
}

func TestRecentRootShorthandNeedsEligibleBaseline(t *testing.T) {
	root := t.TempDir()
	runCmdWithStateRoot(t, root, "conflate", "on")
	_, err := runCmdMayFailWithStateRoot(t, root, "--cfg", "4.3")
	if err == nil || !strings.Contains(err.Error(), "no eligible") {
		t.Fatalf("err = %v", err)
	}
}

func TestRootShorthandIsNotActiveWhenConflationIsOff(t *testing.T) {
	root := t.TempDir()
	_, err := runCmdMayFailWithStateRoot(t, root, "--cfg", "4.3")
	if err == nil || !strings.Contains(err.Error(), "unknown flag") {
		t.Fatalf("err = %v", err)
	}
}

func TestParseRootGenPatchConsumesFlagValues(t *testing.T) {
	for _, tc := range []struct {
		argv []string
		check func(genArgs) bool
	}{
		{argv: []string{"--cfg", "4.3"}, check: func(a genArgs) bool { return a.Cfg != nil && *a.Cfg == 4.3 }},
		{argv: []string{"--prompt", "two horses drinking"}, check: func(a genArgs) bool { return a.Prompt == "two horses drinking" }},
	} {
		patch, err := parseRootGenPatch(tc.argv)
		if err != nil {
			t.Fatalf("parse %v: %v", tc.argv, err)
		}
		if !patch.Active || !tc.check(patch.Args) {
			t.Fatalf("parse %v = %#v", tc.argv, patch)
		}
	}
}

func TestParseRootGenPatchRejectsOnlyTruePositionals(t *testing.T) {
	_, err := parseRootGenPatch([]string{"--cfg", "4.3", "owl"})
	if err == nil || !strings.Contains(err.Error(), "--prompt") {
		t.Fatalf("err = %v", err)
	}
}

func TestRootShorthandFlagSetMatchesGenCommand(t *testing.T) {
	var values genFlagValues
	shorthand := pflag.NewFlagSet("shorthand", pflag.ContinueOnError)
	bindGenFlags(shorthand, &values)
	want := map[string]bool{}
	genCmd.Flags().VisitAll(func(flag *pflag.Flag) {
		if flag.Name != "help" { want[flag.Name] = true }
	})
	got := map[string]bool{}
	shorthand.VisitAll(func(flag *pflag.Flag) { got[flag.Name] = true })
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("shorthand flags=%v gen flags=%v", got, want)
	}
}
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./cmd/st -run 'TestRootShorthandRejectsPositionalPrompt|TestParseFailureStillWritesHistoryEntry|TestRecentRootShorthandNeedsEligibleBaseline|TestRootShorthandIsNotActiveWhenConflationIsOff|TestParseRootGenPatchConsumesFlagValues|TestParseRootGenPatchRejectsOnlyTruePositionals|TestRootShorthandFlagSetMatchesGenCommand' -count=1
```

Expected: FAIL because the current runner bypasses history and does not recognize root shorthand.

- [x] **Step 3: Implement the history-aware runtime**

Create `cli/go/cmd/st/history_runtime.go`:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

type invocationPlan struct {
	family          history.Family
	kind            invocationKind
	rawArgv         []string
	rawDisplay      string
	rootGenPatch    *genPatch
	replayHistoryID int64
}

type invocationState struct {
	store history.Store
	id    int64
	final *invocationResult
}

type invocationResult struct {
	params                 map[string]any
	derivedFromHistoryID   *int64
	replayedFromHistoryID  *int64
	policySnapshot         *history.PolicySnapshot
}

type exitError struct {
	code int
	err  error
}

func (e exitError) Error() string { return e.err.Error() }
func (e exitError) Unwrap() error { return e.err }

func exitCodeOf(err error) int {
	if err == nil { return 0 }
	var coded exitError
	if errors.As(err, &coded) { return coded.code }
	return 1
}

type invocationStateKey struct{}

func withInvocationState(ctx context.Context, state *invocationState) context.Context {
	return context.WithValue(ctx, invocationStateKey{}, state)
}

func stateFromContext(ctx context.Context) (*invocationState, error) {
	state, ok := ctx.Value(invocationStateKey{}).(*invocationState)
	if !ok || state == nil {
		return nil, fmt.Errorf("missing invocation state")
	}
	return state, nil
}

type invocationKind uint8

const (
	invocationCobra invocationKind = iota
	invocationRootGen
	invocationReplay
)

func executeCLI(ctx context.Context, argv []string) error {
	store, err := loadStateStore()
	if err != nil {
		return err
	}
	id, err := store.ReserveID(ctx)
	if err != nil {
		return err
	}
	started := time.Now().UTC()
	state := &invocationState{store: store, id: id}
	plan, err := buildInvocationPlan(ctx, store, argv)
	if err == nil {
		err = dispatchInvocation(withInvocationState(ctx, state), state, plan, argv)
	}
	if appendErr := appendHistory(ctx, state, started, plan, err); appendErr != nil {
		if err != nil {
			return fmt.Errorf("%v; append history: %w", err, appendErr)
		}
		return fmt.Errorf("append history: %w", appendErr)
	}
	return err
}

func buildInvocationPlan(ctx context.Context, store history.Store, argv []string) (invocationPlan, error) {
	plan := invocationPlan{
		family:     classifyFamily(argv),
		rawArgv:    append([]string{"st"}, argv...),
		rawDisplay: history.RenderArgv(append([]string{"st"}, argv...)),
	}
	if len(argv) > 0 && strings.HasPrefix(argv[0], "-") && firstCommandToken(argv) == "" {
		policy, err := store.LoadPolicy(ctx)
		if err != nil {
			return plan, err
		}
		if policy.Enabled {
			patch, err := parseRootGenPatch(argv)
			if err != nil {
				return plan, err
			}
			if patch.Active {
				plan.kind = invocationRootGen
				plan.family = history.FamilyGen
				plan.rootGenPatch = &patch
			}
		}
	}
	if firstCommandToken(argv) == "replay" {
		plan.kind = invocationReplay
		plan.family = history.FamilyGen
	}
	return plan, nil
}

func classifyFamily(argv []string) history.Family {
	command := firstCommandToken(argv)
	if command == "" && len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		command = argv[0]
	}
	if command == "" {
		return history.FamilyUnknown
	}
	switch command {
	case "gen":
		return history.FamilyGen
	case "conflate":
		return history.FamilyConflate
	case "replay":
		// Replay is a dispatch kind, but its persisted semantic family is gen.
		return history.FamilyGen
	default:
		if isKnownTopLevelCommand(command) {
			return history.Family(command)
		}
		return history.FamilyUnknown
	}
}

func dispatchInvocation(ctx context.Context, state *invocationState, plan invocationPlan, argv []string) error {
	if plan.kind == invocationRootGen {
		return fmt.Errorf("%w: run a full st gen command or pin history:<id>", history.ErrNoEligibleEntry)
	}
	rootCmd.SetArgs(argv)
	return rootCmd.ExecuteContext(ctx)
}

func appendHistory(ctx context.Context, state *invocationState, started time.Time, plan invocationPlan, runErr error) error {
	exitCode := exitCodeOf(runErr)
	var summary *string
	if runErr != nil {
		text := runErr.Error()
		summary = &text
	}
	entry := history.Entry{
		SchemaVersion: 1,
		ID: state.id,
		StartedAt: started.Format(time.RFC3339Nano),
		FinishedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Family: plan.family,
		Raw: history.CommandView{Argv: plan.rawArgv, Display: plan.rawDisplay},
		ExitCode: exitCode,
		Error: summary,
	}
	if state.final != nil {
		entry.Family = history.FamilyGen
		entry.Effective = &history.CommandView{
			Argv: history.CanonicalGenArgv(state.final.params),
			Display: history.CanonicalGenDisplay(state.final.params),
			Params: history.CloneParams(state.final.params),
		}
		entry.DerivedFromHistoryID = state.final.derivedFromHistoryID
		entry.ReplayedFromHistoryID = state.final.replayedFromHistoryID
		entry.ConflatePolicy = state.final.policySnapshot
	}
	return state.store.Append(ctx, entry)
}

```

- [x] **Step 4: Route `main()` and test helpers through the new runner**

Modify `cli/go/cmd/st/main.go`:

```go
func main() {
	if err := executeCLI(context.Background(), os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(exitCodeOf(err))
	}
}
```

Also replace the direct `os.Exit(2)` in `requireConfig` with `return nil, exitError{code: 2, err: fmt.Errorf("%s", message)}`. No subcommand may terminate the process; the top-level wrapper must append history before `main` converts the returned error to an OS exit code.

Modify the test harness in `cli/go/cmd/st/gen_test.go`, where `runCmd` and `runCmdMayFail` are currently defined:

```go
func runCmd(t *testing.T, args ...string) string {
	t.Helper()
	out, err := runCmdMayFailWithStateRoot(t, t.TempDir(), args...)
	if err != nil {
		t.Fatalf("execute %v: %v\noutput: %s", args, err, out)
	}
	return out
}

func runCmdMayFail(t *testing.T, args ...string) (string, error) {
	t.Helper()
	return runCmdMayFailWithStateRoot(t, t.TempDir(), args...)
}

func runCmdWithStateRoot(t *testing.T, stateRoot string, args ...string) string {
	t.Helper()
	out, err := runCmdMayFailWithStateRoot(t, stateRoot, args...)
	if err != nil {
		t.Fatalf("execute %v: %v\noutput: %s", args, err, out)
	}
	return out
}

func runCmdMayFailWithStateRoot(t *testing.T, stateRoot string, args ...string) (string, error) {
	t.Helper()
	resetCLIFlagState()
	old := resolveStateRoot
	resolveStateRoot = func() (string, error) { return stateRoot, nil }
	defer func() { resolveStateRoot = old }()
	var sb strings.Builder
	rootCmd.SetOut(&sb)
	rootCmd.SetErr(&sb)
	rootCmd.SetArgs(args)
	err := executeCLI(context.Background(), args)
	return sb.String(), err
}
```

- [x] **Step 5: Add root shorthand validation and verify GREEN**

Modify `cli/go/cmd/st/gen.go` to bind gen flags through one helper and parse shorthand with a real pflag `FlagSet`. The parser must consume option values before deciding whether anything is positional; it must also require at least one changed gen flag so root-only flags such as `--json` do not trigger a generation:

```go
type genPatch struct {
	Active  bool
	Args    genArgs
	Changed map[string]bool
}

func parseRootGenPatch(argv []string) (genPatch, error) {
	if len(argv) == 0 || !strings.HasPrefix(argv[0], "-") {
		return genPatch{}, nil
	}
	if firstCommandToken(argv) != "" {
		return genPatch{}, nil
	}
	var values genFlagValues
	rootValues := currentRootFlagValues()
	f := pflag.NewFlagSet("st gen shorthand", pflag.ContinueOnError)
	f.SetInterspersed(true)
	bindGenFlags(f, &values)
	bindRootPersistentFlags(f, &rootValues)
	if err := f.Parse(argv); err != nil {
		return genPatch{}, err
	}
	changed := changedGenFlags(f)
	if len(changed) == 0 {
		return genPatch{}, nil
	}
	if f.NArg() != 0 {
		if isKnownTopLevelCommand(f.Arg(0)) {
			return genPatch{}, nil
		}
		return genPatch{}, fmt.Errorf("root shorthand is flag-only; pass positional text with --prompt or use explicit st gen")
	}
	applyRootFlagValues(rootValues)
	applyGenExecutionValues(values)
	return genPatch{
		Active:  true,
		Args:    genArgsFromFlagSet(f, values, nil),
		Changed: changed,
	}, nil
}

func firstCommandToken(argv []string) string {
	values := currentRootFlagValues()
	f := pflag.NewFlagSet("st root", pflag.ContinueOnError)
	f.SetInterspersed(false)
	bindRootPersistentFlags(f, &values)
	if err := f.Parse(argv); err != nil || f.NArg() == 0 {
		return ""
	}
	return f.Arg(0)
}

func isKnownTopLevelCommand(name string) bool {
	for _, cmd := range rootCmd.Commands() {
		if cmd.Name() == name {
			return true
		}
	}
	return false
}

type genFlagValues struct {
	Prompt, Negative, Size, Seed, Scheduler, Mode string
	InitImage, Recreate, ControlnetFile, Outfile string
	Steps, SkipStep, SR int
	Cfg, ControlStrength float64
	Controlnets, ControlImages []string
	Stream, Quiet bool
}

func bindGenFlags(f *pflag.FlagSet, v *genFlagValues) {
	f.StringVar(&v.Prompt, "prompt", "", "prompt text")
	f.StringVar(&v.Negative, "negative", "", "negative prompt")
	f.StringVar(&v.Size, "size", "", "image size")
	f.IntVar(&v.Steps, "steps", 0, "inference steps")
	f.IntVar(&v.SkipStep, "skip-step", 0, "timesteps to skip")
	f.Float64Var(&v.Cfg, "cfg", 0, "guidance scale")
	f.StringVar(&v.Seed, "seed", "", `seed integer or "random"`)
	f.StringVar(&v.Scheduler, "scheduler", "", "scheduler id")
	f.StringVar(&v.Mode, "mode", "", "model mode")
	f.IntVar(&v.SR, "sr", 0, "super-resolution magnitude")
	f.StringVar(&v.InitImage, "init-image", "", "img2img source")
	f.StringVar(&v.Recreate, "recreate", "", "PNG recipe source")
	f.StringArrayVar(&v.Controlnets, "controlnet", nil, "ControlNet attachment JSON")
	f.StringVar(&v.ControlnetFile, "controlnet-file", "", "ControlNet JSON file")
	f.StringArrayVar(&v.ControlImages, "control-image", nil, "control image type:path")
	f.Float64Var(&v.ControlStrength, "control-strength", 0, "ControlNet strength")
	f.StringVar(&v.Outfile, "outfile", "", "explicit output path")
	f.BoolVar(&v.Stream, "stream", false, "stream NDJSON")
	f.BoolVar(&v.Quiet, "quiet", false, "suppress diagnostics")
}

func genArgsFromFlagSet(f *pflag.FlagSet, v genFlagValues, args []string) genArgs {
	a := genArgs{
		InitImage: v.InitImage, Recreate: v.Recreate,
		Controlnets: v.Controlnets, ControlnetFile: v.ControlnetFile,
		ControlImages: v.ControlImages, Outfile: v.Outfile,
	}
	if len(args) > 0 { a.Prompt = strings.Join(args, " ") }
	if f.Changed("prompt") { a.Prompt = v.Prompt }
	if f.Changed("negative") { a.Negative = &v.Negative }
	if f.Changed("size") { a.Genres = &v.Size }
	if f.Changed("steps") { a.Steps = &v.Steps }
	if f.Changed("skip-step") { a.SkipStep = &v.SkipStep }
	if f.Changed("cfg") { a.Cfg = &v.Cfg }
	if f.Changed("seed") { a.Seed = &v.Seed }
	if f.Changed("scheduler") { a.Scheduler = &v.Scheduler }
	if f.Changed("mode") { a.Mode = &v.Mode }
	if f.Changed("sr") { a.SR = &v.SR }
	if f.Changed("control-strength") { a.ControlStrength = &v.ControlStrength }
	return a
}

func changedGenFlags(f *pflag.FlagSet) map[string]bool {
	changed := map[string]bool{}
	for _, name := range []string{
		"prompt", "negative", "size", "steps", "skip-step", "cfg", "seed",
		"scheduler", "mode", "sr", "init-image", "recreate", "controlnet",
		"controlnet-file", "control-image", "control-strength", "outfile", "stream", "quiet",
	} {
		if f.Changed(name) { changed[name] = true }
	}
	return changed
}

type rootFlagValues struct {
	Server, Config, OutputDir string
	JSON bool
	Timeout time.Duration
}

func currentRootFlagValues() rootFlagValues {
	return rootFlagValues{Server: flagServer, Config: flagConfig, OutputDir: flagOutputDir, JSON: flagJSON, Timeout: flagTimeout}
}

func bindRootPersistentFlags(f *pflag.FlagSet, v *rootFlagValues) {
	f.StringVar(&v.Server, "server", v.Server, "backend base URL")
	f.StringVar(&v.Config, "config", v.Config, "config file path")
	f.StringVarP(&v.OutputDir, "output-dir", "o", v.OutputDir, "output directory")
	f.BoolVar(&v.JSON, "json", v.JSON, "machine-readable JSON")
	f.DurationVar(&v.Timeout, "timeout", v.Timeout, "per-request timeout")
}

func applyRootFlagValues(v rootFlagValues) {
	flagServer, flagConfig, flagOutputDir = v.Server, v.Config, v.OutputDir
	flagJSON, flagTimeout = v.JSON, v.Timeout
}

func applyGenExecutionValues(v genFlagValues) {
	genOutfile, genStream, genQuiet = v.Outfile, v.Stream, v.Quiet
}

func bindGenExecutionFlags(f *pflag.FlagSet) {
	f.StringVar(&genOutfile, "outfile", "", "explicit output path")
	f.BoolVar(&genStream, "stream", false, "stream progress as NDJSON")
	f.BoolVar(&genQuiet, "quiet", false, "suppress diagnostics")
}
```

Keep `bindGenFlags` synchronized with `genCmd` and add a test comparing their non-help flag-name sets. `bindRootPersistentFlags` binds `--server`, `--config`, `--output-dir`, `--json`, and `--timeout` to a value struct; shorthand applies that struct only after a successful parse. `--stream` and `--quiet` remain gen flags. Do not inspect option-value tokens by prefix after this parse. The `firstCommandToken` helper lets `--server URL replay 5` retain replay dispatch and prevents `--server URL gen --cfg 3` from being misclassified as shorthand.

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./cmd/st -run 'TestRootShorthandRejectsPositionalPrompt|TestParseFailureStillWritesHistoryEntry|TestRecentRootShorthandNeedsEligibleBaseline|TestRootShorthandIsNotActiveWhenConflationIsOff|TestParseRootGenPatchConsumesFlagValues|TestParseRootGenPatchRejectsOnlyTruePositionals|TestRootShorthandFlagSetMatchesGenCommand' -count=1
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
cd /Users/darkbit1001/workspace/Stability-Toys
git add cli/go/cmd/st/main.go cli/go/cmd/st/gen.go cli/go/cmd/st/history_runtime.go cli/go/cmd/st/history_runtime_test.go cli/go/cmd/st/gen_test.go
git commit -m "feat: add st history-aware runtime"
```

---

### Task 5: Thread conflation baselines and replay through the gen executor

**Files:**
- Create: `cli/go/cmd/st/replay.go`
- Create: `cli/go/cmd/st/replay_test.go`
- Modify: `cli/go/cmd/st/gen.go`
- Modify: `cli/go/cmd/st/gen_test.go`
- Modify: `cli/go/cmd/st/history_runtime.go`
- Modify: `cli/go/internal/config/precedence.go`
- Modify: `cli/go/internal/config/precedence_test.go`

- [x] **Step 1: Write failing conflation and replay tests**

Create `cli/go/cmd/st/replay_test.go`:

```go
package main

import (
	"context"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

func TestReplayUsesEffectiveParamsExactly(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	sourceID, _ := store.ReserveID(context.Background())
	_ = store.Append(context.Background(), history.Entry{
		SchemaVersion: 1,
		ID:            sourceID,
		Family:        history.FamilyGen,
		Raw:           history.CommandView{Argv: []string{"st", "gen"}, Display: "st gen"},
		Effective: &history.CommandView{
			Argv:    []string{"st", "gen", "--prompt", "owl", "--cfg", "4.5"},
			Display: "st gen --prompt owl --cfg 4.5",
			Params:  map[string]any{"prompt": "owl", "guidance_scale": 4.5, "seed": 421337},
		},
		ExitCode: 1,
	})

	params, entry, err := loadReplayParams(context.Background(), store, sourceID)
	if err != nil {
		t.Fatal(err)
	}
	if entry.ID != sourceID || params["seed"] != 421337 {
		t.Fatalf("entry=%#v params=%#v", entry, params)
	}
}

func TestReplayRejectsGenerationOverrides(t *testing.T) {
	root := t.TempDir()
	for _, args := range [][]string{
		{"replay", "1", "--cfg", "3"},
		{"replay", "1", "replacement prompt"},
	} {
		_, err := runCmdMayFailWithStateRoot(t, root, args...)
		if err == nil {
			t.Fatalf("%v unexpectedly accepted", args)
		}
	}
}
```

Append to `cli/go/cmd/st/gen_test.go`:

```go
func TestConflatedGenUsesBaselineEffectiveParams(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	baseID, _ := store.ReserveID(context.Background())
	_ = store.Append(context.Background(), history.Entry{
		SchemaVersion: 1,
		ID:            baseID,
		Family:        history.FamilyGen,
		Raw:           history.CommandView{Argv: []string{"st", "gen"}, Display: "st gen"},
		Effective: &history.CommandView{
			Params: map[string]any{"prompt": "horse bartender", "guidance_scale": 4.5, "size": "1024x1024", "seed": 421337},
		},
		ExitCode: 0,
	})

	runCmdWithStateRoot(t, root, "conflate", "history:1")
	patch, err := parseRootGenPatch([]string{"--prompt", "two horses drinking"})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{}
	cfg.Defaults.Generation.Cfg = 7.5
	params, baseline, _, err := buildConflatedParams(context.Background(), store, patch, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if baseline.ID != 1 || params["guidance_scale"] != 4.5 || params["size"] != "1024x1024" || params["seed"] != 421337 || params["prompt"] != "two horses drinking" {
		t.Fatalf("baseline=%#v params=%#v", baseline, params)
	}
}

func TestConflatedGenExplicitZeroAndRandomOverrideBaseline(t *testing.T) {
	baseline := stclient.GenParams{
		"prompt": "owl", "guidance_scale": 4.5, "skip_step": 4,
		"superres": true, "superres_magnitude": 2, "seed": 421337,
	}
	patch := genPatch{
		Active: true,
		Args: genArgs{Cfg: f64p(0), SkipStep: intp(0), SR: intp(0), Seed: strp("random")},
		Changed: map[string]bool{"cfg": true, "skip-step": true, "sr": true, "seed": true},
	}
	got, err := buildGenParamsWithBaseline(nil, patch.Args, baseline, patch.Changed)
	if err != nil {
		t.Fatal(err)
	}
	if got["guidance_scale"] != float64(0) {
		t.Fatalf("cfg = %#v", got["guidance_scale"])
	}
	for _, key := range []string{"skip_step", "superres", "superres_magnitude", "seed"} {
		if _, ok := got[key]; ok {
			t.Fatalf("%s unexpectedly inherited: %#v", key, got)
		}
	}
}

func TestExplicitGenWithoutRecentBaselineFallsBackToNormalResolution(t *testing.T) {
	store := history.NewFSStore(t.TempDir())
	policy := history.DefaultPolicy()
	policy.Enabled = true
	if err := store.SavePolicy(context.Background(), policy); err != nil {
		t.Fatal(err)
	}
	patch := genPatch{Args: genArgs{Prompt: "first run"}}
	got, baseline, _, err := buildConflatedParams(context.Background(), store, patch, &config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if baseline != nil || got["prompt"] != "first run" {
		t.Fatalf("baseline=%#v params=%#v", baseline, got)
	}
}
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./cmd/st -run 'TestReplayUsesEffectiveParamsExactly|TestReplayRejectsGenerationOverrides|TestConflatedGenUsesBaselineEffectiveParams|TestConflatedGenExplicitZeroAndRandomOverrideBaseline|TestExplicitGenWithoutRecentBaselineFallsBackToNormalResolution' -count=1
```

Expected: FAIL with undefined `loadReplayParams`, `buildConflatedParams`, and `buildGenParamsWithBaseline`.

- [x] **Step 3: Add the four-layer config resolver**

Refactor `cli/go/internal/config/precedence.go` so ordinary generation delegates to a baseline-aware resolver. Baseline is copied only between baked params and explicit flags; the explicit application helper must delete inherited values for `--seed random`, `--skip-step 0`, and `--sr 0`:

```go
func ResolveParams(cfg *Config, baked map[string]any, f Flags) map[string]any {
	return ResolveParamsWithBaseline(cfg, baked, nil, f)
}

func resolveDefaults(cfg *Config) map[string]any {
	if cfg == nil {
		cfg = &Config{}
	}
	p := map[string]any{}
	g := cfg.Defaults.Generation
	setStr(p, "size", g.Genres)
	if g.Cfg != 0 {
		p["guidance_scale"] = g.Cfg
	}
	if g.Steps != 0 {
		p["num_inference_steps"] = g.Steps
	}
	if g.SkipStep != 0 {
		p["skip_step"] = g.SkipStep
	}
	applySeed(p, g.Seed)
	setStr(p, "mode", g.Mode)
	return p
}

func ResolveParamsWithBaseline(cfg *Config, baked, baseline map[string]any, f Flags) map[string]any {
	p := resolveDefaults(cfg)
	for k, v := range baked {
		p[k] = v
	}
	for k, v := range baseline {
		p[k] = v
	}
	applyExplicitFlags(p, f)
	return p
}

func applyExplicitFlags(p map[string]any, f Flags) {
	if f.Prompt != "" {
		p["prompt"] = f.Prompt
	}
	if f.Genres != nil {
		p["size"] = *f.Genres
	}
	if f.Steps != nil {
		p["num_inference_steps"] = *f.Steps
	}
	if f.SkipStep != nil {
		if *f.SkipStep > 0 {
			p["skip_step"] = *f.SkipStep
		} else {
			delete(p, "skip_step")
		}
	}
	if f.Cfg != nil {
		p["guidance_scale"] = *f.Cfg
	}
	if f.Negative != nil {
		p["negative_prompt"] = *f.Negative
	}
	if f.Scheduler != nil {
		p["scheduler_id"] = *f.Scheduler
	}
	if f.Mode != nil {
		p["mode"] = *f.Mode
	}
	if f.SRLevel != nil {
		delete(p, "superres")
		delete(p, "superres_magnitude")
		if *f.SRLevel > 0 {
			p["superres"] = true
			p["superres_magnitude"] = clamp(*f.SRLevel, 1, 3)
		}
	}
	if f.Seed != nil {
		applySeed(p, *f.Seed)
	}
}
```

Add to `cli/go/internal/config/precedence_test.go`:

```go
func TestResolveParamsWithBaselinePrecedence(t *testing.T) {
	cfg := &Config{}
	cfg.Defaults.Generation = Generation{Cfg: 2, Steps: 5, Genres: "512x512"}
	baked := map[string]any{"guidance_scale": 3.0, "num_inference_steps": 10, "size": "768x768"}
	baseline := map[string]any{"guidance_scale": 4.5, "num_inference_steps": 20, "size": "1024x1024"}
	got := ResolveParamsWithBaseline(cfg, baked, baseline, Flags{Steps: intp(30)})
	if got["guidance_scale"] != 4.5 || got["size"] != "1024x1024" || got["num_inference_steps"] != 30 {
		t.Fatalf("params = %#v", got)
	}
}

func TestResolveParamsWithBaselineExplicitZeroClearsInheritedFields(t *testing.T) {
	zero := 0
	zeroFloat := 0.0
	baseline := map[string]any{
		"guidance_scale": 4.5,
		"skip_step": 4,
		"superres": true,
		"superres_magnitude": 2,
	}
	got := ResolveParamsWithBaseline(&Config{}, nil, baseline, Flags{Cfg: &zeroFloat, SkipStep: &zero, SRLevel: &zero})
	if got["guidance_scale"] != float64(0) {
		t.Fatalf("guidance_scale = %#v", got["guidance_scale"])
	}
	for _, key := range []string{"skip_step", "superres", "superres_magnitude"} {
		if _, ok := got[key]; ok {
			t.Fatalf("%s unexpectedly present in %#v", key, got)
		}
	}
}

func TestResolveParamsWithBaselineRandomClearsInheritedSeed(t *testing.T) {
	random := "random"
	got := ResolveParamsWithBaseline(&Config{}, nil, map[string]any{"seed": 421337}, Flags{Seed: &random})
	if _, ok := got["seed"]; ok {
		t.Fatalf("seed unexpectedly present in %#v", got)
	}
}
```

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./internal/config -run 'TestResolveParamsWithBaseline' -count=1
```

Expected: PASS.

- [x] **Step 4: Factor common generation execution and baseline merge helpers**

Modify `cli/go/cmd/st/gen.go`:

```go
func buildConflatedParams(ctx context.Context, store history.Store, patch genPatch, cfg *config.Config) (stclient.GenParams, *history.Entry, *history.PolicySnapshot, error) {
	policy, err := store.LoadPolicy(ctx)
	if err != nil {
		return nil, nil, nil, err
	}
	if !policy.Enabled {
		params, err := buildGenParamsWithBaseline(cfg, patch.Args, nil, patch.Changed)
		return params, nil, nil, err
	}
	baseline, err := selectBaseline(ctx, store, policy.Selector)
	if err != nil {
		if errors.Is(err, history.ErrNoEligibleEntry) && !patch.Active {
			params, buildErr := buildGenParamsWithBaseline(cfg, patch.Args, nil, patch.Changed)
			return params, nil, nil, buildErr
		}
		return nil, nil, nil, err
	}
	params, err := buildGenParamsWithBaseline(cfg, patch.Args, baseline.Effective.Params, patch.Changed)
	if err != nil {
		return nil, nil, nil, err
	}
	return params, &baseline, history.SnapshotSelector(policy.Selector), nil
}

func selectBaseline(ctx context.Context, store history.HistoryStore, selector history.Selector) (history.Entry, error) {
	if selector.Kind == history.SelectorHistory {
		entry, err := store.Get(ctx, selector.HistoryID)
		if err != nil {
			return history.Entry{}, err
		}
		if entry.Family != history.FamilyGen || entry.Effective == nil || len(entry.Effective.Params) == 0 {
			return history.Entry{}, fmt.Errorf("history:%d is not an eligible gen baseline", selector.HistoryID)
		}
		return entry, nil
	}
	return store.Latest(ctx, history.Filter{
		Family: history.FamilyGen,
		ExitCodes: selector.ExitCodes,
		RequireEffective: true,
	})
}

func buildGenParamsWithBaseline(cfg *config.Config, a genArgs, baseline stclient.GenParams, changed map[string]bool) (stclient.GenParams, error) {
	baked, err := loadCurrentBakedParams(a)
	if err != nil {
		return nil, err
	}
	p := stclient.GenParams(config.ResolveParamsWithBaseline(cfg, baked, baseline, a.toFlags()))

	if changed["init-image"] {
		delete(p, "init_image_ref")
		if ref, ok := strings.CutPrefix(a.InitImage, "fileref:"); ok && ref != "" {
			p["init_image_ref"] = ref
		}
	}
	if changed["controlnet"] || changed["controlnet-file"] || changed["control-image"] {
		delete(p, "controlnets")
		if err := applyCurrentControlnetInputs(cfg, a, p); err != nil {
			return nil, err
		}
	}
	return p, nil
}

func loadCurrentBakedParams(a genArgs) (map[string]any, error) {
	local, required := localRecipePath(a)
	if local == "" {
		return nil, nil
	}
	data, err := os.ReadFile(local)
	if err == nil {
		var baked map[string]any
		baked, err = pngmeta.BakedParams(data)
		if err == nil {
			return baked, nil
		}
	}
	if required {
		return nil, err
	}
	return nil, nil
}

func applyCurrentControlnetInputs(cfg *config.Config, a genArgs, p stclient.GenParams) error {
	cns := make([]any, 0, len(a.Controlnets)+1)
	for _, raw := range a.Controlnets {
		if presetName, ok := strings.CutPrefix(raw, "@"); ok {
			var preset config.ControlnetPreset
			if cfg != nil {
				preset = cfg.ControlnetPresets[presetName]
			}
			if preset == nil {
				return fmt.Errorf("--controlnet @%s: preset not found in config", presetName)
			}
			cns = append(cns, map[string]any(preset))
			continue
		}
		var cn map[string]any
		if err := json.Unmarshal([]byte(raw), &cn); err != nil {
			return fmt.Errorf("--controlnet %q: %w", raw, err)
		}
		cns = append(cns, cn)
	}
	if a.ControlnetFile != "" {
		data, err := os.ReadFile(a.ControlnetFile)
		if err != nil {
			return fmt.Errorf("--controlnet-file %q: %w", a.ControlnetFile, err)
		}
		var cn map[string]any
		if err := json.Unmarshal(data, &cn); err != nil {
			return fmt.Errorf("--controlnet-file %q: invalid JSON: %w", a.ControlnetFile, err)
		}
		cns = append(cns, cn)
	}
	if len(cns) > 0 {
		p["controlnets"] = cns
	}
	return nil
}

func inferChangedInputs(a genArgs) map[string]bool {
	return map[string]bool{
		"init-image":     a.InitImage != "",
		"controlnet":     len(a.Controlnets) > 0,
		"controlnet-file": a.ControlnetFile != "",
		"control-image":  len(a.ControlImages) > 0,
	}
}

func loadReplayParams(ctx context.Context, store history.HistoryStore, id int64) (stclient.GenParams, history.Entry, error) {
	entry, err := store.Get(ctx, id)
	if err != nil {
		return nil, history.Entry{}, err
	}
	if entry.Family != history.FamilyGen || entry.Effective == nil || len(entry.Effective.Params) == 0 {
		return nil, history.Entry{}, fmt.Errorf("history:%d is not replayable", id)
	}
	return stclient.GenParams(history.CloneParams(entry.Effective.Params)), entry, nil
}
```

Keep `buildGenParams(cfg, a)` as the compatibility wrapper for existing tests and non-conflated callers; it calls `buildGenParamsWithBaseline(cfg, a, nil, inferChangedInputs(a))`. The command path must pass the real `Changed` map from `genArgsFromFlagSet`, so config defaults never masquerade as explicit current values. `resolveControlImages` runs after this merge and appends only the current invocation's newly uploaded attachments.

- [x] **Step 5: Implement `st replay` and runtime dispatch**

Create `cli/go/cmd/st/replay.go`:

```go
package main

import (
	"fmt"
	"strconv"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

var replayCmd = &cobra.Command{
	Use:   "replay <id>",
	Short: "Replay one history-backed generation exactly",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		id, err := strconv.ParseInt(args[0], 10, 64)
		if err != nil || id < 1 {
			return fmt.Errorf("replay id must be a positive integer, got %q", args[0])
		}
		state, err := stateFromContext(cmd.Context())
		if err != nil {
			return err
		}
		params, source, err := loadReplayParams(cmd.Context(), state.store, id)
		if err != nil {
			return err
		}
		if !genQuiet {
			fmt.Fprintf(cmd.ErrOrStderr(), "replaying [id=%d]: %s\n", id, source.Effective.Display)
		}
		cfg, err := requireConfig()
		if err != nil {
			return err
		}
		return executeResolvedGen(cmd, cfg, genArgs{Outfile: genOutfile}, params, func(final stclient.GenParams) {
			state.final = &invocationResult{
				params: final,
				replayedFromHistoryID: &source.ID,
			}
		})
	},
}

func init() {
	bindGenExecutionFlags(replayCmd.Flags())
	rootCmd.AddCommand(replayCmd)
}
```

`bindGenExecutionFlags` adds only `--outfile`, `--stream`, and `--quiet`; replay therefore accepts current output controls but rejects `--cfg`, `--prompt`, and all other generation overrides at Cobra parsing. Refactor `runGen` into `executeResolvedGen` for transport, mode switch, fetch, metadata, and output writing; replay passes its copied params directly and never calls the four-layer resolver.

Modify `cli/go/cmd/st/history_runtime.go` to route replay and shorthand into the common executor:

```go
func dispatchInvocation(ctx context.Context, state *invocationState, plan invocationPlan, argv []string) error {
	switch {
	case plan.kind == invocationRootGen:
		return runConflatedRootGen(ctx, rootCmd, *plan.rootGenPatch)
	default:
		rootCmd.SetArgs(argv)
		return rootCmd.ExecuteContext(ctx)
	}
}
```

- [x] **Step 6: Populate effective history, lineage, and stderr diagnostics; verify GREEN**

The common generation executor invokes `beforeSubmit` after current local uploads have become backend refs and immediately before `client.Generate`. It updates the same params map with `res.Seed` after a successful completion. This makes failed backend runs replayable while ensuring successful history stores the concrete returned seed.

```go
func runGen(cmd *cobra.Command, args []string) error {
	patch := genPatch{
		Args: genArgsFromFlags(cmd, args),
		Changed: changedGenFlags(cmd.Flags()),
	}
	return runPatchedGen(cmd, patch)
}

func runConflatedRootGen(ctx context.Context, cmd *cobra.Command, patch genPatch) error {
	cmd.SetContext(ctx)
	return runPatchedGen(cmd, patch)
}

func runPatchedGen(cmd *cobra.Command, patch genPatch) error {
	state, err := stateFromContext(cmd.Context())
	if err != nil {
		return err
	}
	cfg, err := requireConfig()
	if err != nil {
		return err
	}
	params, baseline, snapshot, err := buildConflatedParams(cmd.Context(), state.store, patch, cfg)
	if err != nil {
		return err
	}
	return executeResolvedGen(cmd, cfg, patch.Args, params, func(final stclient.GenParams) {
		state.final = &invocationResult{params: final, policySnapshot: snapshot}
		if baseline == nil {
			return
		}
		state.final.derivedFromHistoryID = &baseline.ID
		if !genQuiet {
			fmt.Fprintf(cmd.ErrOrStderr(), "initial command [id=%d]: %s\n", baseline.ID, baseline.Effective.Display)
			fmt.Fprintf(cmd.ErrOrStderr(), "next command [id=%d]: %s\n", state.id, history.CanonicalGenDisplay(final))
		}
	})
}
```

Refactor the body currently in `runGen` into `executeResolvedGen(cmd, cfg, args, params, beforeSubmit)`. Keep its current upload, mode-switch, output, `--json`, and `--stream` behavior; call `beforeSubmit(params)` after uploads/mode preparation and before `client.Generate`, and assign `params["seed"] = res.Seed` immediately after a successful result. The `appendHistory` implementation from Task 4 records the resulting effective params and mutually exclusive lineage fields.

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./cmd/st -run 'TestReplayUsesEffectiveParamsExactly|TestReplayRejectsGenerationOverrides|TestConflatedGenUsesBaselineEffectiveParams|TestConflatedGenExplicitZeroAndRandomOverrideBaseline|TestExplicitGenWithoutRecentBaselineFallsBackToNormalResolution|TestGenWritesOutputFile' -count=1
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
cd /Users/darkbit1001/workspace/Stability-Toys
git add cli/go/cmd/st/replay.go cli/go/cmd/st/replay_test.go cli/go/cmd/st/gen.go cli/go/cmd/st/gen_test.go cli/go/cmd/st/history_runtime.go cli/go/internal/config/precedence.go cli/go/internal/config/precedence_test.go
git commit -m "feat: add st replay and gen conflation execution"
```

---

### Task 6: Finish integration coverage, docs, and verification

**Files:**
- Modify: `cli/go/cmd/st/history_runtime_test.go`
- Modify: `cli/go/cmd/st/gen_test.go`
- Modify: `cli/go/README.md`
- Modify: `cli/go/USAGE.md`

- [x] **Step 1: Add a reusable scripted generation server and split-output harness**

Append to `cli/go/cmd/st/gen_test.go`:

```go
type genReply struct {
	Error string
	Seed  int64
}

func newScriptedGenServer(t *testing.T, replies ...genReply) (*httptest.Server, *[]stclient.GenParams) {
	t.Helper()
	var mu sync.Mutex
	index := 0
	captured := []stclient.GenParams{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/storage/") {
			_, _ = w.Write([]byte("PNGBYTES"))
			return
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		if err := wsjson.Read(r.Context(), conn, &sub); err != nil {
			return
		}
		params, _ := sub["params"].(map[string]any)
		mu.Lock()
		captured = append(captured, stclient.GenParams(params))
		reply := replies[index]
		index++
		mu.Unlock()
		_ = wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": sub["id"], "jobId": fmt.Sprintf("J%d", index)})
		if reply.Error != "" {
			_ = wsjson.Write(r.Context(), conn, map[string]any{"type": "job:error", "error": reply.Error})
			return
		}
		_ = wsjson.Write(r.Context(), conn, map[string]any{
			"type": "job:complete", "jobId": fmt.Sprintf("J%d", index),
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta": map[string]any{"seed": reply.Seed},
		})
	}))
	return srv, &captured
}
```

Add the split-output helper beside the existing `runCmd`/`runCmdMayFail` helpers in `cli/go/cmd/st/gen_test.go`:

```go
func runCmdCaptureWithStateRoot(t *testing.T, stateRoot string, args ...string) (string, string, error) {
	t.Helper()
	old := resolveStateRoot
	resolveStateRoot = func() (string, error) { return stateRoot, nil }
	defer func() { resolveStateRoot = old }()
	resetCLIFlagState()
	var stdout, stderr strings.Builder
	rootCmd.SetOut(&stdout)
	rootCmd.SetErr(&stderr)
	rootCmd.SetArgs(args)
	err := executeCLI(context.Background(), args)
	return stdout.String(), stderr.String(), err
}
```

All new multi-invocation tests call through this helper so one invocation's `--with-exit`, `--json`, `--stream`, or gen values cannot contaminate the next.

- [x] **Step 2: Add recency, pinning, diagnostics, and quiet integration tests**

Append to `cli/go/cmd/st/history_runtime_test.go`:

```go
func TestRecentExitOneSelectorRequeriesAfterEveryRun(t *testing.T) {
	root := t.TempDir()
	srv, _ := newScriptedGenServer(t,
		genReply{Error: "A failed"},
		genReply{Error: "A1 failed"},
		genReply{Seed: 12},
		genReply{Error: "A3 failed"},
		genReply{Seed: 14},
	)
	defer srv.Close()
	outDir := t.TempDir()
	cfg := writeTestConfig(t, outDir)
	_, _, _ = runCmdCaptureWithStateRoot(t, root, "gen", "--server", srv.URL, "--config", cfg, "--prompt", "A")
	_, _, err := runCmdCaptureWithStateRoot(t, root, "conflate", "--with-exit", "1")
	if err != nil {
		t.Fatal(err)
	}
	for _, prompt := range []string{"A1", "A2", "A3", "A4"} {
		_, _, _ = runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, "--prompt", prompt)
	}
	store := history.NewFSStore(root)
	for id, wantBase := range map[int64]int64{3: 1, 4: 3, 5: 3, 6: 5} {
		entry, err := store.Get(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		if entry.DerivedFromHistoryID == nil || *entry.DerivedFromHistoryID != wantBase {
			t.Fatalf("entry %d derived_from=%v, want %d", id, entry.DerivedFromHistoryID, wantBase)
		}
	}
}

func TestPinnedBaselineAndDiagnosticsStayFixed(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	ctx := context.Background()
	baseID, _ := store.ReserveID(ctx)
	_ = store.Append(ctx, replayableEntry(baseID, map[string]any{"prompt": "owl", "guidance_scale": 4.5}, 1))
	if _, _, err := runCmdCaptureWithStateRoot(t, root, "conflate", fmt.Sprintf("history:%d", baseID)); err != nil {
		t.Fatal(err)
	}
	srv, _ := newScriptedGenServer(t, genReply{Error: "failed"}, genReply{Seed: 2})
	defer srv.Close()
	cfg := writeTestConfig(t, t.TempDir())
	_, stderr, _ := runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, "--cfg", "5")
	if !strings.Contains(stderr, fmt.Sprintf("initial command [id=%d]:", baseID)) || !strings.Contains(stderr, "next command [id=3]:") {
		t.Fatalf("stderr = %q", stderr)
	}
	_, quietErr, _ := runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, "--quiet", "--cfg", "6")
	if strings.Contains(quietErr, "initial command") || strings.Contains(quietErr, "next command") {
		t.Fatalf("--quiet leaked diagnostics: %q", quietErr)
	}
	entry, err := store.Get(ctx, 4)
	if err != nil {
		t.Fatal(err)
	}
	if entry.DerivedFromHistoryID == nil || *entry.DerivedFromHistoryID != baseID {
		t.Fatalf("pin advanced: %#v", entry)
	}
}

func TestConflateConfirmationUsesStdout(t *testing.T) {
	stdout, stderr, err := runCmdCaptureWithStateRoot(t, t.TempDir(), "conflate", "on")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stdout, "Conflating recent successful gen runs.") || stderr != "" {
		t.Fatalf("stdout=%q stderr=%q", stdout, stderr)
	}
}
```

Add this helper in the same test file:

```go
func replayableEntry(id int64, params map[string]any, exitCode int) history.Entry {
	return history.Entry{
		SchemaVersion: 1,
		ID: id,
		StartedAt: "2026-07-13T00:00:00Z",
		FinishedAt: "2026-07-13T00:00:01Z",
		Family: history.FamilyGen,
		Raw: history.CommandView{Argv: []string{"st", "gen"}, Display: "st gen"},
		Effective: &history.CommandView{
			Argv: history.CanonicalGenArgv(params),
			Display: history.CanonicalGenDisplay(params),
			Params: history.CloneParams(params),
		},
		ExitCode: exitCode,
	}
}
```

IDs in these tests come from actual reservation order: the pinning command consumes ID 2, so its two derived runs are IDs 3 and 4.

- [x] **Step 3: Add replay, expired-reference, and stdout-contract integration tests**

Append to `cli/go/cmd/st/history_runtime_test.go`:

```go
func TestReplayKeepsPolicyAndReportsOnStderr(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	ctx := context.Background()
	id, _ := store.ReserveID(ctx)
	_ = store.Append(ctx, replayableEntry(id, map[string]any{"prompt": "owl", "seed": 42}, 1))
	policy := history.DefaultPolicy()
	policy.Enabled = true
	policy.UpdatedAt = "before"
	_ = store.SavePolicy(ctx, policy)
	srv, _ := newScriptedGenServer(t, genReply{Seed: 42})
	defer srv.Close()
	cfg := writeTestConfig(t, t.TempDir())
	_, stderr, err := runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, "replay", strconv.FormatInt(id, 10))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stderr, "replaying [id=1]:") {
		t.Fatalf("stderr = %q", stderr)
	}
	after, _ := store.LoadPolicy(ctx)
	if after.UpdatedAt != "before" {
		t.Fatalf("replay mutated policy: %#v", after)
	}
	replayed, _ := store.Get(ctx, 2)
	if replayed.Family != history.FamilyGen || replayed.ReplayedFromHistoryID == nil || *replayed.ReplayedFromHistoryID != id || replayed.DerivedFromHistoryID != nil {
		t.Fatalf("replay history = %#v", replayed)
	}
}

func TestInheritedExpiredRefsFailWithoutFallback(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	ctx := context.Background()
	id, _ := store.ReserveID(ctx)
	params := map[string]any{"prompt": "owl", "init_image_ref": "expired-R1"}
	_ = store.Append(ctx, replayableEntry(id, params, 0))
	_, _, _ = runCmdCaptureWithStateRoot(t, root, "conflate", "history:1")
	srv, captured := newScriptedGenServer(t, genReply{Error: "file ref expired"}, genReply{Error: "file ref expired"})
	defer srv.Close()
	cfg := writeTestConfig(t, t.TempDir())
	_, _, conflateErr := runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, "--cfg", "5")
	_, _, replayErr := runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, "replay", "1")
	if conflateErr == nil || replayErr == nil || len(*captured) != 2 {
		t.Fatalf("conflate=%v replay=%v captured=%#v", conflateErr, replayErr, *captured)
	}
	for _, request := range *captured {
		if request["init_image_ref"] != "expired-R1" {
			t.Fatalf("request unexpectedly replaced inherited ref: %#v", request)
		}
	}
}

func TestConflationPreservesJSONAndStreamStdoutShape(t *testing.T) {
	for _, mode := range []string{"--json", "--stream"} {
		t.Run(strings.TrimPrefix(mode, "--"), func(t *testing.T) {
			root := t.TempDir()
			store := history.NewFSStore(root)
			id, _ := store.ReserveID(context.Background())
			_ = store.Append(context.Background(), replayableEntry(id, map[string]any{"prompt": "owl"}, 0))
			_, _, _ = runCmdCaptureWithStateRoot(t, root, "conflate", "history:1")
			srv, _ := newScriptedGenServer(t, genReply{Seed: 7})
			defer srv.Close()
			outDir := t.TempDir()
			cfg := writeTestConfig(t, outDir)
			stdout, stderr, err := runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, mode, "--prompt", "variation")
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(stderr, "initial command [id=1]") {
				t.Fatalf("stderr = %q", stderr)
			}
			lines := strings.Split(strings.TrimSuffix(stdout, "\n"), "\n")
			if mode == "--json" {
				var object map[string]any
				if err := json.Unmarshal([]byte(stdout), &object); err != nil || object["seed"] != float64(7) {
					t.Fatalf("json stdout=%q err=%v", stdout, err)
				}
			} else {
				if len(lines) != 2 {
					t.Fatalf("stream lines=%q", lines)
				}
				for _, line := range lines {
					if !json.Valid([]byte(line)) {
						t.Fatalf("non-NDJSON line %q", line)
					}
				}
			}
		})
	}
}
```

The JSON assertion permits exactly one top-level JSON value; the stream assertion permits only the existing `job_id` and `complete` NDJSON lines. Conflation/replay diagnostics are asserted separately on stderr and therefore cannot alter either stdout contract.

- [x] **Step 4: Document the new command surface**

Modify `cli/go/README.md` command table and config/state section:

```md
| `st conflate ...` | Toggle or configure gen-only parameter inheritance from history |
| `st replay <id>` | Re-run one historical generation entry exactly |

History/state lives under `$XDG_STATE_HOME/st/` (or `~/.local/state/st/`) in
`history.jsonl`, `conflate-policy.json`, `next-id`, and `state.lock`. History is always
written; conflation is opt-in.
```

Modify `cli/go/USAGE.md` with examples:

```md
### Conflation

```bash
st conflate
st --cfg 4.3 --prompt "two horses drinking"
st conflate history:12345
st --prompt "variation prompt"
```

Root shorthand is flag-only while conflation is enabled. Use `--prompt`; bare
positional text at the `st` root is rejected.

### Replay

```bash
st replay 12345
```
```

- [x] **Step 5: Run focused and full verification**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys/cli/go
go test ./cmd/st -count=1
go test ./internal/history -count=1
go test ./... -count=1
```

Expected: PASS for all three commands.

- [x] **Step 6: Run drift check for touched docs/code**

Run:

```bash
cd /Users/darkbit1001/workspace/Stability-Toys
drift check --changed
```

Expected: PASS, or only unrelated pre-existing drift findings.

- [x] **Step 7: Commit**

```bash
cd /Users/darkbit1001/workspace/Stability-Toys
git add cli/go/cmd/st/history_runtime_test.go cli/go/cmd/st/gen_test.go cli/go/README.md cli/go/USAGE.md
git commit -m "docs: document st conflation and replay"
```

---

## Self-Review

- **Precedence:** Task 5 now implements `config < baked < baseline < explicit` directly. The command path carries pflag `Changed` intent, and tests pin config-vs-baseline conflicts, explicit zeros, and `--seed random` deletion.
- **Shorthand parsing:** Task 4 uses a real combined pflag parse, consumes option values correctly, preserves normal Cobra dispatch while conflation is off, rejects only remaining positional args, and keeps typoed subcommands out of generation dispatch.
- **Storage correctness:** Task 2 now includes `state.lock`, short lock scopes, write-fsync-rename plus directory fsync, append fsync, private modes, concurrent subprocess reservation/append proof, and incomplete-final-line tolerance with interior-corruption rejection.
- **Policy transactionality:** Task 3 validates selector combinations, family, exit-code range, positive IDs, and pinned entry eligibility before atomic policy replacement. Failed pinning leaves the old policy intact.
- **Output contract:** `st conflate` confirmations are intentionally stdout. Only diagnostics attached to generation/replay use stderr and honor `--quiet`; Task 6 verifies JSON and NDJSON stdout shapes separately.
- **Spec coverage:** Tasks 1-5 cover XDG state, always-on history, recent/pinned selection, replay, lineage, exact effective params, and gen-only eligibility. Task 6 explicitly covers exit-1 advancement, fixed pins, diagnostics, quiet, replay overrides/policy isolation, expired refs, concurrent IDs, and docs.
- **Placeholder scan:** Clean; every planned test and implementation step is concrete.
- **Type consistency:** Replay is an internal invocation kind but persists as family `gen`; dispatch signatures consistently carry `invocationState`; `stclient.GenParams` remains the authoritative effective request object.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-13-st-gen-conflation-history.md`.

Repo policy forbids subagent-driven development here. The next valid execution mode is inline execution using `superpowers:executing-plans`, task-by-task, with review stops at task boundaries.
