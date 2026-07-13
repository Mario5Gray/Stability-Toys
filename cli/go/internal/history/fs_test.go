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
