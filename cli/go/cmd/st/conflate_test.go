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

func TestConflateInclusiveGenResetsExitSelectorToSuccess(t *testing.T) {
	root := t.TempDir()
	runCmdWithStateRoot(t, root, "conflate", "on", "--with-exit", "1")
	runCmdWithStateRoot(t, root, "conflate", "--inclusive", "gen")
	policy, err := history.NewFSStore(root).LoadPolicy(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !policy.Enabled || policy.Selector.Kind != history.SelectorRecent || fmt.Sprint(policy.Selector.ExitCodes) != "[0]" {
		t.Fatalf("policy = %#v", policy)
	}
}
