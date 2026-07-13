package main

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"

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
		argv  []string
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
	genCmd.LocalFlags().VisitAll(func(flag *pflag.Flag) {
		if flag.Name != "help" {
			want[flag.Name] = true
		}
	})
	got := map[string]bool{}
	shorthand.VisitAll(func(flag *pflag.Flag) { got[flag.Name] = true })
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("shorthand flags=%v gen flags=%v", got, want)
	}
}
