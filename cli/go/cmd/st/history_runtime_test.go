package main

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
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
	_, conflateStderr, conflateErr := runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, "--cfg", "5")
	_, replayStderr, replayErr := runCmdCaptureWithStateRoot(t, root, "--server", srv.URL, "--config", cfg, "replay", "1")
	if conflateErr == nil || replayErr == nil || len(*captured) != 2 {
		t.Fatalf("conflate=%v replay=%v captured=%#v", conflateErr, replayErr, *captured)
	}
	for name, stderr := range map[string]string{"conflate": conflateStderr, "replay": replayStderr} {
		for _, want := range []string{
			"asset ref from history/conflation is no longer available",
			"Upload refs are temporary by default and expire after 300 seconds",
			"ASSET_STORE_PROVIDER=FILESYSTEM",
			"FS_STORAGE_TTL_S=604800",
			"v1 does not expose a public promote endpoint",
		} {
			if !strings.Contains(stderr, want) {
				t.Fatalf("%s stderr missing %q:\n%s", name, want, stderr)
			}
		}
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

func replayableEntry(id int64, params map[string]any, exitCode int) history.Entry {
	return history.Entry{
		SchemaVersion: 1,
		ID:            id,
		StartedAt:     "2026-07-13T00:00:00Z",
		FinishedAt:    "2026-07-13T00:00:01Z",
		Family:        history.FamilyGen,
		Raw:           history.CommandView{Argv: []string{"st", "gen"}, Display: "st gen"},
		Effective: &history.CommandView{
			Argv:    history.CanonicalGenArgv(params),
			Display: history.CanonicalGenDisplay(params),
			Params:  history.CloneParams(params),
		},
		ExitCode: exitCode,
	}
}
