package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

func TestHistDefaultsToAllEntriesNewestFirst(t *testing.T) {
	root := t.TempDir()
	writeHistEntries(t, root, 3)

	stdout, stderr, err := runCmdCaptureWithStateRoot(t, root, "hist")
	if err != nil {
		t.Fatal(err)
	}
	if stderr != "" {
		t.Fatalf("stderr = %q", stderr)
	}
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	if len(lines) != 5 {
		t.Fatalf("lines = %q", lines)
	}
	if !strings.Contains(lines[0], "ID") || !strings.Contains(lines[0], "COMMAND") {
		t.Fatalf("missing header: %q", lines[0])
	}
	for i, want := range []string{"st gen --prompt prompt-3", "st gen --prompt prompt-2", "st gen --prompt prompt-1"} {
		if !strings.Contains(lines[i+2], want) {
			t.Fatalf("line %d = %q, want command %q", i+2, lines[i+2], want)
		}
	}
}

func TestHistLimitsHumanOutput(t *testing.T) {
	root := t.TempDir()
	writeHistEntries(t, root, 3)

	stdout, _, err := runCmdCaptureWithStateRoot(t, root, "hist", "1")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(stdout, "prompt-2") || !strings.Contains(stdout, "prompt-3") {
		t.Fatalf("stdout = %q", stdout)
	}
}

func TestHistJSONLimitsNewestFirst(t *testing.T) {
	root := t.TempDir()
	writeHistEntries(t, root, 3)

	stdout, stderr, err := runCmdCaptureWithStateRoot(t, root, "hist", "2", "--json")
	if err != nil {
		t.Fatal(err)
	}
	if stderr != "" {
		t.Fatalf("stderr = %q", stderr)
	}
	var got []map[string]any
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatalf("json: %v\n%s", err, stdout)
	}
	if len(got) != 2 || got[0]["id"] != float64(3) || got[1]["id"] != float64(2) {
		t.Fatalf("entries = %#v", got)
	}
	if got[0]["command"] != "st gen --prompt prompt-3" || got[0]["exit_code"] != float64(1) {
		t.Fatalf("first entry = %#v", got[0])
	}
}

func TestHistRejectsInvalidLimit(t *testing.T) {
	_, _, err := runCmdCaptureWithStateRoot(t, t.TempDir(), "hist", "0")
	if err == nil || !strings.Contains(err.Error(), "positive integer") {
		t.Fatalf("err = %v", err)
	}
}

func writeHistEntries(t *testing.T, root string, count int) {
	t.Helper()
	store := history.NewFSStore(root)
	for i := 1; i <= count; i++ {
		id, err := store.ReserveID(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		errText := fmt.Sprintf("exit %d", i%2)
		entry := replayableEntry(id, map[string]any{"prompt": fmt.Sprintf("prompt-%d", i)}, i%2)
		if entry.ExitCode != 0 {
			entry.Error = &errText
		}
		if err := store.Append(context.Background(), entry); err != nil {
			t.Fatal(err)
		}
	}
}
