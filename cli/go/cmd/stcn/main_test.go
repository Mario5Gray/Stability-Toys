package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRunEmitsCompactAttachment(t *testing.T) {
	var out bytes.Buffer
	err := run([]string{"canny:Rmap1", "--strength", "0.8"}, &out)
	if err != nil {
		t.Fatal(err)
	}
	line := out.String()
	if !strings.HasSuffix(line, "\n") {
		t.Fatalf("want trailing newline: %q", line)
	}
	trimmed := strings.TrimRight(line, "\n")
	if strings.ContainsAny(trimmed, " \t") {
		t.Fatalf("emitted line is not a single shell token: %q", trimmed)
	}
	var a map[string]any
	if err := json.Unmarshal([]byte(trimmed), &a); err != nil {
		t.Fatal(err)
	}
	if a["control_type"] != "canny" || a["map_asset_ref"] != "Rmap1" ||
		a["attachment_id"] != "canny" || a["strength"].(float64) != 0.8 {
		t.Fatalf("bad attachment: %v", a)
	}
}

func TestRunRequiresPositional(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"--strength", "0.8"}, &out); err == nil {
		t.Fatal("want error when positional missing")
	}
	if out.Len() != 0 {
		t.Fatalf("nothing must be emitted on error, got %q", out.String())
	}
}

func TestRunPropagatesValidationError(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"canny:Rmap1", "--strength", "9"}, &out); err == nil {
		t.Fatal("want validation error for out-of-range strength")
	}
	if out.Len() != 0 {
		t.Fatalf("nothing must be emitted on error, got %q", out.String())
	}
}
