package main

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestBootstrapExitsWithPathMessage: a missing config path triggers a bootstrap
// and a message naming the path the user must edit.
func TestBootstrapExitsWithPathMessage(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	cfg, msg, bootstrapped := resolveConfig(p) // p does not exist yet
	if !bootstrapped {
		t.Fatal("expected bootstrap")
	}
	if cfg != nil {
		t.Fatalf("cfg should be nil on bootstrap, got %+v", cfg)
	}
	if msg == "" || !strings.Contains(msg, p) {
		t.Fatalf("message must state the path: %q", msg)
	}
}

// TestResolveConfigLoadsExisting: once a config exists, resolveConfig loads it
// and does not bootstrap.
func TestResolveConfigLoadsExisting(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	if _, _, bootstrapped := resolveConfig(p); !bootstrapped {
		t.Fatal("first call should bootstrap")
	}
	cfg, msg, bootstrapped := resolveConfig(p)
	if bootstrapped {
		t.Fatal("second call should load, not bootstrap")
	}
	if cfg == nil {
		t.Fatalf("expected a loaded config, msg=%q", msg)
	}
}
