package main

import (
	"os"
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

func TestResolveServerURLFlagWins(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	writeConfigWithServerURL(t, p, "http://from-config:7860")

	got := resolveServerURL("http://from-flag:7860", p)
	if got != "http://from-flag:7860" {
		t.Fatalf("flag should win, got %q", got)
	}
}

func TestResolveServerURLFromConfig(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	writeConfigWithServerURL(t, p, "http://from-config:7860")

	got := resolveServerURL("", p)
	if got != "http://from-config:7860" {
		t.Fatalf("config server_url should be used, got %q", got)
	}
}

func TestResolveServerURLEmptyWhenConfigMissing(t *testing.T) {
	got := resolveServerURL("", filepath.Join(t.TempDir(), "nonexistent.json"))
	if got != "" {
		t.Fatalf("should return empty when config missing, got %q", got)
	}
}

func writeConfigWithServerURL(t *testing.T, path, serverURL string) {
	t.Helper()
	body := `{"config":{"server_url":"` + serverURL + `","defaults":{"generation":{"genres":"512x512"},"output_format":"png","output_directory":"/tmp","include_meta":false}}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
