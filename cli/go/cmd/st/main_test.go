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

	got, err := resolveServerURL("http://from-flag:7860", p)
	if err != nil {
		t.Fatal(err)
	}
	if got != "http://from-flag:7860" {
		t.Fatalf("flag should win, got %q", got)
	}
}

func TestResolveServerURLFromConfig(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	writeConfigWithServerURL(t, p, "http://from-config:7860")

	got, err := resolveServerURL("", p)
	if err != nil {
		t.Fatal(err)
	}
	if got != "http://from-config:7860" {
		t.Fatalf("config server_url should be used, got %q", got)
	}
}

func TestResolveServerURLNormalizesSchemeLessHost(t *testing.T) {
	got, err := resolveServerURL("host.lan:4200", filepath.Join(t.TempDir(), "unused.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "http://host.lan:4200" {
		t.Fatalf("scheme-less host should default to http, got %q", got)
	}
}

func TestResolveServerURLRejectsInvalidHost(t *testing.T) {
	if _, err := resolveServerURL("ftp://host", "/unused.json"); err == nil {
		t.Fatal("expected invalid scheme to be rejected")
	}
}

func TestResolveServerURLConfigMissingIsError(t *testing.T) {
	p := filepath.Join(t.TempDir(), "nonexistent.json")
	_, err := resolveServerURL("", p)
	if err == nil || !strings.Contains(err.Error(), p) {
		t.Fatalf("error must name the config path, got %v", err)
	}
}

func TestResolveServerURLMissingValueNamesSources(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	writeConfigWithServerURL(t, p, "")

	_, err := resolveServerURL("", p)
	if err == nil {
		t.Fatal("expected an error when no URL is configured")
	}
	for _, want := range []string{"--server", "ST_SERVER", "server_url"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error must name %s, got %v", want, err)
		}
	}
}

func writeConfigWithServerURL(t *testing.T, path, serverURL string) {
	t.Helper()
	body := `{"config":{"server_url":"` + serverURL + `","defaults":{"generation":{"genres":"512x512"},"output_format":"png","output_directory":"/tmp","include_meta":false}}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
