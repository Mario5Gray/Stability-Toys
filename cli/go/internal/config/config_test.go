package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadReadsGenerationDefaults(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	os.WriteFile(p, []byte(`{"config":{"defaults":{"generation":{"mode":"m","cfg":2.5,"steps":10,"genres":"512x512","seed":"random"},"output_format":"png","output_directory":"/tmp/out"}}}`), 0o644)

	cfg, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Defaults.Generation.Mode != "m" || cfg.Defaults.Generation.Steps != 10 || cfg.Defaults.OutputDirectory != "/tmp/out" {
		t.Fatalf("got %+v", cfg.Defaults)
	}
}

func TestBootstrapWritesTemplate(t *testing.T) {
	p := filepath.Join(t.TempDir(), "config.json")
	if err := BootstrapTemplate(p); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(p); err != nil {
		t.Fatalf("template not loadable: %v", err)
	}
}

func TestLoadReadsServerURL(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	os.WriteFile(p, []byte(`{"config":{"server_url":"http://myhost:7860","defaults":{"generation":{"genres":"512x512"},"output_format":"png","output_directory":"/tmp"}}}`), 0o644)

	cfg, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ServerURL != "http://myhost:7860" {
		t.Fatalf("server_url=%q, want http://myhost:7860", cfg.ServerURL)
	}
}

// TestResolveDiscoveryOrder pins the --config > $ST_CONFIG > XDG default chain.
func TestResolveDiscoveryOrder(t *testing.T) {
	t.Setenv("ST_CONFIG", "/env/config.json")
	t.Setenv("XDG_CONFIG_HOME", "/xdg")

	if got, _ := Resolve("/flag/config.json"); got != "/flag/config.json" {
		t.Fatalf("flag should win, got %s", got)
	}
	if got, _ := Resolve(""); got != "/env/config.json" {
		t.Fatalf("env should win when no flag, got %s", got)
	}

	t.Setenv("ST_CONFIG", "")
	want := filepath.Join("/xdg", AppDir, "config.json")
	if got, _ := Resolve(""); got != want {
		t.Fatalf("xdg default, got %s want %s", got, want)
	}
}

// --- STABL: single config location under ~/.config/st -----------------------

func TestResolveUsesStDirectoryNotStabilityToys(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("ST_CONFIG", "")

	got, err := Resolve("")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	want := filepath.Join(dir, "st", "config.json")
	if got != want {
		t.Fatalf("Resolve() = %q, want %q", got, want)
	}
}

func TestResolvePrecedenceFlagBeatsEnvBeatsDefault(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("ST_CONFIG", filepath.Join(dir, "from-env.json"))

	if got, _ := Resolve("/explicit/flag.json"); got != "/explicit/flag.json" {
		t.Fatalf("flag should win, got %q", got)
	}
	if got, _ := Resolve(""); got != filepath.Join(dir, "from-env.json") {
		t.Fatalf("ST_CONFIG should win over default, got %q", got)
	}
}

func TestLegacyConfigIsReportedWhenOnlyLegacyExists(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("ST_CONFIG", "")

	legacy := filepath.Join(dir, "stability-toys", "config.json")
	if err := os.MkdirAll(filepath.Dir(legacy), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacy, []byte(`{"config":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}

	err := CheckLegacyLocation("")
	if err == nil {
		t.Fatal("expected an error naming the legacy config, got nil")
	}
	msg := err.Error()
	for _, want := range []string{legacy, filepath.Join(dir, "st", "config.json"), "mv"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("error should mention %q; got:\n%s", want, msg)
		}
	}
}

func TestNoLegacyErrorWhenNewLocationExists(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("ST_CONFIG", "")

	for _, p := range []string{
		filepath.Join(dir, "stability-toys", "config.json"),
		filepath.Join(dir, "st", "config.json"),
	} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(`{"config":{}}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	if err := CheckLegacyLocation(""); err != nil {
		t.Fatalf("migrated installs must not be nagged: %v", err)
	}
}

func TestNoLegacyErrorWhenNeitherExists(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("ST_CONFIG", "")

	if err := CheckLegacyLocation(""); err != nil {
		t.Fatalf("fresh installs must bootstrap normally: %v", err)
	}
}

func TestExplicitPathBypassesLegacyCheck(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("ST_CONFIG", "")

	legacy := filepath.Join(dir, "stability-toys", "config.json")
	os.MkdirAll(filepath.Dir(legacy), 0o755)
	os.WriteFile(legacy, []byte(`{"config":{}}`), 0o644)

	// An explicit --config must never trip the cutover.
	if err := CheckLegacyLocation("/somewhere/else.json"); err != nil {
		t.Fatalf("explicit path must bypass: %v", err)
	}
	t.Setenv("ST_CONFIG", "/from/env.json")
	if err := CheckLegacyLocation(""); err != nil {
		t.Fatalf("$ST_CONFIG must bypass: %v", err)
	}
}
