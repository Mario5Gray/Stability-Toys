package config

import (
	"os"
	"path/filepath"
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
	want := filepath.Join("/xdg", "stability-toys", "config.json")
	if got, _ := Resolve(""); got != want {
		t.Fatalf("xdg default, got %s want %s", got, want)
	}
}
