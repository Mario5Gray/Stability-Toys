package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/config"
)

func TestOutputDestinationRejectsEmptyDirectory(t *testing.T) {
	cfg := &config.Config{}
	_, _, err := outputDestination(cfg)
	if err == nil {
		t.Fatal("expected an error for an empty output_directory")
	}
	for _, want := range []string{"output_directory", "-o"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error must mention %s, got %v", want, err)
		}
	}
}

func TestOutputDestinationRejectsTemplatePlaceholder(t *testing.T) {
	cfg := &config.Config{}
	cfg.Defaults.OutputDirectory = "REPLACE_ME/output"
	_, _, err := outputDestination(cfg)
	if err == nil || !strings.Contains(err.Error(), "REPLACE_ME") {
		t.Fatalf("expected placeholder rejection, got %v", err)
	}
}

func TestOutputDestinationRejectsUnsupportedFormat(t *testing.T) {
	cfg := &config.Config{}
	cfg.Defaults.OutputDirectory = t.TempDir()
	cfg.Defaults.OutputFormat = "jpg"
	_, _, err := outputDestination(cfg)
	if err == nil || !strings.Contains(err.Error(), "PNG") {
		t.Fatalf("expected format rejection, got %v", err)
	}
}

func TestOutputDestinationDefaultsToPNG(t *testing.T) {
	cfg := &config.Config{}
	cfg.Defaults.OutputDirectory = t.TempDir()
	dir, format, err := outputDestination(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if dir != cfg.Defaults.OutputDirectory || format != "png" {
		t.Fatalf("got dir=%q format=%q", dir, format)
	}
}

func TestPreflightCreatesAbsoluteOutfileParent(t *testing.T) {
	dir := t.TempDir()
	outfile := filepath.Join(dir, "nested", "image.png")
	if err := preflightOutputDestination(&config.Config{}, genArgs{Outfile: outfile}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Dir(outfile)); err != nil {
		t.Fatalf("parent directory was not created: %v", err)
	}
}
