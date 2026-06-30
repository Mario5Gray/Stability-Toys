package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModesReloadCmdHitsEndpoint(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/modes/reload" && r.Method == http.MethodPost {
			called = true
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	out := runCmd(t, "--server", srv.URL, "modes", "reload")
	if !called {
		t.Fatal("POST /api/modes/reload was not called")
	}
	if !strings.Contains(out, "reloaded") {
		t.Errorf("expected 'reloaded' in output, got: %q", out)
	}
}

func fakeModeServer(t *testing.T) *httptest.Server {
	t.Helper()
	body := `{"default_mode":"fast","modes":{"fast":{"model":"sdxl-turbo","default_size":"512x512","default_steps":4,"default_guidance":0,"controlnet_policy":{"enabled":true}}}}`
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/modes" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(body))
			return
		}
		http.NotFound(w, r)
	}))
}

func TestModesShowCmdNotFound(t *testing.T) {
	srv := fakeModeServer(t)
	defer srv.Close()
	_, err := runCmdMayFail(t, "--server", srv.URL, "modes", "show", "slow")
	if err == nil {
		t.Fatal("expected error for unknown mode name, got nil")
	}
	if !strings.Contains(err.Error(), "slow") {
		t.Errorf("error should name the missing mode, got: %v", err)
	}
}

func TestModesShowCmdJSON(t *testing.T) {
	srv := fakeModeServer(t)
	defer srv.Close()
	out, err := runCmdMayFail(t, "--server", srv.URL, "modes", "show", "fast")
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if jsonErr := json.Unmarshal([]byte(strings.TrimSpace(out)), &m); jsonErr != nil {
		t.Fatalf("output not valid JSON: %v\noutput: %q", jsonErr, out)
	}
	if m["name"] != "fast" {
		t.Errorf("name = %v, want \"fast\"", m["name"])
	}
	if _, ok := m["model"]; !ok {
		t.Errorf("missing 'model' field in JSON output")
	}
}

func TestModesSwitchCmdHitsEndpoint(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/modes/switch" && r.Method == http.MethodPost {
			called = true
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	out := runCmd(t, "--server", srv.URL, "modes", "switch", "fast")
	if !called {
		t.Fatal("POST /api/modes/switch was not called")
	}
	if !strings.Contains(out, "fast") {
		t.Errorf("expected mode name in output, got: %q", out)
	}
}
