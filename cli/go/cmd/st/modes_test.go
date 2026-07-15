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

func TestModesListRendersSchedulers(t *testing.T) {
	body := `{"default_mode":"fast","modes":{
	  "fast":{"model":"sdxl-turbo","default_size":"512x512","default_steps":4,"default_guidance":0,
	          "default_scheduler_id":"lcm","allowed_scheduler_ids":["lcm","euler","ddim"],
	          "controlnet_policy":{"enabled":true}},
	  "plain":{"model":"sd15","default_size":"512x512","default_steps":20,"default_guidance":7.5,
	           "controlnet_policy":{"enabled":false}}
	}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	out, err := runCmdMayFail(t, "--server", srv.URL, "modes")
	if err != nil {
		t.Fatal(err)
	}
	// The allowed list renders with the default marked.
	if !strings.Contains(out, "schedulers: lcm (default), euler, ddim") {
		t.Errorf("expected rendered schedulers line, got:\n%s", out)
	}
	// A mode with no allowed list must not emit a schedulers line.
	if strings.Count(out, "schedulers:") != 1 {
		t.Errorf("mode without allowed_scheduler_ids should omit the line, got:\n%s", out)
	}
}

func TestModesShowCmdJSONIncludesSchedulers(t *testing.T) {
	body := `{"default_mode":"fast","modes":{"fast":{"model":"sdxl-turbo","default_size":"512x512","default_steps":4,"default_guidance":0,"default_scheduler_id":"lcm","allowed_scheduler_ids":["lcm","euler"],"controlnet_policy":{"enabled":true}}}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	out, err := runCmdMayFail(t, "--server", srv.URL, "modes", "show", "fast")
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if jsonErr := json.Unmarshal([]byte(strings.TrimSpace(out)), &m); jsonErr != nil {
		t.Fatalf("output not valid JSON: %v\noutput: %q", jsonErr, out)
	}
	ids, ok := m["allowed_scheduler_ids"].([]any)
	if !ok || len(ids) != 2 || ids[0] != "lcm" {
		t.Errorf("allowed_scheduler_ids missing/wrong in JSON: %v", m["allowed_scheduler_ids"])
	}
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
