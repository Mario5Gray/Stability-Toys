package main

import (
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
