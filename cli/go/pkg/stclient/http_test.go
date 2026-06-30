package stclient

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// realModesResponse mirrors the live GET /api/modes shape (server/model_routes.py):
// "modes" is an object keyed by mode name, not a list, and there is no "current" key.
const realModesResponse = `{
  "default_mode": "default",
  "resolution_sets": {},
  "modes": {
    "default":  {"model": "sdxl-base", "default_size": "1024x1024", "default_steps": 20, "default_guidance": 7.5,
                 "controlnet_policy": {"enabled": true, "max_attachments": 4}, "chat_enabled": false},
    "cartoony": {"model": "sdxl-cartoon", "default_size": "512x512", "default_steps": 8, "default_guidance": 2.5,
                 "controlnet_policy": {"enabled": false}, "chat_enabled": false}
  }
}`

func TestModesParsesDictKeysSorted(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/modes" || r.Method != http.MethodGet {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		w.Write([]byte(realModesResponse))
	}))
	defer srv.Close()

	modes, err := New(srv.URL).Modes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(modes) != 2 {
		t.Fatalf("got %d modes: %+v", len(modes), modes)
	}
	// Map iteration order is non-deterministic; Modes must return sorted names.
	if modes[0].Name != "cartoony" || modes[1].Name != "default" {
		t.Fatalf("expected sorted [cartoony default], got %+v", modes)
	}
	// "default" is the default_mode; IsDefault must be set on the right entry.
	if modes[1].IsDefault != true {
		t.Fatalf("expected modes[1] (default) to have IsDefault=true, got %+v", modes[1])
	}
	if modes[0].IsDefault != false {
		t.Fatalf("expected modes[0] (cartoony) to have IsDefault=false, got %+v", modes[0])
	}
	// Per-mode config fields must be decoded, not discarded.
	if modes[1].Model != "sdxl-base" || modes[1].DefaultSize != "1024x1024" || modes[1].DefaultSteps != 20 {
		t.Fatalf("default mode config not decoded: %+v", modes[1])
	}
	if !modes[1].ControlNetEnabled {
		t.Fatalf("default mode controlnet_policy.enabled not decoded: %+v", modes[1])
	}
}

func TestCurrentModeReadsModelsStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/status" {
			t.Fatalf("CurrentMode must read /api/models/status, got %s", r.URL.Path)
		}
		w.Write([]byte(`{"backend":"mlx","current_mode":"cartoony","is_loaded":true}`))
	}))
	defer srv.Close()

	got, err := New(srv.URL).CurrentMode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got != "cartoony" {
		t.Fatalf("current mode = %q, want cartoony", got)
	}
}

func TestModelsReturnsStatusMap(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/status" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		w.Write([]byte(`{"backend":"mlx","current_mode":"default","queue_size":0}`))
	}))
	defer srv.Close()

	status, err := New(srv.URL).Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if status["backend"] != "mlx" {
		t.Fatalf("status backend = %v, want mlx", status["backend"])
	}
}

func TestSwitchModePostsJSONBody(t *testing.T) {
	var gotBody map[string]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/modes/switch" || r.Method != http.MethodPost {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Fatalf("Content-Type = %q, want application/json", ct)
		}
		b, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(b, &gotBody); err != nil {
			t.Fatalf("body not JSON: %v (%s)", err, b)
		}
		w.Write([]byte(`{"status":"queued","to_mode":"cartoony"}`))
	}))
	defer srv.Close()

	if err := New(srv.URL).SwitchMode(context.Background(), "cartoony"); err != nil {
		t.Fatal(err)
	}
	if gotBody["mode"] != "cartoony" {
		t.Fatalf("switch body = %+v, want {mode: cartoony}", gotBody)
	}
}

func TestGetJSONReturnsErrorOnNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	if _, err := New(srv.URL).Modes(context.Background()); err == nil {
		t.Fatal("expected error on 500, got nil")
	}
}

func TestReloadModesPostsToEndpoint(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/modes/reload" || r.Method != http.MethodPost {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := New(srv.URL).ReloadModes(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("POST /api/modes/reload not called")
	}
}
