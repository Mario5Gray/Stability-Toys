package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestUploadJSONShowsServerResolvedBucket(t *testing.T) {
	dir := t.TempDir()
	img := filepath.Join(dir, "m.png")
	if err := os.WriteFile(img, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// client sent type=canny; server resolved it to control_map
		_, _ = w.Write([]byte(`{"fileRef":"R1","bucket":"control_map","width":8,"height":6}`))
	}))
	defer srv.Close()

	stdout, _, err := runCmdCaptureWithStateRoot(t, t.TempDir(),
		"upload", "canny:"+img, "--json", "--server", srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(stdout), &out); err != nil {
		t.Fatalf("bad json %q: %v", stdout, err)
	}
	if out["bucket"] != "control_map" || out["fileRef"] != "R1" {
		t.Fatalf("want server-resolved bucket, got %v", out)
	}
	if out["width"].(float64) != 8 {
		t.Fatalf("missing dims: %v", out)
	}
}
