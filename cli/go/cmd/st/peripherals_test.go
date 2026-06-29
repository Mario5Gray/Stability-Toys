package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

func writeTempFile(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestUploadCmdPrintsFileRef(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"fileRef":"R9"}`))
	}))
	defer srv.Close()

	out := runCmd(t, "upload", "--server", srv.URL, writeTempFile(t, "x.png", "img"))
	if !strings.Contains(out, "R9") {
		t.Fatalf("got %q", out)
	}
}

func TestSuperresCmdWritesOutput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("SRBYTES"))
	}))
	defer srv.Close()

	outDir := t.TempDir()
	runCmd(t, "superres", "--server", srv.URL, "-o", outDir, "--magnitude", "2", writeTempFile(t, "in.png", "img"))

	data, err := os.ReadFile(filepath.Join(outDir, "out-0001.png"))
	if err != nil {
		t.Fatalf("superres output not written: %v", err)
	}
	if string(data) != "SRBYTES" {
		t.Fatalf("got %q", data)
	}
}

func TestCancelCmdAcks(t *testing.T) {
	var gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var f map[string]any
		wsjson.Read(r.Context(), conn, &f)
		gotType, _ = f["type"].(string)
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:cancel:ack", "id": f["id"], "jobId": f["jobId"]})
	}))
	defer srv.Close()

	out := runCmd(t, "cancel", "--server", srv.URL, "J1")
	if gotType != "job:cancel" {
		t.Fatalf("server saw %q, want job:cancel", gotType)
	}
	if !strings.Contains(out, "J1") {
		t.Fatalf("output should name the job: %q", out)
	}
}

func TestPriorityCmdAcks(t *testing.T) {
	var gotPriority any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var f map[string]any
		wsjson.Read(r.Context(), conn, &f)
		gotPriority = f["priority"]
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:priority:ack", "id": f["id"]})
	}))
	defer srv.Close()

	runCmd(t, "priority", "--server", srv.URL, "J2", "5")
	if gotPriority != float64(5) {
		t.Fatalf("server saw priority %v, want 5", gotPriority)
	}
}

func TestModelsCmdPrints(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"current_mode":"m1","backend":"mlx"}`))
	}))
	defer srv.Close()

	out := runCmd(t, "models", "--server", srv.URL)
	if !strings.Contains(out, "current_mode") || !strings.Contains(out, "m1") {
		t.Fatalf("got %q", out)
	}
}

func TestModesCmdPrints(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"modes":{"beta":{},"alpha":{}}}`))
	}))
	defer srv.Close()

	out := runCmd(t, "modes", "--server", srv.URL)
	if !strings.Contains(out, "alpha") || !strings.Contains(out, "beta") {
		t.Fatalf("got %q", out)
	}
}
