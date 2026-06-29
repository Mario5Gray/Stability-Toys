package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

func TestGenerateResolvesOnComplete(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/storage/") {
			w.Write([]byte("PNGDATA"))
			return
		}
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		wsjson.Read(r.Context(), conn, &sub)
		corr := sub["id"]
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": corr, "jobId": "J1"})
		wsjson.Write(r.Context(), conn, map[string]any{
			"type": "job:complete", "jobId": "J1",
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta":    map[string]any{"seed": 777},
		})
	}))
	defer srv.Close()

	_, res, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": "owl"})
	if err != nil {
		t.Fatal(err)
	}
	if res.StorageKey != "K1" || res.StorageURL != "/storage/K1" || res.Seed != 777 {
		t.Fatalf("got %+v", res)
	}
}

func TestGenerateReturnsErrorOnJobError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		wsjson.Read(r.Context(), conn, &sub)
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": sub["id"], "jobId": "J1"})
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:error", "jobId": "J1", "error": "Missing prompt"})
	}))
	defer srv.Close()

	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": ""})
	if err == nil {
		t.Fatal("expected error on job:error, got nil")
	}
	if !strings.Contains(err.Error(), "Missing prompt") {
		t.Fatalf("error should carry server message, got: %v", err)
	}
}

// TestGenerateDoesNotDeadlockOnManyProgress guards the buffered-progress design:
// the server can emit unbounded job:progress (server/ws_routes.py _on_job_update),
// but the progress channel is not read until Generate returns. Progress sends
// must not block, or Generate hangs forever past the buffer size.
func TestGenerateDoesNotDeadlockOnManyProgress(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		wsjson.Read(r.Context(), conn, &sub)
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": sub["id"], "jobId": "J1"})
		for range 50 { // far exceeds the 16-slot buffer
			wsjson.Write(r.Context(), conn, map[string]any{"type": "job:progress", "jobId": "J1", "progress": 0.1})
		}
		wsjson.Write(r.Context(), conn, map[string]any{
			"type": "job:complete", "jobId": "J1",
			"outputs": []any{map[string]any{"url": "/storage/K9", "key": "K9"}},
			"meta":    map[string]any{"seed": 1},
		})
	}))
	defer srv.Close()

	done := make(chan *Result, 1)
	go func() {
		_, res, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": "x"})
		if err == nil {
			done <- res
		} else {
			done <- nil
		}
	}()

	select {
	case res := <-done:
		if res == nil || res.StorageKey != "K9" {
			t.Fatalf("got %+v", res)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Generate deadlocked on excess job:progress frames")
	}
}
