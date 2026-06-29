package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

func TestCancelSendsJobCancel(t *testing.T) {
	got := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var f map[string]any
		wsjson.Read(r.Context(), conn, &f)
		got <- f
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:cancel:ack", "id": f["id"], "jobId": f["jobId"], "detail": "canceled"})
	}))
	defer srv.Close()

	if err := New(srv.URL).Cancel(context.Background(), "J1"); err != nil {
		t.Fatal(err)
	}
	f := <-got
	if f["type"] != "job:cancel" {
		t.Fatalf("type = %v, want job:cancel", f["type"])
	}
	if f["jobId"] != "J1" {
		t.Fatalf("jobId = %v, want J1", f["jobId"])
	}
}

func TestSetPrioritySendsJobPriority(t *testing.T) {
	got := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var f map[string]any
		wsjson.Read(r.Context(), conn, &f)
		got <- f
		// Server stub (ws_routes.py:296) acks without reading any priority field.
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:priority:ack", "id": f["id"], "detail": "priority not yet implemented"})
	}))
	defer srv.Close()

	if err := New(srv.URL).SetPriority(context.Background(), "J2", 5); err != nil {
		t.Fatal(err)
	}
	f := <-got
	if f["type"] != "job:priority" {
		t.Fatalf("type = %v, want job:priority", f["type"])
	}
	if f["jobId"] != "J2" {
		t.Fatalf("jobId = %v, want J2", f["jobId"])
	}
	if f["priority"] != float64(5) {
		t.Fatalf("priority = %v, want 5", f["priority"])
	}
}

func TestControlFrameReturnsErrorOnJobError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var f map[string]any
		wsjson.Read(r.Context(), conn, &f)
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:error", "error": "boom"})
	}))
	defer srv.Close()

	if err := New(srv.URL).Cancel(context.Background(), "J1"); err == nil {
		t.Fatal("expected error when server replies job:error")
	}
}
