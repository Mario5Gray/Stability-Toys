package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

// fakeGenServer spins up a test WS server that reads one job:submit and
// then sends the provided frames in order. Handlers that do not match
// /v1/ws fall through (e.g. /storage/ for FetchStorage in gen_test.go).
func fakeGenServer(t *testing.T, frames []any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/ws") {
			http.NotFound(w, r)
			return
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("WS accept: %v", err)
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		wsjson.Read(r.Context(), conn, &sub)
		for _, f := range frames {
			if err := wsjson.Write(r.Context(), conn, f); err != nil {
				return
			}
		}
	}))
}

func TestGenerateResolvesOnComplete(t *testing.T) {
	srv := fakeGenServer(t, []any{
		map[string]any{"type": "job:ack", "id": "corr1", "jobId": "J1"},
		map[string]any{
			"type":    "job:complete",
			"jobId":   "J1",
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta":    map[string]any{"seed": float64(777)},
		},
	})
	defer srv.Close()

	_, res, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": "owl"}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.StorageKey != "K1" || res.StorageURL != "/storage/K1" || res.Seed != 777 {
		t.Fatalf("got %+v", res)
	}
}

func TestGenerateReturnsJobID(t *testing.T) {
	srv := fakeGenServer(t, []any{
		map[string]any{"type": "job:ack", "id": "corr1", "jobId": "job-abc"},
		map[string]any{
			"type":    "job:complete",
			"jobId":   "job-abc",
			"outputs": []any{map[string]any{"url": "/storage/K2", "key": "K2"}},
			"meta":    map[string]any{"seed": float64(1)},
		},
	})
	defer srv.Close()

	var gotAck string
	jobID, _, err := New(srv.URL).Generate(context.Background(), GenParams{}, func(id string) { gotAck = id }, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gotAck != "job-abc" {
		t.Errorf("onAck got %q, want job-abc", gotAck)
	}
	if jobID != "job-abc" {
		t.Errorf("returned jobID = %q, want job-abc", jobID)
	}
}

func TestGenerateReturnsErrorOnJobError(t *testing.T) {
	srv := fakeGenServer(t, []any{
		map[string]any{"type": "job:ack", "id": "corr1", "jobId": "J1"},
		map[string]any{"type": "job:error", "jobId": "J1", "error": "Missing prompt"},
	})
	defer srv.Close()

	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": ""}, nil, nil)
	if err == nil {
		t.Fatal("expected error on job:error, got nil")
	}
	if !strings.Contains(err.Error(), "Missing prompt") {
		t.Fatalf("error should carry server message, got: %v", err)
	}
}

func TestGenerateSkipsOnProgressForEmptyDelta(t *testing.T) {
	srv := fakeGenServer(t, []any{
		map[string]any{"type": "job:ack", "jobId": "J1"},
		// frame with no delta field at all (legacy server / non-image path)
		map[string]any{"type": "job:progress", "jobId": "J1"},
		// frame with explicitly empty delta
		map[string]any{"type": "job:progress", "jobId": "J1", "delta": ""},
		// frame with a real delta — only this one should fire the callback
		map[string]any{"type": "job:progress", "jobId": "J1", "delta": "node 1/4 (25%)"},
		map[string]any{
			"type":    "job:complete",
			"jobId":   "J1",
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta":    map[string]any{"seed": float64(1)},
		},
	})
	defer srv.Close()

	var received []string
	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{}, nil, func(delta string) {
		received = append(received, delta)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(received) != 1 {
		t.Fatalf("expected 1 onProgress call (non-empty delta only), got %d: %v", len(received), received)
	}
	if received[0] != "node 1/4 (25%)" {
		t.Errorf("delta = %q, want %q", received[0], "node 1/4 (25%)")
	}
}

func TestGeneratePassesDeltaValueToCallback(t *testing.T) {
	deltas := []string{"node 1/6 (16%)", "node 2/6 (33%)", "node 3/6 (50%)"}
	frames := []any{
		map[string]any{"type": "job:ack", "jobId": "J1"},
	}
	for _, d := range deltas {
		frames = append(frames, map[string]any{"type": "job:progress", "jobId": "J1", "delta": d})
	}
	frames = append(frames, map[string]any{
		"type":    "job:complete",
		"jobId":   "J1",
		"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
		"meta":    map[string]any{"seed": float64(42)},
	})
	srv := fakeGenServer(t, frames)
	defer srv.Close()

	var received []string
	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{}, nil, func(delta string) {
		received = append(received, delta)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(received) != len(deltas) {
		t.Fatalf("got %d callbacks, want %d", len(received), len(deltas))
	}
	for i, want := range deltas {
		if received[i] != want {
			t.Errorf("callback[%d] = %q, want %q", i, received[i], want)
		}
	}
}

func TestGenerateCallsOnProgressForAllFrames(t *testing.T) {
	const n = 50
	frames := []any{
		map[string]any{"type": "job:ack", "jobId": "J1"},
	}
	for range n {
		frames = append(frames, map[string]any{"type": "job:progress", "jobId": "J1", "delta": "x"})
	}
	frames = append(frames, map[string]any{
		"type":    "job:complete",
		"jobId":   "J1",
		"outputs": []any{map[string]any{"url": "/storage/K9", "key": "K9"}},
		"meta":    map[string]any{},
	})
	srv := fakeGenServer(t, frames)
	defer srv.Close()

	count := 0
	_, _, err := New(srv.URL).Generate(context.Background(), GenParams{"prompt": "x"}, nil, func(string) { count++ })
	if err != nil {
		t.Fatal(err)
	}
	if count != n {
		t.Errorf("onProgress called %d times, want %d", count, n)
	}
}
