package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

// execCmd runs the root command and returns (output, err) without failing the
// test — used to assert non-zero exits (RunE error => main os.Exit(1)).
func execCmd(args ...string) (string, error) {
	var sb strings.Builder
	rootCmd.SetOut(&sb)
	rootCmd.SetErr(&sb)
	rootCmd.SetArgs(args)
	err := rootCmd.Execute()
	return sb.String(), err
}

type track3Calls struct {
	mu              sync.Mutex
	uploaded        bool
	submitSeen      bool
	uploadBeforeSub bool
	controlnetCount int
}

// track3Server mocks the upload + WS generate sequence. When emitArtifacts is
// true the job:complete frame carries a controlnet_artifacts entry.
func track3Server(t *testing.T, emitArtifacts bool) (*httptest.Server, *track3Calls) {
	t.Helper()
	calls := &track3Calls{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/upload" {
			calls.mu.Lock()
			calls.uploaded = true
			calls.mu.Unlock()
			w.Write([]byte(`{"fileRef":"M1"}`))
			return
		}
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		wsjson.Read(r.Context(), conn, &sub)
		calls.mu.Lock()
		calls.submitSeen = true
		calls.uploadBeforeSub = calls.uploaded
		if params, ok := sub["params"].(map[string]any); ok {
			if cns, ok := params["controlnets"].([]any); ok {
				calls.controlnetCount = len(cns)
			}
		}
		calls.mu.Unlock()
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": sub["id"], "jobId": "J1"})
		complete := map[string]any{
			"type": "job:complete", "jobId": "J1",
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta":    map[string]any{"seed": 1},
		}
		if emitArtifacts {
			complete["controlnet_artifacts"] = []any{map[string]any{
				"attachment_id": "track3-1", "asset_ref": "A1", "control_type": "canny",
				"preprocessor_id": "p", "source_asset_ref": "S1",
			}}
		}
		wsjson.Write(r.Context(), conn, complete)
	}))
	return srv, calls
}

func TestValidateTrack3PassesWhenArtifactsPresent(t *testing.T) {
	srv, calls := track3Server(t, true)
	defer srv.Close()

	out, err := execCmd("validate-track3", "--server", srv.URL, "--control-image", writeTempFile(t, "map.png", "x"))
	if err != nil {
		t.Fatalf("expected pass, got error: %v\n%s", err, out)
	}
	if !strings.Contains(out, "PASS") {
		t.Fatalf("expected PASS summary, got %q", out)
	}
	if !calls.uploadBeforeSub {
		t.Fatal("upload must precede job:submit")
	}
	if calls.controlnetCount != 1 {
		t.Fatalf("submit should carry one controlnet, got %d", calls.controlnetCount)
	}
}

func TestValidateTrack3FailsWhenArtifactsAbsent(t *testing.T) {
	srv, _ := track3Server(t, false)
	defer srv.Close()

	_, err := execCmd("validate-track3", "--server", srv.URL, "--control-image", writeTempFile(t, "map.png", "x"))
	if err == nil {
		t.Fatal("expected non-zero exit when no controlnet_artifacts returned")
	}
}
