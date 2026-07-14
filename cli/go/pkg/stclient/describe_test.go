package stclient

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func describeReq() DescribeRequest {
	u1, u2 := "http://x/a.png", "http://x/b.png"
	return DescribeRequest{
		Targets: []DescribeTarget{
			{ID: "t1", URL: &u1},
			{ID: "t2", URL: &u2},
		},
		Tasks: []DescribeTask{
			{ID: "caption", Kind: "caption", Caption: &CaptionParams{}},
		},
	}
}

func TestDescribeSendsWireShapeAndDecodes(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/describe" {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"status": "ok",
			"observations": [
				{"task_id": "caption", "target_id": "t1", "kind": "text",
				 "text": {"content": "stub:caption"}}
			],
			"runs": [
				{"task_id": "caption", "target_id": "t1",
				 "delegate": "vlm_caption", "status": "succeeded"},
				{"task_id": "caption", "target_id": "t2",
				 "delegate": "vlm_caption", "status": "succeeded"}
			]
		}`))
	}))
	defer srv.Close()

	resp, err := New(srv.URL).Describe(context.Background(), describeReq())
	if err != nil {
		t.Fatal(err)
	}
	// Wire-shape pin: request targets serialized in declaration order.
	targets := got["targets"].([]any)
	if len(targets) != 2 || targets[0].(map[string]any)["id"] != "t1" || targets[1].(map[string]any)["id"] != "t2" {
		t.Fatalf("target order not preserved: %v", targets)
	}
	if resp.Status != "ok" || len(resp.Runs) != 2 || resp.Runs[0].Delegate != "vlm_caption" {
		t.Fatalf("bad decode: %+v", resp)
	}
	if resp.Observations[0].Text.Content != "stub:caption" {
		t.Fatalf("bad observation decode: %+v", resp.Observations[0])
	}
}

func TestDescribeMapsAnalysisErrorToAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"code": "analysis_mode_not_found", "message": "unknown mode 'NOPE'"}}`))
	}))
	defer srv.Close()

	_, err := New(srv.URL).Describe(context.Background(), describeReq())
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if apiErr.Code != "analysis_mode_not_found" {
		t.Fatalf("bad code: %q", apiErr.Code)
	}
	if apiErr.Error() != "analysis_mode_not_found: unknown mode 'NOPE'" {
		t.Fatalf("bad Error(): %q", apiErr.Error())
	}
}

func TestDescribeNonJSONErrorBodyFallsBackToPlainError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("upstream exploded"))
	}))
	defer srv.Close()

	_, err := New(srv.URL).Describe(context.Background(), describeReq())
	if err == nil {
		t.Fatal("want error")
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		t.Fatalf("plain error expected for untyped body, got APIError %v", apiErr)
	}
}

func TestDescribeValidatesBeforeSending(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer srv.Close()

	_, err := New(srv.URL).Describe(context.Background(), DescribeRequest{})
	if err == nil {
		t.Fatal("want validation error")
	}
	if called {
		t.Fatal("invalid request must never leave the client")
	}
}
