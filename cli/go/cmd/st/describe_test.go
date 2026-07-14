package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

func TestClassifyTargetsAssignsPositionalIDsInArgOrder(t *testing.T) {
	specs := classifyTargets([]string{"./b.png", "https://x/a.png", "./c.png"})
	ids := []string{specs[0].id, specs[1].id, specs[2].id}
	if !reflect.DeepEqual(ids, []string{"t1", "t2", "t3"}) {
		t.Fatalf("positional IDs broken: %v", ids)
	}
	if specs[0].isURL || !specs[1].isURL || specs[2].isURL {
		t.Fatalf("URL classification broken: %+v", specs)
	}
	// Order is contract: arg order, never sorted.
	if specs[0].arg != "./b.png" || specs[2].arg != "./c.png" {
		t.Fatalf("arg order not preserved: %+v", specs)
	}
}

func TestBuildDescribeTasksCanonicalOrderRegardlessOfFlagOrder(t *testing.T) {
	tasks := buildDescribeTasks(describeOptions{detect: true, caption: true})
	if len(tasks) != 2 {
		t.Fatalf("want 2 tasks, got %d", len(tasks))
	}
	// Canonical TaskKind order: caption before detect, task id = kind string.
	if tasks[0].ID != "caption" || string(tasks[0].Kind) != "caption" || tasks[0].Caption == nil {
		t.Fatalf("task 0 not caption: %+v", tasks[0])
	}
	if tasks[1].ID != "detect" || string(tasks[1].Kind) != "detect" || tasks[1].Detect == nil {
		t.Fatalf("task 1 not detect: %+v", tasks[1])
	}
}

func TestBuildDescribeTasksCarriesParams(t *testing.T) {
	tasks := buildDescribeTasks(describeOptions{
		caption: true, prompt: "focus on lighting",
		detect: true, labels: []string{"person", "car"},
		minConfidence: 0.4, minConfidenceSet: true,
	})
	if tasks[0].Caption.Prompt == nil || *tasks[0].Caption.Prompt != "focus on lighting" {
		t.Fatalf("prompt not carried: %+v", tasks[0].Caption)
	}
	if !reflect.DeepEqual(tasks[1].Detect.Labels, []string{"person", "car"}) {
		t.Fatalf("labels not carried: %+v", tasks[1].Detect)
	}
	if tasks[1].Detect.MinConfidence == nil || *tasks[1].Detect.MinConfidence != 0.4 {
		t.Fatalf("min confidence not carried: %+v", tasks[1].Detect)
	}
}

func TestValidateDescribeFlagsUsageErrors(t *testing.T) {
	cases := []struct {
		name string
		o    describeOptions
	}{
		{"no task flags", describeOptions{}},
		{"prompt without caption", describeOptions{detect: true, prompt: "x"}},
		{"labels without detect", describeOptions{caption: true, labels: []string{"a"}}},
		{"min-confidence without detect", describeOptions{caption: true, minConfidenceSet: true}},
	}
	for _, tc := range cases {
		if err := validateDescribeFlags(tc.o); err == nil {
			t.Fatalf("%s: want usage error", tc.name)
		}
	}
	if err := validateDescribeFlags(describeOptions{caption: true}); err != nil {
		t.Fatalf("valid flags rejected: %v", err)
	}
}

func sampleResponse() *stclient.DescribeResponse {
	conf := 0.92
	return &stclient.DescribeResponse{
		Status: "partial",
		Observations: []stclient.DescribeObservation{
			{TaskID: "caption", TargetID: "t1", Kind: "text",
				Text: &stclient.TextObservation{Content: "a red bicycle"}},
			{TaskID: "detect", TargetID: "t1", Kind: "detection",
				Detection: &stclient.DetectionObservation{
					Label: "bicycle", Confidence: conf,
					Box: stclient.Box{X: 0.1, Y: 0.2, W: 0.5, H: 0.4},
				}},
		},
		Runs: []stclient.DescribeRun{
			{TaskID: "caption", TargetID: "t1", Delegate: "vlm_caption", Status: "succeeded"},
			{TaskID: "detect", TargetID: "t1", Delegate: "yolo_detect", Status: "succeeded"},
			{TaskID: "detect", TargetID: "t2", Delegate: "yolo_detect", Status: "failed",
				Error: &stclient.RunError{Code: "analysis_run_failed", Message: "conn refused"}},
		},
	}
}

func TestRenderDescribeHumanShowsCaptionAndDetections(t *testing.T) {
	var buf strings.Builder
	renderDescribeHuman(&buf, sampleResponse())
	out := buf.String()
	for _, want := range []string{"caption (t1): a red bicycle", "bicycle", "0.92"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestRenderDescribeFailuresListsEveryNonSucceededRun(t *testing.T) {
	var buf strings.Builder
	renderDescribeFailures(&buf, sampleResponse())
	out := buf.String()
	// Frozen content requirement: task, target, delegate, status, code, message.
	for _, want := range []string{"detect", "t2", "yolo_detect", "failed", "analysis_run_failed", "conn refused"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "t1") {
		t.Fatalf("succeeded runs must not be rendered as failures:\n%s", out)
	}
}

func TestDescribeStatusErrExitCodes(t *testing.T) {
	ok := &stclient.DescribeResponse{Status: "ok"}
	if err := describeStatusErr(ok); err != nil {
		t.Fatalf("ok must be exit 0, got %v", err)
	}
	partial := &stclient.DescribeResponse{Status: "partial"}
	if code := exitCodeOf(describeStatusErr(partial)); code != 3 {
		t.Fatalf("partial must exit 3, got %d", code)
	}
	failed := &stclient.DescribeResponse{Status: "failed"}
	if code := exitCodeOf(describeStatusErr(failed)); code != 2 {
		t.Fatalf("failed must exit 2, got %d", code)
	}
}

func TestDescribeEndToEndUploadsAndExitsPartial(t *testing.T) {
	dir := t.TempDir()
	img := filepath.Join(dir, "a.png")
	if err := os.WriteFile(img, []byte("fakepng"), 0o644); err != nil {
		t.Fatal(err)
	}

	var describeBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/upload":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"fileRef": "ref-123"}`))
		case "/v1/describe":
			_ = json.NewDecoder(r.Body).Decode(&describeBody)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"status": "partial",
				"observations": [
					{"task_id": "caption", "target_id": "t1", "kind": "text",
					 "text": {"content": "stub"}}
				],
				"runs": [
					{"task_id": "caption", "target_id": "t1", "delegate": "vlm_caption", "status": "succeeded"},
					{"task_id": "detect", "target_id": "t1", "delegate": "yolo_detect", "status": "failed",
					 "error": {"code": "analysis_run_failed", "message": "boom"}}
				]
			}`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	// runCmdCaptureWithStateRoot (gen_test.go) resets global flag state and
	// overrides the history state root — required for any CLI-driving test,
	// or globals leak across the ./cmd/st suite and history writes hit the
	// real user state dir.
	stdout, stderr, err := runCmdCaptureWithStateRoot(t, t.TempDir(),
		"describe", img, "--caption", "--detect", "--server", srv.URL)
	if code := exitCodeOf(err); code != 3 {
		t.Fatalf("partial must exit 3, got %d (%v)", code, err)
	}
	// The uploaded ref must land as target t1's asset_ref.
	target := describeBody["targets"].([]any)[0].(map[string]any)
	if target["id"] != "t1" || target["asset_ref"] != "ref-123" {
		t.Fatalf("upload not wired into target: %v", target)
	}
	if !strings.Contains(stdout, "stub") {
		t.Fatalf("caption not rendered to stdout:\n%s", stdout)
	}
	// Frozen failure-rendering contract, asserted on captured stderr.
	for _, want := range []string{"detect", "t1", "yolo_detect", "failed", "analysis_run_failed", "boom"} {
		if !strings.Contains(stderr, want) {
			t.Fatalf("failure line missing %q on stderr:\n%s", want, stderr)
		}
	}
}

func TestDescribeTransportFailureExitsOneWithMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close() // connection refused from here on

	_, _, err := runCmdCaptureWithStateRoot(t, t.TempDir(),
		"describe", "http://x/a.png", "--caption", "--server", srv.URL)
	if err == nil {
		t.Fatal("want transport error")
	}
	if code := exitCodeOf(err); code != 1 {
		t.Fatalf("transport failure must exit 1, got %d", code)
	}
	var apiErr *stclient.APIError
	if errors.As(err, &apiErr) {
		t.Fatalf("transport failure must not carry a synthetic code: %v", apiErr)
	}
	if err.Error() == "" {
		t.Fatal("transport failure must carry a human-readable message")
	}
}
