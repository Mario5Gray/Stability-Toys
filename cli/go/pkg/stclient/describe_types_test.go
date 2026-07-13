package stclient

import (
	"encoding/json"
	"strings"
	"testing"
)

func strp(s string) *string { return &s }

func validDescribeRequest() DescribeRequest {
	return DescribeRequest{
		Targets: []DescribeTarget{{ID: "t1", AssetRef: strp("asset-1")}},
		Tasks:   []DescribeTask{{ID: "cap1", Kind: TaskKindCaption, Caption: &CaptionParams{}}},
	}
}

// Pins the wire shape of a full request per the spec's Request Contract.
func TestDescribeRequestWireShape(t *testing.T) {
	req := validDescribeRequest()
	req.Mode = strp("SDXL")
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if m["mode"] != "SDXL" {
		t.Fatalf("mode: %s", b)
	}
	tg := m["targets"].([]any)[0].(map[string]any)
	if tg["id"] != "t1" || tg["asset_ref"] != "asset-1" {
		t.Fatalf("target: %s", b)
	}
	if _, ok := tg["url"]; ok {
		t.Fatalf("nil url must be omitted: %s", b)
	}
	tk := m["tasks"].([]any)[0].(map[string]any)
	if tk["id"] != "cap1" || tk["kind"] != "caption" {
		t.Fatalf("task: %s", b)
	}
	if _, ok := tk["detect"]; ok {
		t.Fatalf("unset params blocks must be omitted: %s", b)
	}
}

// Pins the inbound response shape: status/runs/observations decode with
// task_id/target_id correlation and normalized payload blocks.
func TestDescribeResponseDecodes(t *testing.T) {
	raw := `{
		"status": "partial",
		"observations": [
			{"task_id":"cap1","target_id":"t1","kind":"text","text":{"content":"an owl"}},
			{"task_id":"det1","target_id":"t1","kind":"detection",
			 "detection":{"label":"owl","confidence":0.93,"box":{"x":0.1,"y":0.2,"w":0.3,"h":0.4}}}
		],
		"artifacts": [
			{"task_id":"emb1","target_id":"t1","kind":"embedding_ref","ref":"asset-9","dims":768}
		],
		"runs": [
			{"task_id":"cap1","target_id":"t1","delegate":"vlm_caption","status":"succeeded"},
			{"task_id":"det1","target_id":"t1","delegate":"yolo_detect","status":"succeeded"},
			{"task_id":"ocr1","target_id":"t1","delegate":"","status":"skipped",
			 "error":{"code":"analysis_no_supported_delegate","message":"no route for kind ocr"}}
		]
	}`
	var resp DescribeResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Status != StatusPartial {
		t.Fatalf("status: %+v", resp.Status)
	}
	if resp.Summary != nil {
		t.Fatalf("summary must be nil when absent")
	}
	if len(resp.Observations) != 2 || len(resp.Runs) != 3 {
		t.Fatalf("counts: %+v", resp)
	}
	o0 := resp.Observations[0]
	if o0.TaskID != "cap1" || o0.TargetID != "t1" || o0.Kind != ObsKindText || o0.Text.Content != "an owl" {
		t.Fatalf("text obs: %+v", o0)
	}
	o1 := resp.Observations[1]
	if o1.Detection == nil || o1.Detection.Box.W != 0.3 || o1.Detection.Confidence != 0.93 {
		t.Fatalf("detection obs: %+v", o1)
	}
	a0 := resp.Artifacts[0]
	if a0.Kind != ArtifactKindEmbeddingRef || a0.Ref != "asset-9" || a0.Dims == nil || *a0.Dims != 768 {
		t.Fatalf("artifact: %+v", a0)
	}
	r2 := resp.Runs[2]
	if r2.Status != RunSkipped || r2.Error == nil || r2.Error.Code != "analysis_no_supported_delegate" {
		t.Fatalf("skipped run: %+v", r2)
	}
}

func TestDescribeRequestValidate(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*DescribeRequest)
		wantErr string // substring; "" means valid
	}{
		{"valid", func(r *DescribeRequest) {}, ""},
		{"no targets", func(r *DescribeRequest) { r.Targets = nil }, "analysis_invalid_request"},
		{"no tasks", func(r *DescribeRequest) { r.Tasks = nil }, "analysis_invalid_request"},
		{"target both sources", func(r *DescribeRequest) {
			r.Targets[0].URL = strp("http://x/i.png")
		}, "analysis_invalid_request"},
		{"target neither source", func(r *DescribeRequest) {
			r.Targets[0].AssetRef = nil
		}, "analysis_invalid_request"},
		{"duplicate target id", func(r *DescribeRequest) {
			r.Targets = append(r.Targets, DescribeTarget{ID: "t1", URL: strp("http://x/i.png")})
		}, "analysis_invalid_request"},
		{"unknown kind", func(r *DescribeRequest) {
			r.Tasks[0].Kind = "segment"
		}, "analysis_invalid_request"},
		{"params kind mismatch", func(r *DescribeRequest) {
			r.Tasks[0].Caption = nil
			r.Tasks[0].Detect = &DetectParams{}
		}, "analysis_invalid_request"},
		{"two params blocks", func(r *DescribeRequest) {
			r.Tasks[0].Detect = &DetectParams{}
		}, "analysis_invalid_request"},
		{"unknown target id", func(r *DescribeRequest) {
			r.Tasks[0].TargetIDs = []string{"nope"}
		}, "analysis_target_binding_invalid"},
		{"zero-run: no primary targets", func(r *DescribeRequest) {
			r.Targets[0].Role = "reference"
		}, "analysis_target_binding_invalid"},
		{"explicit binding to non-primary is allowed", func(r *DescribeRequest) {
			r.Targets[0].Role = "reference"
			r.Tasks[0].TargetIDs = []string{"t1"}
		}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := validDescribeRequest()
			tc.mutate(&req)
			err := req.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected valid, got %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("want %q in error, got %v", tc.wantErr, err)
			}
		})
	}
}
