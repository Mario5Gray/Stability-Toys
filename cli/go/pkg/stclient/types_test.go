package stclient

import (
	"encoding/json"
	"testing"
)

func TestSubmitFrameShape(t *testing.T) {
	p := GenParams{"prompt": "owl", "size": "512x512"}
	f := newSubmitFrame("corr-1", p)
	b, _ := json.Marshal(f)
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if m["type"] != "job:submit" || m["jobType"] != "generate" || m["id"] != "corr-1" {
		t.Fatalf("bad envelope: %s", b)
	}
	if m["params"].(map[string]any)["prompt"] != "owl" {
		t.Fatalf("params not nested: %s", b)
	}
}

// TestInFrameDecodesComplete pins the inbound job:complete shape the WS client
// (T5) will match by jobId: outputs[{url,key}] + meta{seed,backend,sr}.
func TestInFrameDecodesComplete(t *testing.T) {
	raw := `{
		"type":"job:complete",
		"jobId":"abc123",
		"outputs":[{"url":"/storage/out-key.png","key":"out-key.png"}],
		"meta":{"seed":42,"backend":"mlx","sr":false}
	}`
	var f inFrame
	if err := json.Unmarshal([]byte(raw), &f); err != nil {
		t.Fatal(err)
	}
	if f.Type != "job:complete" || f.JobID != "abc123" {
		t.Fatalf("envelope: %+v", f)
	}
	if len(f.Outputs) != 1 || f.Outputs[0].Key != "out-key.png" || f.Outputs[0].URL != "/storage/out-key.png" {
		t.Fatalf("outputs: %+v", f.Outputs)
	}
	if f.Meta["backend"] != "mlx" {
		t.Fatalf("meta: %+v", f.Meta)
	}
}

func TestInFrameDecodesAckCorrelation(t *testing.T) {
	// Ack echoes the submit correlation id and supplies the server jobId.
	var f inFrame
	if err := json.Unmarshal([]byte(`{"type":"job:ack","id":"corr-1","jobId":"abc123"}`), &f); err != nil {
		t.Fatal(err)
	}
	if f.ID != "corr-1" || f.JobID != "abc123" {
		t.Fatalf("ack correlation: %+v", f)
	}
}
