package openapi

import "testing"

func TestGenerateRequestTypeExists(t *testing.T) {
	var r GenerateRequest
	r.Prompt = "x"
	if r.Prompt != "x" {
		t.Fatalf("Prompt field not wired")
	}
}
