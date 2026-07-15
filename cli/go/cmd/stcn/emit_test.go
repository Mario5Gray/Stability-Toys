package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/openapi"
)

func TestMarshalCompactHasNoWhitespace(t *testing.T) {
	a, err := buildAttachment(attachOpts{
		controlType: "canny", mapAssetRef: "fileref:MAP1",
		strength: 0.8, strengthSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := marshalCompact(a)
	if err != nil {
		t.Fatal(err)
	}
	// Single-token pin: no space/tab/newline anywhere in the emitted bytes.
	if bytes.ContainsAny(out, " \t\n\r") {
		t.Fatalf("output has whitespace, would word-split: %q", out)
	}
}

func TestMarshalCompactRoundTripsIntoSchemaType(t *testing.T) {
	a, err := buildAttachment(attachOpts{
		controlType: "depth", mapAssetRef: "Rmap2", model: "sdxl-depth", id: "d0",
		strength: 0.5, strengthSet: true, start: 0.2, startSet: true, end: 0.8, endSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := marshalCompact(a)
	if err != nil {
		t.Fatal(err)
	}
	var back openapi.ControlNetAttachment
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatalf("does not round-trip into schema type: %v", err)
	}
	if back.AttachmentId != "d0" || back.ControlType != "depth" ||
		back.MapAssetRef == nil || *back.MapAssetRef != "Rmap2" ||
		back.ModelId == nil || *back.ModelId != "sdxl-depth" ||
		back.Strength == nil || back.StartPercent == nil || back.EndPercent == nil {
		t.Fatalf("fields lost in round-trip: %+v", back)
	}
	// Unset optionals must not appear as keys (omitempty).
	if strings.Contains(string(out), "source_asset_ref") || strings.Contains(string(out), "preprocess") {
		t.Fatalf("nil optionals leaked into JSON: %s", out)
	}
}
