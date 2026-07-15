package main

import (
	"testing"
)

func TestParseHeadSplitsOnFirstColon(t *testing.T) {
	ct, ref, err := parseHead("canny:fileref:MAP1")
	if err != nil {
		t.Fatal(err)
	}
	if ct != "canny" || ref != "fileref:MAP1" {
		t.Fatalf("got (%q, %q)", ct, ref)
	}
}

func TestParseHeadRejectsMissingColon(t *testing.T) {
	if _, _, err := parseHead("canny"); err == nil {
		t.Fatal("want error for missing colon")
	}
}

func TestBuildAttachmentMinimalDefaultsIdToControlType(t *testing.T) {
	a, err := buildAttachment(attachOpts{controlType: "canny", mapAssetRef: "Rmap1"})
	if err != nil {
		t.Fatal(err)
	}
	if a.AttachmentId != "canny" || a.ControlType != "canny" {
		t.Fatalf("ids: %+v", a)
	}
	if a.MapAssetRef == nil || *a.MapAssetRef != "Rmap1" {
		t.Fatalf("map: %+v", a.MapAssetRef)
	}
	// Unset optionals stay nil (omitted from JSON).
	if a.Strength != nil || a.StartPercent != nil || a.EndPercent != nil || a.ModelId != nil {
		t.Fatalf("optionals should be nil: %+v", a)
	}
}

func TestBuildAttachmentCarriesAllFields(t *testing.T) {
	a, err := buildAttachment(attachOpts{
		controlType: "depth", mapAssetRef: "Rmap2", model: "sdxl-depth", id: "d0",
		strength: 0.8, strengthSet: true,
		start: 0.1, startSet: true, end: 0.9, endSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if a.AttachmentId != "d0" || *a.ModelId != "sdxl-depth" {
		t.Fatalf("ids/model: %+v", a)
	}
	if a.Strength == nil || a.StartPercent == nil || a.EndPercent == nil {
		t.Fatalf("floats nil: %+v", a)
	}
	if *a.Strength != float32(0.8) || *a.StartPercent != float32(0.1) || *a.EndPercent != float32(0.9) {
		t.Fatalf("floats: %+v", a)
	}
}

func TestBuildAttachmentRejectsOutOfRange(t *testing.T) {
	cases := []attachOpts{
		{controlType: "canny", mapAssetRef: "R", strength: 2.5, strengthSet: true},
		{controlType: "canny", mapAssetRef: "R", strength: -0.1, strengthSet: true},
		{controlType: "canny", mapAssetRef: "R", start: 1.5, startSet: true},
		{controlType: "canny", mapAssetRef: "R", end: -0.1, endSet: true},
		{controlType: "canny", mapAssetRef: "R", start: 0.9, startSet: true, end: 0.1, endSet: true},
		{controlType: "", mapAssetRef: "R"},
		{controlType: "canny", mapAssetRef: ""},
	}
	for i, o := range cases {
		if _, err := buildAttachment(o); err == nil {
			t.Fatalf("case %d: want error", i)
		}
	}
}

func TestBuildAttachmentRejectsShellUnsafeFields(t *testing.T) {
	cases := []attachOpts{
		{controlType: "canny", mapAssetRef: "R map"},          // space in ref
		{controlType: "ca nny", mapAssetRef: "R"},             // space in type
		{controlType: "canny", mapAssetRef: "R", model: "m*"}, // glob in model
		{controlType: "canny", mapAssetRef: "R", id: "my id"}, // space in id
		{controlType: "canny", mapAssetRef: `R"x`},            // quote
		{controlType: "canny", mapAssetRef: "R$x"},            // dollar
	}
	for i, o := range cases {
		if _, err := buildAttachment(o); err == nil {
			t.Fatalf("case %d: want shell-safety error", i)
		}
	}
	// Colon, slash, dot, dash are allowed.
	if _, err := buildAttachment(attachOpts{controlType: "canny", mapAssetRef: "fileref:MAP-1/a.png"}); err != nil {
		t.Fatalf("safe ref rejected: %v", err)
	}
}
