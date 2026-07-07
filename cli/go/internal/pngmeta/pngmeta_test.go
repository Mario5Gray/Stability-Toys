package pngmeta

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

func makePNGWithText(t *testing.T, key, val string) []byte {
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	out, err := WriteText(buf.Bytes(), key, val)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestWriteThenReadLCM(t *testing.T) {
	pngBytes := makePNGWithText(t, "lcm", `{"prompt":"owl","seed":42,"cfg":2.5}`)
	m, err := ReadLCM(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	if m["prompt"] != "owl" {
		t.Fatalf("got %+v", m)
	}
}

// TestWriteTextKeepsValidPNG guards that inserting the tEXt chunk before IEND
// produces bytes the stdlib decoder still accepts (correct length + CRC).
func TestWriteTextKeepsValidPNG(t *testing.T) {
	pngBytes := makePNGWithText(t, "lcm", `{"prompt":"x"}`)
	if _, err := png.Decode(bytes.NewReader(pngBytes)); err != nil {
		t.Fatalf("output not a valid PNG: %v", err)
	}
}

// TestBakedParamsMapsToRequestFields pins the lcm->GenerateRequest renaming used
// by precedence layer 2 (cfg->guidance_scale, steps->num_inference_steps).
func TestParseThenFindLCM(t *testing.T) {
	pngBytes := makePNGWithText(t, "lcm", `{"prompt":"owl","seed":42}`)
	chunks, err := Parse(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	m, ok, err := chunks.FindLCM()
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected FindLCM ok=true")
	}
	if m["prompt"] != "owl" {
		t.Fatalf("got %+v", m)
	}
}

func TestFindLCMAbsentIsNotError(t *testing.T) {
	pngBytes := makePNGWithText(t, "other", `{"x":1}`)
	chunks, err := Parse(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	_, ok, err := chunks.FindLCM()
	if err != nil {
		t.Fatalf("absence must not be an error: %v", err)
	}
	if ok {
		t.Fatal("expected FindLCM ok=false when lcm chunk absent")
	}
}

func TestReadLCMErrorsWhenAbsent(t *testing.T) {
	pngBytes := makePNGWithText(t, "other", `{"x":1}`)
	if _, err := ReadLCM(pngBytes); err == nil {
		t.Fatal("expected error when lcm chunk absent")
	}
}

func TestBakedParamsMapsToRequestFields(t *testing.T) {
	pngBytes := makePNGWithText(t, "lcm", `{"prompt":"owl","cfg":2.5,"steps":10,"unrelated":"drop"}`)
	out, err := BakedParams(pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	if out["guidance_scale"] != 2.5 || out["num_inference_steps"] != float64(10) || out["prompt"] != "owl" {
		t.Fatalf("got %+v", out)
	}
	if _, ok := out["unrelated"]; ok {
		t.Fatalf("only mapped keys should survive: %+v", out)
	}
}
