package main

import (
	"bytes"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/pngmeta"
)

func pngWithText(t *testing.T, keyword, jsonText string) string {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	out, err := pngmeta.WriteText(buf.Bytes(), keyword, jsonText)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "in.png")
	if err := os.WriteFile(p, out, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func pngWithLCM(t *testing.T, lcmJSON string) string {
	return pngWithText(t, "lcm", lcmJSON)
}

func TestReadPrintsLCMWrapped(t *testing.T) {
	path := pngWithLCM(t, `{"prompt":"owl","seed":42}`)
	out := runCmd(t, "read", path)
	if !strings.Contains(out, `"lcm"`) {
		t.Fatalf("read output missing lcm wrapper key: %q", out)
	}
	if !strings.Contains(out, `"prompt"`) || !strings.Contains(out, "owl") {
		t.Fatalf("read output missing lcm fields: %q", out)
	}
}

func TestReadPrintsControlNetMap(t *testing.T) {
	path := pngWithText(t, "controlnet_map", `{"tool":"canny_map","control_type":"canny"}`)
	out := runCmd(t, "read", path)
	if !strings.Contains(out, `"controlnet_map"`) || !strings.Contains(out, "canny_map") {
		t.Fatalf("read output missing controlnet_map fields: %q", out)
	}
}

func TestReadPrintsLCMAndControlNetTogether(t *testing.T) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	out1, err := pngmeta.WriteText(buf.Bytes(), "lcm", `{"prompt":"owl"}`)
	if err != nil {
		t.Fatal(err)
	}
	out2, err := pngmeta.WriteText(out1, "controlnet", `[{"attachment_id":"cn_1","control_type":"canny"}]`)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "both.png")
	if err := os.WriteFile(p, out2, 0o644); err != nil {
		t.Fatal(err)
	}

	got := runCmd(t, "read", p)
	if !strings.Contains(got, `"lcm"`) || !strings.Contains(got, `"controlnet"`) {
		t.Fatalf("read output missing one of lcm/controlnet: %q", got)
	}
	if !strings.Contains(got, "cn_1") {
		t.Fatalf("read output missing controlnet entry: %q", got)
	}
}

func TestReadErrorsWhenNoKnownChunks(t *testing.T) {
	path := pngWithText(t, "unrelated", `{"x":1}`)
	if _, err := runCmdMayFail(t, "read", path); err == nil {
		t.Fatal("expected error when no known metadata chunk present")
	}
}

// TestRecreateSeedsParams characterizes the --recreate baked layer wired in T12:
// the PNG's lcm params seed generation, and explicit flags still override them.
func TestRecreateSeedsParams(t *testing.T) {
	path := pngWithLCM(t, `{"prompt":"base","cfg":9}`)

	p, err := buildGenParams(nil, genArgs{Recreate: path})
	if err != nil {
		t.Fatal(err)
	}
	if p["guidance_scale"] != float64(9) || p["prompt"] != "base" {
		t.Fatalf("baked layer not applied: %+v", p)
	}
	if _, ok := p["init_image_ref"]; ok {
		t.Fatalf("recreate is recipe-only, must not set init_image_ref: %+v", p)
	}

	p2, err := buildGenParams(nil, genArgs{Recreate: path, Cfg: f64p(1)})
	if err != nil {
		t.Fatal(err)
	}
	if p2["guidance_scale"] != 1.0 {
		t.Fatalf("flag should override baked cfg: %+v", p2)
	}
}
