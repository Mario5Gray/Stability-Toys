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

func pngWithLCM(t *testing.T, lcmJSON string) string {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	out, err := pngmeta.WriteText(buf.Bytes(), "lcm", lcmJSON)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "in.png")
	if err := os.WriteFile(p, out, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestReadPrintsLCM(t *testing.T) {
	path := pngWithLCM(t, `{"prompt":"owl","seed":42}`)
	out := runCmd(t, "read", path)
	if !strings.Contains(out, `"prompt"`) || !strings.Contains(out, "owl") {
		t.Fatalf("read output missing lcm fields: %q", out)
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
