package output

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNextPathIncrements(t *testing.T) {
	dir := t.TempDir()
	p1, _ := NextPath(dir, "png")
	if filepath.Base(p1) != "out-0001.png" {
		t.Fatalf("got %s", p1)
	}
	os.WriteFile(p1, []byte("x"), 0o644)
	p2, _ := NextPath(dir, "png")
	if filepath.Base(p2) != "out-0002.png" {
		t.Fatalf("got %s", p2)
	}
}

func TestResolveOutfileAppendsExt(t *testing.T) {
	got, _ := Resolve("/tmp/pic", "/tmp", "png")
	if got != "/tmp/pic.png" {
		t.Fatalf("got %s", got)
	}
}

// TestResolveRelativeOutfileJoinsDir pins that a bare (non-abs) --outfile with an
// extension is placed under the output directory.
func TestResolveRelativeOutfileJoinsDir(t *testing.T) {
	got, _ := Resolve("pic.png", "/tmp/out", "png")
	if got != filepath.Join("/tmp/out", "pic.png") {
		t.Fatalf("got %s", got)
	}
}
