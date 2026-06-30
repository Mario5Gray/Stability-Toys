package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"

	"github.com/darkbit/stability-toys/cli/st/internal/config"
)

func TestBuildGenParamsFromArgs(t *testing.T) {
	args := genArgs{Prompt: "an owl", Cfg: f64p(3.0), Genres: strp("768x768"), InitImage: "fileref:R1"}
	p, err := buildGenParams(nil /*cfg*/, args)
	if err != nil {
		t.Fatal(err)
	}
	if p["prompt"] != "an owl" || p["guidance_scale"] != 3.0 || p["size"] != "768x768" {
		t.Fatalf("params: %+v", p)
	}
	if p["init_image_ref"] != "R1" {
		t.Fatalf("fileref not threaded: %+v", p)
	}
}

// TestBuildGenParamsControlnetsJSON pins that each --controlnet value is parsed
// as a structured ControlNetAttachment (JSON) and threaded under controlnets.
func TestBuildGenParamsControlnetsJSON(t *testing.T) {
	cn := `{"attachment_id":"a1","control_type":"canny","map_asset_ref":"fileref:M1"}`
	args := genArgs{Prompt: "x", Controlnets: []string{cn}}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("controlnets: %+v", p["controlnets"])
	}
	entry, _ := list[0].(map[string]any)
	if entry["control_type"] != "canny" || entry["attachment_id"] != "a1" {
		t.Fatalf("entry: %+v", entry)
	}
}

// TestGenWritesOutputFile is the end-to-end spine: gen submits over WS, fetches
// the storage bytes, and writes them to the auto-incremented output path.
func TestGenWritesOutputFile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/storage/") {
			w.Write([]byte("PNGBYTES"))
			return
		}
		conn, _ := websocket.Accept(w, r, nil)
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		wsjson.Read(r.Context(), conn, &sub)
		wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": sub["id"], "jobId": "J1"})
		wsjson.Write(r.Context(), conn, map[string]any{
			"type": "job:complete", "jobId": "J1",
			"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
			"meta":    map[string]any{"seed": 1},
		})
	}))
	defer srv.Close()

	outDir := t.TempDir()
	cfgPath := writeTestConfig(t, outDir)

	runCmd(t, "gen", "--server", srv.URL, "--config", cfgPath, "-o", outDir, "an owl")

	data, err := os.ReadFile(filepath.Join(outDir, "out-0001.png"))
	if err != nil {
		t.Fatalf("output not written: %v", err)
	}
	if string(data) != "PNGBYTES" {
		t.Fatalf("wrong bytes written: %q", data)
	}
}

func writeTestConfig(t *testing.T, outDir string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.json")
	body := `{"config":{"defaults":{"generation":{"genres":"512x512"},"output_format":"png","output_directory":"` + outDir + `","include_meta":false}}}`
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func runCmd(t *testing.T, args ...string) string {
	t.Helper()
	var sb strings.Builder
	rootCmd.SetOut(&sb)
	rootCmd.SetErr(&sb)
	rootCmd.SetArgs(args)
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("execute %v: %v\noutput: %s", args, err, sb.String())
	}
	return sb.String()
}

func TestBuildGenParamsControlnetFile(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "cn-*.json")
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(`{"attachment_id":"a2","control_type":"depth","map_asset_ref":"fileref:D1"}`)
	f.Close()

	args := genArgs{Prompt: "x", ControlnetFile: f.Name()}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("controlnets: %+v", p["controlnets"])
	}
	entry, _ := list[0].(map[string]any)
	if entry["control_type"] != "depth" {
		t.Fatalf("control_type = %v, want depth", entry["control_type"])
	}
}

func TestBuildGenParamsControlnetFileMergesWithFlag(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "cn-*.json")
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(`{"attachment_id":"file-cn","control_type":"depth"}`)
	f.Close()

	cn := `{"attachment_id":"flag-cn","control_type":"canny"}`
	args := genArgs{Prompt: "x", Controlnets: []string{cn}, ControlnetFile: f.Name()}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 2 {
		t.Fatalf("expected 2 controlnets, got: %+v", p["controlnets"])
	}
}

func TestBuildGenParamsControlnetPreset(t *testing.T) {
	cfg := &config.Config{}
	cfg.ControlnetPresets = map[string]config.ControlnetPreset{
		"owl-canny": {"attachment_id": "cn-1", "control_type": "canny", "map_asset_ref": "fileref:D1"},
	}
	args := genArgs{Prompt: "x", Controlnets: []string{"@owl-canny"}}
	p, err := buildGenParams(cfg, args)
	if err != nil {
		t.Fatal(err)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("controlnets: %+v", p["controlnets"])
	}
	entry, _ := list[0].(map[string]any)
	if entry["control_type"] != "canny" {
		t.Fatalf("control_type = %v, want canny", entry["control_type"])
	}
}

func TestBuildGenParamsControlnetPresetMissingErrors(t *testing.T) {
	args := genArgs{Prompt: "x", Controlnets: []string{"@unknown"}}
	_, err := buildGenParams(nil, args)
	if err == nil {
		t.Fatal("expected error for unknown preset, got nil")
	}
	if !strings.Contains(err.Error(), "@unknown") {
		t.Errorf("error should name the preset, got: %v", err)
	}
}

func strp(s string) *string   { return &s }
func f64p(f float64) *float64 { return &f }
