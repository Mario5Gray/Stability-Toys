package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"

	"github.com/darkbit/stability-toys/cli/st/internal/config"
	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
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

// runCmdMayFail is like runCmd but returns the error instead of fataling.
func runCmdMayFail(t *testing.T, args ...string) (string, error) {
	t.Helper()
	var sb strings.Builder
	rootCmd.SetOut(&sb)
	rootCmd.SetErr(&sb)
	rootCmd.SetArgs(args)
	err := rootCmd.Execute()
	return sb.String(), err
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
	// flag entries are first, file entry is last
	first, _ := list[0].(map[string]any)
	if first["attachment_id"] != "flag-cn" {
		t.Errorf("list[0] should be flag entry (flag-cn), got attachment_id=%v", first["attachment_id"])
	}
	second, _ := list[1].(map[string]any)
	if second["attachment_id"] != "file-cn" {
		t.Errorf("list[1] should be file entry (file-cn), got attachment_id=%v", second["attachment_id"])
	}
}

func TestBuildGenParamsControlnetFileBadPath(t *testing.T) {
	args := genArgs{Prompt: "x", ControlnetFile: "/nonexistent/path/cn.json"}
	_, err := buildGenParams(nil, args)
	if err == nil {
		t.Fatal("expected error for bad --controlnet-file path, got nil")
	}
	if !strings.Contains(err.Error(), "/nonexistent/path/cn.json") {
		t.Errorf("error should name the bad path, got: %v", err)
	}
}

func TestBuildGenParamsControlnetFileInvalidJSON(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "cn-*.json")
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("not valid json {{{")
	f.Close()

	args := genArgs{Prompt: "x", ControlnetFile: f.Name()}
	_, err = buildGenParams(nil, args)
	if err == nil {
		t.Fatal("expected error for invalid JSON in --controlnet-file, got nil")
	}
	if !strings.Contains(err.Error(), "invalid JSON") {
		t.Errorf("error should mention 'invalid JSON', got: %v", err)
	}
}

// TestGenControlImageUploadsAndAttaches verifies that --control-image depth:./f.png
// uploads the file then injects a controlnet attachment with map_asset_ref set.
func TestGenControlImageUploadsAndAttaches(t *testing.T) {
	imgFile := filepath.Join(t.TempDir(), "ctrl.png")
	if err := os.WriteFile(imgFile, []byte("FAKEPNG"), 0o644); err != nil {
		t.Fatal(err)
	}

	var capturedControlnets []any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/upload":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"fileRef":"fileref:CTRLREF"}`))
		case strings.HasPrefix(r.URL.Path, "/storage/"):
			w.Write([]byte("PNGBYTES"))
		default:
			conn, _ := websocket.Accept(w, r, nil)
			defer conn.Close(websocket.StatusNormalClosure, "")
			var sub map[string]any
			wsjson.Read(r.Context(), conn, &sub)
			if params, ok := sub["params"].(map[string]any); ok {
				if cns, ok := params["controlnets"].([]any); ok {
					capturedControlnets = cns
				}
			}
			wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": sub["id"], "jobId": "J1"})
			wsjson.Write(r.Context(), conn, map[string]any{
				"type": "job:complete", "jobId": "J1",
				"outputs": []any{map[string]any{"url": "/storage/K1", "key": "K1"}},
				"meta":    map[string]any{"seed": 1},
			})
		}
	}))
	defer srv.Close()

	outDir := t.TempDir()
	cfgPath := writeTestConfig(t, outDir)
	runCmd(t, "gen", "--server", srv.URL, "--config", cfgPath, "-o", outDir,
		"--control-image", "depth:"+imgFile, "an owl")

	if len(capturedControlnets) != 1 {
		t.Fatalf("expected 1 controlnet attachment, got %d: %v", len(capturedControlnets), capturedControlnets)
	}
	entry, _ := capturedControlnets[0].(map[string]any)
	if entry["control_type"] != "depth" {
		t.Errorf("control_type = %v, want depth", entry["control_type"])
	}
	if entry["map_asset_ref"] != "fileref:CTRLREF" {
		t.Errorf("map_asset_ref = %v, want fileref:CTRLREF", entry["map_asset_ref"])
	}
	if entry["attachment_id"] == "" {
		t.Error("attachment_id must not be empty")
	}
}

func TestControlImageRequiresTypePrefix(t *testing.T) {
	err := resolveControlImages(context.Background(), nil, []string{"/some/file.png"}, stclient.GenParams{})
	if err == nil || !strings.Contains(err.Error(), "control_type") {
		t.Errorf("expected error about missing control_type, got: %v", err)
	}
}

func TestBuildObservationCallbacksStreamQuietReturnsBothNil(t *testing.T) {
	onAck, onProg := buildObservationCallbacks(genCmd, true /* quiet */, true /* stream */)
	if onAck != nil || onProg != nil {
		t.Fatal("--stream --quiet: expected both callbacks nil (quiet takes precedence over stream)")
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

func TestBuildGenParamsSkipStep(t *testing.T) {
	args := genArgs{Prompt: "an owl", SkipStep: intp(4)}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	if p["skip_step"] != 4 {
		t.Fatalf("skip_step=%v, want 4", p["skip_step"])
	}
}

func TestBuildGenParamsSkipStepZeroOmitted(t *testing.T) {
	args := genArgs{Prompt: "an owl"}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := p["skip_step"]; ok {
		t.Fatalf("skip_step should be absent when nil, got %v", p["skip_step"])
	}
}

func strp(s string) *string   { return &s }
func f64p(f float64) *float64 { return &f }
func intp(i int) *int         { return &i }
