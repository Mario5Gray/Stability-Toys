package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/darkbit/stability-toys/cli/st/internal/config"
	"github.com/darkbit/stability-toys/cli/st/internal/history"
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

func TestBuildGenParamsControlRef(t *testing.T) {
	args := genArgs{
		Prompt:          "x",
		ControlRefs:     []string{"canny:fileref:M1"},
		ControlStrength: f64p(0.8),
	}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("controlnets: %+v", p["controlnets"])
	}
	entry, _ := list[0].(map[string]any)
	if entry["control_type"] != "canny" || entry["map_asset_ref"] != "fileref:M1" {
		t.Fatalf("entry: %+v", entry)
	}
	if entry["strength"] != 0.8 {
		t.Fatalf("strength = %v, want 0.8", entry["strength"])
	}
	if entry["attachment_id"] == "" {
		t.Fatalf("attachment_id must not be empty: %+v", entry)
	}
}

func TestBuildGenParamsControlRefRequiresTypePrefix(t *testing.T) {
	args := genArgs{Prompt: "x", ControlRefs: []string{":fileref:M1"}}
	_, err := buildGenParams(nil, args)
	if err == nil || !strings.Contains(err.Error(), "missing control_type") {
		t.Fatalf("err = %v", err)
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

type genReply struct {
	Error string
	Seed  int64
}

func newScriptedGenServer(t *testing.T, replies ...genReply) (*httptest.Server, *[]stclient.GenParams) {
	t.Helper()
	var mu sync.Mutex
	index := 0
	captured := []stclient.GenParams{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/storage/") {
			_, _ = w.Write([]byte("PNGBYTES"))
			return
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "")
		var sub map[string]any
		if err := wsjson.Read(r.Context(), conn, &sub); err != nil {
			return
		}
		params, _ := sub["params"].(map[string]any)
		mu.Lock()
		captured = append(captured, stclient.GenParams(params))
		reply := replies[index]
		index++
		mu.Unlock()
		_ = wsjson.Write(r.Context(), conn, map[string]any{"type": "job:ack", "id": sub["id"], "jobId": fmt.Sprintf("J%d", index)})
		if reply.Error != "" {
			_ = wsjson.Write(r.Context(), conn, map[string]any{"type": "job:error", "error": reply.Error})
			return
		}
		_ = wsjson.Write(r.Context(), conn, map[string]any{
			"type":  "job:complete",
			"jobId": fmt.Sprintf("J%d", index),
			"outputs": []any{map[string]any{
				"url": "/storage/K1",
				"key": "K1",
			}},
			"meta": map[string]any{"seed": reply.Seed},
		})
	}))
	return srv, &captured
}

func runCmd(t *testing.T, args ...string) string {
	t.Helper()
	out, err := runCmdMayFailWithStateRoot(t, t.TempDir(), args...)
	if err != nil {
		t.Fatalf("execute %v: %v\noutput: %s", args, err, out)
	}
	return out
}

// runCmdMayFail is like runCmd but returns the error instead of fataling.
func runCmdMayFail(t *testing.T, args ...string) (string, error) {
	t.Helper()
	return runCmdMayFailWithStateRoot(t, t.TempDir(), args...)
}

func runCmdWithStateRoot(t *testing.T, stateRoot string, args ...string) string {
	t.Helper()
	out, err := runCmdMayFailWithStateRoot(t, stateRoot, args...)
	if err != nil {
		t.Fatalf("execute %v: %v\noutput: %s", args, err, out)
	}
	return out
}

func runCmdMayFailWithStateRoot(t *testing.T, stateRoot string, args ...string) (string, error) {
	t.Helper()
	resetCLIFlagState()
	old := resolveStateRoot
	resolveStateRoot = func() (string, error) { return stateRoot, nil }
	defer func() { resolveStateRoot = old }()
	var sb strings.Builder
	rootCmd.SetOut(&sb)
	rootCmd.SetErr(&sb)
	rootCmd.SetArgs(args)
	err := executeCLI(context.Background(), args)
	return sb.String(), err
}

func runCmdCaptureWithStateRoot(t *testing.T, stateRoot string, args ...string) (string, string, error) {
	t.Helper()
	old := resolveStateRoot
	resolveStateRoot = func() (string, error) { return stateRoot, nil }
	defer func() { resolveStateRoot = old }()
	resetCLIFlagState()
	var stdout, stderr strings.Builder
	rootCmd.SetOut(&stdout)
	rootCmd.SetErr(&stderr)
	rootCmd.SetArgs(args)
	err := executeCLI(context.Background(), args)
	return stdout.String(), stderr.String(), err
}

func resetCLIFlagState() {
	flagServer = os.Getenv("ST_SERVER")
	flagConfig, flagOutputDir = "", ""
	flagJSON = false
	flagTimeout = 0
	genPrompt, genNegative, genSize = "", "", ""
	genSeed, genScheduler, genMode, genInitImage, genRecreate = "", "", "", "", ""
	genControlnetFile, genOutfile = "", ""
	genSteps, genSkipStep, genSR = 0, 0, 0
	genCfg, genControlStrength = 0, 0
	genStream, genQuiet = false, false
	genControlnets, genControlRefs, genControlImages = nil, nil, nil
	conflateInclusive, conflateExitCodes = nil, nil
	describeOpts = describeOptions{}

	var clearChanged func(*cobra.Command)
	clearChanged = func(cmd *cobra.Command) {
		cmd.Flags().VisitAll(func(flag *pflag.Flag) { flag.Changed = false })
		cmd.PersistentFlags().VisitAll(func(flag *pflag.Flag) { flag.Changed = false })
		for _, child := range cmd.Commands() {
			clearChanged(child)
		}
	}
	clearChanged(rootCmd)
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
	err := resolveControlImages(context.Background(), nil, []string{"/some/file.png"}, nil, stclient.GenParams{})
	if err == nil || !strings.Contains(err.Error(), "control_type") {
		t.Errorf("expected error about missing control_type, got: %v", err)
	}
}

// TestControlImageAppliesStrength verifies --control-strength sets the strength
// field on every --control-image attachment; nil leaves it unset (mode default).
func TestControlImageAppliesStrength(t *testing.T) {
	imgFile := filepath.Join(t.TempDir(), "ctrl.png")
	os.WriteFile(imgFile, []byte("FAKEPNG"), 0o644)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"fileRef":"fileref:R"}`))
	}))
	defer srv.Close()
	client := stclient.New(srv.URL)

	strength := 0.65
	params := stclient.GenParams{}
	if err := resolveControlImages(context.Background(), client, []string{"depth:" + imgFile}, &strength, params); err != nil {
		t.Fatal(err)
	}
	list, _ := params["controlnets"].([]any)
	entry, _ := list[0].(map[string]any)
	if entry["strength"] != 0.65 {
		t.Errorf("strength = %v, want 0.65", entry["strength"])
	}

	// nil strength: key must be absent so the server applies the mode default.
	params2 := stclient.GenParams{}
	if err := resolveControlImages(context.Background(), client, []string{"depth:" + imgFile}, nil, params2); err != nil {
		t.Fatal(err)
	}
	list2, _ := params2["controlnets"].([]any)
	entry2, _ := list2[0].(map[string]any)
	if _, ok := entry2["strength"]; ok {
		t.Errorf("strength should be absent when flag unset, got %v", entry2["strength"])
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

// TestBuildGenParamsCombinedInitImageAndControlnet pins that st gen can already
// build a request carrying both init_image_ref and controlnets — the CLI has never
// needed a change for the combined case; only the server rejected it (STABL-ztaxgbhv).
func TestBuildGenParamsCombinedInitImageAndControlnet(t *testing.T) {
	cn := `{"attachment_id":"a1","control_type":"canny","map_asset_ref":"fileref:M1"}`
	args := genArgs{Prompt: "an owl", InitImage: "fileref:R1", Controlnets: []string{cn}}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	if p["init_image_ref"] != "R1" {
		t.Fatalf("init_image_ref not threaded: %+v", p)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("controlnets not threaded: %+v", p["controlnets"])
	}
}

// jpegFile writes a minimal real JPEG to a temp path. The backend decodes init
// images via PIL auto-detection, so the CLI must not require PNG here.
func jpegFile(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1)), nil); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "init.jpg")
	if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// plainPNGFile writes a valid PNG with no lcm tEXt chunk.
func plainPNGFile(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "plain.png")
	if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestBuildGenParamsInitImageJPEG: a local non-PNG init image is not a recipe
// source; missing baked params must mean "no baked layer", not a hard error
// (STABL-eccqgurq).
func TestBuildGenParamsInitImageJPEG(t *testing.T) {
	p, err := buildGenParams(nil, genArgs{Prompt: "an owl", InitImage: jpegFile(t)})
	if err != nil {
		t.Fatalf("local JPEG init image must not fail param build: %v", err)
	}
	if p["prompt"] != "an owl" {
		t.Fatalf("params: %+v", p)
	}
	if _, ok := p["init_image_ref"]; ok {
		t.Fatalf("local path uploads in the runner, must not set init_image_ref: %+v", p)
	}
}

// TestBuildGenParamsInitImagePlainPNG: a PNG without an lcm chunk contributes
// no baked layer but must not error (STABL-eccqgurq).
func TestBuildGenParamsInitImagePlainPNG(t *testing.T) {
	p, err := buildGenParams(nil, genArgs{Prompt: "an owl", InitImage: plainPNGFile(t)})
	if err != nil {
		t.Fatalf("plain PNG init image must not fail param build: %v", err)
	}
	if p["prompt"] != "an owl" {
		t.Fatalf("params: %+v", p)
	}
}

// TestBuildGenParamsInitImageWithLCMStillSeeds pins that a generated PNG used
// as init image keeps seeding baked params (existing behavior, must survive fix).
func TestBuildGenParamsInitImageWithLCMStillSeeds(t *testing.T) {
	path := pngWithLCM(t, `{"prompt":"base","cfg":9}`)
	p, err := buildGenParams(nil, genArgs{InitImage: path})
	if err != nil {
		t.Fatal(err)
	}
	if p["guidance_scale"] != float64(9) || p["prompt"] != "base" {
		t.Fatalf("baked layer not applied from init image: %+v", p)
	}
}

func TestConflatedGenUsesBaselineEffectiveParams(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	baseID, _ := store.ReserveID(context.Background())
	_ = store.Append(context.Background(), history.Entry{
		SchemaVersion: 1,
		ID:            baseID,
		Family:        history.FamilyGen,
		Raw:           history.CommandView{Argv: []string{"st", "gen"}, Display: "st gen"},
		Effective: &history.CommandView{
			Params: map[string]any{"prompt": "horse bartender", "guidance_scale": 4.5, "size": "1024x1024", "seed": 421337},
		},
		ExitCode: 0,
	})

	runCmdWithStateRoot(t, root, "conflate", "history:1")
	patch, err := parseRootGenPatch([]string{"--prompt", "two horses drinking"})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{}
	cfg.Defaults.Generation.Cfg = 7.5
	params, baseline, _, err := buildConflatedParams(context.Background(), store, patch, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if baseline.ID != 1 || params["guidance_scale"] != 4.5 || params["size"] != "1024x1024" || numericParam(params["seed"]) != 421337 || params["prompt"] != "two horses drinking" {
		t.Fatalf("baseline=%#v params=%#v", baseline, params)
	}
}

func TestConflatedGenExplicitZeroAndRandomOverrideBaseline(t *testing.T) {
	baseline := stclient.GenParams{
		"prompt": "owl", "guidance_scale": 4.5, "skip_step": 4,
		"superres": true, "superres_magnitude": 2, "seed": 421337,
	}
	patch := genPatch{
		Active:  true,
		Args:    genArgs{Cfg: f64p(0), SkipStep: intp(0), SR: intp(0), Seed: strp("random")},
		Changed: map[string]bool{"cfg": true, "skip-step": true, "sr": true, "seed": true},
	}
	got, err := buildGenParamsWithBaseline(nil, patch.Args, baseline, patch.Changed)
	if err != nil {
		t.Fatal(err)
	}
	if got["guidance_scale"] != float64(0) {
		t.Fatalf("cfg = %#v", got["guidance_scale"])
	}
	for _, key := range []string{"skip_step", "superres", "superres_magnitude", "seed"} {
		if _, ok := got[key]; ok {
			t.Fatalf("%s unexpectedly inherited: %#v", key, got)
		}
	}
}

func TestExplicitGenWithoutRecentBaselineFallsBackToNormalResolution(t *testing.T) {
	store := history.NewFSStore(t.TempDir())
	policy := history.DefaultPolicy()
	policy.Enabled = true
	if err := store.SavePolicy(context.Background(), policy); err != nil {
		t.Fatal(err)
	}
	patch := genPatch{Args: genArgs{Prompt: "first run"}}
	got, baseline, _, err := buildConflatedParams(context.Background(), store, patch, &config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if baseline != nil || got["prompt"] != "first run" {
		t.Fatalf("baseline=%#v params=%#v", baseline, got)
	}
}

func numericParam(v any) float64 {
	switch n := v.(type) {
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case float64:
		return n
	default:
		return 0
	}
}

// TestRecreateStillRequiresLCM: --recreate exists to read the recipe, so a
// non-recipe file must remain a hard error (both non-PNG and lcm-less PNG).
func TestRecreateStillRequiresLCM(t *testing.T) {
	if _, err := buildGenParams(nil, genArgs{Recreate: jpegFile(t)}); err == nil {
		t.Fatal("--recreate on a JPEG must error")
	}
	if _, err := buildGenParams(nil, genArgs{Recreate: plainPNGFile(t)}); err == nil {
		t.Fatal("--recreate on a plain PNG must error")
	}
}

// TestRecreateRequiredEvenWithInitImage: when both are set, recreate is the
// recipe source and stays mandatory even if the init image is fine.
func TestRecreateRequiredEvenWithInitImage(t *testing.T) {
	args := genArgs{Recreate: plainPNGFile(t), InitImage: jpegFile(t)}
	if _, err := buildGenParams(nil, args); err == nil {
		t.Fatal("--recreate without lcm chunk must error even alongside --init-image")
	}
}
