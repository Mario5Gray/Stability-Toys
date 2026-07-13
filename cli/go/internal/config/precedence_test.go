package config

import "testing"

func TestPrecedenceFlagsBeatBakedBeatConfig(t *testing.T) {
	cfg := &Config{}
	cfg.Defaults.Generation = Generation{Cfg: 2.0, Steps: 5, Genres: "512x512", Seed: "random"}
	baked := map[string]any{"guidance_scale": 7.0, "num_inference_steps": 20}
	flags := Flags{Steps: intp(30)} // only steps set on CLI

	got := ResolveParams(cfg, baked, flags)
	if got["num_inference_steps"] != 30 { // flag wins
		t.Fatalf("steps=%v", got["num_inference_steps"])
	}
	if got["guidance_scale"] != 7.0 { // baked wins over config
		t.Fatalf("cfg=%v", got["guidance_scale"])
	}
	if got["size"] != "512x512" { // config base
		t.Fatalf("size=%v", got["size"])
	}
	if _, ok := got["seed"]; ok { // random -> omitted
		t.Fatalf("seed should be omitted")
	}
}

// TestPrecedenceSeedAndSRFlags pins the explicit-flag branches the spine relies
// on: a numeric seed flag is kept verbatim, and --sr maps to superres + clamped
// magnitude.
func TestPrecedenceSeedAndSRFlags(t *testing.T) {
	cfg := &Config{}
	flags := Flags{Seed: strp("12345"), SRLevel: intp(9)}

	got := ResolveParams(cfg, nil, flags)
	if got["seed"] != "12345" {
		t.Fatalf("seed=%v, want 12345", got["seed"])
	}
	if got["superres"] != true {
		t.Fatalf("superres=%v", got["superres"])
	}
	if got["superres_magnitude"] != 3 { // clamped to max
		t.Fatalf("magnitude=%v, want 3", got["superres_magnitude"])
	}
}

func TestPrecedenceSkipStepFromConfig(t *testing.T) {
	cfg := &Config{}
	cfg.Defaults.Generation = Generation{SkipStep: 3}

	got := ResolveParams(cfg, nil, Flags{})
	if got["skip_step"] != 3 {
		t.Fatalf("skip_step=%v, want 3", got["skip_step"])
	}
}

func TestPrecedenceSkipStepFlagOverridesConfig(t *testing.T) {
	cfg := &Config{}
	cfg.Defaults.Generation = Generation{SkipStep: 2}

	got := ResolveParams(cfg, nil, Flags{SkipStep: intp(5)})
	if got["skip_step"] != 5 {
		t.Fatalf("skip_step=%v, want 5", got["skip_step"])
	}
}

func TestPrecedenceSkipStepZeroOmitted(t *testing.T) {
	cfg := &Config{}
	cfg.Defaults.Generation = Generation{SkipStep: 0}

	got := ResolveParams(cfg, nil, Flags{})
	if _, ok := got["skip_step"]; ok {
		t.Fatalf("skip_step should be omitted when zero, got %v", got["skip_step"])
	}
}

func TestResolveParamsWithBaselinePrecedence(t *testing.T) {
	cfg := &Config{}
	cfg.Defaults.Generation = Generation{Cfg: 2, Steps: 5, Genres: "512x512"}
	baked := map[string]any{"guidance_scale": 3.0, "num_inference_steps": 10, "size": "768x768"}
	baseline := map[string]any{"guidance_scale": 4.5, "num_inference_steps": 20, "size": "1024x1024"}
	got := ResolveParamsWithBaseline(cfg, baked, baseline, Flags{Steps: intp(30)})
	if got["guidance_scale"] != 4.5 || got["size"] != "1024x1024" || got["num_inference_steps"] != 30 {
		t.Fatalf("params = %#v", got)
	}
}

func TestResolveParamsWithBaselineExplicitZeroClearsInheritedFields(t *testing.T) {
	zero := 0
	zeroFloat := 0.0
	baseline := map[string]any{
		"guidance_scale":     4.5,
		"skip_step":          4,
		"superres":           true,
		"superres_magnitude": 2,
	}
	got := ResolveParamsWithBaseline(&Config{}, nil, baseline, Flags{Cfg: &zeroFloat, SkipStep: &zero, SRLevel: &zero})
	if got["guidance_scale"] != float64(0) {
		t.Fatalf("guidance_scale = %#v", got["guidance_scale"])
	}
	for _, key := range []string{"skip_step", "superres", "superres_magnitude"} {
		if _, ok := got[key]; ok {
			t.Fatalf("%s unexpectedly present in %#v", key, got)
		}
	}
}

func TestResolveParamsWithBaselineRandomClearsInheritedSeed(t *testing.T) {
	random := "random"
	got := ResolveParamsWithBaseline(&Config{}, nil, map[string]any{"seed": 421337}, Flags{Seed: &random})
	if _, ok := got["seed"]; ok {
		t.Fatalf("seed unexpectedly present in %#v", got)
	}
}

func intp(i int) *int       { return &i }
func strp(s string) *string { return &s }
