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

func intp(i int) *int       { return &i }
func strp(s string) *string { return &s }
