package config

// Flags carries the explicit CLI generation values. Pointers distinguish "unset"
// (nil — fall through to baked/config) from a deliberate zero value.
type Flags struct {
	Prompt    string
	Negative  *string
	Genres    *string
	Steps     *int
	Cfg       *float64
	Seed      *string // "random" or integer text
	Scheduler *string
	Mode      *string
	SRLevel   *int
}

// ResolveParams layers generation parameters by precedence — config defaults <
// baked PNG params < explicit CLI flags — and returns the merged GenerateRequest
// field map. Named ResolveParams (not Resolve) to avoid colliding with the
// config-path Resolve in config.go; the gen command (Task 12) consumes this.
func ResolveParams(cfg *Config, baked map[string]any, f Flags) map[string]any {
	p := map[string]any{}
	g := cfg.Defaults.Generation

	// layer 1: config defaults
	setStr(p, "size", g.Genres)
	if g.Cfg != 0 {
		p["guidance_scale"] = g.Cfg
	}
	if g.Steps != 0 {
		p["num_inference_steps"] = g.Steps
	}
	applySeed(p, g.Seed)
	setStr(p, "mode", g.Mode)

	// layer 2: baked params (already in GenerateRequest field names)
	for k, v := range baked {
		p[k] = v
	}

	// layer 3: explicit CLI flags
	if f.Prompt != "" {
		p["prompt"] = f.Prompt
	}
	if f.Genres != nil {
		p["size"] = *f.Genres
	}
	if f.Steps != nil {
		p["num_inference_steps"] = *f.Steps
	}
	if f.Cfg != nil {
		p["guidance_scale"] = *f.Cfg
	}
	if f.Negative != nil {
		p["negative_prompt"] = *f.Negative
	}
	if f.Scheduler != nil {
		p["scheduler_id"] = *f.Scheduler
	}
	if f.Mode != nil {
		p["mode"] = *f.Mode
	}
	if f.SRLevel != nil && *f.SRLevel > 0 {
		p["superres"] = true
		p["superres_magnitude"] = clamp(*f.SRLevel, 1, 3)
	}
	if f.Seed != nil {
		applySeed(p, *f.Seed)
	}
	return p
}

func setStr(p map[string]any, k, v string) {
	if v != "" {
		p[k] = v
	}
}

// applySeed writes a concrete seed, or omits it for "random"/empty so the
// backend chooses one. A non-random prior seed is cleared when overridden by a
// random flag.
func applySeed(p map[string]any, seed any) {
	switch s := seed.(type) {
	case string:
		if s == "" || s == "random" {
			delete(p, "seed")
			return
		}
	}
	if seed != nil && seed != "random" && seed != "" {
		p["seed"] = seed
	}
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
