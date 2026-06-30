// Package config loads the user's persistent CLI configuration and resolves
// where that file lives. The on-disk schema mirrors the operations-CLI design
// spec: a top-level "config" wrapper around generation defaults and output
// settings. These defaults form precedence layer 1 (see precedence.go); baked
// PNG params and explicit CLI flags override them.
package config

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

// ErrBootstrapped signals that no config existed and a template was written;
// callers should print the path and exit so the user can edit it.
var ErrBootstrapped = errors.New("config bootstrapped")

// Generation holds the default generation parameters. Seed is `any` because it
// is either an integer or the string "random"; the precedence resolver treats
// "random"/empty as "omit so the backend picks one".
type Generation struct {
	Mode     string  `json:"mode"`
	Cfg      float64 `json:"cfg"`
	Steps    int     `json:"steps"`
	SkipStep int     `json:"skip_step,omitempty"`
	Genres   string  `json:"genres"`
	Seed     any     `json:"seed"` // int or "random"
}

// Meta holds client-side image metadata defaults written into output PNGs.
type Meta struct {
	ProducerName string           `json:"producer_name"`
	IncludeDate  bool             `json:"include_date"`
	Misc         []map[string]any `json:"misc"`
}

// Defaults is the resolved default block applied to every generation.
type Defaults struct {
	Generation      Generation `json:"generation"`
	OutputFormat    string     `json:"output_format"`
	OutputDirectory string     `json:"output_directory"`
	IncludeMeta     bool       `json:"include_meta"`
	Meta            Meta       `json:"meta"`
}

// ControlnetPreset is a named ControlNetAttachment stored in config and
// referenced by --controlnet @name. Values are passed verbatim to the
// backend as a controlnet attachment object.
type ControlnetPreset map[string]any

// Config is the root document (unwrapped from the "config" key on disk).
type Config struct {
	ServerURL         string                      `json:"server_url,omitempty"`
	Defaults          Defaults                    `json:"defaults"`
	ControlnetPresets map[string]ControlnetPreset `json:"controlnet_presets,omitempty"`
}

// Load reads and unwraps a config file from disk.
func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var wrap struct {
		Config Config `json:"config"`
	}
	if err := json.Unmarshal(b, &wrap); err != nil {
		return nil, err
	}
	return &wrap.Config, nil
}

// Resolve picks the config path: explicit --config flag, then $ST_CONFIG, then
// the XDG default ($XDG_CONFIG_HOME or ~/.config)/stability-toys/config.json.
func Resolve(flagPath string) (string, error) {
	if flagPath != "" {
		return flagPath, nil
	}
	if env := os.Getenv("ST_CONFIG"); env != "" {
		return env, nil
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "stability-toys", "config.json"), nil
}

// BootstrapTemplate writes a loadable placeholder config to path (creating
// parent dirs). The REPLACE_ME markers signal fields the user must edit.
func BootstrapTemplate(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmpl := `{
  "config": {
    "defaults": {
      "generation": { "mode": "default", "cfg": 2.5, "steps": 10, "genres": "512x512", "seed": "random" },
      "output_format": "png",
      "output_directory": "REPLACE_ME/output",
      "include_meta": true,
      "meta": { "producer_name": "REPLACE_ME", "include_date": true, "misc": [] }
    }
  }
}
`
	return os.WriteFile(path, []byte(tmpl), 0o644)
}
