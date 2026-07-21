// Package config loads the user's persistent CLI configuration and resolves
// where that file lives. The on-disk schema mirrors the operations-CLI design
// spec: a top-level "config" wrapper around generation defaults and output
// settings. These defaults form precedence layer 1 (see precedence.go); baked
// PNG params and explicit CLI flags override them.
package config

import (
	"fmt"
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

// AppDir is the single directory name this CLI uses under every XDG root, so
// config and state are found in the same place under different roots:
//
//	$XDG_CONFIG_HOME/st/config.json   settings, hand-edited
//	$XDG_STATE_HOME/st/              history, locks, counters (managed)
//
// Config stays under the config root rather than moving in beside state: the
// state directory holds lock files and an ID counter that nobody should edit
// or sync, and XDG_DATA_HOME is for application data, not settings.
const AppDir = "st"

// legacyAppDir is the pre-consolidation config directory name. Only the config
// root ever used it; state has always been under "st".
const legacyAppDir = "stability-toys"

func configRoot() (string, error) {
	if base := os.Getenv("XDG_CONFIG_HOME"); base != "" {
		return base, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config"), nil
}

// Resolve picks the config path: explicit --config flag, then $ST_CONFIG, then
// the XDG default ($XDG_CONFIG_HOME or ~/.config)/st/config.json.
func Resolve(flagPath string) (string, error) {
	if flagPath != "" {
		return flagPath, nil
	}
	if env := os.Getenv("ST_CONFIG"); env != "" {
		return env, nil
	}
	base, err := configRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, AppDir, "config.json"), nil
}

// CheckLegacyLocation reports a config left in the pre-consolidation directory
// when the default location is in use.
// It returns an error only when the legacy file exists and the current one does
// not, so a fresh install bootstraps normally and a migrated install is never
// nagged. The error carries the exact command to run: this is a hard cutover,
// and silently reading the old path would leave both locations live forever.
func CheckLegacyLocation(flagPath string) error {
	// An explicit path is the user telling us where the config is; the cutover
	// only governs the default location. Nagging here would break every caller
	// that passes --config or $ST_CONFIG.
	if flagPath != "" || os.Getenv("ST_CONFIG") != "" {
		return nil
	}
	base, err := configRoot()
	if err != nil {
		return nil
	}
	current := filepath.Join(base, AppDir, "config.json")
	if _, err := os.Stat(current); err == nil {
		return nil
	}
	legacy := filepath.Join(base, legacyAppDir, "config.json")
	if _, err := os.Stat(legacy); err != nil {
		return nil
	}
	return fmt.Errorf(
		"config found at the old location %s\n\n"+
			"st now keeps configuration in %s.\nMove it with:\n\n"+
			"  mkdir -p %s\n  mv %s %s\n  rmdir %s\n",
		legacy, current,
		filepath.Dir(current), legacy, current, filepath.Dir(legacy),
	)
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
