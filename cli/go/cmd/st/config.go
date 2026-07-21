package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/darkbit/stability-toys/cli/st/internal/config"
	"github.com/darkbit/stability-toys/cli/st/internal/history"
	"github.com/spf13/cobra"
)

// The config file is not discoverable by guessing: it is JSON rather than the
// more common YAML, and it lives under an XDG root that is usually unset in the
// environment, so there is nothing to grep for. These commands exist so the
// answer is always one command away instead of a hunt through ~/.config.

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Show where st keeps its configuration and state",
	RunE: func(cmd *cobra.Command, args []string) error {
		return runConfigPaths(cmd)
	},
}

var configPathCmd = &cobra.Command{
	Use:   "path",
	Short: "Print the resolved config file path",
	RunE: func(cmd *cobra.Command, args []string) error {
		path, err := config.Resolve(flagConfig)
		if err != nil {
			return err
		}
		fmt.Fprintln(cmd.OutOrStdout(), path)
		return nil
	},
}

var configShowCmd = &cobra.Command{
	Use:   "show",
	Short: "Print the active config file contents",
	RunE: func(cmd *cobra.Command, args []string) error {
		path, err := config.Resolve(flagConfig)
		if err != nil {
			return err
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		fmt.Fprint(cmd.OutOrStdout(), string(b))
		return nil
	},
}

// runConfigPaths reports every location st reads or writes, marking which the
// user is expected to edit. Existence is shown so a missing file is obvious
// rather than being mistaken for a wrong path.
func runConfigPaths(cmd *cobra.Command) error {
	out := cmd.OutOrStdout()

	cfgPath, err := config.Resolve(flagConfig)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "config  %s%s   (edit this)\n", cfgPath, existsMark(cfgPath))

	stateRoot, err := history.ResolveStateRoot()
	if err == nil {
		fmt.Fprintf(out, "state   %s%s   (managed, do not edit)\n", stateRoot, existsMark(stateRoot))
		for _, name := range []string{"history.jsonl", "next-id", "state.lock"} {
			p := filepath.Join(stateRoot, name)
			fmt.Fprintf(out, "          %s%s\n", p, existsMark(p))
		}
	}

	fmt.Fprintf(out, "\noverride with --config <path> or $ST_CONFIG\n")
	return nil
}

func existsMark(path string) string {
	if _, err := os.Stat(path); err != nil {
		return "  (missing)"
	}
	return ""
}

// validateConfigJSON is used by tests to assert the shipped template parses.
func validateConfigJSON(b []byte) error {
	var v map[string]any
	return json.Unmarshal(b, &v)
}

func init() {
	configCmd.AddCommand(configPathCmd, configShowCmd)
	rootCmd.AddCommand(configCmd)
}
