package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

var modesCmd = &cobra.Command{
	Use:   "modes",
	Short: "List or manage model modes",
	Args:  cobra.NoArgs,
	RunE:  runModes,
}

var modesSwitchCmd = &cobra.Command{
	Use:   "switch <name>",
	Short: "Switch to a model mode",
	Args:  cobra.ExactArgs(1),
	RunE:  runModesSwitch,
}

var modesShowCmd = &cobra.Command{
	Use:   "show <name>",
	Short: "Print configuration for a mode as JSON",
	Args:  cobra.ExactArgs(1),
	RunE:  runModesShow,
}

var modesReloadCmd = &cobra.Command{
	Use:   "reload",
	Short: "Hot-reload modes.yaml on the server",
	Args:  cobra.NoArgs,
	RunE:  runModesReload,
}

func init() {
	modesCmd.AddCommand(modesSwitchCmd, modesShowCmd, modesReloadCmd)
	rootCmd.AddCommand(modesCmd)
}

func runModes(cmd *cobra.Command, args []string) error {
	modes, err := newClient().Modes(cmd.Context())
	if err != nil {
		return err
	}
	if flagJSON {
		return emitJSON(cmd, modes)
	}
	for _, m := range modes {
		name := m.Name
		if m.IsDefault {
			name += " (default)"
		}
		extra := ""
		if m.ControlNetEnabled {
			extra += "  controlnet"
		}
		if m.ChatEnabled {
			extra += "  chat"
		}
		fmt.Fprintf(cmd.OutOrStdout(), "%s\n  model=%s  size=%s  steps=%d  cfg=%.1f%s\n",
			name, m.Model, m.DefaultSize, m.DefaultSteps, m.DefaultGuidance, extra)
	}
	return nil
}

func runModesSwitch(cmd *cobra.Command, args []string) error {
	if err := newClient().SwitchMode(cmd.Context(), args[0]); err != nil {
		return err
	}
	fmt.Fprintf(cmd.OutOrStdout(), "switched to %s\n", args[0])
	return nil
}

func runModesShow(cmd *cobra.Command, args []string) error {
	modes, err := newClient().Modes(cmd.Context())
	if err != nil {
		return err
	}
	for _, m := range modes {
		if m.Name == args[0] {
			return emitJSON(cmd, m)
		}
	}
	return fmt.Errorf("mode %q not found", args[0])
}

func runModesReload(cmd *cobra.Command, args []string) error {
	if err := newClient().ReloadModes(cmd.Context()); err != nil {
		return err
	}
	fmt.Fprintln(cmd.OutOrStdout(), "modes reloaded")
	return nil
}
