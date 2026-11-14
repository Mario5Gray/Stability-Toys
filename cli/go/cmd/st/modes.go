package main

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
)

var modesCmd = &cobra.Command{
	Use:   "modes",
	Short: "List or manage model modes",
	Args:  noArgs,
	RunE:  runModes,
}

var modesSwitchCmd = &cobra.Command{
	Use:   "switch <name>",
	Short: "Switch to a model mode",
	Args:  exactArgs(1),
	RunE:  runModesSwitch,
}

var modesShowCmd = &cobra.Command{
	Use:   "show <name>",
	Short: "Print configuration for a mode as JSON",
	Args:  exactArgs(1),
	RunE:  runModesShow,
}

var modesReloadCmd = &cobra.Command{
	Use:   "reload",
	Short: "Hot-reload modes.yaml on the server",
	Args:  noArgs,
	RunE:  runModesReload,
}

func init() {
	modesCmd.AddCommand(modesSwitchCmd, modesShowCmd, modesReloadCmd)
	rootCmd.AddCommand(modesCmd)
}

func runModes(cmd *cobra.Command, args []string) error {
	client, err := newClient()
	if err != nil {
		return err
	}
	modes, err := client.Modes(cmd.Context())
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
		if len(m.AllowedSchedulerIDs) > 0 {
			labeled := make([]string, len(m.AllowedSchedulerIDs))
			for i, id := range m.AllowedSchedulerIDs {
				if id == m.DefaultSchedulerID {
					labeled[i] = id + " (default)"
				} else {
					labeled[i] = id
				}
			}
			fmt.Fprintf(cmd.OutOrStdout(), "  schedulers: %s\n", strings.Join(labeled, ", "))
		}
		if len(m.Loras) > 0 {
			loras := make([]string, len(m.Loras))
			for i, l := range m.Loras {
				loras[i] = fmt.Sprintf("%s(%g)", l.Path, l.Strength)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "  loras: %s\n", strings.Join(loras, ", "))
		}
	}
	return nil
}

func runModesSwitch(cmd *cobra.Command, args []string) error {
	client, err := newClient()
	if err != nil {
		return err
	}
	if err := client.SwitchMode(cmd.Context(), args[0]); err != nil {
		return err
	}
	fmt.Fprintf(cmd.OutOrStdout(), "switched to %s\n", args[0])
	return nil
}

func runModesShow(cmd *cobra.Command, args []string) error {
	client, err := newClient()
	if err != nil {
		return err
	}
	modes, err := client.Modes(cmd.Context())
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
	client, err := newClient()
	if err != nil {
		return err
	}
	if err := client.ReloadModes(cmd.Context()); err != nil {
		return err
	}
	fmt.Fprintln(cmd.OutOrStdout(), "modes reloaded")
	return nil
}
