package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

var modesCmd = &cobra.Command{
	Use:   "modes",
	Short: "List available model modes",
	Args:  cobra.NoArgs,
	RunE:  runModes,
}

func init() {
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
		fmt.Fprintln(cmd.OutOrStdout(), m.Name)
	}
	return nil
}
