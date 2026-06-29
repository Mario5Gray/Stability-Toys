package main

import (
	"github.com/spf13/cobra"
)

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "Show model/backend status",
	Args:  cobra.NoArgs,
	RunE:  runModels,
}

func init() {
	rootCmd.AddCommand(modelsCmd)
}

func runModels(cmd *cobra.Command, args []string) error {
	status, err := newClient().Models(cmd.Context())
	if err != nil {
		return err
	}
	return emitJSON(cmd, status)
}
