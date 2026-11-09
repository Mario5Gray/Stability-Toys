package main

import (
	"github.com/spf13/cobra"
)

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "Show model/backend status",
	Args:  noArgs,
	RunE:  runModels,
}

func init() {
	rootCmd.AddCommand(modelsCmd)
}

func runModels(cmd *cobra.Command, args []string) error {
	client, err := newClient()
	if err != nil {
		return err
	}
	status, err := client.Models(cmd.Context())
	if err != nil {
		return err
	}
	return emitJSON(cmd, status)
}
