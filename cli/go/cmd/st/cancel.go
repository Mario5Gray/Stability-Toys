package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

var cancelCmd = &cobra.Command{
	Use:   "cancel <jobId>",
	Short: "Cancel a running job",
	Args:  cobra.ExactArgs(1),
	RunE:  runCancel,
}

func init() {
	rootCmd.AddCommand(cancelCmd)
}

func runCancel(cmd *cobra.Command, args []string) error {
	jobID := args[0]
	if err := newClient().Cancel(cmd.Context(), jobID); err != nil {
		return err
	}
	if flagJSON {
		return emitJSON(cmd, map[string]any{"canceled": jobID})
	}
	fmt.Fprintf(cmd.OutOrStdout(), "canceled %s\n", jobID)
	return nil
}
