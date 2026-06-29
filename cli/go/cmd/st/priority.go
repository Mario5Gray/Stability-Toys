package main

import (
	"fmt"
	"strconv"

	"github.com/spf13/cobra"
)

var priorityCmd = &cobra.Command{
	Use:   "priority <jobId> <level>",
	Short: "Set a job's priority",
	Args:  cobra.ExactArgs(2),
	RunE:  runPriority,
}

func init() {
	rootCmd.AddCommand(priorityCmd)
}

func runPriority(cmd *cobra.Command, args []string) error {
	level, err := strconv.Atoi(args[1])
	if err != nil {
		return fmt.Errorf("level must be an integer: %w", err)
	}
	jobID := args[0]
	if err := newClient().SetPriority(cmd.Context(), jobID, level); err != nil {
		return err
	}
	if flagJSON {
		return emitJSON(cmd, map[string]any{"job": jobID, "priority": level})
	}
	fmt.Fprintf(cmd.OutOrStdout(), "priority of %s set to %d\n", jobID, level)
	return nil
}
