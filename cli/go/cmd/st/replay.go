package main

import (
	"fmt"
	"strconv"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

var replayCmd = &cobra.Command{
	Use:   "replay <id>",
	Short: "Replay one history-backed generation exactly",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		id, err := strconv.ParseInt(args[0], 10, 64)
		if err != nil || id < 1 {
			return fmt.Errorf("replay id must be a positive integer, got %q", args[0])
		}
		state, err := stateFromContext(cmd.Context())
		if err != nil {
			return err
		}
		params, source, err := loadReplayParams(cmd.Context(), state.store, id)
		if err != nil {
			return err
		}
		if !genQuiet {
			fmt.Fprintf(cmd.ErrOrStderr(), "replaying [id=%d]: %s\n", id, source.Effective.Display)
		}
		cfg, err := requireConfig()
		if err != nil {
			return err
		}
		return executeResolvedGen(cmd, cfg, genArgs{Outfile: genOutfile}, params, func(final stclient.GenParams) {
			state.final = &invocationResult{
				params:                final,
				replayedFromHistoryID: &source.ID,
			}
		})
	},
}

func init() {
	bindGenExecutionFlags(replayCmd.Flags())
	rootCmd.AddCommand(replayCmd)
}
