package main

import (
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"
)

// Positional-argument validators. They wrap Cobra's message in usageError so
// main can point at the failing command's help instead of leaving the user to
// guess which flag or positional form was expected.

func exactArgs(n int) cobra.PositionalArgs {
	validate := cobra.ExactArgs(n)
	return func(cmd *cobra.Command, args []string) error {
		if err := validate(cmd, args); err != nil {
			return usageError{cmd: cmd, err: err}
		}
		return nil
	}
}

func minArgs(n int) cobra.PositionalArgs {
	validate := cobra.MinimumNArgs(n)
	return func(cmd *cobra.Command, args []string) error {
		if err := validate(cmd, args); err != nil {
			return usageError{cmd: cmd, err: err}
		}
		return nil
	}
}

func maxArgs(n int) cobra.PositionalArgs {
	validate := cobra.MaximumNArgs(n)
	return func(cmd *cobra.Command, args []string) error {
		if err := validate(cmd, args); err != nil {
			return usageError{cmd: cmd, err: err}
		}
		return nil
	}
}

func noArgs(cmd *cobra.Command, args []string) error {
	if err := cobra.NoArgs(cmd, args); err != nil {
		return usageError{cmd: cmd, err: err}
	}
	return nil
}

// emitJSON writes v as indented JSON to the command's stdout.
func emitJSON(cmd *cobra.Command, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintln(cmd.OutOrStdout(), string(b))
	return nil
}
