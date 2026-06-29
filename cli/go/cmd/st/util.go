package main

import (
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"
)

// emitJSON writes v as indented JSON to the command's stdout.
func emitJSON(cmd *cobra.Command, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintln(cmd.OutOrStdout(), string(b))
	return nil
}
