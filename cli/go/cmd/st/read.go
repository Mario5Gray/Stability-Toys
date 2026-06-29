package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/pngmeta"
)

var readCmd = &cobra.Command{
	Use:   "read <image.png>",
	Short: "Print the lcm generation metadata embedded in a PNG",
	Args:  cobra.ExactArgs(1),
	RunE:  runRead,
}

func init() {
	rootCmd.AddCommand(readCmd)
}

func runRead(cmd *cobra.Command, args []string) error {
	data, err := os.ReadFile(args[0])
	if err != nil {
		return err
	}
	m, err := pngmeta.ReadLCM(data)
	if err != nil {
		return err
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintln(cmd.OutOrStdout(), string(b))
	return nil
}
