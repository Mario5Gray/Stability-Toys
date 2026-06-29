package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

var uploadCmd = &cobra.Command{
	Use:   "upload <file>",
	Short: "Upload a file and print its fileref",
	Args:  cobra.ExactArgs(1),
	RunE:  runUpload,
}

func init() {
	rootCmd.AddCommand(uploadCmd)
}

func runUpload(cmd *cobra.Command, args []string) error {
	data, err := os.ReadFile(args[0])
	if err != nil {
		return err
	}
	ref, err := newClient().Upload(cmd.Context(), filepath.Base(args[0]), data)
	if err != nil {
		return err
	}
	if flagJSON {
		return emitJSON(cmd, map[string]any{"fileRef": ref})
	}
	fmt.Fprintln(cmd.OutOrStdout(), ref)
	return nil
}
