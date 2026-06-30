package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var uploadCmd = &cobra.Command{
	Use:   "upload [type:]<file>",
	Short: "Upload a file and print its fileref",
	Long: `Upload a local file to the server and print the assigned fileref.

Optionally prefix the path with a type label to declare intent:

  st upload image:./owl.png      # general image upload
  st upload canny:./map.png      # declare as a canny control map

The type label is sent as a "type" form field. Without a prefix the file
is uploaded with no type declared.`,
	Args: cobra.ExactArgs(1),
	RunE: runUpload,
}

func init() {
	rootCmd.AddCommand(uploadCmd)
}

func runUpload(cmd *cobra.Command, args []string) error {
	bucket, filePath := parseUploadArg(args[0])
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}
	ref, err := newClient().Upload(cmd.Context(), filepath.Base(filePath), data, bucket)
	if err != nil {
		return err
	}
	if flagJSON {
		return emitJSON(cmd, map[string]any{"fileRef": ref, "bucket": bucket})
	}
	fmt.Fprintln(cmd.OutOrStdout(), ref)
	return nil
}

// parseUploadArg splits "type:path" into (type, path). If no colon is
// present, returns ("", arg) — no bucket, plain path.
func parseUploadArg(arg string) (bucket, path string) {
	if before, after, ok := strings.Cut(arg, ":"); ok {
		return before, after
	}
	return "", arg
}
