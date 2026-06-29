package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/output"
)

var (
	srMagnitude int
	srOutfile   string
)

var superresCmd = &cobra.Command{
	Use:   "superres <file>",
	Short: "Upscale an image via the backend super-resolution endpoint",
	Args:  cobra.ExactArgs(1),
	RunE:  runSuperres,
}

func init() {
	f := superresCmd.Flags()
	f.IntVarP(&srMagnitude, "magnitude", "m", 2, "upscale magnitude (1-3)")
	f.StringVar(&srOutfile, "outfile", "", "explicit output path (else auto out-####)")
	rootCmd.AddCommand(superresCmd)
}

func runSuperres(cmd *cobra.Command, args []string) error {
	data, err := os.ReadFile(args[0])
	if err != nil {
		return err
	}
	img, err := newClient().SuperRes(cmd.Context(), data, srMagnitude)
	if err != nil {
		return err
	}
	dir := flagOutputDir
	if dir == "" {
		dir = "."
	}
	path, err := output.Resolve(srOutfile, dir, "png")
	if err != nil {
		return err
	}
	if err := output.Write(path, img); err != nil {
		return err
	}
	if flagJSON {
		return emitJSON(cmd, map[string]any{"output": path})
	}
	fmt.Fprintf(cmd.OutOrStdout(), "wrote %s\n", path)
	return nil
}
