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
	Short: "Print embedded PNG metadata (lcm, controlnet, controlnet_map)",
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

	chunks, err := pngmeta.Parse(data)
	if err != nil {
		return err
	}

	out := map[string]any{}
	if v, ok, err := chunks.FindLCM(); err != nil {
		return err
	} else if ok {
		out["lcm"] = v
	}
	if v, ok, err := chunks.FindControlNet(); err != nil {
		return err
	} else if ok {
		out["controlnet"] = v
	}
	if v, ok, err := chunks.FindControlNetMap(); err != nil {
		return err
	} else if ok {
		out["controlnet_map"] = v
	}

	if len(out) == 0 {
		return fmt.Errorf("no known metadata chunk (lcm, controlnet, controlnet_map) found in %s", args[0])
	}

	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintln(cmd.OutOrStdout(), string(b))
	return nil
}
