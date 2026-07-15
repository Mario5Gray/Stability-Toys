// Command stcn forms a single ControlNet attachment from flags and prints it
// as compact JSON suitable for `st gen --controlnet $(stcn ...)`. It is a
// pure, offline tool: it never contacts the server, uploads, or reads config.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
)

func newRootCmd(out io.Writer, opts *attachOpts) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "stcn <control_type>:<map_asset_ref>",
		Short: "Form one ControlNet attachment as compact JSON",
		Long: `stcn forms a single ControlNet attachment from flags and prints it as
compact JSON. Compose into a generation by repeating the flag:

  st gen --prompt "..." \
    --controlnet $(stcn canny:Rmap1 --strength 0.8) \
    --controlnet $(stcn depth:Rmap2 --strength 0.4)

Only map_asset_ref (a pre-made control map) is supported in v1. Emitted
string fields must be shell-token-safe (A-Z a-z 0-9 . _ : / -) so the
unquoted $(stcn ...) form is a single argv token.`,
		Args:          cobra.ExactArgs(1),
		SilenceUsage:  true,
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			ct, ref, err := parseHead(args[0])
			if err != nil {
				return err
			}
			opts.controlType = ct
			opts.mapAssetRef = ref
			opts.strengthSet = cmd.Flags().Changed("strength")
			opts.startSet = cmd.Flags().Changed("start")
			opts.endSet = cmd.Flags().Changed("end")

			a, err := buildAttachment(*opts)
			if err != nil {
				return err
			}
			b, err := marshalCompact(a)
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "%s\n", b)
			return nil
		},
	}
	f := cmd.Flags()
	f.Float64Var(&opts.strength, "strength", 0, "conditioning strength (0.0-2.0; unset = mode default)")
	f.Float64Var(&opts.start, "start", 0, "start_percent (0.0-1.0)")
	f.Float64Var(&opts.end, "end", 0, "end_percent (0.0-1.0)")
	f.StringVar(&opts.model, "model", "", "model_id override (default = mode policy)")
	f.StringVar(&opts.id, "id", "", "attachment_id (default = control_type)")
	return cmd
}

// run is the testable entrypoint: parse args, emit to out, return any error.
func run(args []string, out io.Writer) error {
	var opts attachOpts
	cmd := newRootCmd(out, &opts)
	cmd.SetArgs(args)
	return cmd.Execute()
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
