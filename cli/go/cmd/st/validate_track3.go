package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

var (
	t3ControlImage string
	t3ControlType  string
	t3Prompt       string
)

// validateTrack3Cmd scripts the ControlNet Track 3 acceptance checklist
// (docs/TESTING_CONTROLNET_TRACK3.md): upload a control map, run a generation
// with one controlnet attachment, and assert the backend returns
// controlnet_artifacts. Point it at a live backend with --server.
var validateTrack3Cmd = &cobra.Command{
	Use:   "validate-track3",
	Short: "Validate ControlNet Track 3 against a live server",
	Args:  cobra.NoArgs,
	RunE:  runValidateTrack3,
}

func init() {
	f := validateTrack3Cmd.Flags()
	f.StringVar(&t3ControlImage, "control-image", "", "control map PNG to upload (required)")
	f.StringVar(&t3ControlType, "control-type", "canny", "control type for the attachment")
	f.StringVar(&t3Prompt, "prompt", "a controlnet validation render", "prompt to use")
	rootCmd.AddCommand(validateTrack3Cmd)
}

func runValidateTrack3(cmd *cobra.Command, args []string) error {
	if t3ControlImage == "" {
		return fmt.Errorf("--control-image is required")
	}
	ctx := cmd.Context()
	client := newClient()
	out := cmd.OutOrStdout()
	fmt.Fprintln(out, "validate-track3:")

	data, err := os.ReadFile(t3ControlImage)
	if err != nil {
		return err
	}
	ref, err := client.Upload(ctx, filepath.Base(t3ControlImage), data)
	if err != nil {
		return fmt.Errorf("upload control map: %w", err)
	}
	fmt.Fprintf(out, "  [PASS] uploaded control map -> %s\n", ref)

	params := stclient.GenParams{
		"prompt": t3Prompt,
		"controlnets": []any{map[string]any{
			"attachment_id": "track3-1",
			"control_type":  t3ControlType,
			"map_asset_ref": ref,
		}},
	}
	_, res, err := client.Generate(ctx, params)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}

	if len(res.CNArtifacts) == 0 {
		fmt.Fprintln(out, "  [FAIL] result carried no controlnet_artifacts")
		return fmt.Errorf("validate-track3 FAILED: backend returned no controlnet_artifacts")
	}
	fmt.Fprintf(out, "  [PASS] generation returned %d controlnet_artifacts\n", len(res.CNArtifacts))
	fmt.Fprintln(out, "PASS")
	return nil
}
