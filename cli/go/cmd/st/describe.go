package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

type describeOptions struct {
	caption          bool
	prompt           string
	detect           bool
	labels           []string
	minConfidence    float64
	minConfidenceSet bool
}

type targetSpec struct {
	arg   string
	id    string
	isURL bool
}

// classifyTargets maps positional args to targets in exact arg order with
// positional IDs t1..tN (1-based). Ordering is contract; never sort.
func classifyTargets(args []string) []targetSpec {
	specs := make([]targetSpec, len(args))
	for i, arg := range args {
		specs[i] = targetSpec{
			arg:   arg,
			id:    fmt.Sprintf("t%d", i+1),
			isURL: strings.HasPrefix(arg, "http://") || strings.HasPrefix(arg, "https://"),
		}
	}
	return specs
}

// buildDescribeTasks emits tasks in canonical TaskKind order regardless of
// flag order; task ID is the kind string.
func buildDescribeTasks(o describeOptions) []stclient.DescribeTask {
	var tasks []stclient.DescribeTask
	if o.caption {
		params := &stclient.CaptionParams{}
		if o.prompt != "" {
			prompt := o.prompt
			params.Prompt = &prompt
		}
		tasks = append(tasks, stclient.DescribeTask{
			ID:      string(stclient.TaskKindCaption),
			Kind:    stclient.TaskKindCaption,
			Caption: params,
		})
	}
	if o.detect {
		params := &stclient.DetectParams{}
		if len(o.labels) > 0 {
			params.Labels = o.labels
		}
		if o.minConfidenceSet {
			minConfidence := o.minConfidence
			params.MinConfidence = &minConfidence
		}
		tasks = append(tasks, stclient.DescribeTask{
			ID:     string(stclient.TaskKindDetect),
			Kind:   stclient.TaskKindDetect,
			Detect: params,
		})
	}
	return tasks
}

func validateDescribeFlags(o describeOptions) error {
	if !o.caption && !o.detect {
		return fmt.Errorf("at least one task flag required (--caption, --detect)")
	}
	if o.prompt != "" && !o.caption {
		return fmt.Errorf("--prompt requires --caption")
	}
	if len(o.labels) > 0 && !o.detect {
		return fmt.Errorf("--labels requires --detect")
	}
	if o.minConfidenceSet && !o.detect {
		return fmt.Errorf("--min-confidence requires --detect")
	}
	return nil
}

var describeOpts describeOptions

var describeCmd = &cobra.Command{
	Use:   "describe <file-or-url> [more...]",
	Short: "Run analysis tasks (caption, detect) against images",
	Long: `Describe images through the server's analysis capability.

Positional arguments are targets, in order: local files are uploaded
first (target IDs t1..tN follow argument order); http(s):// arguments
pass through as URL targets. Task flags select what runs:

  st describe ./photo.png --caption
  st describe ./photo.png --detect --labels person,car
  st describe ./a.png ./b.png --caption --detect

Exit codes: 0 ok, 1 transport/usage/validation error, 2 failed, 3 partial.`,
	Args: cobra.MinimumNArgs(1),
	RunE: runDescribe,
}

func init() {
	f := describeCmd.Flags()
	f.BoolVar(&describeOpts.caption, "caption", false, "add a caption task")
	f.StringVar(&describeOpts.prompt, "prompt", "", "caption guidance prompt (requires --caption)")
	f.BoolVar(&describeOpts.detect, "detect", false, "add a detection task")
	f.StringSliceVar(&describeOpts.labels, "labels", nil, "detection label filter (requires --detect)")
	f.Float64Var(&describeOpts.minConfidence, "min-confidence", 0, "minimum detection confidence (requires --detect)")
	rootCmd.AddCommand(describeCmd)
}

func runDescribe(cmd *cobra.Command, args []string) error {
	describeOpts.minConfidenceSet = cmd.Flags().Changed("min-confidence")
	if err := validateDescribeFlags(describeOpts); err != nil {
		return err // usage error: exit 1 via main's default
	}

	client := newClient()
	ctx := cmd.Context()
	specs := classifyTargets(args)
	targets := make([]stclient.DescribeTarget, len(specs))
	for i, spec := range specs {
		if spec.isURL {
			u := spec.arg
			targets[i] = stclient.DescribeTarget{ID: spec.id, URL: &u}
			continue
		}
		data, err := os.ReadFile(spec.arg)
		if err != nil {
			return err
		}
		ref, err := client.Upload(ctx, filepath.Base(spec.arg), data, "upload")
		if err != nil {
			return fmt.Errorf("upload %s: %w", spec.arg, err)
		}
		targets[i] = stclient.DescribeTarget{ID: spec.id, AssetRef: &ref}
	}

	req := stclient.DescribeRequest{Targets: targets, Tasks: buildDescribeTasks(describeOpts)}
	resp, err := client.Describe(ctx, req)
	if err != nil {
		// *APIError prints "code: message", transport errors print their
		// message — both via main's "error:" stderr line. Exit 1.
		return err
	}

	if flagJSON {
		if err := emitJSON(cmd, resp); err != nil {
			return err
		}
	} else {
		renderDescribeHuman(cmd.OutOrStdout(), resp)
	}
	renderDescribeFailures(cmd.ErrOrStderr(), resp)
	return describeStatusErr(resp)
}

// renderDescribeHuman writes the human default rendering: caption text
// lines and a detection table. Scripts must not parse this — the frozen
// machine surface is --json.
func renderDescribeHuman(w io.Writer, resp *stclient.DescribeResponse) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	for _, obs := range resp.Observations {
		switch {
		case obs.Text != nil:
			fmt.Fprintf(tw, "%s (%s): %s\n", obs.TaskID, obs.TargetID, obs.Text.Content)
		case obs.Detection != nil:
			d := obs.Detection
			fmt.Fprintf(tw, "%s (%s):\t%s\t%.2f\tbox(%.3f, %.3f, %.3f, %.3f)\n",
				obs.TaskID, obs.TargetID, d.Label, d.Confidence, d.Box.X, d.Box.Y, d.Box.W, d.Box.H)
		}
	}
	tw.Flush()
}

// renderDescribeFailures writes every non-succeeded run to w (stderr in
// practice). Presence and content are frozen by the spec: task, target,
// delegate, status, error code, error message.
func renderDescribeFailures(w io.Writer, resp *stclient.DescribeResponse) {
	for _, run := range resp.Runs {
		if run.Status == stclient.RunSucceeded {
			continue
		}
		code, msg := "", ""
		if run.Error != nil {
			code, msg = run.Error.Code, run.Error.Message
		}
		fmt.Fprintf(w, "run %s/%s (%s) %s: %s: %s\n",
			run.TaskID, run.TargetID, run.Delegate, run.Status, code, msg)
	}
}

// describeStatusErr maps the frozen exit-code contract: ok=0, failed=2,
// partial=3 (transport/usage/validation errors exit 1 elsewhere).
func describeStatusErr(resp *stclient.DescribeResponse) error {
	switch resp.Status {
	case stclient.StatusFailed:
		return exitError{code: 2, err: errors.New("describe failed: no run succeeded")}
	case stclient.StatusPartial:
		return exitError{code: 3, err: errors.New("describe partial: some runs did not succeed")}
	default:
		return nil
	}
}
