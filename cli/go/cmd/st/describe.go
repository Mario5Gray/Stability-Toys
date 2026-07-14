package main

import (
	"fmt"
	"strings"

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
