package main

import (
	"reflect"
	"testing"
)

func TestClassifyTargetsAssignsPositionalIDsInArgOrder(t *testing.T) {
	specs := classifyTargets([]string{"./b.png", "https://x/a.png", "./c.png"})
	ids := []string{specs[0].id, specs[1].id, specs[2].id}
	if !reflect.DeepEqual(ids, []string{"t1", "t2", "t3"}) {
		t.Fatalf("positional IDs broken: %v", ids)
	}
	if specs[0].isURL || !specs[1].isURL || specs[2].isURL {
		t.Fatalf("URL classification broken: %+v", specs)
	}
	// Order is contract: arg order, never sorted.
	if specs[0].arg != "./b.png" || specs[2].arg != "./c.png" {
		t.Fatalf("arg order not preserved: %+v", specs)
	}
}

func TestBuildDescribeTasksCanonicalOrderRegardlessOfFlagOrder(t *testing.T) {
	tasks := buildDescribeTasks(describeOptions{detect: true, caption: true})
	if len(tasks) != 2 {
		t.Fatalf("want 2 tasks, got %d", len(tasks))
	}
	// Canonical TaskKind order: caption before detect, task id = kind string.
	if tasks[0].ID != "caption" || string(tasks[0].Kind) != "caption" || tasks[0].Caption == nil {
		t.Fatalf("task 0 not caption: %+v", tasks[0])
	}
	if tasks[1].ID != "detect" || string(tasks[1].Kind) != "detect" || tasks[1].Detect == nil {
		t.Fatalf("task 1 not detect: %+v", tasks[1])
	}
}

func TestBuildDescribeTasksCarriesParams(t *testing.T) {
	tasks := buildDescribeTasks(describeOptions{
		caption: true, prompt: "focus on lighting",
		detect: true, labels: []string{"person", "car"},
		minConfidence: 0.4, minConfidenceSet: true,
	})
	if tasks[0].Caption.Prompt == nil || *tasks[0].Caption.Prompt != "focus on lighting" {
		t.Fatalf("prompt not carried: %+v", tasks[0].Caption)
	}
	if !reflect.DeepEqual(tasks[1].Detect.Labels, []string{"person", "car"}) {
		t.Fatalf("labels not carried: %+v", tasks[1].Detect)
	}
	if tasks[1].Detect.MinConfidence == nil || *tasks[1].Detect.MinConfidence != 0.4 {
		t.Fatalf("min confidence not carried: %+v", tasks[1].Detect)
	}
}

func TestValidateDescribeFlagsUsageErrors(t *testing.T) {
	cases := []struct {
		name string
		o    describeOptions
	}{
		{"no task flags", describeOptions{}},
		{"prompt without caption", describeOptions{detect: true, prompt: "x"}},
		{"labels without detect", describeOptions{caption: true, labels: []string{"a"}}},
		{"min-confidence without detect", describeOptions{caption: true, minConfidenceSet: true}},
	}
	for _, tc := range cases {
		if err := validateDescribeFlags(tc.o); err == nil {
			t.Fatalf("%s: want usage error", tc.name)
		}
	}
	if err := validateDescribeFlags(describeOptions{caption: true}); err != nil {
		t.Fatalf("valid flags rejected: %v", err)
	}
}
