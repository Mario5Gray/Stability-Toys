package main

import (
	"context"
	"testing"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

func TestReplayUsesEffectiveParamsExactly(t *testing.T) {
	root := t.TempDir()
	store := history.NewFSStore(root)
	sourceID, _ := store.ReserveID(context.Background())
	_ = store.Append(context.Background(), history.Entry{
		SchemaVersion: 1,
		ID:            sourceID,
		Family:        history.FamilyGen,
		Raw:           history.CommandView{Argv: []string{"st", "gen"}, Display: "st gen"},
		Effective: &history.CommandView{
			Argv:    []string{"st", "gen", "--prompt", "owl", "--cfg", "4.5"},
			Display: "st gen --prompt owl --cfg 4.5",
			Params:  map[string]any{"prompt": "owl", "guidance_scale": 4.5, "seed": 421337},
		},
		ExitCode: 1,
	})

	params, entry, err := loadReplayParams(context.Background(), store, sourceID)
	if err != nil {
		t.Fatal(err)
	}
	if entry.ID != sourceID || numericParam(params["seed"]) != 421337 {
		t.Fatalf("entry=%#v params=%#v", entry, params)
	}
}

func TestReplayRejectsGenerationOverrides(t *testing.T) {
	root := t.TempDir()
	for _, args := range [][]string{
		{"replay", "1", "--cfg", "3"},
		{"replay", "1", "replacement prompt"},
	} {
		_, err := runCmdMayFailWithStateRoot(t, root, args...)
		if err == nil {
			t.Fatalf("%v unexpectedly accepted", args)
		}
	}
}
