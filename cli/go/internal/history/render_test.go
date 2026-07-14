package history

import (
	"reflect"
	"strings"
	"testing"
)

func TestRenderRawDisplayShellEscapesArgv(t *testing.T) {
	tests := []struct {
		name string
		argv []string
		want string
	}{
		{name: "spaces apostrophes and empty", argv: []string{"st", "--prompt", "bartender's horse", ""}, want: "st --prompt 'bartender'\\''s horse' ''"},
		{name: "shell metacharacters", argv: []string{"st", "--prompt", "horse;echo"}, want: "st --prompt 'horse;echo'"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RenderArgv(tt.argv); got != tt.want {
				t.Fatalf("display = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCanonicalGenArgvNormalizesPromptAndStableOrder(t *testing.T) {
	params := map[string]any{
		"prompt":              "horse bartender",
		"guidance_scale":      4.5,
		"size":                "1024x1024",
		"num_inference_steps": 20,
		"seed":                int64(421337),
	}
	got := CanonicalGenArgv(params)
	want := []string{"st", "gen", "--prompt", "horse bartender", "--size", "1024x1024", "--steps", "20", "--cfg", "4.5", "--seed", "421337"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("argv = %#v, want %#v", got, want)
	}
	if display := CanonicalGenDisplay(params); display != "st gen --prompt 'horse bartender' --size 1024x1024 --steps 20 --cfg 4.5 --seed 421337" {
		t.Fatalf("display = %q", display)
	}
}

func TestSelectorSnapshotForPinnedHistory(t *testing.T) {
	got := SnapshotSelector(Selector{Kind: SelectorHistory, HistoryID: 12345})
	want := &PolicySnapshot{Selector: "history", HistoryID: 12345}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("snapshot = %#v, want %#v", got, want)
	}
}

func TestSelectorSnapshotForRecentFamily(t *testing.T) {
	got := SnapshotSelector(Selector{Kind: SelectorRecent, Family: FamilyGen})
	want := &PolicySnapshot{Selector: "recent:gen"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("snapshot = %#v, want %#v", got, want)
	}
	if got := SnapshotSelector(Selector{Kind: "future"}); got != nil {
		t.Fatalf("unknown selector snapshot = %#v, want nil", got)
	}
}

func TestValidatePolicy(t *testing.T) {
	validRecent := DefaultPolicy()
	validHistory := Policy{
		SchemaVersion: 1,
		Enabled:       true,
		Selector:      Selector{Kind: SelectorHistory, HistoryID: 9},
	}

	tests := []struct {
		name    string
		policy  Policy
		wantErr string
	}{
		{name: "default recent", policy: validRecent},
		{name: "pinned history", policy: validHistory},
		{name: "schema", policy: Policy{SchemaVersion: 2}, wantErr: "schema_version"},
		{name: "history id", policy: Policy{SchemaVersion: 1, Selector: Selector{Kind: SelectorHistory}}, wantErr: "positive history_id"},
		{name: "recent family", policy: Policy{SchemaVersion: 1, Selector: Selector{Kind: SelectorRecent, Family: FamilyUnknown, ExitCodes: []int{0}}}, wantErr: "family gen"},
		{name: "recent exits", policy: Policy{SchemaVersion: 1, Selector: Selector{Kind: SelectorRecent, Family: FamilyGen}}, wantErr: "exit code"},
		{name: "exit below range", policy: Policy{SchemaVersion: 1, Selector: Selector{Kind: SelectorRecent, Family: FamilyGen, ExitCodes: []int{-1}}}, wantErr: "invalid exit code"},
		{name: "exit above range", policy: Policy{SchemaVersion: 1, Selector: Selector{Kind: SelectorRecent, Family: FamilyGen, ExitCodes: []int{256}}}, wantErr: "invalid exit code"},
		{name: "selector", policy: Policy{SchemaVersion: 1, Selector: Selector{Kind: "future"}}, wantErr: "selector kind"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidatePolicy(tt.policy)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidatePolicy() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ValidatePolicy() error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestCloneParamsAndNormalizeExitCodes(t *testing.T) {
	original := map[string]any{"prompt": "horse", "seed": int64(42)}
	clone := CloneParams(original)
	clone["prompt"] = "owl"
	if original["prompt"] != "horse" {
		t.Fatalf("CloneParams aliased source map: %#v", original)
	}

	if got, want := NormalizeExitCodes([]int{1, 0, 1, 255, 0}), []int{0, 1, 255}; !reflect.DeepEqual(got, want) {
		t.Fatalf("NormalizeExitCodes() = %#v, want %#v", got, want)
	}
}
