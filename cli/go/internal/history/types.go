package history

import "fmt"

type Family string

const (
	FamilyGen      Family = "gen"
	FamilyConflate Family = "conflate"
	FamilyUnknown  Family = "unknown"
)

type SelectorKind string

const (
	SelectorRecent  SelectorKind = "recent"
	SelectorHistory SelectorKind = "history"
)

type CommandView struct {
	Argv    []string       `json:"argv"`
	Display string         `json:"display"`
	Params  map[string]any `json:"params,omitempty"`
}

type Selector struct {
	Kind      SelectorKind `json:"kind"`
	Family    Family       `json:"family,omitempty"`
	ExitCodes []int        `json:"exit_codes,omitempty"`
	HistoryID int64        `json:"history_id,omitempty"`
}

type Policy struct {
	SchemaVersion int      `json:"schema_version"`
	Enabled       bool     `json:"enabled"`
	Selector      Selector `json:"selector"`
	UpdatedAt     string   `json:"updated_at"`
}

type PolicySnapshot struct {
	Selector  string `json:"selector"`
	HistoryID int64  `json:"history_id,omitempty"`
}

type Entry struct {
	SchemaVersion         int             `json:"schema_version"`
	ID                    int64           `json:"id"`
	StartedAt             string          `json:"started_at"`
	FinishedAt            string          `json:"finished_at"`
	Family                Family          `json:"family"`
	Raw                   CommandView     `json:"raw"`
	Effective             *CommandView    `json:"effective,omitempty"`
	ExitCode              int             `json:"exit_code"`
	DerivedFromHistoryID  *int64          `json:"derived_from_history_id,omitempty"`
	ReplayedFromHistoryID *int64          `json:"replayed_from_history_id,omitempty"`
	ConflatePolicy        *PolicySnapshot `json:"conflate_policy,omitempty"`
	Error                 *string         `json:"error"`
}

func DefaultPolicy() Policy {
	return Policy{
		SchemaVersion: 1,
		Enabled:       false,
		Selector: Selector{
			Kind:      SelectorRecent,
			Family:    FamilyGen,
			ExitCodes: []int{0},
		},
	}
}

func ValidatePolicy(policy Policy) error {
	if policy.SchemaVersion != 1 {
		return fmt.Errorf("unsupported policy schema_version %d", policy.SchemaVersion)
	}

	switch policy.Selector.Kind {
	case SelectorHistory:
		if policy.Selector.HistoryID < 1 {
			return fmt.Errorf("history selector requires a positive history_id")
		}
	case SelectorRecent:
		if policy.Selector.Family != FamilyGen || len(policy.Selector.ExitCodes) == 0 {
			return fmt.Errorf("recent selector requires family gen and at least one exit code")
		}
		for _, code := range policy.Selector.ExitCodes {
			if code < 0 || code > 255 {
				return fmt.Errorf("invalid exit code %d", code)
			}
		}
	default:
		return fmt.Errorf("unsupported selector kind %q", policy.Selector.Kind)
	}

	return nil
}
