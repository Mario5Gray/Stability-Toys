package main

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

type histSummary struct {
	ID                    int64                   `json:"id"`
	ExitCode              int                     `json:"exit_code"`
	Family                history.Family          `json:"family"`
	StartedAt             string                  `json:"started_at"`
	FinishedAt            string                  `json:"finished_at"`
	Command               string                  `json:"command"`
	Raw                   history.CommandView     `json:"raw"`
	Effective             *history.CommandView    `json:"effective,omitempty"`
	DerivedFromHistoryID  *int64                  `json:"derived_from_history_id,omitempty"`
	ReplayedFromHistoryID *int64                  `json:"replayed_from_history_id,omitempty"`
	ConflatePolicy        *history.PolicySnapshot `json:"conflate_policy,omitempty"`
	Error                 *string                 `json:"error"`
}

var histCmd = &cobra.Command{
	Use:   "hist [n]",
	Short: "Show local command history",
	Args:  cobra.MaximumNArgs(1),
	RunE:  runHist,
}

func init() {
	rootCmd.AddCommand(histCmd)
}

func runHist(cmd *cobra.Command, args []string) error {
	limit := 0
	if len(args) == 1 {
		n, err := strconv.Atoi(args[0])
		if err != nil || n < 1 {
			return fmt.Errorf("hist limit must be a positive integer")
		}
		limit = n
	}

	store, err := loadStateStore()
	if err != nil {
		return err
	}
	entries, err := store.List(cmd.Context())
	if err != nil {
		return err
	}
	summaries := summarizeHistory(entries, limit)
	if flagJSON {
		encoded, err := json.MarshalIndent(summaries, "", "  ")
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(cmd.OutOrStdout(), string(encoded))
		return err
	}
	return writeHistHuman(cmd, summaries)
}

func summarizeHistory(entries []history.Entry, limit int) []histSummary {
	out := make([]histSummary, 0, len(entries))
	for i := len(entries) - 1; i >= 0; i-- {
		if limit > 0 && len(out) >= limit {
			break
		}
		entry := entries[i]
		out = append(out, histSummary{
			ID:                    entry.ID,
			ExitCode:              entry.ExitCode,
			Family:                entry.Family,
			StartedAt:             entry.StartedAt,
			FinishedAt:            entry.FinishedAt,
			Command:               histCommand(entry),
			Raw:                   entry.Raw,
			Effective:             entry.Effective,
			DerivedFromHistoryID:  entry.DerivedFromHistoryID,
			ReplayedFromHistoryID: entry.ReplayedFromHistoryID,
			ConflatePolicy:        entry.ConflatePolicy,
			Error:                 entry.Error,
		})
	}
	return out
}

func histCommand(entry history.Entry) string {
	if entry.Effective != nil && entry.Effective.Display != "" {
		return entry.Effective.Display
	}
	if entry.Raw.Display != "" {
		return entry.Raw.Display
	}
	return history.RenderArgv(entry.Raw.Argv)
}

func writeHistHuman(cmd *cobra.Command, entries []histSummary) error {
	out := cmd.OutOrStdout()
	if _, err := fmt.Fprintln(out, "ID      EXIT  FAMILY    STARTED              COMMAND"); err != nil {
		return err
	}
	if _, err := fmt.Fprintln(out, "------  ----  --------  -------------------  -------"); err != nil {
		return err
	}
	for _, entry := range entries {
		if _, err := fmt.Fprintf(out, "%-6d  %-4d  %-8s  %-19s  %s\n", entry.ID, entry.ExitCode, entry.Family, histWhen(entry.StartedAt), entry.Command); err != nil {
			return err
		}
	}
	return nil
}

func histWhen(startedAt string) string {
	when := startedAt
	if dot := strings.Index(when, "."); dot >= 0 && strings.HasSuffix(when, "Z") {
		when = when[:dot] + "Z"
	}
	when = strings.Replace(when, "T", " ", 1)
	return strings.TrimSuffix(when, "Z")
}
