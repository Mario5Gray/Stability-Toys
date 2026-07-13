package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

var (
	conflateInclusive []string
	conflateExitCodes []int
)

var conflateCmd = &cobra.Command{
	Use:   "conflate [on|off|status|history:<id>]",
	Short: "Toggle or configure generation conflation policy",
	Args:  cobra.MaximumNArgs(1),
	RunE:  runConflate,
}

func init() {
	f := conflateCmd.Flags()
	f.StringArrayVar(&conflateInclusive, "inclusive", nil, "eligible family selector (v1: gen only)")
	f.IntSliceVar(&conflateExitCodes, "with-exit", nil, "eligible exit code (repeatable)")
	rootCmd.AddCommand(conflateCmd)
}

func runConflate(cmd *cobra.Command, args []string) error {
	store, err := loadStateStore()
	if err != nil {
		return err
	}
	ctx := cmd.Context()
	policy, err := store.LoadPolicy(ctx)
	if err != nil {
		return err
	}

	verb := ""
	if len(args) == 1 {
		verb = args[0]
	}
	hasSelectors := len(conflateInclusive) > 0 || len(conflateExitCodes) > 0
	if verb == "status" {
		if hasSelectors {
			return fmt.Errorf("conflate status does not accept selector arguments")
		}
		return writeConflateStatus(cmd, ctx, store, policy)
	}

	next, err := deriveConflatePolicy(ctx, store, policy, verb, conflateInclusive, conflateExitCodes)
	if err != nil {
		return err
	}
	next.UpdatedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if err := store.SavePolicy(ctx, next); err != nil {
		return err
	}
	return writeConflateStatus(cmd, ctx, store, next)
}

func deriveConflatePolicy(ctx context.Context, store history.HistoryStore, current history.Policy, verb string, inclusive []string, exits []int) (history.Policy, error) {
	next := current
	hasSelectors := len(inclusive) > 0 || len(exits) > 0
	if verb == "off" && hasSelectors {
		return history.Policy{}, fmt.Errorf("conflate off does not accept selector arguments")
	}
	if strings.HasPrefix(verb, "history:") && hasSelectors {
		return history.Policy{}, fmt.Errorf("history selector is mutually exclusive with --inclusive and --with-exit")
	}
	for _, family := range inclusive {
		if family != string(history.FamilyGen) {
			return history.Policy{}, fmt.Errorf("--inclusive accepts only gen in v1, got %q", family)
		}
	}
	for _, code := range exits {
		if code < 0 || code > 255 {
			return history.Policy{}, fmt.Errorf("--with-exit must be in 0..255, got %d", code)
		}
	}

	switch {
	case verb == "" && !hasSelectors:
		next.Enabled = !current.Enabled
	case verb == "on":
		next.Enabled = true
	case verb == "off":
		next.Enabled = false
	case strings.HasPrefix(verb, "history:"):
		id, err := strconv.ParseInt(strings.TrimPrefix(verb, "history:"), 10, 64)
		if err != nil || id < 1 {
			return history.Policy{}, fmt.Errorf("invalid history selector %q", verb)
		}
		entry, err := store.Get(ctx, id)
		if err != nil {
			return history.Policy{}, fmt.Errorf("history:%d: %w", id, err)
		}
		if entry.Family != history.FamilyGen || entry.Effective == nil || len(entry.Effective.Params) == 0 {
			return history.Policy{}, fmt.Errorf("history:%d is not an eligible gen baseline", id)
		}
		next.Enabled = true
		next.Selector = history.Selector{Kind: history.SelectorHistory, HistoryID: id}
	case verb == "" && hasSelectors:
		next.Enabled = true
	default:
		return history.Policy{}, fmt.Errorf("unknown conflate action %q", verb)
	}

	if hasSelectors {
		next.Enabled = true
		next.Selector = history.Selector{
			Kind:      history.SelectorRecent,
			Family:    history.FamilyGen,
			ExitCodes: history.NormalizeExitCodes(defaultExitCodes(exits)),
		}
	}
	return next, nil
}

func defaultExitCodes(exits []int) []int {
	if len(exits) == 0 {
		return []int{0}
	}
	return exits
}

func writeConflateStatus(cmd *cobra.Command, ctx context.Context, store history.HistoryStore, policy history.Policy) error {
	if !policy.Enabled {
		fmt.Fprintln(cmd.OutOrStdout(), "Conflation off.")
		return nil
	}
	if policy.Selector.Kind == history.SelectorHistory {
		entry, err := store.Get(ctx, policy.Selector.HistoryID)
		if err != nil {
			return err
		}
		fmt.Fprintln(cmd.OutOrStdout(), "Conflating only the selected history reference:")
		fmt.Fprintln(cmd.OutOrStdout(), entry.Effective.Display)
		return nil
	}
	if len(policy.Selector.ExitCodes) == 1 && policy.Selector.ExitCodes[0] == 0 {
		fmt.Fprintln(cmd.OutOrStdout(), "Conflating recent successful gen runs.")
		return nil
	}
	fmt.Fprintf(cmd.OutOrStdout(), "Conflating recent gen runs with exit code(s) %v.\n", policy.Selector.ExitCodes)
	return nil
}
