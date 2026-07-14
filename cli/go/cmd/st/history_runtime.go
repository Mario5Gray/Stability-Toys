package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/history"
)

type invocationPlan struct {
	family          history.Family
	kind            invocationKind
	rawArgv         []string
	rawDisplay      string
	rootGenPatch    *genPatch
	replayHistoryID int64
}

type invocationState struct {
	store history.Store
	id    int64
	final *invocationResult
}

type invocationResult struct {
	params                map[string]any
	displaySeed           *string
	derivedFromHistoryID  *int64
	replayedFromHistoryID *int64
	policySnapshot        *history.PolicySnapshot
}

type exitError struct {
	code int
	err  error
}

func (e exitError) Error() string {
	return e.err.Error()
}

func (e exitError) Unwrap() error {
	return e.err
}

func exitCodeOf(err error) int {
	if err == nil {
		return 0
	}
	var coded exitError
	if errors.As(err, &coded) {
		return coded.code
	}
	return 1
}

type invocationStateKey struct{}

func withInvocationState(ctx context.Context, state *invocationState) context.Context {
	return context.WithValue(ctx, invocationStateKey{}, state)
}

func stateFromContext(ctx context.Context) (*invocationState, error) {
	state, ok := ctx.Value(invocationStateKey{}).(*invocationState)
	if !ok || state == nil {
		return nil, fmt.Errorf("missing invocation state")
	}
	return state, nil
}

type invocationKind uint8

const (
	invocationCobra invocationKind = iota
	invocationRootGen
	invocationReplay
)

func executeCLI(ctx context.Context, argv []string) error {
	store, err := loadStateStore()
	if err != nil {
		return err
	}
	id, err := store.ReserveID(ctx)
	if err != nil {
		return err
	}

	started := time.Now().UTC()
	state := &invocationState{store: store, id: id}
	plan, err := buildInvocationPlan(ctx, store, argv)
	if err == nil {
		err = dispatchInvocation(withInvocationState(ctx, state), state, plan, argv)
	}
	if appendErr := appendHistory(ctx, state, started, plan, err); appendErr != nil {
		if err != nil {
			return fmt.Errorf("%v; append history: %w", err, appendErr)
		}
		return fmt.Errorf("append history: %w", appendErr)
	}
	return err
}

func buildInvocationPlan(ctx context.Context, store history.Store, argv []string) (invocationPlan, error) {
	rawArgv := append([]string{"st"}, argv...)
	plan := invocationPlan{
		family:     classifyFamily(argv),
		rawArgv:    rawArgv,
		rawDisplay: history.RenderArgv(rawArgv),
	}

	if len(argv) > 0 && strings.HasPrefix(argv[0], "-") && firstCommandToken(argv) == "" {
		policy, err := store.LoadPolicy(ctx)
		if err != nil {
			return plan, err
		}
		if policy.Enabled {
			patch, err := parseRootGenPatch(argv)
			if err != nil {
				return plan, err
			}
			if patch.Active {
				plan.kind = invocationRootGen
				plan.family = history.FamilyGen
				plan.rootGenPatch = &patch
			}
		}
	}

	if firstCommandToken(argv) == "replay" {
		plan.kind = invocationReplay
		plan.family = history.FamilyGen
	}
	return plan, nil
}

func classifyFamily(argv []string) history.Family {
	command := firstCommandToken(argv)
	if command == "" && len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		command = argv[0]
	}
	if command == "" {
		return history.FamilyUnknown
	}
	switch command {
	case "gen":
		return history.FamilyGen
	case "conflate":
		return history.FamilyConflate
	case "replay":
		return history.FamilyGen
	default:
		if isKnownTopLevelCommand(command) {
			return history.Family(command)
		}
		return history.FamilyUnknown
	}
}

func dispatchInvocation(ctx context.Context, _ *invocationState, plan invocationPlan, argv []string) error {
	setCommandContext(rootCmd, ctx)
	if plan.kind == invocationRootGen {
		return runConflatedRootGen(ctx, rootCmd, *plan.rootGenPatch)
	}
	rootCmd.SetArgs(argv)
	return rootCmd.ExecuteContext(ctx)
}

func setCommandContext(cmd interface {
	SetContext(context.Context)
	Commands() []*cobra.Command
}, ctx context.Context) {
	cmd.SetContext(ctx)
	for _, child := range cmd.Commands() {
		setCommandContext(child, ctx)
	}
}

func appendHistory(ctx context.Context, state *invocationState, started time.Time, plan invocationPlan, runErr error) error {
	exitCode := exitCodeOf(runErr)
	var summary *string
	if runErr != nil {
		text := runErr.Error()
		summary = &text
	}
	entry := history.Entry{
		SchemaVersion: 1,
		ID:            state.id,
		StartedAt:     started.Format(time.RFC3339Nano),
		FinishedAt:    time.Now().UTC().Format(time.RFC3339Nano),
		Family:        plan.family,
		Raw:           history.CommandView{Argv: plan.rawArgv, Display: plan.rawDisplay},
		ExitCode:      exitCode,
		Error:         summary,
	}
	if state.final != nil {
		displayParams := displayParamsWithSeedIntent(state.final.params, state.final.displaySeed)
		entry.Family = history.FamilyGen
		entry.Effective = &history.CommandView{
			Argv:    history.CanonicalGenArgv(displayParams),
			Display: history.CanonicalGenDisplay(displayParams),
			Params:  history.CloneParams(state.final.params),
		}
		entry.DerivedFromHistoryID = state.final.derivedFromHistoryID
		entry.ReplayedFromHistoryID = state.final.replayedFromHistoryID
		entry.ConflatePolicy = state.final.policySnapshot
	}
	return state.store.Append(ctx, entry)
}
