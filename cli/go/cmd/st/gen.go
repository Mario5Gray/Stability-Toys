package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/darkbit/stability-toys/cli/st/internal/config"
	"github.com/darkbit/stability-toys/cli/st/internal/history"
	"github.com/darkbit/stability-toys/cli/st/internal/output"
	"github.com/darkbit/stability-toys/cli/st/internal/pngmeta"
	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

const uploadAssetTTLSeconds = 300

// gen command flag backing vars (pointer-or-nil resolved via flag.Changed).
var (
	genPrompt          string
	genNegative        string
	genSize            string
	genSteps           int
	genSkipStep        int
	genCfg             float64
	genSeed            string
	genScheduler       string
	genMode            string
	genSR              int
	genInitImage       string
	genRecreate        string
	genControlnets     []string
	genControlnetFile  string
	genControlRefs     []string
	genControlImages   []string
	genControlStrength float64
	genOutfile         string
	genStream          bool
	genReset           bool
	genQuiet           bool
)

// genArgs is the resolved set of generation inputs. Pointer fields are nil when
// the user left the corresponding flag unset, so config/baked layers show
// through (see config.ResolveParams).
type genArgs struct {
	Prompt          string
	Negative        *string
	Genres          *string
	Steps           *int
	SkipStep        *int
	Cfg             *float64
	Seed            *string
	Scheduler       *string
	Mode            *string
	SR              *int
	InitImage       string
	Recreate        string
	Controlnets     []string
	ControlnetFile  string
	ControlRefs     []string
	ControlImages   []string
	ControlStrength *float64
	Outfile         string
}

func (a genArgs) toFlags() config.Flags {
	return config.Flags{
		Prompt:    a.Prompt,
		Negative:  a.Negative,
		Genres:    a.Genres,
		Steps:     a.Steps,
		SkipStep:  a.SkipStep,
		Cfg:       a.Cfg,
		Seed:      a.Seed,
		Scheduler: a.Scheduler,
		Mode:      a.Mode,
		SRLevel:   a.SR,
	}
}

type genPatch struct {
	Active  bool
	Reset   bool
	Args    genArgs
	Changed map[string]bool
}

type genFlagValues struct {
	Prompt          string
	Negative        string
	Size            string
	Steps           int
	SkipStep        int
	Cfg             float64
	Seed            string
	Scheduler       string
	Mode            string
	SR              int
	InitImage       string
	Recreate        string
	Controlnets     []string
	ControlnetFile  string
	ControlRefs     []string
	ControlImages   []string
	ControlStrength float64
	Outfile         string
	Stream          bool
	Quiet           bool
	Reset           bool
}

func parseRootGenPatch(argv []string) (genPatch, error) {
	if len(argv) == 0 || !strings.HasPrefix(argv[0], "-") {
		return genPatch{}, nil
	}
	if firstCommandToken(argv) != "" {
		return genPatch{}, nil
	}

	var values genFlagValues
	rootValues := currentRootFlagValues()
	f := pflag.NewFlagSet("st gen shorthand", pflag.ContinueOnError)
	f.SetInterspersed(true)
	bindGenFlags(f, &values)
	bindRootPersistentFlags(f, &rootValues)
	if err := f.Parse(argv); err != nil {
		return genPatch{}, err
	}
	changed := changedGenFlags(f)
	if len(changed) == 0 && !values.Reset {
		return genPatch{}, nil
	}
	if f.NArg() != 0 {
		if isKnownTopLevelCommand(f.Arg(0)) {
			return genPatch{}, nil
		}
		return genPatch{}, fmt.Errorf("root shorthand is flag-only; pass positional text with --prompt or use explicit st gen")
	}

	applyRootFlagValues(rootValues)
	applyGenExecutionValues(values)
	return genPatch{
		Active:  true,
		Reset:   values.Reset,
		Args:    genArgsFromFlagSet(f, values, nil),
		Changed: changed,
	}, nil
}

func firstCommandToken(argv []string) string {
	values := currentRootFlagValues()
	f := pflag.NewFlagSet("st root", pflag.ContinueOnError)
	f.SetInterspersed(false)
	bindRootPersistentFlags(f, &values)
	if err := f.Parse(argv); err != nil || f.NArg() == 0 {
		return ""
	}
	return f.Arg(0)
}

func isKnownTopLevelCommand(name string) bool {
	for _, cmd := range rootCmd.Commands() {
		if cmd.Name() == name {
			return true
		}
	}
	return false
}

func bindGenFlags(f *pflag.FlagSet, v *genFlagValues) {
	f.StringVar(&v.Prompt, "prompt", "", "prompt text (else positional args)")
	f.StringVar(&v.Negative, "negative", "", "negative prompt")
	f.StringVar(&v.Size, "size", "", "image size, e.g. 512x512")
	f.IntVar(&v.Steps, "steps", 0, "inference steps")
	f.IntVar(&v.SkipStep, "skip-step", 0, "number of timesteps to skip (LCM skip_step)")
	f.Float64Var(&v.Cfg, "cfg", 0, "guidance scale")
	f.StringVar(&v.Seed, "seed", "", `seed integer or "random"`)
	f.StringVar(&v.Scheduler, "scheduler", "", "scheduler id")
	f.StringVar(&v.Mode, "mode", "", "model mode to switch to before generating")
	f.IntVar(&v.SR, "sr", 0, "super-resolution magnitude (1-3; 0 = off)")
	f.StringVar(&v.InitImage, "init-image", "", "img2img source: local PNG path or fileref:ID")
	f.StringVar(&v.Recreate, "recreate", "", "local PNG whose lcm params seed this generation (recipe only)")
	f.StringArrayVar(&v.Controlnets, "controlnet", nil, "ControlNetAttachment as JSON (repeatable)")
	f.StringVar(&v.ControlnetFile, "controlnet-file", "", "ControlNetAttachment JSON file (merged with --controlnet entries)")
	f.StringArrayVar(&v.ControlRefs, "control-ref", nil, "reuse a control map ref directly: type:<asset-ref> (repeatable)")
	f.StringArrayVar(&v.ControlImages, "control-image", nil, "auto-upload a control image and attach it: type:<path> (repeatable)")
	f.Float64Var(&v.ControlStrength, "control-strength", 0, "ControlNet conditioning strength for --control-ref/--control-image attachments (0.0-2.0; unset = mode default)")
	f.StringVar(&v.Outfile, "outfile", "", "explicit output path (else auto out-####)")
	f.BoolVar(&v.Stream, "stream", false, "stream progress as NDJSON to stdout (job_id, progress events, complete)")
	f.BoolVar(&v.Quiet, "quiet", false, "suppress progress and job_id output on stderr")
	f.BoolVar(&v.Reset, "reset", false, "clean slate: ignore the conflation baseline for this run (conflation stays on)")
}

func genArgsFromFlagSet(f *pflag.FlagSet, v genFlagValues, args []string) genArgs {
	a := genArgs{
		InitImage:      v.InitImage,
		Recreate:       v.Recreate,
		Controlnets:    v.Controlnets,
		ControlnetFile: v.ControlnetFile,
		ControlRefs:    v.ControlRefs,
		ControlImages:  v.ControlImages,
		Outfile:        v.Outfile,
	}
	if len(args) > 0 {
		a.Prompt = strings.Join(args, " ")
	}
	if f.Changed("prompt") {
		a.Prompt = v.Prompt
	}
	if f.Changed("negative") {
		a.Negative = &v.Negative
	}
	if f.Changed("size") {
		a.Genres = &v.Size
	}
	if f.Changed("steps") {
		a.Steps = &v.Steps
	}
	if f.Changed("skip-step") {
		a.SkipStep = &v.SkipStep
	}
	if f.Changed("cfg") {
		a.Cfg = &v.Cfg
	}
	if f.Changed("seed") {
		a.Seed = &v.Seed
	}
	if f.Changed("scheduler") {
		a.Scheduler = &v.Scheduler
	}
	if f.Changed("mode") {
		a.Mode = &v.Mode
	}
	if f.Changed("sr") {
		a.SR = &v.SR
	}
	if f.Changed("control-strength") {
		a.ControlStrength = &v.ControlStrength
	}
	return a
}

func changedGenFlags(f *pflag.FlagSet) map[string]bool {
	changed := map[string]bool{}
	// Output-only flags are intentionally gen flags: with conflation enabled,
	// they can rerun the selected baseline while changing observation/output behavior.
	for _, name := range []string{
		"prompt", "negative", "size", "steps", "skip-step", "cfg", "seed",
		"scheduler", "mode", "sr", "init-image", "recreate", "controlnet",
		"controlnet-file", "control-ref", "control-image", "control-strength", "outfile", "stream", "quiet",
	} {
		if f.Changed(name) {
			changed[name] = true
		}
	}
	return changed
}

type rootFlagValues struct {
	Server    string
	Config    string
	OutputDir string
	JSON      bool
	Timeout   time.Duration
}

func currentRootFlagValues() rootFlagValues {
	return rootFlagValues{
		Server:    flagServer,
		Config:    flagConfig,
		OutputDir: flagOutputDir,
		JSON:      flagJSON,
		Timeout:   flagTimeout,
	}
}

func bindRootPersistentFlags(f *pflag.FlagSet, v *rootFlagValues) {
	f.StringVar(&v.Server, "server", v.Server, "backend base URL (or $ST_SERVER)")
	f.StringVar(&v.Config, "config", v.Config, configFlagHelp())
	f.StringVarP(&v.OutputDir, "output-dir", "o", v.OutputDir, "directory for generated images (overrides config)")
	f.BoolVar(&v.JSON, "json", v.JSON, "emit machine-readable JSON")
	f.DurationVar(&v.Timeout, "timeout", v.Timeout, "per-request timeout (0 = client default)")
}

func applyRootFlagValues(v rootFlagValues) {
	flagServer = v.Server
	flagConfig = v.Config
	flagOutputDir = v.OutputDir
	flagJSON = v.JSON
	flagTimeout = v.Timeout
}

func applyGenExecutionValues(v genFlagValues) {
	genOutfile = v.Outfile
	genStream = v.Stream
	genQuiet = v.Quiet
}

func bindGenExecutionFlags(f *pflag.FlagSet) {
	f.StringVar(&genOutfile, "outfile", "", "explicit output path (else auto out-####)")
	f.BoolVar(&genStream, "stream", false, "stream progress as NDJSON to stdout (job_id, progress events, complete)")
	f.BoolVar(&genQuiet, "quiet", false, "suppress progress and job_id output on stderr")
}

var genCmd = &cobra.Command{
	Use:   "gen [prompt]",
	Short: "Generate an image",
	Args:  cobra.ArbitraryArgs,
	RunE:  runGen,
}

func init() {
	f := genCmd.Flags()
	f.StringVar(&genPrompt, "prompt", "", "prompt text (else positional args)")
	f.StringVar(&genNegative, "negative", "", "negative prompt")
	f.StringVar(&genSize, "size", "", "image size, e.g. 512x512")
	f.IntVar(&genSteps, "steps", 0, "inference steps")
	f.IntVar(&genSkipStep, "skip-step", 0, "number of timesteps to skip (LCM skip_step)")
	f.Float64Var(&genCfg, "cfg", 0, "guidance scale")
	f.StringVar(&genSeed, "seed", "", `seed integer or "random"`)
	f.StringVar(&genScheduler, "scheduler", "", "scheduler id")
	f.StringVar(&genMode, "mode", "", "model mode to switch to before generating")
	f.IntVar(&genSR, "sr", 0, "super-resolution magnitude (1-3; 0 = off)")
	f.StringVar(&genInitImage, "init-image", "", "img2img source: local PNG path or fileref:ID")
	f.StringVar(&genRecreate, "recreate", "", "local PNG whose lcm params seed this generation (recipe only)")
	f.StringArrayVar(&genControlnets, "controlnet", nil, "ControlNetAttachment as JSON (repeatable)")
	f.StringVar(&genControlnetFile, "controlnet-file", "", "ControlNetAttachment JSON file (merged with --controlnet entries)")
	f.StringArrayVar(&genControlRefs, "control-ref", nil, "reuse a control map ref directly: type:<asset-ref> (repeatable)")
	f.StringArrayVar(&genControlImages, "control-image", nil, "auto-upload a control image and attach it: type:<path> (repeatable)")
	f.Float64Var(&genControlStrength, "control-strength", 0, "ControlNet conditioning strength for --control-ref/--control-image attachments (0.0-2.0; unset = mode default)")
	f.StringVar(&genOutfile, "outfile", "", "explicit output path (else auto out-####)")
	f.BoolVar(&genStream, "stream", false, "stream progress as NDJSON to stdout (job_id, progress events, complete)")
	f.BoolVar(&genQuiet, "quiet", false, "suppress progress and job_id output on stderr")
	f.BoolVar(&genReset, "reset", false, "clean slate: ignore the conflation baseline for this run (conflation stays on)")
	rootCmd.AddCommand(genCmd)
}

// genArgsFromFlags reads the cobra flag state into a genArgs. flag.Changed lets
// an explicitly-set zero value (e.g. --cfg 0) stay distinct from "unset".
func genArgsFromFlags(cmd *cobra.Command, args []string) genArgs {
	f := cmd.Flags()
	return genArgsFromFlagSet(f, genFlagValues{
		Prompt:          genPrompt,
		Negative:        genNegative,
		Size:            genSize,
		Steps:           genSteps,
		SkipStep:        genSkipStep,
		Cfg:             genCfg,
		Seed:            genSeed,
		Scheduler:       genScheduler,
		Mode:            genMode,
		SR:              genSR,
		InitImage:       genInitImage,
		Recreate:        genRecreate,
		Controlnets:     genControlnets,
		ControlnetFile:  genControlnetFile,
		ControlRefs:     genControlRefs,
		ControlImages:   genControlImages,
		ControlStrength: genControlStrength,
		Outfile:         genOutfile,
		Stream:          genStream,
		Quiet:           genQuiet,
	}, args)
}

// localRecipePath returns a local image (recreate first, then init-image) whose
// lcm metadata should seed precedence layer 2, and whether baked params are
// mandatory. Reading the recipe is --recreate's entire purpose, so it is
// required; an init image may be any decodable format, so missing/unreadable
// baked params just mean no baked layer. fileref: inputs and non-existent
// paths are ignored.
func localRecipePath(a genArgs) (path string, required bool) {
	for _, cand := range []string{a.Recreate, a.InitImage} {
		if cand == "" || strings.HasPrefix(cand, "fileref:") {
			continue
		}
		if _, err := os.Stat(cand); err == nil {
			return cand, cand == a.Recreate
		}
	}
	return "", false
}

// buildGenParams layers config < baked PNG < flags into the WS params. It sets
// init_image_ref only for the fileref: case; local-file upload happens in the
// command runner. --recreate contributes baked params but never an image ref.
func buildGenParams(cfg *config.Config, a genArgs) (stclient.GenParams, error) {
	return buildGenParamsWithBaseline(cfg, a, nil, inferChangedInputs(a))
}

func buildConflatedParams(ctx context.Context, store history.Store, patch genPatch, cfg *config.Config) (stclient.GenParams, *history.Entry, *history.PolicySnapshot, error) {
	policy, err := store.LoadPolicy(ctx)
	if err != nil {
		return nil, nil, nil, err
	}
	if !policy.Enabled {
		params, err := buildGenParamsWithBaseline(cfg, patch.Args, nil, patch.Changed)
		return params, nil, nil, err
	}
	if patch.Reset {
		// Clean slate: conflation stays enabled, but this run inherits nothing.
		// The run is still recorded, so the next gen conflates forward from it.
		params, err := buildGenParamsWithBaseline(cfg, patch.Args, nil, patch.Changed)
		return params, nil, nil, err
	}
	baseline, err := selectBaseline(ctx, store, policy.Selector)
	if err != nil {
		if errors.Is(err, history.ErrNoEligibleEntry) && !patch.Active {
			params, buildErr := buildGenParamsWithBaseline(cfg, patch.Args, nil, patch.Changed)
			return params, nil, nil, buildErr
		}
		return nil, nil, nil, err
	}
	params, err := buildGenParamsWithBaseline(cfg, patch.Args, baseline.Effective.Params, patch.Changed)
	if err != nil {
		return nil, nil, nil, err
	}
	return params, &baseline, history.SnapshotSelector(policy.Selector), nil
}

func selectBaseline(ctx context.Context, store history.HistoryStore, selector history.Selector) (history.Entry, error) {
	if selector.Kind == history.SelectorHistory {
		entry, err := store.Get(ctx, selector.HistoryID)
		if err != nil {
			return history.Entry{}, err
		}
		if entry.Family != history.FamilyGen || entry.Effective == nil || len(entry.Effective.Params) == 0 {
			return history.Entry{}, fmt.Errorf("history:%d is not an eligible gen baseline", selector.HistoryID)
		}
		return entry, nil
	}
	return store.Latest(ctx, history.Filter{
		Family:           history.FamilyGen,
		ExitCodes:        selector.ExitCodes,
		RequireEffective: true,
	})
}

func buildGenParamsWithBaseline(cfg *config.Config, a genArgs, baseline stclient.GenParams, changed map[string]bool) (stclient.GenParams, error) {
	baked, err := loadCurrentBakedParams(a)
	if err != nil {
		return nil, err
	}
	if changed == nil {
		changed = map[string]bool{}
	}
	p := stclient.GenParams(config.ResolveParamsWithBaseline(cfg, baked, baseline, a.toFlags()))

	if changed["init-image"] {
		delete(p, "init_image_ref")
	}
	if ref, ok := strings.CutPrefix(a.InitImage, "fileref:"); ok && a.InitImage != "" {
		p["init_image_ref"] = ref
	}

	if changed["controlnet"] || changed["controlnet-file"] || changed["control-ref"] || changed["control-image"] {
		delete(p, "controlnets")
		if err := applyCurrentControlnetInputs(cfg, a, p); err != nil {
			return nil, err
		}
	}
	return p, nil
}

func loadCurrentBakedParams(a genArgs) (map[string]any, error) {
	local, required := localRecipePath(a)
	if local == "" {
		return nil, nil
	}
	data, err := os.ReadFile(local)
	if err == nil {
		var baked map[string]any
		baked, err = pngmeta.BakedParams(data)
		if err == nil {
			return baked, nil
		}
	}
	if required {
		return nil, err
	}
	return nil, nil
}

func applyCurrentControlnetInputs(cfg *config.Config, a genArgs, p stclient.GenParams) error {
	cns := make([]any, 0, len(a.Controlnets)+len(a.ControlRefs)+1)
	for _, raw := range a.Controlnets {
		if presetName, ok := strings.CutPrefix(raw, "@"); ok {
			var preset config.ControlnetPreset
			if cfg != nil {
				preset = cfg.ControlnetPresets[presetName]
			}
			if preset == nil {
				return fmt.Errorf("--controlnet @%s: preset not found in config", presetName)
			}
			cns = append(cns, map[string]any(preset))
			continue
		}
		var cn map[string]any
		if err := json.Unmarshal([]byte(raw), &cn); err != nil {
			return fmt.Errorf("--controlnet %q: %w", raw, err)
		}
		cns = append(cns, cn)
	}
	for _, raw := range a.ControlRefs {
		controlType, mapAssetRef := parseControlRefArg(raw)
		if controlType == "" {
			return fmt.Errorf("--control-ref %q: missing control_type prefix (use type:<asset-ref>, e.g. canny:fileref:MAP1)", raw)
		}
		if mapAssetRef == "" {
			return fmt.Errorf("--control-ref %q: missing asset ref", raw)
		}
		attachment := map[string]any{
			"attachment_id": fmt.Sprintf("ctrl-%d", len(cns)),
			"control_type":  controlType,
			"map_asset_ref": mapAssetRef,
		}
		if a.ControlStrength != nil {
			attachment["strength"] = *a.ControlStrength
		}
		cns = append(cns, attachment)
	}
	if a.ControlnetFile != "" {
		data, err := os.ReadFile(a.ControlnetFile)
		if err != nil {
			return fmt.Errorf("--controlnet-file %q: %w", a.ControlnetFile, err)
		}
		var cn map[string]any
		if err := json.Unmarshal(data, &cn); err != nil {
			return fmt.Errorf("--controlnet-file %q: invalid JSON: %w", a.ControlnetFile, err)
		}
		cns = append(cns, cn)
	}
	if len(cns) > 0 {
		p["controlnets"] = cns
	}
	return nil
}

func inferChangedInputs(a genArgs) map[string]bool {
	return map[string]bool{
		"init-image":      a.InitImage != "",
		"controlnet":      len(a.Controlnets) > 0,
		"controlnet-file": a.ControlnetFile != "",
		"control-ref":     len(a.ControlRefs) > 0,
		"control-image":   len(a.ControlImages) > 0,
	}
}

func loadReplayParams(ctx context.Context, store history.HistoryStore, id int64) (stclient.GenParams, history.Entry, error) {
	entry, err := store.Get(ctx, id)
	if err != nil {
		return nil, history.Entry{}, err
	}
	if entry.Family != history.FamilyGen || entry.Effective == nil || len(entry.Effective.Params) == 0 {
		return nil, history.Entry{}, fmt.Errorf("history:%d is not replayable", id)
	}
	return stclient.GenParams(history.CloneParams(entry.Effective.Params)), entry, nil
}

// buildObservationCallbacks returns onAck and onProgress callbacks for
// Generate based on the active output flags.
//   - quiet:  both nil (silent)
//   - stream: NDJSON to stdout — job_id line on ack, progress lines per frame
//   - default: job_id + progress delta to stderr
func buildObservationCallbacks(cmd *cobra.Command, quiet, stream bool) (func(string), func(string)) {
	if quiet {
		return nil, nil
	}
	if stream {
		onAck := func(id string) {
			b, _ := json.Marshal(map[string]any{"job_id": id})
			fmt.Fprintln(cmd.OutOrStdout(), string(b))
		}
		onProg := func(delta string) {
			b, _ := json.Marshal(map[string]any{"event": "progress", "delta": delta})
			fmt.Fprintln(cmd.OutOrStdout(), string(b))
		}
		return onAck, onProg
	}
	onAck := func(id string) { fmt.Fprintf(cmd.ErrOrStderr(), "job_id=%s\n", id) }
	onProg := func(delta string) { fmt.Fprint(cmd.ErrOrStderr(), delta) }
	return onAck, onProg
}

func runGen(cmd *cobra.Command, args []string) error {
	patch := genPatch{
		Reset:   cmd.Flags().Changed("reset"),
		Args:    genArgsFromFlags(cmd, args),
		Changed: changedGenFlags(cmd.Flags()),
	}
	return runPatchedGen(cmd, patch)
}

func runConflatedRootGen(ctx context.Context, cmd *cobra.Command, patch genPatch) error {
	cmd.SetContext(ctx)
	return runPatchedGen(cmd, patch)
}

func runPatchedGen(cmd *cobra.Command, patch genPatch) error {
	state, err := stateFromContext(cmd.Context())
	if err != nil {
		return err
	}
	cfg, err := requireConfig()
	if err != nil {
		return err
	}
	params, baseline, snapshot, err := buildConflatedParams(cmd.Context(), state.store, patch, cfg)
	if err != nil {
		if patch.Active && errors.Is(err, history.ErrNoEligibleEntry) {
			return fmt.Errorf("%w: run a full st gen command or pin history:<id>", err)
		}
		return err
	}
	return executeResolvedGen(cmd, cfg, patch.Args, params, func(final stclient.GenParams) {
		displaySeed := explicitSeedDisplayIntent(patch.Args)
		state.final = &invocationResult{params: final, displaySeed: displaySeed, policySnapshot: snapshot}
		if baseline == nil {
			return
		}
		state.final.derivedFromHistoryID = &baseline.ID
		if !genQuiet {
			fmt.Fprintf(cmd.ErrOrStderr(), "initial command [id=%d]: %s\n", baseline.ID, baseline.Effective.Display)
			fmt.Fprintf(cmd.ErrOrStderr(), "next command [id=%d]: %s\n", state.id, history.CanonicalGenDisplay(displayParamsWithSeedIntent(final, displaySeed)))
		}
	})
}

func explicitSeedDisplayIntent(a genArgs) *string {
	if a.Seed == nil || *a.Seed == "" {
		return nil
	}
	seed := *a.Seed
	return &seed
}

func displayParamsWithSeedIntent(params map[string]any, displaySeed *string) map[string]any {
	displayParams := history.CloneParams(params)
	if displaySeed != nil {
		if _, ok := displayParams["seed"]; !ok {
			displayParams["seed"] = *displaySeed
		}
	}
	return displayParams
}

func executeResolvedGen(cmd *cobra.Command, cfg *config.Config, a genArgs, params stclient.GenParams, beforeSubmit func(stclient.GenParams)) error {
	if cfg == nil {
		cfg = &config.Config{}
	}

	ctx := cmd.Context()
	client := newClient()

	// --control-image type:<path>: upload each file and inject a controlnet attachment.
	if len(a.ControlImages) > 0 {
		if err := resolveControlImages(cmd.Context(), client, a.ControlImages, a.ControlStrength, params); err != nil {
			return err
		}
	}

	// Local init-image: upload to get a fileref the backend can resolve.
	if a.InitImage != "" && !strings.HasPrefix(a.InitImage, "fileref:") {
		data, err := os.ReadFile(a.InitImage)
		if err != nil {
			return err
		}
		ref, err := client.Upload(ctx, filepath.Base(a.InitImage), data, "")
		if err != nil {
			return err
		}
		params["init_image_ref"] = ref
	}

	// Switch model mode only when the resolved mode differs from the live one.
	if m, ok := params["mode"].(string); ok && m != "" {
		if cur, err := client.CurrentMode(ctx); err == nil && cur != m {
			if err := client.SwitchMode(ctx, m); err != nil {
				return err
			}
		}
	}

	if genStream && flagJSON {
		return fmt.Errorf("--stream and --json are mutually exclusive")
	}
	if beforeSubmit != nil {
		beforeSubmit(params)
	}
	onAck, onProgress := buildObservationCallbacks(cmd, genQuiet, genStream)
	jobID, res, err := client.Generate(ctx, params, onAck, onProgress)
	_ = jobID // surfaced to caller via onAck; reserved for future st watch composition
	if err != nil {
		writeExpiredAssetRefAdvisory(cmd, err)
		return err
	}
	params["seed"] = res.Seed

	img, err := client.FetchStorage(ctx, res.StorageKey)
	if err != nil {
		return err
	}

	if cfg.Defaults.IncludeMeta {
		// Best-effort client metadata; skip if the bytes are not a PNG.
		if stamped, err := pngmeta.WriteText(img, "lcm-client", clientMeta(cfg, res)); err == nil {
			img = stamped
		}
	}

	dir := flagOutputDir
	if dir == "" {
		dir = cfg.Defaults.OutputDirectory
	}
	format := cfg.Defaults.OutputFormat
	if format == "" {
		format = "png"
	}
	path, err := output.Resolve(a.Outfile, dir, format)
	if err != nil {
		return err
	}
	if err := output.Write(path, img); err != nil {
		return err
	}

	return printGenResult(cmd, path, res)
}

func writeExpiredAssetRefAdvisory(cmd *cobra.Command, err error) {
	if !isExpiredAssetRefError(err) {
		return
	}
	fmt.Fprintf(cmd.ErrOrStderr(), `
asset advisory:
  An asset ref from history/conflation is no longer available.
  Upload refs are temporary by default and expire after %d seconds.
  If you expect to reuse asset refs across longer conflation or replay sessions,
  run the backend with asset persistence enabled and retention configured:

    ASSET_STORE_PROVIDER=FILESYSTEM
    FS_STORAGE_DIR=/data/image-cache
    FS_STORAGE_TTL_S=604800
    FS_STORAGE_CLEANUP_INTERVAL_S=3600

  Tiered asset persistence is server-side infrastructure in v1; v1 does not expose a public promote endpoint or st asset promote command.
  If this ref has already expired, re-upload the original asset and start from a fresh baseline.

`, uploadAssetTTLSeconds)
}

func isExpiredAssetRefError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "file ref expired") {
		return true
	}
	if strings.Contains(msg, "asset ref") && strings.Contains(msg, "not found or evicted") {
		return true
	}
	return strings.Contains(msg, "asset") && strings.Contains(msg, "expired")
}

// resolveControlImages uploads each --control-image entry and appends the
// resulting controlnet attachment to params["controlnets"].
// Each entry must be "type:<path>"; omitting the type prefix is an error
// because control_type is required by the server.
func resolveControlImages(ctx context.Context, client *stclient.Client, entries []string, strength *float64, params stclient.GenParams) error {
	cns, _ := params["controlnets"].([]any)
	baseIndex := len(cns)
	for i, entry := range entries {
		bucket, filePath := parseUploadArg(entry)
		if bucket == "" {
			return fmt.Errorf("--control-image %q: missing control_type prefix (use type:<path>, e.g. depth:./map.png)", entry)
		}
		data, err := os.ReadFile(filePath)
		if err != nil {
			return fmt.Errorf("--control-image %q: %w", entry, err)
		}
		ref, err := client.Upload(ctx, filepath.Base(filePath), data, bucket)
		if err != nil {
			return fmt.Errorf("--control-image %q: upload: %w", entry, err)
		}
		attachment := map[string]any{
			"attachment_id": fmt.Sprintf("ctrl-%d", baseIndex+i),
			"control_type":  bucket,
			"map_asset_ref": ref,
		}
		// Leave strength unset when the flag wasn't passed so the server applies
		// the mode policy's default_strength.
		if strength != nil {
			attachment["strength"] = *strength
		}
		cns = append(cns, attachment)
	}
	params["controlnets"] = cns
	return nil
}

func parseControlRefArg(arg string) (controlType, ref string) {
	before, after, ok := strings.Cut(arg, ":")
	if !ok {
		return "", ""
	}
	return before, after
}

func clientMeta(cfg *config.Config, res *stclient.Result) string {
	m := map[string]any{
		"producer_name": cfg.Defaults.Meta.ProducerName,
		"seed":          res.Seed,
	}
	b, _ := json.Marshal(m)
	return string(b)
}

func printGenResult(cmd *cobra.Command, path string, res *stclient.Result) error {
	if genStream {
		out := map[string]any{
			"event":       "complete",
			"output":      path,
			"seed":        res.Seed,
			"storage_key": res.StorageKey,
			"storage_url": res.StorageURL,
		}
		b, err := json.Marshal(out)
		if err != nil {
			return err
		}
		fmt.Fprintln(cmd.OutOrStdout(), string(b))
		return nil
	}
	if flagJSON {
		out := map[string]any{
			"output":      path,
			"storage_key": res.StorageKey,
			"storage_url": res.StorageURL,
			"seed":        res.Seed,
		}
		b, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			return err
		}
		fmt.Fprintln(cmd.OutOrStdout(), string(b))
		return nil
	}
	fmt.Fprintf(cmd.OutOrStdout(), "wrote %s (seed %d)\n", path, res.Seed)
	return nil
}
