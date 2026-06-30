package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/config"
	"github.com/darkbit/stability-toys/cli/st/internal/output"
	"github.com/darkbit/stability-toys/cli/st/internal/pngmeta"
	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

// gen command flag backing vars (pointer-or-nil resolved via flag.Changed).
var (
	genPrompt      string
	genNegative    string
	genSize        string
	genSteps       int
	genCfg         float64
	genSeed        string
	genScheduler   string
	genMode        string
	genSR          int
	genInitImage   string
	genRecreate    string
	genControlnets    []string
	genControlnetFile string
	genOutfile        string
	genStream         bool
	genQuiet          bool
)

// genArgs is the resolved set of generation inputs. Pointer fields are nil when
// the user left the corresponding flag unset, so config/baked layers show
// through (see config.ResolveParams).
type genArgs struct {
	Prompt      string
	Negative    *string
	Genres      *string
	Steps       *int
	Cfg         *float64
	Seed        *string
	Scheduler   *string
	Mode        *string
	SR          *int
	InitImage   string
	Recreate    string
	Controlnets    []string
	ControlnetFile string
	Outfile        string
}

func (a genArgs) toFlags() config.Flags {
	return config.Flags{
		Prompt:    a.Prompt,
		Negative:  a.Negative,
		Genres:    a.Genres,
		Steps:     a.Steps,
		Cfg:       a.Cfg,
		Seed:      a.Seed,
		Scheduler: a.Scheduler,
		Mode:      a.Mode,
		SRLevel:   a.SR,
	}
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
	f.Float64Var(&genCfg, "cfg", 0, "guidance scale")
	f.StringVar(&genSeed, "seed", "", `seed integer or "random"`)
	f.StringVar(&genScheduler, "scheduler", "", "scheduler id")
	f.StringVar(&genMode, "mode", "", "model mode to switch to before generating")
	f.IntVar(&genSR, "sr", 0, "super-resolution magnitude (1-3; 0 = off)")
	f.StringVar(&genInitImage, "init-image", "", "img2img source: local PNG path or fileref:ID")
	f.StringVar(&genRecreate, "recreate", "", "local PNG whose lcm params seed this generation (recipe only)")
	f.StringArrayVar(&genControlnets, "controlnet", nil, "ControlNetAttachment as JSON (repeatable)")
	f.StringVar(&genControlnetFile, "controlnet-file", "", "ControlNetAttachment JSON file (merged with --controlnet entries)")
	f.StringVar(&genOutfile, "outfile", "", "explicit output path (else auto out-####)")
	f.BoolVar(&genStream, "stream", false, "stream progress as NDJSON to stdout (job_id, progress events, complete)")
	f.BoolVar(&genQuiet, "quiet", false, "suppress progress and job_id output on stderr")
	rootCmd.AddCommand(genCmd)
}

// genArgsFromFlags reads the cobra flag state into a genArgs. flag.Changed lets
// an explicitly-set zero value (e.g. --cfg 0) stay distinct from "unset".
func genArgsFromFlags(cmd *cobra.Command, args []string) genArgs {
	f := cmd.Flags()
	a := genArgs{
		InitImage:      genInitImage,
		Recreate:       genRecreate,
		Controlnets:    genControlnets,
		ControlnetFile: genControlnetFile,
		Outfile:        genOutfile,
	}
	if len(args) > 0 {
		a.Prompt = strings.Join(args, " ")
	}
	if f.Changed("prompt") {
		a.Prompt = genPrompt
	}
	if f.Changed("negative") {
		a.Negative = &genNegative
	}
	if f.Changed("size") {
		a.Genres = &genSize
	}
	if f.Changed("steps") {
		a.Steps = &genSteps
	}
	if f.Changed("cfg") {
		a.Cfg = &genCfg
	}
	if f.Changed("seed") {
		a.Seed = &genSeed
	}
	if f.Changed("scheduler") {
		a.Scheduler = &genScheduler
	}
	if f.Changed("mode") {
		a.Mode = &genMode
	}
	if f.Changed("sr") {
		a.SR = &genSR
	}
	return a
}

// localRecipePath returns a local PNG (recreate first, then init-image) whose
// lcm metadata should seed precedence layer 2. fileref: inputs and non-existent
// paths are ignored.
func localRecipePath(a genArgs) string {
	for _, cand := range []string{a.Recreate, a.InitImage} {
		if cand == "" || strings.HasPrefix(cand, "fileref:") {
			continue
		}
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	return ""
}

// buildGenParams layers config < baked PNG < flags into the WS params. It sets
// init_image_ref only for the fileref: case; local-file upload happens in the
// command runner. --recreate contributes baked params but never an image ref.
func buildGenParams(cfg *config.Config, a genArgs) (stclient.GenParams, error) {
	if cfg == nil {
		cfg = &config.Config{}
	}
	var baked map[string]any
	if local := localRecipePath(a); local != "" {
		data, err := os.ReadFile(local)
		if err != nil {
			return nil, err
		}
		baked, err = pngmeta.BakedParams(data)
		if err != nil {
			return nil, err
		}
	}
	p := config.ResolveParams(cfg, baked, a.toFlags())

	if ref, ok := strings.CutPrefix(a.InitImage, "fileref:"); ok && a.InitImage != "" {
		p["init_image_ref"] = ref
	}
	if len(a.Controlnets) > 0 {
		cns := make([]any, 0, len(a.Controlnets))
		for _, raw := range a.Controlnets {
			var cn map[string]any
			if err := json.Unmarshal([]byte(raw), &cn); err != nil {
				return nil, fmt.Errorf("--controlnet %q: %w", raw, err)
			}
			cns = append(cns, cn)
		}
		p["controlnets"] = cns
	}
	if a.ControlnetFile != "" {
		data, err := os.ReadFile(a.ControlnetFile)
		if err != nil {
			return nil, fmt.Errorf("--controlnet-file %q: %w", a.ControlnetFile, err)
		}
		var cn map[string]any
		if err := json.Unmarshal(data, &cn); err != nil {
			return nil, fmt.Errorf("--controlnet-file %q: invalid JSON: %w", a.ControlnetFile, err)
		}
		cns, _ := p["controlnets"].([]any)
		p["controlnets"] = append(cns, cn)
	}
	return stclient.GenParams(p), nil
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
	cfg, err := requireConfig()
	if err != nil {
		return err
	}
	a := genArgsFromFlags(cmd, args)
	params, err := buildGenParams(cfg, a)
	if err != nil {
		return err
	}

	ctx := cmd.Context()
	client := newClient()

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
	onAck, onProgress := buildObservationCallbacks(cmd, genQuiet, genStream)
	jobID, res, err := client.Generate(ctx, params, onAck, onProgress)
	_ = jobID // surfaced to caller via onAck; reserved for future st watch composition
	if err != nil {
		return err
	}

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
