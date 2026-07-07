# `st read` — ControlNet Metadata Support

**Date:** 2026-07-06
**Status:** Design (approved; ready for spec review)

## Motivation

`st read <image.png>` (`cli/go/cmd/st/read.go`) currently prints only the `lcm` tEXt
chunk — generation parameters stamped onto generation-output PNGs by the CUDA worker
(`backends/cuda_worker.py`). It cannot surface ControlNet provenance, which lives in two
other tEXt chunks the backend already writes:

| Chunk keyword | Written onto | Shape | Written by |
|---|---|---|---|
| `lcm` | generation-output PNGs | flat dict of generation params | `backends/cuda_worker.py` |
| `controlnet` | generation-output PNGs, **alongside `lcm`** whenever the generation used a ControlNet binding | **list** of per-attachment entries (`attachment_id`, `control_type`, `generation` params, and the source map's `controlnet_map` payload inline under `source`) | `backends/cuda_worker.py:_controlnet_metadata` |
| `controlnet_map` | standalone control-map PNGs (the map image itself) | flat dict (`tool`, `version`, `control_type`, `source_width`, `source_height`, `created_at`, plus tool params) | `scripts/cn_metadata.py` |

`lcm` + `controlnet` co-occurring is the **normal** case whenever a generation used
ControlNet — not an edge case. This design makes `st read` detect all three chunks and
print whichever are present, so both control-map files and ControlNet-generated outputs
are inspectable from the CLI, matching the project's CLI-first delivery philosophy.

## Design

### `internal/pngmeta` — generalize chunk lookup, one shared walk

Today `ReadLCM` hardcodes the `lcm` keyword and both parses chunks and hard-errors when
absent in a single function. `BakedParams` (used by `st gen --recreate`) depends on that
error-on-absent behavior and must be unaffected.

To make "one PNG chunk walk, not three" actually true for `runRead` (which needs all
three keywords from the same file), parsing is split from lookup: a `Chunks` value
parses the PNG once, and each `Find*` method queries the already-parsed result — no
re-walk per keyword.

```go
// Chunks is a PNG parsed once so its tEXt chunks can be queried by keyword without
// re-walking the file for each lookup.
type Chunks struct{ chunks []chunk }

// Parse walks pngBytes once into a queryable Chunks value.
func Parse(pngBytes []byte) (Chunks, error)

// text returns the raw tEXt payload for keyword, or ok=false if absent. Absence is
// not an error; a present-but-malformed chunk's JSON error surfaces from the
// Find* method that decodes it.
func (c Chunks) text(keyword string) ([]byte, bool)

func (c Chunks) FindLCM() (map[string]any, bool, error)            // new
func (c Chunks) FindControlNetMap() (map[string]any, bool, error)  // new
func (c Chunks) FindControlNet() ([]any, bool, error)              // new — list, not dict

func ReadLCM(pngBytes []byte) (map[string]any, error) // UNCHANGED signature/behavior;
                                                        // reimplemented as Parse + FindLCM
```

`runRead` calls `pngmeta.Parse(data)` once, then all three `Find*` methods against the
same `Chunks` value — genuinely one walk.

`ReadLCM` keeps its exact current contract (`fmt.Errorf("no lcm tEXt chunk")` when
absent), reimplemented internally as `Parse` + `FindLCM`, so `BakedParams` and
`st gen --recreate` are untouched by this refactor.

### `st read` — detect three chunks, one top-level key per chunk found

```go
func runRead(cmd *cobra.Command, args []string) error {
    data, err := os.ReadFile(args[0])
    if err != nil { return err }

    chunks, err := pngmeta.Parse(data)
    if err != nil { return err }

    out := map[string]any{}
    if v, ok, err := chunks.FindLCM(); err != nil {
        return err
    } else if ok {
        out["lcm"] = v
    }
    if v, ok, err := chunks.FindControlNet(); err != nil {
        return err
    } else if ok {
        out["controlnet"] = v
    }
    if v, ok, err := chunks.FindControlNetMap(); err != nil {
        return err
    } else if ok {
        out["controlnet_map"] = v
    }

    if len(out) == 0 {
        return fmt.Errorf("no known metadata chunk (lcm, controlnet, controlnet_map) found in %s", args[0])
    }

    b, err := json.MarshalIndent(out, "", "  ")
    if err != nil { return err }
    fmt.Fprintln(cmd.OutOrStdout(), string(b))
    return nil
}
```

Rules:
- Every chunk found gets its own top-level key, named after its own keyword. No
  schema coupling between `controlnet` (list) and `controlnet_map` (dict) — each is
  passed through as-is, never merged or normalized.
- If a present chunk's JSON is malformed, `read` fails loud immediately (propagates
  the JSON error) rather than silently omitting it.
- If none of the three chunks are present, `read` errors — this is a breaking
  behavior change in error *message* only (today's message is `"no lcm tEXt chunk"`);
  the exit-nonzero behavior for "nothing to show" is unchanged.
- Output is **always wrapped** under the chunk-keyword key, including the single-`lcm`
  case (breaking change from today's flat `{"prompt":...}` output — approved).

### Example outputs

```
st read ./output.png                 # lcm only (no controlnet used)
{ "lcm": { "prompt": "...", "seed": 42 } }

st read ./output_with_controlnet.png # lcm + controlnet (normal case when CN was used)
{
  "lcm": { "prompt": "...", "seed": 42 },
  "controlnet": [
    { "attachment_id": "cn_1", "control_type": "canny",
      "generation": { "model_id": "...", "strength": 0.8, "start_percent": 0.0, "end_percent": 1.0 },
      "source": { "tool": "canny_map", "control_type": "canny", "source_width": 1024, "source_height": 1024, "created_at": "..." } }
  ]
}

st read ./control_map.png            # standalone control-map file
{ "controlnet_map": { "tool": "canny_map", "control_type": "canny", "source_width": 1024, "source_height": 1024, "created_at": "..." } }
```

### Docs

- `cli/go/README.md`: update the one-line `st read` description to mention all three
  chunks.
- `cli/go/USAGE.md`, "Reading PNG metadata" section:
  - update the example output block (currently shows flat `lcm` JSON) to the wrapped
    shape
  - update the prose line "Returns the raw `lcm` tEXt chunk as JSON. No server call;
    works offline." — it currently describes only the old flat-`lcm` behavior and must
    be revised to cover all three chunks (the "no server call; works offline" part
    stays true and unchanged).

### Testing

CLI-level (`cli/go/cmd/st/read_test.go`):
- `TestReadPrintsLCM` (existing): tighten to assert the `"lcm"` wrapper key present,
  not just substring match on `prompt`/`owl`.
- New: PNG with only `controlnet_map` → output wrapped under `controlnet_map`.
- New: PNG with `lcm` + `controlnet` → both keys present, `controlnet` value is a list.
- New: PNG with none of the three chunks → error.
- New: PNG with a chunk present but malformed JSON → error propagates (not silently
  dropped).
- `TestRecreateSeedsParams` (existing, `--recreate`): must stay green unmodified,
  confirming `ReadLCM`/`BakedParams` behavior is unaffected by the refactor.

Package-level (`cli/go/internal/pngmeta/pngmeta_test.go`), exercising the refactored
seam directly — both must stay green unmodified since `ReadLCM`'s contract doesn't
change:
- `TestWriteThenReadLCM` (existing): `ReadLCM` round-trip via `Parse`+`FindLCM`.
- `TestBakedParamsMapsToRequestFields` (existing): `BakedParams` → `ReadLCM`, unaffected.
- New: `Parse` + `FindControlNet` on a chunk with a JSON list payload.
- New: `Parse` + `FindControlNetMap` on a chunk with a JSON dict payload.
- New: a single `Parse` call whose `Chunks` value answers all three `Find*` lookups
  correctly on a PNG carrying `lcm` + `controlnet` together (the co-occurring case).

## Out of scope

- No new command/subcommand — this extends the existing `st read`.
- No change to how the backend writes any of the three chunks.
- No fetch-by-asset-ref-from-server path (explicitly deferred; this design is
  local-file-only, per the scoping decision made during brainstorming).
- No normalization/merging of `controlnet` list entries with `controlnet_map` dicts —
  they are printed as independent, differently-shaped values.
