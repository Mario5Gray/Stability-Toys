# ControlNet Metadata Transfer — Design

**Date:** 2026-07-04
**Status:** Approved (brainstorm), pending implementation plan

## Problem

ControlNet control maps carry no record of how they were produced, and the
generation image records nothing about the ControlNet parameters used to
render it. Two gaps:

1. The standalone map tools (`scripts/canny_map.py`, `scripts/depth_map.py`,
   `scripts/pose_map.py`) `save()` their output PNGs with **no embedded
   metadata**. The parameters that shaped the map (thresholds, model choice,
   etc.) are lost the moment the file is written.
2. The generation PNG records only an `lcm` `tEXt` chunk (prompt/seed/size/
   steps/cfg/negative/scheduler). Nothing about which control maps were used,
   how they were made, or the ControlNet generation parameters.

## Goal

1. Each of the three map tools stamps a JSON metadata chunk into its emitted
   PNG describing the tool, its parameters, and the source dimensions.
2. At render time the server reads any embedded metadata off each control map
   and writes a new `controlnet` chunk into the generation PNG, combining the
   map's own provenance (**if available**) with the ControlNet generation
   parameters actually used for that render.

## Approach — read at the byte seam

The worker already holds each map's raw PNG bytes via
`ControlNetBinding.control_image_bytes` (`server/controlnet_execution.py`). At
PNG-encode time it reads the embedded chunk directly from those bytes and
writes the combined `controlnet` chunk.

- No new plumbing through the job or the frozen `ControlNetBinding` dataclass.
- Works uniformly for any map carrying the chunk, regardless of how it reached
  the server (uploaded via `map_asset_ref`, reused emitted map).
- "If available" falls out naturally: no chunk on the map → `source: null`.

**Rejected alternative — thread through AssetStore metadata + binding fields.**
Extending `ControlNetBinding` with a provenance dict populated from
`AssetStore` metadata only works for server-generated maps and misses the
script-made uploaded maps that are the actual target, while adding fields to a
frozen dataclass and its resolver for no gain.

## Part 1 — Scripts embed their own metadata

New shared helper `scripts/cn_metadata.py`:

- `build_map_metadata(control_type, tool, params, source_size) -> dict`
  assembles the payload.
- `save_with_metadata(pil_image, destination, payload)` writes the PNG with a
  `tEXt` chunk keyed `controlnet_map` holding `json.dumps(payload)`.

Each tool calls `save_with_metadata(...)` in place of `result.save(destination)`.

### Packaging

`scripts/pyproject.toml` ships a **flat module list**, not a package:
`py-modules = ["depth_map", "pose_map", "canny_map"]`. A new top-level
`cn_metadata` module must be added to that list, otherwise `pip install
"./scripts[all]"` and the `st-canny-map` / `st-depth-map` / `st-pose-map`
entry points fail with `ModuleNotFoundError` at `import cn_metadata`. (Direct
`python scripts/<tool>.py` runs resolve it via the script's own directory on
`sys.path`, but the installed surface would not.)

The scripts import it as a top-level module (`import cn_metadata`), matching
how the flat layout already works.

`tests/test_canny_map.py::test_pyproject_exposes_canny_install_surface`
asserts the `py-modules` contents and must be updated to include
`cn_metadata`.

### `--colorize` preview is excluded

`depth_map.py --colorize` writes a second `_color.png` jet-colormap
**visualization**. That file is a human preview, not a control map fed to
ControlNet, and is **not** stamped with `controlnet_map`. Only the primary
grayscale map carries metadata.

### Chunk key

`controlnet_map` — a single PNG `tEXt` chunk on the control-map image.

### Payload schema

Common fields (all tools):

| field | meaning |
|---|---|
| `tool` | `"canny_map"` \| `"depth_map"` \| `"pose_map"` |
| `version` | integer schema version, starts at `1` |
| `control_type` | `"canny"` \| `"depth"` \| `"pose"` |
| `source_width` | source image width after any `--max-res` scaling |
| `source_height` | source image height after any `--max-res` scaling |
| `created_at` | ISO-8601 UTC timestamp |

Tool-specific fields:

- **canny**: `low_threshold`, `high_threshold`, `blur`, `invert`, `max_res`
- **depth**: `model`, `size`, `device`, `invert`, `max_res`
- **pose**: `model`, `parts`, `device`, `max_res`

`max_res` is recorded as given (may be `null`). `source_width`/`source_height`
reflect the image actually processed (post-`--max-res`).

`device` records the **requested `--device` CLI argument** (provenance = what
the operator asked for), for depth and pose. Note: `pose_map.py` currently
accepts `--device` but does not apply it to any of its detectors; the field
still records the requested value, and the divergence is a pre-existing script
behavior, not something this work changes.

## Part 2 — Worker transfers into the generation PNG

### Reader

New function `read_control_map_metadata(png_bytes) -> dict | None`:

- Opens the PNG, reads the `controlnet_map` `tEXt` chunk, `json.loads` it.
- Tolerant: missing chunk, malformed JSON, or decode failure all return
  `None`. Never raises into the render path.

Location: colocated with the reader's consumer. Placed in a small shared module
so both worker classes and tests import it (candidate:
`server/controlnet_metadata.py`; final location decided in the plan).

### Metadata assembly

New method on `CudaWorkerBase`, `_controlnet_metadata(bindings) -> list`:

For each binding, produce:

```json
{
  "attachment_id": "cn-1",
  "control_type": "canny",
  "generation": {
    "model_id": "sdxl-canny",
    "strength": 0.8,
    "start_percent": 0.0,
    "end_percent": 0.7
  },
  "source": { "...embedded controlnet_map payload, or null..." }
}
```

`generation` is drawn from the binding fields (`model_id`, `strength`,
`start_percent`, `end_percent`). `source` is
`read_control_map_metadata(binding.control_image_bytes)`.

### Chunk write

Both controlnet-capable render paths — `DiffusersCudaWorker.run_job`
(`backends/cuda_worker.py`, `lcm` chunk site) and
`DiffusersSDXLCudaWorker.run_job` (its `lcm` chunk site) — add a second `tEXt`
chunk:

```python
pnginfo.add_text("controlnet", json.dumps(self._controlnet_metadata(bindings)))
```

Written only when `bindings` is non-empty. The existing `lcm` chunk is
unchanged. The two `run_job_with_latents` paths never carry bindings and are
not touched.

## Data flow

```
canny_map/depth_map/pose_map
    └─ save_with_metadata → PNG + controlnet_map chunk
         └─ (upload) → AssetStore control_map/upload asset
              └─ resolve_controlnet_bindings → ControlNetBinding.control_image_bytes
                   └─ run_job: read_control_map_metadata(bytes) → source
                        + binding gen params → generation
                        └─ pnginfo.add_text("controlnet", [...]) on output PNG
```

## Non-goals

- **Server preprocessors unchanged.** `server/controlnet_preprocessors.py`
  (the `source_asset_ref + preprocess` path) is not modified. Maps produced
  there carry no `controlnet_map` chunk, so their render entry records
  `source: null`. Only the three standalone scripts stamp provenance, per
  agreed scope.
- **No change to `ControlNetBinding`** (frozen dataclass) or the
  `X-ControlNet-Artifacts` / `job:complete.controlnet_artifacts` response
  surface.
- **No new control types.** `pose` metadata is stamped by the tool; server
  execution support for `pose` is orthogonal and out of scope here.

## Testing

Runtime-model isolation is a hard constraint: `depth_map.py` and `pose_map.py`
call model-backed code (`depth_anything`/`midas`/`zoe`,
`openpose`/`dwpose`/`mediapipe`). Tests must not download or run models. Seams:

- **`cn_metadata` unit tests** (no models): call `build_map_metadata(...)` and
  assert payload shape/values; call `save_with_metadata(pil_image, dest,
  payload)` on a synthetic in-memory `PIL.Image`, reopen, assert the
  `controlnet_map` chunk round-trips. This is the primary coverage for the
  stamping logic and is fully offline.
- **canny script test**: canny is pure-CPU (`cv2`), so extend the existing
  `tests/test_canny_map.py` subprocess pattern — run on a fixture image,
  reopen output, assert the `controlnet_map` chunk contains the expected
  canny keys and post-`--max-res` `source_width`/`source_height`.
- **depth / pose script tests**: patch the model-invoking function
  (`monkeypatch` `depth_map.depth_anything` / `pose_map.dwpose` to return a
  small dummy `PIL.Image`), then invoke `main()` in-process and assert the
  saved PNG carries the expected `controlnet_map` payload. No subprocess, no
  model, no network.
- **Reader test**: valid chunk → dict; absent chunk → `None`; malformed JSON →
  `None`; non-PNG bytes → `None`.
- **Worker metadata test**: build bindings with (a) a script-stamped map and
  (b) a bare map; assert `_controlnet_metadata` returns two entries with
  correct `generation` params and `source` populated vs. `null` respectively.
- **Chunk-presence test**: render path with bindings emits a `controlnet`
  chunk; render path without bindings emits none.
