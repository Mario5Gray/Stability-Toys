# ControlNet Operator Guide

Ship-now guide for enabling and operating ControlNet in Stability-Toys.

This doc covers:

- what ControlNet support exists today
- which model artifacts operators must install
- how to wire `conf/modes.yml` and `conf/controlnets.yaml`
- which request patterns are supported
- what remains intentionally out of scope in v1

For CLI request syntax, see [`cli/go/USAGE.md`](cli/go/USAGE.md#controlnet).

## Support Boundary

Current v1 boundary:

- ControlNet execution supported on CUDA backend
- mode system required
- HTTP `/generate` and WebSocket `job:submit` support ControlNet
- built-in preprocessors: `canny`, `depth`
- configured direct-map control types: `canny`, `depth`, `pose`
- direct reusable control-map path supported via `map_asset_ref`
- preprocess path supported via `source_asset_ref + preprocess`
- multiple attachments supported, request order preserved
- img2img + ControlNet in same request, WS/CLI only, CUDA backend (SD1.5 and SDXL;
  shipped under `STABL-ztaxgbhv`) — capability-gated on
  `supports_img2img_and_controlnet`; non-capable backends reject fail-fast before
  preprocessing. Control-map aspect ratio must match the init image within 2%.

Not supported in v1:

- CPU backend ControlNet execution
- RKNN backend ControlNet execution
- MLX backend ControlNet execution
- img2img + ControlNet on non-CUDA backends (explicit non-goal, not a future v1.x item)
- img2img + ControlNet over HTTP `/generate` (WS/CLI only; HTTP has no `init_image_ref`)
- server-side preprocessor families beyond `canny` and `depth`

Important runtime rule:

- server rejects ControlNet request unless backend provider reports `supports_controlnet=True`
- in this repo, that means CUDA path

## Possible `control_type` Values

Shipped in repo now:

- `canny`
- `depth`
- `pose`

How server decides what is legal:

- mode policy in `conf/modes.yml` lists allowed `control_type` entries under `controlnet_policy.allowed_control_types`
- registry entry in `conf/controlnets.yaml` lists which `control_types` each `model_id` supports
- request must satisfy both

So effective allowed set for one mode is:

- values present in that mode's `allowed_control_types`
- and backed by a registry model whose `control_types` contains same value

Not shipped by default in repo config:

- `normal`
- `segmentation`

Important nuance:

- `control_type` is not hardcoded to closed enum in request model
- repo default mode/registry config ships `canny`, `depth`, and `pose`
- built-in server preprocessors only ship `canny` and `depth`
- `pose` maps can be generated with `scripts/pose_map.py` / `st-pose-map` and submitted via `map_asset_ref`
- additional types can be wired if operator adds matching mode policy and registry entries
- preprocess-driven use also needs matching server preprocessor registration

## Operator Model Checklist

Operator needs two model layers:

1. Base generation models referenced by `conf/modes.yml`
2. ControlNet models referenced by `conf/controlnets.yaml`

Families must match:

- SDXL mode -> use `compatible_with: [sdxl]` ControlNet models
- SD1.5 mode -> use `compatible_with: [sd15]` ControlNet models
- SD2 variants currently map to `sd15` ControlNet family for compatibility checks

Example shipped registry ids:

- `sdxl-canny`
- `sdxl-depth`
- `sdxl-openpose`
- `sd15-canny`
- `sd15-depth`
- `sd15-openpose`

Typical source models:

- `diffusers/controlnet-canny-sdxl-1.0`
- `diffusers/controlnet-depth-sdxl-1.0`
- `diffusers/controlnet-openpose-sdxl-1.0`
- `lllyasviel/sd-controlnet-canny`
- `lllyasviel/sd-controlnet-depth` or `lllyasviel/control_v11f1p_sd15_depth`
- `lllyasviel/control_v11p_sd15_openpose`

Preprocessor requirements:

- `canny`: OpenCV only, no extra model checkpoint
- `depth`: Hugging Face depth-estimation pipeline, default model `LiheYoung/depth-anything-small-hf`
- `pose`: offline helper only in v1 (`st-pose-map`); no built-in server-side pose preprocessor

If operator enables `allow_preprocess: true` for `depth`, host must be able to download or already have that depth model available.
If operator uses `pose`, generate the map first and submit it via `map_asset_ref`.

## Model Layout

Base model layout follows repo convention in [`docs/COMFYUI_MODEL_LAYOUT.md`](docs/COMFYUI_MODEL_LAYOUT.md).

ControlNet registry paths can live anywhere local, but practical convention is:

```text
/models/
  checkpoints/
  diffusers/
  loras/
  controlnets/
    sdxl-canny/
    sdxl-depth/
    controlnet-openpose-sdxl-1.0-safetensors/
    sd15-canny/
    sd15-depth/
    control_v11p_sd15_openpose_diffusers/
```

Each `conf/controlnets.yaml` entry points at local resolved path for one ControlNet model.

## `conf/controlnets.yaml`

`conf/controlnets.yaml` is global ControlNet registry. It maps `model_id` to local path and compatibility metadata.

Example shipped shape:

```yaml
models:
  sdxl-canny:
    path: /models/controlnets/sdxl-canny
    control_types: [canny]
    compatible_with: [sdxl]

  sdxl-depth:
    path: /models/controlnets/sdxl-depth
    control_types: [depth]
    compatible_with: [sdxl]

  sdxl-openpose:
    path: /models/controlnets/controlnet-openpose-sdxl-1.0-safetensors
    control_types: [pose]
    compatible_with: [sdxl]

  sd15-canny:
    path: /models/controlnets/sd15-canny
    control_types: [canny]
    compatible_with: [sd15]

  sd15-depth:
    path: /models/controlnets/sd15-depth
    control_types: [depth]
    compatible_with: [sd15]

  sd15-openpose:
    path: /models/controlnets/control_v11p_sd15_openpose_diffusers
    control_types: [pose]
    compatible_with: [sd15]
```

Field meaning:

- `path`: local filesystem path to ControlNet model artifact
- `control_types`: which attachment `control_type` values model accepts
- `compatible_with`: which active base-model families may use it

Registry rules:

- `model_id` in request must exist in registry
- requested `control_type` must appear in `control_types`
- active mode family must appear in `compatible_with`

Startup validation:

- default mode: strict
- server validates registry paths and mode references during startup
- missing path or bad family wiring fails fast at startup

Relevant env vars:

- `MODE_CONFIG_PATH`: config dir root; default registry path becomes `${MODE_CONFIG_PATH}/controlnets.yaml`
- `CONTROLNET_REGISTRY_PATH`: explicit registry file override
- `CONTROLNET_REGISTRY_VALIDATION`: `strict` or `lazy`

Recommended operator stance:

- use `strict` in normal deployments
- use `lazy` only for staged bring-up when files may mount later

## `conf/modes.yml`

`conf/modes.yml` owns per-mode policy, not global model lookup.

Each mode that should allow ControlNet needs `controlnet_policy`.

Example SDXL mode:

```yaml
model_root: /models
lora_root: /models/loras
default_mode: SDXL

resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
  sdxl:
    - size: 1024x1024
      aspect_ratio: "1:1"

modes:
  SDXL:
    model: checkpoints/sdxl-base.safetensors
    checkpoint_variant: sdxl-base
    resolution_set: sdxl
    default_size: 1024x1024
    default_steps: 20
    default_guidance: 7.5
    controlnet_policy:
      enabled: true
      max_attachments: 2
      allow_reuse_emitted_maps: true
      allowed_control_types:
        canny:
          default_model_id: sdxl-canny
          allowed_model_ids: [sdxl-canny]
          allow_preprocess: true
          default_strength: 1.0
          min_strength: 0.0
          max_strength: 2.0
        depth:
          default_model_id: sdxl-depth
          allowed_model_ids: [sdxl-depth]
          allow_preprocess: true
          default_strength: 1.0
          min_strength: 0.0
          max_strength: 2.0
```

Field meaning:

- `enabled`: hard gate; false means all ControlNet requests rejected
- `max_attachments`: upper bound per request
- `allow_reuse_emitted_maps`: policy signal that mode allows reuse of emitted control maps in later requests
- `allowed_control_types`: per-type policy table

Per-control-type fields:

- `default_model_id`: auto-filled when request omits `model_id`
- `allowed_model_ids`: optional allowlist; if set, request model must be in list
- `allow_preprocess`: allow `source_asset_ref + preprocess`
- `default_strength`: auto-filled when request omits `strength`
- `min_strength` / `max_strength`: allowed range

Important wiring rule:

- `default_model_id` and every `allowed_model_ids` entry must also exist in `conf/controlnets.yaml`

## Request Shapes

Operator should expect two legal attachment forms.

### Reusable control map

Client already has control map:

```json
{
  "attachment_id": "cn-1",
  "control_type": "canny",
  "model_id": "sdxl-canny",
  "map_asset_ref": "fileref:Rabc123",
  "strength": 0.8,
  "start_percent": 0.0,
  "end_percent": 0.7
}
```

### Source image plus preprocessing

Server derives control map first:

```json
{
  "attachment_id": "cn-1",
  "control_type": "depth",
  "source_asset_ref": "fileref:Rsrc123",
  "preprocess": {
    "id": "depth",
    "options": {}
  }
}
```

Attachment rules:

- exactly one of `map_asset_ref` or `source_asset_ref`
- `source_asset_ref` requires `preprocess`
- `map_asset_ref` must not include `preprocess`
- `start_percent <= end_percent`
- request may contain multiple attachments
- duplicate `attachment_id` rejected

## Server Behavior

Flow for valid request:

1. active mode resolved
2. mode `controlnet_policy` enforced
3. preprocess runs for any `source_asset_ref + preprocess` attachments
4. emitted control maps stored as `control_map` assets
5. attachment normalized to `map_asset_ref`
6. registry resolves `model_id -> path`
7. family compatibility checked
8. CUDA worker builds ordered ControlNet bindings
9. generation runs
10. response returns image plus emitted `controlnet_artifacts`

Response surfaces:

- HTTP success: `X-ControlNet-Artifacts` header
- WS success: `job:complete.controlnet_artifacts`

Failure behavior:

- policy/config/input errors -> fail fast before worker generation
- unsupported backend -> reject request
- registry mismatch -> reject request

## Operator Bring-Up

Recommended bring-up sequence:

1. Install base model used by target mode
2. Install matching ControlNet model directories
3. Fill `conf/controlnets.yaml`
4. Fill `controlnet_policy` in `conf/modes.yml`
5. Start server with strict registry validation
6. Confirm startup passes
7. Run one `canny` request from `source_asset_ref`
8. Run one `depth` request from `source_asset_ref`
9. Reuse emitted `map_asset_ref`
10. Run two-attachment request and confirm order-sensitive behavior

CLI check path:

```bash
st validate-track3 \
  --server http://host:4200 \
  --control-image ./canny-map.png \
  --control-type canny \
  --prompt "controlnet validation"
```

Full validation checklist: [`docs/TESTING_CONTROLNET_TRACK3.md`](docs/TESTING_CONTROLNET_TRACK3.md).

## Example Deployment Notes

### SDXL-only host

- install SDXL base checkpoint or diffusers pipeline
- install `sdxl-canny`, `sdxl-depth`, and `sdxl-openpose`
- only reference `sdxl-*` ControlNet model ids in mode policies

### SD1.5-only host

- install SD1.5 base model
- install `sd15-canny`, `sd15-depth`, and `sd15-openpose`
- set mode `checkpoint_variant` or detected family so server recognizes `sd15`

### Mixed host

- keep both base-model families and both ControlNet families
- per-mode policy decides which ids legal for each mode

## Current v1 Objectives and Limits

Shipped-now operator objective:

- stable CUDA ControlNet operation with explicit mode policy and explicit registry wiring
- combined img2img + ControlNet on CUDA over WS/CLI (`STABL-ztaxgbhv`), governed by
  `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md`

Still-open or intentionally deferred as built-in/default surfaces:

- non-CUDA backends
- img2img + ControlNet on RKNN/MLX/CPU (explicit non-goal — compounds onto the
  existing non-CUDA ControlNet deferral above, not a separate gap) and over HTTP
  `/generate` (would need a separate API decision to add `init_image_ref` there)
- more built-in server preprocessors like pose / normal / segmentation
- richer model-registry metadata beyond ControlNet-specific fields
- MLX runtime wiring; see [`docs/CONTROLNET_MLX_CONVERSION.md`](docs/CONTROLNET_MLX_CONVERSION.md)

## Troubleshooting

`unknown ControlNet model_id '...'`

- missing or misspelled registry entry in `conf/controlnets.yaml`

`model_id '...' is incompatible with active mode family '...'`

- wrong family pairing, like SD1.5 ControlNet on SDXL mode

`mode '...' does not enable ControlNet`

- mode missing `controlnet_policy.enabled: true`

`preprocessing not allowed for control_type '...'`

- mode policy has `allow_preprocess: false`

`ControlNet model path does not exist: ...`

- registry path wrong, or strict validation catching missing mount

`ControlNet provider not yet implemented on this backend`

- request reached non-CUDA or non-ControlNet-capable backend path
