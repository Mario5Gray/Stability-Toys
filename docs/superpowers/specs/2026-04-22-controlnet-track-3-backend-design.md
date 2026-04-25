# ControlNet Track 3 Backend Design

## Summary

This spec narrows ControlNet Track 3 to backend execution only. Track 2 is already merged and provides the request contract, mode-owned policy, preprocessors, typed asset store, and HTTP/WS artifact emission on the current stub path. Track 3 replaces that stub on the CUDA mode-system path with real ControlNet execution, a global local-path model registry, ordered multi-attachment bindings, and successful HTTP/WS response emission. Frontend work moves out of this track and is explicitly deferred to Track 4.

The backend objective is simple: a ControlNet request that passes validation should execute successfully on the CUDA path, honor all requested attachments in order, and return both the generated image/result and any emitted `controlnet_artifacts` on the success path. If any attachment is invalid or incompatible, the whole request must fail fast before worker execution.

## Goals

- Replace the current ControlNet dispatch stub with real CUDA execution on the mode-system path.
- Support both preprocess-driven attachments (`source_asset_ref + preprocess`) and direct reusable `map_asset_ref` attachments.
- Add a global, mode-agnostic `conf/controlnets.yaml` registry for backend model resolution.
- Enforce fail-fast compatibility validation between the active mode's base model family and every requested ControlNet attachment.
- Preserve attachment order and support multiple simultaneous attachments from day one.
- Return `controlnet_artifacts` on successful HTTP `/generate` responses and successful WS `job:complete` frames.
- Add a process-local runtime cache for loaded ControlNet models with pinning during active generations.
- Support configurable registry validation policy: strict startup validation or lazy first-use validation.

## Non-Goals

- Frontend/UI controls, rendering, or reuse flows. Those move to Track 4.
- RKNN or other non-CUDA execution paths.
- Remote downloads or Hugging Face identifiers in the registry.
- Persistent cache across restarts.
- Drawing/sketch input, img2img ControlNet, or additional preprocessors beyond the Track 2 seam.
- Best-effort partial attachment execution. Any invalid attachment fails the whole request.

## Current State

- [`server/controlnet_constraints.py`](/Users/darkbit1001/workspace/Stability-Toys/server/controlnet_constraints.py) still raises `NotImplementedError` for any validated ControlNet request.
- [`server/controlnet_preprocessing.py`](/Users/darkbit1001/workspace/Stability-Toys/server/controlnet_preprocessing.py) already resolves `source_asset_ref`, runs preprocessors through `DEFAULT_REGISTRY`, emits `control_map` assets, and backfills `attachment.map_asset_ref`.
- [`server/lcm_sr_server.py`](/Users/darkbit1001/workspace/Stability-Toys/server/lcm_sr_server.py) and [`server/ws_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/server/ws_routes.py) already thread `controlnet_artifacts` through the current stub/error path.
- [`server/mode_config.py`](/Users/darkbit1001/workspace/Stability-Toys/server/mode_config.py) already exposes `controlnet_policy` through `/api/modes`.
- Track 2 deliberately proved the backend seam up to “validated + preprocessed + emitted artifact refs” without executing a real ControlNet provider.

## Proposed Boundary

Track 3 backend owns:

- ControlNet backend registry loading and validation.
- Request-time model and attachment compatibility checks.
- CUDA provider execution and ordered ControlNet binding.
- Resolution-aware control-map decode/resize at generation time.
- Process-local loaded-model caching and pinning.
- Success-path response emission for HTTP and WS.

Track 4 owns:

- Rendering the legal ControlNet choices in the UI.
- Attachment add/remove/edit flows.
- Emitted-artifact browsing and reuse from the frontend.

This keeps Track 3 “backend complete” without coupling it to frontend delivery.

## Registry Design

Track 3 introduces a global, mode-agnostic local-file registry:

`conf/controlnets.yaml`

Recommended shape:

```yaml
models:
  sdxl-canny:
    path: models/controlnet/sdxl-canny
    control_types: [canny]
    compatible_with: [sdxl]

  sdxl-depth:
    path: models/controlnet/sdxl-depth
    control_types: [depth]
    compatible_with: [sdxl]
```

Rules:

- `model_id` is the stable external key already referenced by mode policy.
- `path` must be a local path already present on disk.
- `control_types` declares which attachment types the model may serve.
- `compatible_with` declares which base-model families may use the model.
- No remote identifiers or on-demand download behavior are supported in v1.

## Validation Policy

Validation mode is configurable:

- `strict`
  - load and validate `conf/controlnets.yaml` at startup
  - fail startup if any registry entry is malformed
  - fail startup if any mode references a missing or incompatible `model_id`

- `lazy`
  - parse enough configuration to boot
  - defer registry-entry path/compatibility failures to first request that needs them

Recommended implementation shape:

- a dedicated registry loader module
- a configuration toggle such as `CONTROLNET_REGISTRY_VALIDATION=strict|lazy`
- shared validation logic reused by both startup and request-time code paths

## Compatibility Model

Track 3 must fail fast.

For every requested attachment, before execution begins:

1. resolve `attachment.model_id` in `conf/controlnets.yaml`
2. verify the registry entry exists
3. verify the registry entry supports the attachment’s `control_type`
4. verify the registry entry is compatible with the active mode’s base-model family

If any one attachment fails those checks:

- reject the whole request
- do not run best-effort partial generation
- do not silently drop attachments

This preserves trust in the request contract and avoids misleading output that only honors part of what the caller asked for.

## Provider Design

Track 3 adds a real CUDA-side provider seam.

Recommended runtime binding shape:

```python
@dataclass
class ControlNetBinding:
    attachment_id: str
    control_type: str
    model_id: str
    model_path: str
    control_image_bytes: bytes
    strength: float
    start_percent: float
    end_percent: float
```

Provider responsibilities:

- accept an ordered list of bindings
- preserve request order exactly
- resolve one or more Diffusers ControlNet modules
- decode and resize each control map to match generation resolution
- invoke the CUDA pipeline with all requested bindings
- return the normal generation result without mutating attachment order

If implementation complexity forces a phased rollout, the provider interface must still remain ordered-list based even if an early internal milestone temporarily executes only one attachment. The public contract must not need redesign later.

## Cache Design

Track 3 cache is process-local only.

It exists to reuse loaded ControlNet model instances during the server process lifetime. It is not a persistence feature.

Recommended properties:

- keyed by `model_id`
- bounded by count and/or estimated VRAM budget
- pin/unpin while a generation is in flight
- eviction only for unpinned entries
- no attempt to serialize or restore warmed CUDA objects across restart

The persistent layer is simply the local model file on disk referenced by `conf/controlnets.yaml`.

## Request Flow

For a valid ControlNet generation on the CUDA/mode-system path:

1. request arrives with `controlnets`
2. Track 1 policy enforcement runs
3. Track 2 preprocessing runs if any attachment uses `source_asset_ref + preprocess`
4. every attachment entering Track 3 now has a usable `map_asset_ref`
5. Track 3 resolves and validates registry entries for every attachment
6. Track 3 resolves control-map bytes from the `AssetStore`
7. Track 3 decodes and resizes control maps as needed
8. Track 3 builds ordered provider bindings
9. Track 3 acquires/pins required ControlNet model instances from the process-local cache
10. CUDA provider executes generation
11. cache pins are released
12. response is emitted with the generated result plus any `controlnet_artifacts`

## Transport Contract

Track 3 backend must complete the successful response path, not just the stub/error path.

HTTP:

- successful `/generate` must include the generated image as it does today
- successful response metadata must also expose `controlnet_artifacts`

WS:

- successful `job:complete` must include the same `controlnet_artifacts` array
- field name and semantics must match the HTTP success path

The backend must remain usable without Track 4. A direct HTTP or WS client should be able to execute and inspect the full ControlNet backend contract.

## Testing Strategy

Automated Track 3 backend tests must cover:

- valid registry parse
- strict startup validation failure on bad registry or bad mode references
- lazy validation defers the same failures to first use
- fail-fast incompatibility for wrong-family or wrong-control-type attachments
- preprocess-driven successful generation on the CUDA seam
- direct `map_asset_ref` reuse on the CUDA seam
- ordered multi-attachment binding behavior
- successful HTTP response artifact emission
- successful WS `job:complete` artifact emission
- process-local cache pin/unpin and bounded eviction behavior

Automated tests should target the CUDA execution seam first. Use controlled doubles where needed, but ensure the provider integration path itself is the first-class subject of the tests.

## Manual GPU Validation Gate

Track 3 backend is not complete until a real CUDA host passes a manual validation checklist:

1. successful `canny` request from `source_asset_ref`
2. successful `depth` request from `source_asset_ref`
3. reuse emitted artifact via `map_asset_ref`
4. successful multi-attachment generation with visible effect from both attachments
5. incompatible attachment request fails before generation
6. repeated requests show cache reuse and bounded eviction without OOM
7. successful HTTP response includes `controlnet_artifacts`
8. successful WS `job:complete` frame includes `controlnet_artifacts`

These steps exist because mocked tests cannot prove real Diffusers/ControlNet loading, VRAM behavior, or actual CUDA runtime execution.

## Acceptance

Track 3 backend is complete when:

- the CUDA mode-system path executes validated ControlNet requests successfully
- both preprocess-driven and direct reusable-map paths work
- multiple attachments are supported in ordered form
- every attachment must be compatible or the request fails fast
- `conf/controlnets.yaml` is the single backend model registry
- local-path-only registry entries are enforced
- registry validation policy is configurable between strict and lazy
- process-local cache reuse works without restart-persistent state
- successful HTTP and WS responses emit `controlnet_artifacts`
- automated CUDA execution tests pass
- manual GPU validation passes on real hardware

## Track 4 Follow-On

Frontend work is intentionally excluded from this track. Track 4 should assume Track 3 backend is already functional and then focus on:

- mode-aware UI for legal ControlNet choices
- attachment editing and ordering
- emitted-artifact browsing and reuse from result views
- end-user workflows on top of the now-functional backend contract
