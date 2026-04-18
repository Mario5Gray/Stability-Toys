# ControlNet Design

## Summary

This design adds ControlNet as a first-class generation control for the existing mode-based generation stack. The v1 target is CUDA plus Diffusers only, with a provider seam that allows RKNN to add platform support later without changing the request or UI contract.

The user-facing contract is intentionally simple:

- generation requests may include a `controlnets` list
- each active mode exposes a `controlnet_policy`
- each ControlNet attachment can reuse an existing control-map asset or derive one from a source image through a backend preprocessor

The backend remains responsible for policy enforcement, preprocessing, derived asset emission, model resolution, and Diffusers execution. The frontend remains responsible for selecting attachments, showing emitted control maps, and reusing them in later generations.

V1 supports:

- CUDA plus Diffusers only
- text-to-image generation only
- multiple simultaneous ControlNet attachments
- built-in `canny` and `depth` preprocessors
- upload or existing-image-derived control sources
- emitted reusable control-map assets returned to the frontend

V2 and later may add drawing, img2img, more preprocessors, and RKNN support.

## Goals

- Add ControlNet to the existing generation path without introducing a second workflow system
- Keep the request and UI contract backend-agnostic
- Support multiple simultaneous ControlNet attachments in one generation
- Support backend-derived `canny` and `depth` control maps in v1
- Make derived control maps reusable frontend assets rather than hidden worker internals
- Keep ControlNet policy mode-owned so the UI only renders legal choices
- Create a provider seam that CUDA implements now and RKNN can adopt later

## Non-Goals

- Add a full node graph or workflow runtime in v1
- Add manual drawing or sketch tooling in v1
- Add img2img or other non-text generation targets in v1
- Deliver RKNN ControlNet execution in v1
- Support every ControlNet preprocessor family in v1
- Build a large persistent asset platform before proving the generation path

## Current State

- [`server/lcm_sr_server.py`](/Users/darkbit1001/workspace/Stability-Toys/server/lcm_sr_server.py) defines `GenerateRequest` and already carries mode-owned generation controls such as `negative_prompt` and `scheduler_id`, but it has no ControlNet request field.
- [`server/ws_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/server/ws_routes.py) translates frontend generation params into `GenerateRequest` and currently forwards negative prompt, scheduler, size, and img2img source references, but not ControlNet attachments.
- [`server/mode_config.py`](/Users/darkbit1001/workspace/Stability-Toys/server/mode_config.py) and [`server/model_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/server/model_routes.py) already expose the pattern this repo uses for mode-owned policy: parse it in config, serialize it through `/api/modes`, and let the frontend stay dumb.
- [`server/generation_constraints.py`](/Users/darkbit1001/workspace/Stability-Toys/server/generation_constraints.py) already centralizes mode-aware backend enforcement for generation requests, but only for size and existing defaults.
- [`backends/cuda_worker.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/cuda_worker.py) already has a CUDA worker base that resolves scheduler policy and applies request-specific generation settings. That is the right general layer for ControlNet execution, but not the right place to hide preprocessor state or asset emission.
- [`server/upload_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/server/upload_routes.py) only provides ephemeral `fileRef -> bytes` resolution. That is insufficient for reusable derived control maps.
- The frontend already uses a mode-aware generation control path through [`lcm-sr-ui/src/hooks/useGenerationParams.js`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/hooks/useGenerationParams.js), [`lcm-sr-ui/src/utils/generationControls.js`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/utils/generationControls.js), and [`lcm-sr-ui/src/components/options/OptionsPanel.jsx`](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/options/OptionsPanel.jsx) for negative prompt and scheduler selection.

## Proposed Approach

Treat ControlNet as another first-class generation control with three layers:

1. `Request and policy layer`
   - requests carry `controlnets`
   - modes carry `controlnet_policy`
2. `Preprocess and asset layer`
   - backend preprocessors resolve source images into reusable control-map assets
3. `Provider execution layer`
   - the active backend provider resolves and applies ControlNet models and maps to the underlying runtime

The important design choice is that preprocessors are asset-producing steps, not hidden worker internals. This keeps ControlNet reusable and inspectable without forcing the repo into a workflow-graph design.

The canonical runtime becomes:

`request -> mode policy validation -> source asset resolution -> optional map preprocessing -> derived asset emission -> backend provider attachment resolution -> final generation -> response with image + emitted control-map metadata`

## Design

### 1. Request contract

Add a new request field to `GenerateRequest` and the matching WebSocket serialization path:

```json
{
  "prompt": "plant on a windowsill, morning light",
  "mode": "sdxl-general",
  "controlnets": [
    {
      "attachment_id": "cn_1",
      "control_type": "canny",
      "map_asset_ref": "asset_controlmap_abc123",
      "model_id": "sdxl-canny",
      "strength": 0.8,
      "start_percent": 0.0,
      "end_percent": 1.0
    },
    {
      "attachment_id": "cn_2",
      "control_type": "depth",
      "source_asset_ref": "asset_source_xyz789",
      "preprocess": {
        "id": "depth",
        "options": {}
      },
      "model_id": "sdxl-depth",
      "strength": 0.55,
      "start_percent": 0.0,
      "end_percent": 0.8
    }
  ]
}
```

Rules:

- `controlnets` is optional
- each attachment must specify exactly one input path:
  - `map_asset_ref`, or
  - `source_asset_ref` plus `preprocess`
- `control_type` is canonical UI and API vocabulary such as `canny` or `depth`
- `model_id` is optional at the API layer if the mode policy supplies a default
- `strength`, `start_percent`, and `end_percent` are validated server-side
- attachment order is preserved and treated as meaningful runtime input

Recommended dataclasses or Pydantic models:

- `ControlNetAttachment`
- `ControlNetPreprocessRequest`
- `ControlNetArtifactRef`

This preserves the "simple drop-in" property for callers: one extra list in the existing request payload rather than a second endpoint family or graph DSL.

### 2. Mode-owned ControlNet policy

Extend `ModeConfig` and `/api/modes` with a `controlnet_policy` block.

Recommended shape:

```yaml
controlnet_policy:
  enabled: true
  max_attachments: 4
  allow_reuse_emitted_maps: true
  allowed_control_types:
    canny:
      default_model_id: sdxl-canny
      allowed_model_ids:
        - sdxl-canny
      allow_preprocess: true
      default_strength: 0.8
      min_strength: 0.0
      max_strength: 2.0
    depth:
      default_model_id: sdxl-depth
      allowed_model_ids:
        - sdxl-depth
      allow_preprocess: true
      default_strength: 0.6
      min_strength: 0.0
      max_strength: 2.0
```

Rules:

- if `enabled` is false or absent, the backend rejects `controlnets`
- only listed `control_type` values are allowed for that mode
- `max_attachments` is enforced before worker execution
- requested `model_id` values must be in the allowed list for the selected control type
- the mode policy may supply defaults when the client leaves them unset

This follows the existing repository pattern:

- config is the source of truth
- `/api/modes` tells the UI what is legal
- backend constraints enforce the same policy on HTTP and WebSocket paths

### 3. Derived control-map assets

V1 needs a reusable asset concept for emitted control maps. The existing temp upload table in `upload_routes.py` is not enough because it only resolves short-lived bytes.

Add a small asset layer for control maps with these responsibilities:

- store an emitted control-map image and metadata
- return a stable asset reference for the current session or storage lifetime
- resolve asset refs back to bytes for generation reuse
- describe emitted artifacts in generation responses so the frontend can render them

Recommended artifact metadata:

```json
{
  "asset_ref": "asset_controlmap_abc123",
  "kind": "control_map",
  "control_type": "canny",
  "origin": "derived",
  "source_asset_ref": "asset_source_xyz789",
  "preprocessor_id": "canny",
  "media_type": "image/png",
  "width": 1024,
  "height": 1024
}
```

V1 storage expectations:

- emitted control-map asset refs must remain valid for the current user session at minimum
- generation responses must surface emitted artifacts
- frontend can reuse them in later requests during that session without recomputation

This is intentionally smaller than a full gallery asset platform. The only hard requirement in v1 is that derived maps survive long enough to be inspected and reused.

### 4. Preprocessor seam

Add a preprocessor layer independent from Diffusers model execution.

Recommended protocol:

```python
class ControlMapPreprocessor(Protocol):
    preprocessor_id: str
    control_type: str

    def run(self, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult: ...
```

Suggested v1 implementations:

- `CannyPreprocessor`
- `DepthPreprocessor`

Responsibilities:

- validate and normalize source image input
- produce the control-map image bytes
- report output metadata
- emit a reusable control-map asset through the asset layer

Non-responsibilities:

- loading or applying ControlNet models
- deciding whether a mode is allowed to use that control type
- mutating the final Diffusers generation request directly

This separation matters for future RKNN support. RKNN may use different preprocess libraries or pipelines, but it should not need a different top-level request shape.

### 5. Provider seam

Add a ControlNet provider seam under the active backend provider layer.

Recommended protocol:

```python
class ControlNetProvider(Protocol):
    backend_id: str

    def validate_policy(self, mode: ModeConfig, attachments: list[ControlNetAttachment]) -> None: ...
    def resolve_attachments(
        self,
        mode: ModeConfig,
        attachments: list[ResolvedControlNetAttachment],
    ) -> list[BackendControlNetBinding]: ...
    def apply(
        self,
        pipe: Any,
        bindings: list[BackendControlNetBinding],
        req: GenerateRequest,
    ) -> Any: ...
```

V1 behavior:

- CUDA implements the provider using Diffusers `ControlNetModel` and the appropriate pipeline wiring for multiple simultaneous attachments
- non-supporting backends report capability absence clearly and do not pretend to partially support ControlNet

Why this seam is right for this repo:

- it mirrors the explicit backend-provider direction already used for backend selection
- it avoids pushing ControlNet knowledge into routes
- it keeps CUDA-specific model loading and caching out of generic request validation

### 6. CUDA execution path

The CUDA implementation should own:

- loading and caching ControlNet models keyed by `model_id`
- resolving multiple attachments into backend bindings
- preserving attachment order
- translating bindings into the active Diffusers generation call
- reporting clear errors for unsupported or failed model loads

The CUDA worker should not own:

- long-lived policy truth
- preprocessor choice
- asset storage semantics

Practical note:

- `backends/cuda_worker.py` is still the right place to consume resolved ControlNet bindings during generation
- it is not the right place to discover user intent or to hide preprocessor outputs

### 7. Backend enforcement

Extend the existing `finalize_mode_generate_request(...)` pattern or add a sibling helper for ControlNet enforcement.

The enforcement layer should:

- reject ControlNet use when the active mode does not enable it
- apply mode defaults for omitted `model_id` or strength fields where defined
- reject unsupported `control_type` and `model_id` values
- reject more than `max_attachments`
- reject malformed attachment percentages or inverted ranges

This logic must run for both:

- HTTP `/generate`
- WebSocket generation job submission

The backend should fail early and specifically. Silent dropping of invalid attachments is not allowed.

### 8. Frontend integration

V1 should extend the existing options and generation-parameter path instead of adding a parallel ControlNet app.

Frontend responsibilities:

- maintain a `controlnets` draft list alongside prompt, negative prompt, scheduler, size, and seed
- render a `ControlNet` section in the options panel when the active mode exposes `controlnet_policy.enabled`
- allow multiple attachments to be added and removed
- let each attachment select:
  - `control_type`
  - `source_asset_ref` or `map_asset_ref`
  - `model_id` if the mode allows multiple models
  - `strength`
  - `start_percent`
  - `end_percent`
- show emitted control-map artifacts after generation and offer reuse in later requests

V1 source selection should support:

- uploaded images
- existing generation results
- previously emitted control-map assets

V1 explicitly does not include:

- freehand drawing or sketch canvas input
- img2img-specific ControlNet controls

### 9. Response contract

Generation responses and WebSocket completion events should include emitted control-map artifacts when present.

Recommended shape:

```json
{
  "image": "...",
  "seed": 12345678,
  "controlnet_artifacts": [
    {
      "attachment_id": "cn_2",
      "asset_ref": "asset_controlmap_depth_001",
      "control_type": "depth",
      "preprocessor_id": "depth",
      "source_asset_ref": "asset_source_xyz789"
    }
  ]
}
```

This allows the frontend to:

- display the emitted maps alongside the generated result
- store them in message metadata or local cache
- reuse them later without reparsing English or calling a second route

## Failure Handling

V1 failure behavior should be narrow and visible:

- reject the entire request if the mode does not allow ControlNet
- reject the entire request if any attachment is invalid
- reject the entire request if preprocessing fails for any attachment
- preserve already emitted artifact refs in the error payload when practical
- fail clearly when a backend does not support ControlNet rather than falling back silently

Rationale:

- silent attachment dropping makes prompt-to-image debugging impossible
- partial success semantics can be revisited later, but they would complicate v1 without enough evidence

## Testing Strategy

### Backend tests

- mode-config parsing and round-trip tests for `controlnet_policy`
- `/api/modes` route tests for ControlNet policy serialization
- request validation tests for `controlnets`
- generation-constraint tests for rejected invalid attachments
- preprocessor tests for `canny` and `depth`
- derived asset tests for emitted map storage and reuse
- CUDA provider tests with mocked Diffusers seams for multi-ControlNet ordering and binding resolution

### Frontend tests

- mode-config hook tests for surfaced `controlnet_policy`
- generation params tests for `controlnets` draft and effective values
- options panel tests for conditional ControlNet UI rendering
- request serialization tests for `controlnets`
- result handling tests for emitted artifact reuse

### Manual validation

1. Upload or select an existing image, derive a `canny` or `depth` control map, and submit a generation.
2. Confirm the generated control map is visible in the UI and has a reusable asset ref.
3. Re-run generation using the emitted map directly without recomputing it.
4. Submit multiple simultaneous attachments and confirm ordering and strengths are honored.
5. Submit an illegal attachment under the active mode and confirm the request fails before worker execution.

## Risks And Tradeoffs

### Multi-ControlNet introduces runtime and memory pressure

That is expected. The mode policy should cap attachment count and allowed model IDs so the UI and backend do not expose unsafe combinations casually.

### Derived assets create a second image lifecycle

That is also intentional. Control maps need to exist as inspectable, reusable artifacts or the feature becomes opaque and wasteful. V1 should keep the lifecycle minimal, not avoid it entirely.

### The provider seam may feel heavier than a direct CUDA-only implementation

It is still the right trade. This repo already has explicit backend evolution pressure. Baking ControlNet directly into a CUDA-only request model would make RKNN support a redesign instead of a follow-up provider.

### A full node-graph abstraction remains tempting

It should be deferred. The current user value is classical SD workflow support, not proving a graph runtime. V1 should ship ControlNet that people can use before generalizing the execution model.

## Rollout

Implement in this order:

1. Extend request models and WebSocket serialization with `controlnets`
2. Add `controlnet_policy` to mode config and `/api/modes`
3. Add backend ControlNet validation alongside existing generation constraints
4. Add the derived control-map asset layer
5. Implement `canny` and `depth` preprocessors
6. Add the CUDA ControlNet provider and worker integration
7. Extend the frontend options and generation state with ControlNet attachments
8. Surface emitted control-map artifacts and enable reuse
9. Add tests and manual validation

## V1 And V2 Split

### V1

- CUDA plus Diffusers only
- text-to-image only
- multiple simultaneous attachments
- `canny` and `depth`
- upload and existing-image-derived sources
- emitted reusable control-map assets
- mode-owned policy
- provider seam for later RKNN support

### V2

- drawing and sketch input
- img2img integration
- more preprocessors such as pose, normal, and segmentation
- richer persistence and gallery workflows for control assets
- RKNN ControlNet provider
- broader workflow or node abstractions if usage data justifies them

## Acceptance

This work is complete when:

- generation requests can include a validated `controlnets` list
- active modes expose `controlnet_policy` through `/api/modes`
- backend preprocessors can produce reusable `canny` and `depth` control-map assets
- CUDA generation can consume multiple simultaneous ControlNet attachments
- generation responses expose emitted control-map artifacts for frontend reuse
- invalid ControlNet requests fail clearly before worker execution
- the UI can add, remove, inspect, and reuse ControlNet attachments without a parallel app surface
