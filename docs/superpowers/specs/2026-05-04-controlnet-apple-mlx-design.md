# ControlNet Apple Backend Design

## Summary

This spec defines how to make ControlNet work on Apple Silicon in this repo without breaking the existing backend-neutral request contract. The implementation target is a real `BACKEND=mlx` provider that supports text-to-image generation, mode switching, LoRA composition, and ordered multi-ControlNet attachments on Apple Silicon.

The important platform constraint is that "Apple backend" is not one runtime:

- `MLX` is the right path for native Python execution on Apple Silicon CPU/GPU with unified memory.
- `Core ML` is the right path for explicit Apple Neural Engine usage.
- `MLX` should not be described as NPU/ANE execution. ANE support must be modeled as a separate engine inside the Apple backend.

Because of that, this design treats Apple support as one backend provider with two internal execution engines:

- `mlx`: default, Python-native, GPU-first, supports SD1.5 and SDXL if the Apple-local model set exists
- `coreml`: optional, ANE-capable, targeted first at SD1.5 ControlNet only

The external API stays the same:

- `BACKEND=mlx`
- same `GenerateRequest`
- same `controlnets` attachment contract
- same `controlnet_policy`
- same `controlnet_artifacts`

## Platform Facts

These facts drive the design:

- MLX officially supports CPU and GPU execution on Apple Silicon and uses unified memory.
- Apple's Core ML stack is the official path for selecting compute units such as `CPU_AND_GPU` and `CPU_AND_NE`.
- Apple's `ml-stable-diffusion` project supports ControlNet, but explicitly notes that ControlNet for SDXL is not supported there.

Design consequence:

- SDXL ControlNet on Apple should target the MLX GPU path first.
- ANE-targeted ControlNet should start with SD1.5 only unless the team is ready to own custom SDXL Core ML conversion and runtime work.

## Goals

- Implement a working Apple Silicon ControlNet backend behind the existing provider seam.
- Preserve current request, mode-policy, asset-store, and artifact-emission contracts.
- Support ordered multi-ControlNet attachments.
- Keep dynamic mode loading and queued mode switching behavior intact.
- Support M1-and-newer Apple Silicon Macs.
- Support LoRA ordering and strength deterministically on Apple as on CUDA.
- Provide an honest ANE story without pretending MLX itself runs on ANE.

## Non-Goals

- Replace CUDA as the primary reference implementation.
- Deliver img2img ControlNet in this track.
- Deliver super-resolution on Apple in the same track.
- Promise parity with every CUDA optimization on day one.
- Build a custom Core ML SDXL ControlNet conversion toolchain in v1.
- Add iOS/iPadOS runtime support in this repo.

## Support Matrix

| Area | v1 target | Notes |
|---|---|---|
| Apple hardware | M1 and newer Mac | macOS only |
| Base backend selector | `BACKEND=mlx` | Existing provider slot becomes real |
| Default execution engine | `mlx` | GPU-first, CPU fallback |
| Optional execution engine | `coreml` | For ANE-capable flows |
| SD1.5 generation | Yes | `mlx` first, `coreml` optional |
| SDXL generation | Yes | `mlx` only in v1 |
| SD1.5 ControlNet | Yes | `mlx` first, `coreml` optional |
| SDXL ControlNet | Yes via `mlx` | Not via `coreml` in v1 |
| Multi-ControlNet | Yes | Preserve attachment order |
| LoRA support | Yes | Deterministic ordering required |
| Superres | No | Keep placeholder behavior |

## Proposed Architecture

### 1. Provider boundary

`backends/platforms/mlx.py` stops being a placeholder and becomes the Apple backend provider.

It should report:

- `supports_generation=True`
- `supports_modes=True`
- `supports_model_registry_stats=False`
- `supports_img2img=False`
- `supports_superres=False`
- `supports_controlnet=False` in Phase 1
- `supports_controlnet=True` starting at the end of Phase 2

Recommended new modules:

- `backends/apple/runtime.py`
- `backends/apple/model_registry.py`
- `backends/apple/controlnet_runtime.py`
- `backends/apple/pipelines/mlx_pipeline.py`
- `backends/apple/pipelines/coreml_pipeline.py`
- `backends/apple/conversion.py`

The provider owns engine selection, model loading, ControlNet binding, and capability reporting. Route code must not branch on Apple-specific details.

`supports_model_registry_stats=False` is intentional. Apple uses unified memory, and this backend must not expose fake CUDA-style VRAM totals or per-model VRAM accounting.

### 2. Internal engine selector

Keep `BACKEND=mlx` as the repo-level selector, but add an Apple-specific runtime selector:

```bash
APPLE_EXECUTION_ENGINE=auto   # auto | mlx | coreml
APPLE_COMPUTE_UNITS=cpu_and_gpu   # coreml-only: cpu_only | cpu_and_gpu | cpu_and_ne | all
```

Rules:

- `mlx`
  - use MLX runtime only
  - target CPU/GPU, prefer GPU execution
- `coreml`
  - require Core ML converted assets
  - allow ANE-capable compute-unit selection
- `auto`
  - prefer `coreml` only when:
    - active family is `sd15`
    - required base and ControlNet models have Core ML assets
    - request shape is supported by the converted package
  - otherwise fall back to `mlx`

This keeps the public backend model simple while accurately representing Apple runtime differences.

Compute-unit mapping for the Core ML engine:

| Env token | Apple constant | Hardware intent |
|---|---|---|
| `cpu_only` | `MLComputeUnitsCPUOnly` | CPU only |
| `cpu_and_gpu` | `MLComputeUnitsCPUAndGPU` | CPU + GPU, no ANE |
| `cpu_and_ne` | `MLComputeUnitsCPUAndNeuralEngine` | CPU + ANE, no GPU |
| `all` | `MLComputeUnitsAll` | runtime-selected mix of CPU, GPU, and ANE |

### 3. Runtime flow

For Apple generation with ControlNet:

1. request arrives with `controlnets`
2. existing mode policy validation runs
3. existing preprocessing emits or resolves `map_asset_ref`
4. Apple runtime resolves the active mode and base-model family
5. Apple runtime resolves backend-specific base model assets
6. Apple runtime resolves backend-specific ControlNet assets
7. Apple runtime decodes and resizes control maps to target generation size
8. Apple runtime builds ordered ControlNet bindings
9. Apple runtime executes the selected engine
10. success response returns the generated image and any `controlnet_artifacts`

The mode API and request schema do not change.

## Model Packaging

### 1. Base-model registry

The current mode system should remain the source of truth for logical model identity, but Apple execution needs backend-specific resolved assets.

Recommended mode extension:

```yaml
modes:
  sdxl-general:
    model: sdxl/sdxl-base-1.0
    backend_overrides:
      mlx:
        model_path: /models/apple/mlx/sdxl/sdxl-base-1.0
        loader_format: mlx-diffusers-port
      coreml:
        unsupported: true
```

For SD1.5:

```yaml
modes:
  sd15-general:
    model: sd15/dreamshaper
    backend_overrides:
      mlx:
        model_path: /models/apple/mlx/sd15/dreamshaper
        loader_format: mlx-diffusers-port
      coreml:
        model_path: /models/apple/coreml/sd15/dreamshaper.mlpackage
        loader_format: coreml-package
```

If the team does not want backend-specific mode config yet, the Apple runtime may instead derive paths from a separate registry file. The important requirement is that resolved Apple assets must not be guessed from CUDA paths.

### 2. ControlNet registry

Extend `conf/controlnets.yaml` so one logical `model_id` can resolve differently per backend and engine:

```yaml
models:
  sdxl-canny:
    path: /models/controlnets/sdxl-canny
    format: diffusers
    control_types: [canny]
    compatible_with: [sdxl]
    backends:
      cuda:
        path: /models/controlnets/sdxl-canny
        format: diffusers
      mlx:
        path: /models/apple/mlx/controlnets/sdxl-canny
        format: mlx-controlnet-bundle

  sd15-canny:
    path: /models/controlnets/sd15-canny
    format: diffusers
    control_types: [canny]
    compatible_with: [sd15]
    backends:
      cuda:
        path: /models/controlnets/sd15-canny
        format: diffusers
      mlx:
        path: /models/apple/mlx/controlnets/sd15-canny
        format: mlx-controlnet-bundle
      coreml:
        path: /models/apple/coreml/controlnets/sd15-canny.mlpackage
        format: coreml-package
```

### 3. Registry migration

This registry change is breaking unless the migration is explicit.

Migration plan:

- keep top-level `path` and `format` as the CUDA/default compatibility path during the migration window
- add optional `backends.<backend_id>` overrides for MLX and Core ML
- update the loader so:
  - existing entries without `backends` still work for CUDA exactly as they do today
  - MLX and Core ML must resolve through their backend-specific override
  - CUDA may continue to use the top-level path or an explicit `backends.cuda` override

That gives one file shape that supports both old CUDA-only deployments and new multi-backend entries without a flag day. A follow-up can remove the implicit top-level CUDA fallback after the repo and deployments have migrated.

Validation rules:

- every requested attachment must resolve for the active Apple engine
- `coreml` entries are required only when the selected engine is `coreml`
- missing `coreml` support must fall back only in `auto`, never in explicit `coreml`
- `format: diffusers` means a raw Diffusers/Hugging Face-loadable artifact and is valid for CUDA only
- `format: mlx-controlnet-bundle` means MLX-ported weights and metadata produced by the MLX conversion flow; loading this as raw Diffusers weights must be a hard error

## MLX Engine Design

### 1. Scope

The MLX engine is the default Apple path and the only v1 path for SDXL ControlNet.

Responsibilities:

- load base pipeline weights from Apple-local model paths
- load LoRA adapters in deterministic request order
- load one or more ControlNet modules
- preserve ordered ControlNet attachment binding
- execute generation on Apple GPU where available

### 2. Implementation strategy

Recommended implementation order:

1. establish a minimal text-to-image MLX path for SD1.5
2. add SDXL text-to-image parity
3. add single ControlNet binding
4. add multi-ControlNet binding with preserved order
5. add LoRA composition
6. add caching and mode reload behavior

Do not attempt to land SDXL multi-ControlNet plus LoRA plus ANE support in one pass.

### 3. Runtime contract

Add an Apple generation runtime that mirrors `CudaGenerationRuntime`:

```python
class AppleGenerationRuntime:
    def submit_generate(self, req: Any, *, timeout_s: float | None = None) -> Any: ...
    def switch_mode(self, mode_name: str, force: bool = False) -> Any: ...
    def get_current_mode(self) -> str | None: ...
    def is_model_loaded(self) -> bool: ...
    def get_queue_size(self) -> int: ...
    def shutdown(self) -> None: ...
```

This runtime should reuse the current queue-and-mode semantics instead of creating a parallel orchestration system.

Implementation note:

- prefer reusing `WorkerPool` with `num_workers=1` in v1
- do not build a second Apple-only orchestration layer if the existing queue/mode-switch machinery can be reused directly

### 4. Worker/pipeline shape

Recommended internal objects:

```python
class ApplePipeline(Protocol):
    def load_mode(self, mode: Any) -> None: ...
    def generate(self, req: Any, bindings: list[ControlNetBinding]) -> tuple[bytes, int]: ...
```

Apple should reuse the shared backend binding contract rather than fork a second Apple-only dataclass. If the MLX path needs richer in-memory payloads than `ControlNetBinding` currently carries, evolve the shared binding type or wrap it internally after resolution instead of creating a parallel public binding shape.

The Apple runtime may use a single-worker model in v1. Concurrency beyond one active generation should be queue-based until memory behavior is understood on real hardware.

## Core ML / ANE Engine Design

### 1. Scope

The Core ML engine exists to provide an honest ANE path. It is not required for the first MLX GPU delivery.

v1 scope:

- SD1.5 only
- text-to-image only
- ControlNet only for control models that have converted Core ML packages

Explicitly out of scope for v1:

- SDXL ControlNet on Core ML
- generic conversion of arbitrary Hugging Face ControlNet repos at request time

### 2. Conversion boundary

Core ML support requires offline conversion.

Required assets:

- base UNet / text encoder / VAE Core ML packages
- one Core ML package per supported ControlNet model
- metadata describing supported resolution buckets and scheduler compatibility

Recommended conversion command ownership:

- add repo scripts under `scripts/apple/`
- conversion is a build-time or operator-time step, never a request-time step

See [docs/CONTROLNET_MLX_CONVERSION.md](/home/hdd/workspace/Stability-Toys/docs/CONTROLNET_MLX_CONVERSION.md) for the MLX-side artifact procedure. That guide produces the MLX bundles this spec expects the Apple backend to consume.

### 3. Compute units

The engine should support:

- `cpu_and_gpu`
- `cpu_and_ne`
- `all`

Rules:

- compute-unit choice is runtime configuration, not part of the request contract
- if `cpu_and_ne` is selected but the converted model or operation set is incompatible, fail clearly or fall back only in `auto`
- logs must state the selected engine and compute-unit combination for each generation

## Preprocessing

The existing preprocessing seam should remain backend-agnostic.

v1 recommendation:

- keep `canny` preprocessing on CPU
- keep `depth` preprocessing on CPU unless a strong Apple-native path is proven faster

Reason:

- preprocessing is not the main performance bottleneck
- this minimizes first-pass risk
- it preserves identical artifact semantics across CUDA and Apple backends

Later optimization options:

- MLX-native depth estimator
- Core Image edge preprocessing for canny-like maps
- Core ML depth estimator for Apple-only fast path

## Caching And Memory Policy

Apple Silicon uses unified memory, so CUDA-style VRAM accounting cannot be copied directly.

Recommended policy:

- single active generation worker in v1
- process-local base model cache keyed by mode name + engine
- process-local ControlNet cache keyed by `model_id` + engine
- soft cache budget driven by resident size estimates, not VRAM APIs
- evict unpinned ControlNet models first
- mode switch pins current assets until in-flight generation completes

New capability reporting should expose truthful Apple metrics, for example:

- active engine
- loaded model ids
- estimated resident bytes
- process RSS
- optional unified-memory pressure snapshot if available

Do not expose fake VRAM totals for Apple.

## Mode Switching

The existing queue-first semantics must stay intact:

- mode switches are queued
- in-flight generations complete on the old mode
- new requests wait behind the mode switch

Required behavior for Apple:

- switching from one mode to another unloads prior LoRAs and ControlNet state cleanly
- switching engine from `mlx` to `coreml` requires full pipeline rebuild
- hot reload of mode config must invalidate stale Apple asset resolutions

## Configuration

Recommended new environment variables:

```bash
BACKEND=mlx
APPLE_EXECUTION_ENGINE=auto
APPLE_COMPUTE_UNITS=cpu_and_gpu
APPLE_MAX_ACTIVE_GENERATIONS=1
APPLE_MODEL_CACHE_LIMIT_BYTES=0
APPLE_CONTROLNET_CACHE_LIMIT_BYTES=0
APPLE_ENABLE_SDXL=1
APPLE_ENABLE_COREML_SD15=0
```

Rules:

- `0` cache limits mean "runtime default"
- `APPLE_ENABLE_COREML_SD15=1` is required before `auto` may select `coreml`
- `APPLE_ENABLE_SDXL=0` may be used to keep early bring-up focused on SD1.5 only

Capability note:

- `capabilities()` should report the static union this backend is designed to support
- feature flags such as `APPLE_ENABLE_SDXL=0` narrow request-time resolution and allowed active families, but do not need to mutate the provider's capability shape dynamically

## File Plan

Expected touched files:

- `backends/platforms/mlx.py`
- `backends/platforms/base.py`
- `backends/platform_registry.py`
- `backends/apple/runtime.py`
- `backends/apple/model_registry.py`
- `backends/apple/controlnet_runtime.py`
- `backends/apple/pipelines/mlx_pipeline.py`
- `backends/apple/pipelines/coreml_pipeline.py`
- `server/controlnet_execution.py`
- `server/controlnet_registry.py`
- `conf/controlnets.yaml`
- `server/mode_config.py`
- `modes.yaml.example`

Expected new tests:

- `tests/test_apple_provider.py`
- `tests/test_apple_controlnet_runtime.py`
- `tests/test_apple_controlnet_registry.py`
- `tests/test_apple_mode_switching.py`

## Delivery Plan

### Phase 1: Real MLX Provider

- replace the placeholder `MLXProvider` in [`backends/platforms/mlx.py`](/home/hdd/workspace/Stability-Toys/backends/platforms/mlx.py:10), which currently raises `NotImplementedError`
- implement Apple runtime skeleton
- support mode loading and plain text-to-image generation
- keep `supports_controlnet=False` until execution is real

### Phase 2: MLX ControlNet

- add Apple backend resolution in `controlnets.yaml`
- support one ControlNet attachment
- add ordered multi-ControlNet support
- set `supports_controlnet=True`

### Phase 3: LoRA + Stability Work

- add deterministic LoRA loading for Apple
- add cache eviction and mode reload correctness
- tune default memory policy for M1/M2/M3 classes

### Phase 4: Optional Core ML / ANE Path

- add offline conversion scripts
- add `APPLE_EXECUTION_ENGINE=coreml|auto`
- support SD1.5 ControlNet on `cpu_and_ne`

## Test Strategy

Automated tests must cover:

- `BACKEND=mlx` provider wiring
- mode loading and queued mode switching
- Apple-specific controlnet registry resolution
- failure when a requested ControlNet model lacks Apple assets
- ordered multi-ControlNet binding
- artifact emission on successful HTTP and WS generation
- fallback behavior in `APPLE_EXECUTION_ENGINE=auto`
- explicit failure behavior in `APPLE_EXECUTION_ENGINE=coreml`
- all existing CUDA ControlNet tests continue to pass with the migrated registry schema

Manual validation must cover:

1. M1 or newer Mac boots with `BACKEND=mlx`
2. SD1.5 text-to-image works on MLX
3. SD1.5 ControlNet `canny` works on MLX
4. SDXL ControlNet works on MLX
5. repeated generations do not leak memory unboundedly
6. mode switch during queued work behaves like CUDA
7. if enabled, SD1.5 ControlNet works on Core ML with `cpu_and_ne`
8. unsupported SDXL Core ML requests fail clearly and do not silently drop ControlNet

## Acceptance Criteria

This work is complete when:

- `BACKEND=mlx` is a real generation backend
- Apple ControlNet works through the existing request contract
- ordered multi-ControlNet attachments are preserved
- LoRA ordering remains deterministic
- mode switching remains queued and non-blocking for in-flight jobs
- Apple capability reporting is truthful
- SDXL uses MLX rather than pretending Core ML support exists
- ANE support, if enabled, is explicitly modeled as Core ML and not mislabeled as MLX
- all existing CUDA ControlNet tests still pass after the registry migration

## Risks

- MLX feature gaps may require owning more diffusion-pipeline code than CUDA does today.
- Unified-memory pressure on 16 GB machines may make SDXL practical only with one active generation and conservative cacheing.
- Core ML conversion for ControlNet may be operationally expensive and should remain optional until the MLX path is stable.
- Backend-specific asset packaging can sprawl unless registry ownership stays explicit and validated.

## Recommendation

Ship Apple ControlNet in two honest steps:

1. deliver MLX GPU support first, including SDXL
2. add optional Core ML / ANE support second, limited to SD1.5 until proven otherwise

That sequence matches the current repo architecture, matches Apple's published runtime boundaries, and avoids promising SDXL ANE support that the upstream Apple Stable Diffusion stack does not currently claim.

## References

- MLX documentation on supported devices and unified memory: https://ml-explore.github.io/mlx/build/html/index.html
- MLX unified memory guide: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- Apple Core ML overview: https://developer.apple.com/machine-learning/core-ml/
- Apple `ml-stable-diffusion` README and ControlNet notes: https://github.com/apple/ml-stable-diffusion
- MLX conversion guide: [docs/CONTROLNET_MLX_CONVERSION.md](/home/hdd/workspace/Stability-Toys/docs/CONTROLNET_MLX_CONVERSION.md)
- RKNN conversion guide: [docs/CONTROLNET_RKNN_CONVERSION.md](/home/hdd/workspace/Stability-Toys/docs/CONTROLNET_RKNN_CONVERSION.md)
