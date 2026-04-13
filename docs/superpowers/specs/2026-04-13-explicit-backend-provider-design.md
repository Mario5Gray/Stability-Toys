# Explicit Backend Provider Design

## Summary

This design replaces the current ad hoc CUDA-vs-RKNN branching with one explicit backend provider boundary selected by `BACKEND`. The selector remains named `BACKEND`, but it becomes required and non-automatic. Supported values become `cuda`, `rknn`, `mlx`, and `cpu`.

The provider owns generation worker creation, model/resource registry behavior, startup policy, and super-resolution service construction. `cuda` and `rknn` remain the only working runtime backends in the first implementation. `mlx` and `cpu` are introduced as first-class backend entries with scaffolded placeholder implementations that fail clearly when generation or super-resolution is first initialized or used.

This design keeps the API surface backend-agnostic, removes runtime autodetection, and creates the extension seams needed for future Apple Silicon MLX work without pretending that CPU generation is already supported.

Related follow-up issue: `STABL-lpvbcfbd`

## Goals

- Keep `BACKEND` as the single runtime selector
- Remove `BACKEND=auto` and all runtime backend autodetection
- Introduce a provider boundary that owns generation, registry, and super-resolution behavior
- Make the mode API backend-agnostic at the route and orchestration layer
- Preserve current working CUDA and RKNN behavior
- Add explicit `mlx` and `cpu` backend entries as scaffolded future paths
- Ensure unsupported backends fail with clear, backend-specific messages

## Non-Goals

- Implement MLX image generation in this first refactor
- Implement CPU image generation in this first refactor
- Redesign prompt, mode, or scheduler semantics
- Introduce dynamic plugin loading or out-of-process backend discovery
- Change cross-repo build architecture documentation ownership

## Current State

- The repo now resolves one explicit backend provider from `BACKEND` and stores provider-owned generation and super-resolution runtimes on FastAPI app state.
- [`backends/worker_factory.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/worker_factory.py) only creates CUDA workers.
- [`backends/worker_pool.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/worker_pool.py) accepts injected factories and registries, but its default path is still CUDA-specific and the module imports `torch` eagerly at import time.
- [`backends/model_registry.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/model_registry.py) now exposes both the CUDA VRAM-aware registry and a backend-neutral placeholder registry, but the CUDA implementation still owns the real allocator accounting path.
- [`server/superres_http.py`](/Users/darkbit1001/workspace/Stability-Toys/server/superres_http.py) and [`server/superres_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/superres_service.py) now require explicit backend selection and no longer accept `auto`.
- [`server/lcm_sr_server.py`](/Users/darkbit1001/workspace/Stability-Toys/server/lcm_sr_server.py) contains `PipelineService`, which mixes queueing, worker creation, singleton lifecycle, and backend branching in one class.
- [`server/lcm_sr_server.py`](/Users/darkbit1001/workspace/Stability-Toys/server/lcm_sr_server.py) now routes compat generation helpers through `app.state.generation_runtime`, but the legacy `PipelineService` class is still present as the RKNN-owned runtime implementation.
- The repo currently has two separate job abstractions: one legacy job dataclass in [`backends/base.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/base.py) and one extensible job hierarchy in [`backends/worker_pool.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/worker_pool.py).
- The Docker test path uses `BACKEND=cpu` as an honest build and smoke-test scaffold. It is still not a supported inference backend.

## Proposed Approach

Introduce a backend provider layer and make it the single source of truth for runtime behavior.

The runtime flow becomes:

1. Read explicit `BACKEND`
2. Resolve one backend provider
3. Ask that provider for:
   - generation worker factory
   - model/resource registry
   - generation startup strategy
   - super-resolution service factory
   - backend capability description
4. Initialize the server and routes against those backend-neutral interfaces

This is intentionally narrower than a full plugin system. Providers are static in-repo modules, not dynamic runtime extensions.

## Design

### 1. Explicit backend selection

`BACKEND` remains the only top-level selector. Valid values are:

- `cuda`
- `rknn`
- `mlx`
- `cpu`

Rules:

- `BACKEND` must be set explicitly
- `auto` is not a valid backend value
- unknown values fail immediately during startup configuration
- runtime behavior must not infer dependencies from local hardware

Migration note:

- removing `auto` from the accepted backend set is a breaking configuration change
- existing deployments that do not set `BACKEND` explicitly will fail at startup
- startup errors should report an unsupported backend and list the supported values

Rationale:

- dependency availability is a build and packaging concern, not a runtime guess
- explicit selection makes local, CI, and production behavior reproducible
- `mlx` and `cpu` can be introduced now without overloading the meaning of `cuda`

### 2. Backend provider contract

Add a backend-neutral provider contract in a new module such as:

- `backends/platforms/base.py`

The provider should expose:

- backend identifier
- capability flags
- worker factory for generation
- model/resource registry factory
- generation startup adapter
- super-resolution service factory

Suggested shape:

```python
class BackendProvider(Protocol):
    backend_id: str

    def capabilities(self) -> BackendCapabilities: ...
    def create_model_registry(self) -> ModelRegistryProtocol: ...
    def create_worker_factory(self) -> WorkerFactory: ...
    def create_generation_runtime(self, ...) -> GenerationRuntimeProtocol: ...
    def create_superres_runtime(self, ...) -> Optional[SuperResServiceProtocol]: ...
```

`BackendCapabilities` should describe backend truth explicitly, for example:

- `supports_modes`
- `supports_generation`
- `supports_superres`
- `supports_model_registry_stats`
- `supports_img2img`

This lets routes and status payloads remain backend-agnostic without lying about capabilities.

### 3. Generation boundary

The generation worker contract already exists in [`backends/base.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/base.py) as `PipelineWorker`. That should remain the reusable execution interface.

What changes:

- [`backends/worker_factory.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/worker_factory.py) stops being the implicit global CUDA factory
- backend-specific worker factories move behind provider modules
- [`backends/worker_pool.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/worker_pool.py) receives its default worker factory from the resolved provider instead of hardcoding CUDA

Important scoping rule:

- `WorkerPool` should not be treated as the mandatory shared runtime for every backend in the first refactor
- in the first pass, `WorkerPool` is best treated as the CUDA-oriented mode-capable generation runtime that gets adapted behind the provider boundary
- RKNN may keep or wrap `PipelineService` temporarily behind its own provider until a later unification pass
- `mlx` and `cpu` can use placeholder generation runtimes without importing CUDA-oriented orchestration code

Backend expectations:

- `cuda` uses the current diffusers worker family
- `rknn` continues using its RKNN generation path
- `mlx` returns a placeholder generation runtime or worker that raises a clear MLX-not-implemented error
- `cpu` returns a placeholder generation runtime or worker that raises a clear CPU-not-implemented error

The placeholder behavior should occur when generation is initialized or first requested, not at backend parsing time. That keeps the future implementation path stable: later work only swaps the placeholder with a real backend implementation.

### 4. Generation runtime migration

`PipelineService` needs explicit treatment in this design because it currently owns:

- backend selection
- worker construction
- queueing
- singleton lifecycle
- the legacy RKNN runtime path

Recommended migration approach:

- treat `PipelineService` as an implementation artifact, not as the lasting abstraction
- allow the RKNN provider to wrap or temporarily reuse `PipelineService` in the first pass
- keep the provider contract above it so later RKNN cleanup can replace it without changing server startup again
- do not expand `PipelineService` to learn about `mlx` or `cpu`

The intent is to avoid one more round of central branching in `lcm_sr_server.py`.

### 5. Model and resource registry boundary

The current [`backends/model_registry.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/model_registry.py) mixes generic model registration with CUDA-specific VRAM accounting. Split that into:

- a backend-neutral registry contract
- backend-specific accounting implementations

Suggested direction:

- `ModelRegistryProtocol` defines shared operations:
  - register model
  - unregister model
  - lookup loaded models
  - report resource statistics
  - estimate fit if supported
- `CudaModelRegistry` keeps today’s VRAM accounting behavior
- `RknnModelRegistry` can report simpler device/runtime state first
- `MlxModelRegistry` and `CpuModelRegistry` can initially expose minimal placeholder or process-memory stats

Important constraint:

- the registry API must not force every backend to pretend it has VRAM semantics
- status payloads should expose generic resource fields where possible and backend-specific detail only where truthful
- eager `torch` imports and constructor-time CUDA probing must move behind the CUDA provider or a CUDA-specific registry implementation so non-CUDA backends can import the backend layer cleanly

### 6. Super-resolution boundary

Super-resolution should move under the same backend provider boundary instead of resolving its backend independently.

That means:

- [`server/superres_http.py`](/Users/darkbit1001/workspace/Stability-Toys/server/superres_http.py) stops choosing CUDA versus RKNN itself
- [`server/superres_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/superres_service.py) keeps reusable service implementations, but provider selection owns which one is used

Backend expectations:

- `cuda` provides the current CUDA SR service
- `rknn` provides the current RKNN SR service
- `mlx` exposes an explicit placeholder SR path for now
- `cpu` exposes an explicit placeholder SR path for now

This keeps generation and SR aligned under one selector and prevents mismatched backend truth across subsystems.

Specific cleanup required:

- remove the `auto` branch from [`server/superres_http.py`](/Users/darkbit1001/workspace/Stability-Toys/server/superres_http.py)
- remove the `auto` branch from [`server/superres_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/superres_service.py)
- stop deriving SR backend choice from `torch.cuda.is_available()`

### 7. Server startup wiring

[`server/lcm_sr_server.py`](/Users/darkbit1001/workspace/Stability-Toys/server/lcm_sr_server.py) should become orchestration only.

Instead of branching on CUDA versus RKNN, startup should:

1. resolve `BACKEND`
2. load one provider
3. create provider-owned registry and generation runtime
4. create provider-owned SR runtime
5. store provider capability metadata in app state for routes and status reporting

Mode behavior:

- the mode API remains present regardless of backend
- backends may support different mode capabilities internally
- unsupported mode operations fail with backend-specific messages, not missing routes

This keeps the user-facing control plane stable while backend implementations evolve underneath it.

Existing compat bug to fold into this work:

- `_run_generate_from_dict()` currently assumes `app.state.service` always exists, which is false for the CUDA mode-system path
- the provider refactor should route compat generation through the provider-owned generation runtime so both legacy and mode-system paths behave consistently

### 8. Runtime status and capabilities

The current runtime status shape is too CUDA-specific in places. In particular, status building currently reaches directly into `torch.cuda` for allocator fields.

Under the provider design:

- runtime status should be composed from provider capabilities plus provider-owned registry/runtime reporting
- inline `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` calls should be removed from backend-neutral status assembly
- backends that cannot truthfully expose VRAM details should expose a smaller honest resource payload

### 9. Job abstraction cleanup

The repo currently has two different job models:

- legacy `Job` in [`backends/base.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/base.py)
- extensible queue `Job` hierarchy in [`backends/worker_pool.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/worker_pool.py)

The provider refactor should select one durable orchestration job model and treat the other as transitional compatibility code. The first pass does not need to delete both immediately, but the design should not assume they coexist forever.

### 10. Backend behavior matrix

Initial truth after this refactor:

| Backend | Generation | Modes API | Super-Resolution | Registry |
|---------|------------|-----------|------------------|----------|
| `cuda` | implemented | implemented | implemented | CUDA VRAM accounting |
| `rknn` | implemented | implemented through shared API, narrower runtime behavior allowed | implemented | RKNN-specific or minimal accounting |
| `mlx` | scaffold only, fails clearly at init/use time | route surface present | scaffold only, fails clearly | minimal placeholder |
| `cpu` | scaffold only, fails clearly at init/use time | route surface present | scaffold only, fails clearly | minimal placeholder |

The important property is honesty:

- unsupported backends are explicit
- unsupported operations fail with precise messages
- the system no longer implies that `cpu` is a real inference backend just because Docker can build it

### 11. File layout

Recommended initial layout:

- `backends/platforms/base.py`
- `backends/platforms/cuda.py`
- `backends/platforms/rknn.py`
- `backends/platforms/mlx.py`
- `backends/platforms/cpu.py`
- `backends/platform_registry.py`

Existing modules to adapt rather than replace immediately:

- [`backends/worker_factory.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/worker_factory.py)
- [`backends/worker_pool.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/worker_pool.py)
- [`backends/model_registry.py`](/Users/darkbit1001/workspace/Stability-Toys/backends/model_registry.py)
- [`server/lcm_sr_server.py`](/Users/darkbit1001/workspace/Stability-Toys/server/lcm_sr_server.py)
- [`server/superres_http.py`](/Users/darkbit1001/workspace/Stability-Toys/server/superres_http.py)
- [`server/superres_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/superres_service.py)

The first pass should prefer thin wrappers and interface extraction over large code motion. The goal is to establish the provider boundary without destabilizing the working CUDA and RKNN paths.

## Testing Strategy

### Unit tests

- backend registry resolution:
  - valid `BACKEND` values resolve the expected provider
  - missing or unsupported `BACKEND` values fail clearly
- worker pool default wiring:
  - provider-owned worker factory is used instead of hardcoded CUDA
- model registry behavior:
  - backend-neutral registry contract works for CUDA and non-CUDA implementations
- super-resolution selection:
  - provider-owned SR runtime is selected from the same backend choice
- placeholder behavior:
  - `mlx` and `cpu` return explicit not-implemented failures with stable messages

### Integration tests

- CUDA startup still initializes the mode-capable generation path
- RKNN startup still initializes the legacy-compatible generation path through the provider boundary
- status routes expose backend identity and truthful capability flags
- mode routes remain reachable under all backends, with unsupported operations failing cleanly where appropriate
- compat generation helpers work under both provider-backed legacy and mode-capable backends

### Manual validation

Validate:

1. `BACKEND=cuda` still supports current generation and SR behavior
2. `BACKEND=rknn` still supports current generation and SR behavior
3. `BACKEND=cpu` starts with explicit backend identity but fails clearly when generation or SR is initialized or invoked
4. `BACKEND=mlx` behaves the same way as a scaffolded future path
5. `BACKEND=auto` fails the same way as any other unsupported backend value

## Risks And Tradeoffs

### Provider indirection adds a new abstraction layer

That is acceptable because the current branching is already spread across multiple files. The provider layer centralizes truth rather than adding arbitrary abstraction.

### Backend-neutral APIs can become too generic

This should be controlled by keeping the shared contracts narrow. The shared layer should expose only what all backends can honestly support, with capability flags for optional behavior.

### RKNN may require some re-wrapping

That is expected. The goal is not to force RKNN into a CUDA-shaped implementation, but to give it a provider that satisfies the same orchestration contract.

### Placeholder backends can be mistaken for complete support

This is why the design requires explicit capability reporting and clear not-implemented failures. A scaffolded backend must be honest in status, logs, docs, and exceptions.

## Rollout

Implement in this order:

1. Introduce backend provider and platform registry contracts
2. Remove `auto` from accepted backend values and require explicit backend selection
3. Extract or guard eager `torch` imports so the backend selection layer can import cleanly for non-CUDA backends
4. Wrap existing CUDA generation, registry, and SR logic behind a CUDA provider
5. Wrap existing RKNN generation, registry, and SR logic behind an RKNN provider, allowing temporary reuse or wrapping of `PipelineService`
6. Add `mlx` and `cpu` scaffold providers with explicit placeholder behavior
7. Refactor [`server/lcm_sr_server.py`](/Users/darkbit1001/workspace/Stability-Toys/server/lcm_sr_server.py) to use the resolved provider
8. Refactor SR initialization to use the same provider and remove all `auto` fallback logic
9. Route compat generation through the provider-owned runtime
10. Reconcile the durable job abstraction and update docs and tests to reflect the explicit backend contract

### Task 1 Scope

Task 1 is the explicit backend provider contract boundary. It consists of the modules and tests that make `BACKEND` the only valid selection point and enforce the capabilities surface that every backend must expose:

1. Add `backends/platforms/base.py` with `BackendCapabilities`, `ModelRegistryProtocol`, `GenerationRuntimeProtocol`, and `BackendProvider` so downstream providers have a shared interface.
2. Create `backends/platform_registry.py` that reads `BACKEND`, raises `RuntimeError` when the value is missing or unsupported, and caches a provider instance from the supported set (`cuda`, `rknn`, `mlx`, `cpu`), with a `reset_backend_provider()` helper for tests.
3. Wire up `tests/test_platform_registry.py` with the three failing cases (missing backend, unsupported backend, known backend) and then rerun `python3 -m pytest tests/test_platform_registry.py -q` after implementation.

This section ensures we cover the TDD loop for Task 1 before touching downstream modules.

## Acceptance

This work is complete when:

- `BACKEND` is the single explicit runtime selector
- `auto` is not an accepted backend value
- generation, registry, and super-resolution behavior are all selected through one backend provider
- CUDA and RKNN continue to function through the new provider boundary
- `mlx` and `cpu` exist as explicit scaffolded backend entries
- placeholder backends fail with clear not-implemented messages at generation/SR init or use time
- the mode API remains backend-agnostic at the route layer
- docs describe CPU as a scaffolded runtime backend, not as a supported inference implementation
