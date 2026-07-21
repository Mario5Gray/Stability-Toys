# HunyuanDiT Family-Profile Implementation Plan

> **For agentic workers:** Execute via `superpowers:executing-plans` inline. Do
> not use subagent-driven development (forbidden by `AGENTS.md`). Checkboxes are
> step markers only; waveplan and FP own task state for `STABL-ichgkgno`.

**Goal:** Replace scattered SD-family dispatch with one portable family-profile
resolution, preserve SD1.5/SD2.x and SDXL behavior, and add a Canny-first
HunyuanDiT CUDA family proven through the production `WorkerPool` path.

**Architecture:** Detection produces neutral architecture facts. A pure-data
`FamilyProfile` registry resolves exactly one execution family before mode
policy is overlaid. `resolve_model()` emits a wire-safe `ResolvedModel` plus a
node-local `LocalModelBinding`; CUDA selects a lazy worker reference from one
family-by-platform table. `WorkerPool` publishes one deep active snapshot and
stamps its epoch onto jobs so request admission and execution cannot cross a
mode switch silently. Thin worker subclasses retain actual denoiser behavior.

**Spec:**
`docs/superpowers/specs/2026-07-16-hunyuandit-family-profile-design.md` at
approved revision `f22e336`.

**Tech Stack:** Python 3.12, dataclasses, RFC 8785 JSON Canonicalization via
`rfc8785==0.1.4`, Diffusers `>=0.39.0`, Transformers `>=4.30.0,<5.0`,
SentencePiece `>=0.2.0`, PyTorch CUDA, pytest, Docker Compose `test-cuda`.

## Global Constraints

- FP issue: `STABL-ichgkgno`. Every implementation commit includes the issue ID
  and names the next task or review gate.
- Work only the task popped by waveplan. Stop after commit, FP revision/comment,
  drift check, and ready-for-review report. Do not self-pop or self-finish the
  next task.
- Use TDD for every code task: commit or show the focused RED first, implement
  only enough for GREEN, then run the task regression command.
- Use Miniforge for local Python commands:
  `source /Users/darkbit1001/miniforge3/bin/activate base`.
- Run `drift refs <path>` before editing any bound file. Review bound prose
  before every `drift link`; never refresh provenance mechanically.
- Keep `backends/family_profiles.py`, `backends/model_resolution.py`, and
  `backends/platforms/cuda_bindings.py` import-clean. Importing them must not
  import Torch, Diffusers, or CUDA worker modules.
- Preserve `sd15` as the execution family for SD1.5, SD2.0, and SD2.1; preserve
  `sdxl` for Base and Refiner. Do not split model lineage at the family layer.
- `ResolvedModel` is portable data authority. `LocalModelBinding` and
  `resolution_epoch` are local authority and must never serialize.
- Do not use pickle. Do not use `ModelInfo.to_dict()` as the snapshot codec.
- Do not re-detect after `ResolvedModel` is emitted. Mode policy must not
  influence family predicates, directly or through `checkpoint_variant`.
- Do not implement deferred work: remote processors, content-digest population,
  Hunyuan materialized conditioning, Depth/Pose, Hunyuan img2img/combined,
  scheduler normalization, runtime watchdogs, status-family exposure, or VRAM
  admission prediction.
- Phase 3 cannot start until Task 8's unchanged SD/SDXL CUDA gate is green.
- Live Hunyuan acceptance runs only on linux/amd64 with an NVIDIA GPU through
  the rendered `test-cuda` service.

## Delivery Map

| Phase | Tasks | Gate |
| --- | --- | --- |
| Phase 0: detector facts | 1 | Hunyuan fixture is not SDXL; existing SD outputs differ only by additive architecture facts |
| Phase 1: neutral contract | 2 | exact-one registry and open validated family strings |
| Phase 2: thread, bind, de-string | 3-8 | one resolution, one binding authority, one active snapshot; unchanged SD/SDXL CUDA suite green |
| Phase 3: Hunyuan family | 9-10 | production WorkerPool Canny generation at 1024x1024 and explicit unsupported-operation rejection |

---

### Task 1: Correct detector architecture facts

**Files:**
- Create: `tests/fixtures/models/hunyuandit-v1.1-diffusers/model_index.json`
- Create: `tests/fixtures/models/hunyuandit-v1.1-diffusers/transformer/config.json`
- Create: `tests/fixtures/models/hunyuandit-v1.1-diffusers/text_encoder/config.json`
- Create: `tests/fixtures/models/hunyuandit-v1.1-diffusers/text_encoder_2/config.json`
- Create: `tests/test_hunyuandit_detector.py`
- Modify: `utils/model_detector.py`
- Modify: existing detector expectations under `tests/`

**Produces:** `ModelInfo.base_arch` and `ModelInfo.transformer_kind` as detector
facts. It does not introduce a family string or worker selection.

- [ ] **Step 1: Add the metadata-only Hunyuan fixture**

Copy only architecture-bearing JSON fields from the proven Tencent artifact:
pipeline class, transformer class and both transformer attention dimensions,
BERT encoder class, and T5 encoder class. Do not commit weights, tokenizer
files, cache metadata, absolute paths, or inferred family values.

- [ ] **Step 2: Write the corrective RED tests**

```python
def test_hunyuandit_directory_is_not_classified_as_sdxl(hunyuandit_dir):
    info = detect_model(str(hunyuandit_dir))
    assert info.variant not in {ModelVariant.SDXL_BASE, ModelVariant.SDXL_REFINER}


def test_hunyuandit_transformer_cad_does_not_populate_unet_cad(hunyuandit_dir):
    info = detect_model(str(hunyuandit_dir))
    assert info.cross_attention_dim is None
```

Also snapshot representative existing Diffusers, Safetensors, and checkpoint SD
results before changing the detector. The expected delta after GREEN is only
`base_arch="unet"` (and `transformer_kind=None`).

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_hunyuandit_detector.py -q
```

Expected: Hunyuan is currently classified as SDXL through the ungated
dual-encoder fallback.

- [ ] **Step 3: Add neutral facts and gate the heuristic**

Extend `ModelInfo`:

```python
base_arch: str = "unknown"
transformer_kind: str | None = None
```

In `DiffusersDetector`, read `model_index.json` before component configs. Set
`base_arch="unet"` only for a declared UNet component and
`base_arch="transformer"` only for a declared transformer component. Set
`transformer_kind="hunyuandit"` only when `transformer/config.json` declares
`HunyuanDiT2DModel`. Never copy transformer attention dimensions into the UNet
`cross_attention_dim` field.

Gate the entire variant classification on architecture, not only the
dual-encoder fallback. Classification runs only for a UNet family; a transformer
or ambiguous/unknown architecture returns early as `UNKNOWN`:

```python
if info.base_arch != "unet":
    return info  # variant stays UNKNOWN; family resolution owns non-UNet arches
```

Gating only the dual-encoder heuristic is insufficient: an ambiguous
`unet`+`transformer` directory still carries the declared UNet
`cross_attention_dim` (e.g. 2048) into the CAD branch and would classify as a
dispatchable SDXL. Set `base_arch="unet"` from `SafetensorsDetector` /
`CheckpointDetector` only when their UNet key/config extraction actually
succeeds (evidence-gated); ambiguous `model_index.json` declaring both a UNet
and a transformer leaves `base_arch="unknown"`. Do not add a new lineage
heuristic.

- [ ] **Step 4: Run focused and detector regressions**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_hunyuandit_detector.py \
  tests/test_worker_factory.py \
  tests/test_model_lifecycle.py -q
```

Confirm the Hunyuan result is not SDXL or SD2.x, transformer CAD did not become
UNet CAD, and existing SD outputs changed only by additive facts.

- [ ] **Step 5: Commit and update authority**

```bash
git add utils/model_detector.py tests/fixtures/models/hunyuandit-v1.1-diffusers tests/test_hunyuandit_detector.py tests/test_worker_factory.py tests/test_model_lifecycle.py
git commit -m "fix(detector): add UNet versus transformer facts (STABL-ichgkgno) - next: neutral family registry"
```

Assign the revision and post one FP `STOP/NEXT` comment. Stop for review.

---

### Task 2: Add the neutral family registry and open family contracts

**Depends on:** Task 1

**Files:**
- Create: `backends/family_profiles.py`
- Create: `tests/test_family_profiles.py`
- Modify: `backends/conditioning/contracts.py`
- Modify: `backends/conditioning/artifacts.py`
- Modify: `tests/test_conditioning_contracts.py`
- Modify: `tests/test_conditioning_registry.py`

**Produces:** Pure comparable `FamilyProfile` values, registry-local predicates,
exact-one resolution, and immediate validation of open string family IDs.

- [ ] **Step 1: Write registry RED tests**

Cover:

```python
def test_hunyuandit_profile_is_pure_data(): ...
def test_every_known_fixture_matches_exactly_one_family(): ...
def test_zero_matches_raise_family_resolution_error(): ...
def test_multiple_matches_raise_family_resolution_error(): ...
def test_checkpoint_variant_is_not_read_by_predicates(): ...
def test_unknown_conditioning_family_fails_at_construction(): ...
```

Assert SD1.5/2.x map to `sd15` and SDXL Base/Refiner map to `sdxl`. Keep the
Task 1 transformer fixture as a zero-match unsupported case until Task 9 adds
the Hunyuan data row; Phase 3 must not leak ahead of the SD no-op gate.

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_family_profiles.py tests/test_conditioning_contracts.py tests/test_conditioning_registry.py -q
```

- [ ] **Step 2: Implement the import-clean contract**

```python
@dataclass(frozen=True)
class FamilyProfile:
    family_id: str
    encoder_roles: tuple[str, ...]
    pooled_required: bool
    pooled_projection_role: str | None
    control_image_kwarg: str


@dataclass(frozen=True)
class FamilyRegistration:
    profile: FamilyProfile
    detect: Callable[[ModelInfo], bool]
```

Define canonical `SD15_PROFILE` and `SDXL_PROFILE` objects and exactly their
predicates from the design. Task 9 adds `HUNYUANDIT_PROFILE` after the Phase 2
gate. Put detection only on `FamilyRegistration`; reject
callable/non-JSON-safe profile fields in the registry self-check.

`resolve_family()` must collect all matches, require exactly one, and use
registry order only for deterministic error rendering. Add
`validate_family_id()` and explicit `FamilyResolutionError` /
`UnknownFamilyError` types.

- [ ] **Step 3: Open both conditioning strings without weakening validation**

Change both `Literal["sd15", "sdxl"]` annotations to `str` and validate in
frozen-dataclass `__post_init__`:

```python
def __post_init__(self) -> None:
    validate_family_id(self.model_family)
```

Do not add fallback behavior for unknown families. Do not change
`compel_service.py`.

- [ ] **Step 4: Verify registry, conditioning, and import hygiene**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_family_profiles.py \
  tests/test_conditioning_contracts.py \
  tests/test_conditioning_registry.py \
  tests/test_conditioning_compel.py -q
python -c 'import sys; import backends.family_profiles; assert "torch" not in sys.modules and "diffusers" not in sys.modules'
```

- [ ] **Step 5: Commit and update authority**

```bash
git add backends/family_profiles.py backends/conditioning tests/test_family_profiles.py tests/test_conditioning_contracts.py tests/test_conditioning_registry.py
git commit -m "feat(family): add neutral exact-one family registry (STABL-ichgkgno) - next: portable resolved model"
```

Assign the revision, post `STOP/NEXT`, and stop for review.

---

### Task 3: Emit a portable ResolvedModel and local binding

**Depends on:** Task 2

**Files:**
- Create: `backends/model_resolution.py`
- Create: `tests/test_model_resolution.py`
- Modify: `backends/worker_pool.py` (move `merge_mode_capabilities` only)
- Modify: `requirements.txt`

**Produces:** Path-free wire data, deterministic descriptor identity, explicit
weak/strong artifact identity, and one pre-overlay resolver entrypoint.

- [ ] **Step 1: Add the RFC 8785 dependency and codec RED tests**

Pin `rfc8785==0.1.4`; this encoding participates in an external wire identity,
so do not leave it unbounded. Write tests for:

- all `ModelInfo` fields except `path` freeze into immutable JSON-safe data
- `ModelVariant` round-trips by value
- serialized bytes contain no source path or `LocalModelBinding`
- non-JSON metadata fails during freeze
- `thaw_model_info(snapshot, binding)` restores only the local path authority
- fixed payload produces committed canonical bytes and exact SHA-256 golden hash
- every profile field contributes to `resolution_id`
- identical local refs remain stable across path and mtime differences
- local manifests use NFC POSIX relative paths, bytewise sorting, regular files
  only, and reject any symlink
- weak fingerprint traces remain readable for diagnostics but fail execution
  validation without a digest or immutable hub commit revision
- schema, family, and canonical-profile mismatch consumption failures do not
  call detection

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_model_resolution.py -q
```

- [ ] **Step 2: Implement data classes and explicit JSON codecs**

```python
@dataclass(frozen=True)
class ModelInfoSnapshot: ...

@dataclass(frozen=True)
class ModelArtifactRef:
    kind: str
    name: str
    revision: str | None
    fingerprint: str
    digest: str | None

@dataclass(frozen=True)
class LocalModelBinding:
    model_path: str

@dataclass(frozen=True)
class ResolvedModel:
    schema_version: int
    resolution_id: str
    model_ref: ModelArtifactRef
    raw_info: ModelInfoSnapshot
    profile: FamilyProfile
    info: ModelInfoSnapshot
```

Implement named `to_json_dict`/`from_json_dict` functions for each wire value;
do not recursively dump `__dict__`, call `asdict()` as an implicit contract, or
reuse `ModelInfo.to_dict()`. Validate incoming keys/types, schema version,
family ID, and complete canonical-profile equality.

- [ ] **Step 3: Implement deterministic artifact descriptors**

Use `rfc8785.dumps(payload)` bytes directly for hashing. Include
`schema_version`, `model_ref`, `raw_info`, every `FamilyProfile` field, and
enriched `info` in the resolution payload.

For local directories, recursively inspect entries and fail if any entry is a
symlink before filtering regular files. Normalize relative names to NFC, use
POSIX separators, encode names to UTF-8 for bytewise ordering, and hash the
sorted `[relative_path, byte_size]` manifest. The fingerprint describes
structure, not bytes; do not treat it as execution authorization.

For hub refs, only a full 40-character hexadecimal commit hash is an immutable
revision in this schema. Tags, branches, short hashes, and absent revisions are
weak references and cannot authorize cross-node execution.

Expose separate validation intent, for example:

```python
validate_resolved_model_trace(resolved, for_execution=False)
```

`for_execution=True` requires `digest` or immutable hub commit hash and raises
`ResolutionCompatibilityError` before any local binding/load attempt.

- [ ] **Step 4: Move overlay ownership and implement resolve_model**

Move `merge_mode_capabilities()` unchanged from `worker_pool.py` into the new
component, then enforce this order:

```python
raw = detect_model(model_path)
profile = resolve_family(raw)
enriched = merge_mode_capabilities(raw, mode)
return build_resolved(...), LocalModelBinding(model_path)
```

The task may leave `WorkerPool` calling the moved function through an import;
factory and pool threading happen in Tasks 4-5. Do not duplicate overlay logic.

- [ ] **Step 5: Verify codec and existing overlay behavior**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_model_resolution.py tests/test_worker_pool.py tests/test_model_lifecycle.py -q
```

- [ ] **Step 6: Commit and update authority**

```bash
git add requirements.txt backends/model_resolution.py backends/worker_pool.py tests/test_model_resolution.py
git commit -m "feat(resolution): emit portable model resolution values (STABL-ichgkgno) - next: platform bindings"
```

Assign the revision, post `STOP/NEXT`, and stop for review.

---

### Task 4: Add family-platform bindings and migrate the factory

**Depends on:** Task 3

**Files:**
- Modify: `backends/platforms/base.py`
- Create: `backends/platforms/cuda_bindings.py`
- Modify: `backends/platforms/cuda.py`
- Modify: `backends/platforms/cpu.py`
- Modify: `backends/platforms/mlx.py`
- Modify: `backends/platforms/rknn.py`
- Modify: `backends/worker_factory.py`
- Modify: `backends/worker_pool.py` (`WorkerFactory` protocol only)
- Modify: `server/model_routes.py` (preserve current status response semantics)
- Modify: `tests/test_worker_factory.py`
- Modify: `tests/test_backend_runtimes.py`
- Modify: `tests/test_model_routes.py`

**Produces:** One eager capability/lazy-worker table and final factory input of
`ResolvedModel + LocalModelBinding`, with no duplicate detector fallback.

- [ ] **Step 1: Write binding and lazy-import RED tests**

Cover:

```python
def test_cuda_binding_reads_do_not_import_torch_diffusers_or_cuda_worker(): ...
def test_every_neutral_family_has_one_cuda_binding(): ...
def test_worker_ref_resolves_only_inside_create_cuda_worker(): ...
def test_factory_never_calls_detect_model(): ...
def test_known_family_without_platform_binding_is_unsupported(): ...
```

Also test exact Phase 2 execution capabilities:

| Family | img2img | ControlNet | combined |
| --- | --- | --- | --- |
| sd15 | true | true | true |
| sdxl | true | true | true |

Task 9 adds the Hunyuan row `(false, true, false)` after the no-op gate.

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_worker_factory.py tests/test_backend_runtimes.py tests/test_model_routes.py -q
```

- [ ] **Step 2: Split platform from execution capabilities**

Keep only generation, modes, super-resolution, and registry statistics in
`BackendCapabilities`. Add:

```python
@dataclass(frozen=True)
class ExecutionCapabilities: ...

@dataclass(frozen=True)
class FamilyPlatformBinding:
    worker_ref: str
    execution_capabilities: ExecutionCapabilities
```

Add `BackendProvider.family_binding(family_id)`. Unsupported CPU/MLX/RKNN cells
return no binding; they do not claim CUDA-family operations.

- [ ] **Step 3: Add the single import-clean CUDA table**

Create `CUDA_FAMILY_BINDINGS` with the `sd15` and `sdxl` rows from the spec.
Values contain dotted strings and booleans only. Task 9 adds the third Hunyuan
row after Task 8 is accepted. Do not add a parallel capability map or import
worker classes into this module.

- [ ] **Step 4: Replace factory dispatch and duplicate inspection**

Final protocol and factory signature:

```python
def create_cuda_worker(
    worker_id: int,
    resolved: ResolvedModel,
    binding: LocalModelBinding,
) -> CudaWorkerBase: ...
```

Look up the canonical CUDA cell from `resolved.profile.family_id`, import
`worker_ref` with `importlib` only inside this function, and instantiate using
`binding.model_path`, `resolved.info`, and `resolved.profile`. Delete
`inspect_model()`, `_worker_type_from_info()`, `detect_worker_type()`, old
`model_path/model_info` compatibility, and the local family if/else in this
same task. Unknown neutral family and missing platform cell remain distinct
errors.

Registry self-tests that resolve all worker refs belong to the CUDA test
environment after Task 9 introduces the Hunyuan dotted path.

- [ ] **Step 5: Preserve current model-status API without expanding scope**

Move removed execution booleans to active binding lookup where the existing
status route still needs them. If no active snapshot/binding exists, preserve
the route's current unloaded behavior explicitly. Do not add `family_id` to
`/models/status`; authoritative status-family exposure is deferred.

- [ ] **Step 6: Verify platform and factory regressions**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_worker_factory.py \
  tests/test_backend_runtimes.py \
  tests/test_model_routes.py \
  tests/test_cuda_packaging_contract.py -q
python -c 'import sys; from backends.platforms.cuda_bindings import CUDA_FAMILY_BINDINGS; assert "torch" not in sys.modules and "diffusers" not in sys.modules'
```

- [ ] **Step 7: Commit and update authority**

```bash
git add backends/platforms/base.py backends/platforms/cuda_bindings.py backends/platforms/cuda.py backends/platforms/cpu.py backends/platforms/mlx.py backends/platforms/rknn.py backends/worker_factory.py backends/worker_pool.py server/model_routes.py tests/test_worker_factory.py tests/test_backend_runtimes.py tests/test_model_routes.py
git commit -m "refactor(platform): bind workers and capabilities by family cell (STABL-ichgkgno) - next: active snapshot"
```

Assign the revision, post `STOP/NEXT`, and stop for review.

---

### Task 5: Publish one active snapshot and enforce resolution epochs

**Depends on:** Task 4

**Files:**
- Modify: `backends/worker_pool.py`
- Modify: `tests/test_worker_pool.py`
- Modify: `tests/test_model_lifecycle.py`

**Produces:** Atomic model authority for admission and a stale-job barrier
immediately before worker execution.

- [ ] **Step 1: Write snapshot lifecycle RED tests**

Cover:

- successful load detects once and publishes one deep-copied mode, resolved
  value, binding, worker, and incremented epoch atomically
- mutating the source `ModeConfig` after publication cannot mutate the snapshot
- explicit switch/unload invalidates before loading; failed load leaves no
  snapshot and no worker
- `get_active_model_snapshot()` returns one coherent value under the state lock
- idle worker eviction retains snapshot/epoch and demand reload uses the
  retained `ResolvedModel` without detection
- a deliberately re-resolved idle reload installs a new epoch
- a stale `GenerationJob` raises before fake `run_job()` is called

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_worker_pool.py tests/test_model_lifecycle.py -q
```

- [ ] **Step 2: Add immutable snapshot and epoch types**

```python
@dataclass(frozen=True)
class ActiveModelSnapshot:
    mode_name: str
    mode: ModeConfig
    resolved: ResolvedModel
    binding: LocalModelBinding
    resolution_epoch: int


class StaleResolutionError(RuntimeError): ...
```

Add `resolution_epoch` as a keyword-only required field on `GenerationJob`.
Update all tests/builders immediately; do not use an implicit default that lets
unstamped production jobs enter the queue.

- [ ] **Step 3: Refactor load publication**

`_load_mode` must:

1. deep-copy the selected mode
2. call `resolve_model()` once
3. construct/configure through the new factory
4. register resource observations
5. under the existing state lock, increment epoch and publish worker, mode, and
   snapshot together

Invalidate an old snapshot before explicit replacement. On any failure, clear
both worker and active snapshot. Preserve idle eviction as a separate path that
retains the snapshot and reconstructs the worker from its retained resolved
value and binding.

- [ ] **Step 4: Enforce epoch at the last safe boundary**

Immediately before `job.execute(self._worker)`/`worker.run_job()` in the serial
worker loop, compare the job epoch to the current snapshot epoch under the lock.
On mismatch, fail the job with `StaleResolutionError` and prove the worker was
not invoked. Do not compare only at enqueue time.

- [ ] **Step 5: Run pool/lifecycle regressions**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_worker_pool.py tests/test_model_lifecycle.py tests/test_worker_factory.py -q
```

- [ ] **Step 6: Commit and update authority**

```bash
git add backends/worker_pool.py tests/test_worker_pool.py tests/test_model_lifecycle.py tests/test_worker_factory.py
git commit -m "feat(pool): publish active resolution snapshots and epochs (STABL-ichgkgno) - next: admission authority"
```

Assign the revision, post `STOP/NEXT`, and stop for review.

---

### Task 6: Make request admission and ControlNet consume the snapshot

**Depends on:** Task 5

**Files:**
- Modify: `backends/platforms/cuda.py`
- Modify: `server/ws_routes.py`
- Modify: `server/lcm_sr_server.py`
- Modify: `server/controlnet_execution.py`
- Modify: `server/controlnet_registry.py`
- Modify: `tests/test_backend_runtimes.py`
- Modify: `tests/test_ws_routes.py`
- Modify: `tests/test_controlnet_http_contract.py`
- Modify: `tests/test_controlnet_success_contract.py`
- Modify: `tests/test_controlnet_execution.py`
- Modify: `tests/test_controlnet_registry.py`

**Produces:** One snapshot read before preprocessing for every mode-backed
submission surface, exact family-cell admission, and registry startup authority
from raw family resolution.

- [ ] **Step 1: Write admission coherence RED tests**

For WebSocket, `generate()`, and `_run_generate_from_dict()` separately, prove:

- exactly one `get_active_model_snapshot()` call occurs before preprocessing
- mode defaults, family capability, ControlNet compatibility, and job epoch all
  originate from that object
- no route calls `detect_model()`, `get_current_mode()`, or `get_mode_config()`
  after snapshot capture
- all four operation combinations use the correct cell capability with SD/SDXL
  plus a synthetic unsupported-family cell
- a queued mode switch produces `StaleResolutionError`, never cross-family run

Add startup tests showing an unknown `compatible_with` value fails config load
and a conflicting `mode.checkpoint_variant` cannot alter detected family.

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_ws_routes.py \
  tests/test_backend_runtimes.py \
  tests/test_controlnet_http_contract.py \
  tests/test_controlnet_success_contract.py \
  tests/test_controlnet_execution.py \
  tests/test_controlnet_registry.py -q
```

- [ ] **Step 2: Thread the captured snapshot through CUDA submission**

Change the mode-backed runtime entrypoint to accept the already captured
`ActiveModelSnapshot` and resolved ControlNet bindings. It must not read ambient
mode state or detect the model internally. Stamp `snapshot.resolution_epoch`
onto every `GenerationJob`.

- [ ] **Step 3: Replace provider-wide execution guards**

At admission, use:

```python
snapshot = worker_pool.get_active_model_snapshot()
binding = provider.family_binding(snapshot.resolved.profile.family_id)
caps = binding.execution_capabilities
```

Apply the operation matrix from the spec before any ControlNet preprocessing.
Base txt2img requires platform generation only; the other combinations use the
three cell booleans exactly. Preserve existing user-facing request error types,
but include stable family and operation names.

- [ ] **Step 4: Remove runtime string derivation and validate startup data**

Runtime ControlNet binding resolution receives
`snapshot.resolved.profile.family_id`; delete calls to
`active_model_family_from_variant()` from mode-backed paths. Startup
`validate_controlnet_mode_references()` calls
`resolve_family(detect_model(mode.model_path))` and never prefers
`checkpoint_variant`.

During `load_controlnet_registry()`, validate every `compatible_with` entry with
`validate_family_id()` so typos fail at startup. Retain any lineage-specific
assertions that are facts rather than family dispatch.

- [ ] **Step 5: Verify admission and registry regressions**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_ws_routes.py \
  tests/test_backend_runtimes.py \
  tests/test_controlnet_http_contract.py \
  tests/test_controlnet_success_contract.py \
  tests/test_controlnet_execution.py \
  tests/test_controlnet_registry.py \
  tests/test_model_routes.py -q
```

- [ ] **Step 6: Commit and update authority**

```bash
git add backends/platforms/cuda.py server/ws_routes.py server/lcm_sr_server.py server/controlnet_execution.py server/controlnet_registry.py tests/test_backend_runtimes.py tests/test_ws_routes.py tests/test_controlnet_http_contract.py tests/test_controlnet_success_contract.py tests/test_controlnet_execution.py tests/test_controlnet_registry.py tests/test_model_routes.py
git commit -m "refactor(admission): consume one active family snapshot (STABL-ichgkgno) - next: worker profile migration"
```

Assign the revision, post `STOP/NEXT`, and stop for review.

---

### Task 7: Refactor SD workers onto profile data and behavioral hooks

**Depends on:** Task 6

**Files:**
- Modify: `backends/cuda_worker.py`
- Modify: `tests/test_cuda_worker_base.py`
- Modify: `tests/test_cuda_worker_capabilities.py`
- Modify: `tests/test_cuda_worker_controlnet.py`
- Modify: conditioning worker tests under `tests/`

**Produces:** Profile-driven conditioning/control-map variance and narrow
behavior hooks while preserving SD/SDXL pipeline assembly and run paths.

- [ ] **Step 1: Write profile/hook RED tests**

Cover:

- base worker profile exists before `_set_native_conditioning_defaults()` first
  reads it
- direct SD and SDXL construction defaults to canonical registry objects, not
  copied profile declarations
- encoder roles, pooled requirements, projection role, and artifact slots come
  from the profile with no SDXL fallback branch
- SD quantization targets only `pipe.unet`
- SDXL targets `pipe.unet` plus `pipe.text_encoder_2`
- generic workers select `ControlNetModel` through the hook
- control-image kwargs come from `profile.control_image_kwarg`

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_cuda_worker_base.py \
  tests/test_cuda_worker_capabilities.py \
  tests/test_cuda_worker_controlnet.py \
  tests/test_conditioning_compel.py -q
```

- [ ] **Step 2: Assign profile before native-conditioning initialization**

Add a `family_profile` constructor parameter with canonical direct-construction
defaults: base/SD uses `SD15_PROFILE`, SDXL uses `SDXL_PROFILE`. The factory
always passes the resolved profile. Assign the attribute before calling
`_set_native_conditioning_defaults()`.

Replace family-string branches in `_conditioning_components()`,
`_pooled_projection_dimension()`, `_describe_conditioning_consumer()`,
`_build_conditioning_context()`, and artifact validation with profile fields.
Unknown family fallback must be impossible because profile construction was
already validated.

- [ ] **Step 3: Add narrow behavior hooks**

```python
def _quantization_targets(self, pipe: Any) -> tuple[Any, ...]: ...
def _controlnet_model_cls(self) -> type[Any]: ...
```

Make `_setup_pipe_memory_opts()` iterate the hook targets instead of assuming
`pipe.unet`/SDXL `text_encoder_2`. Make `_load_controlnet_model()` call the
family worker's model-class hook. Preserve current SD fp8 behavior exactly.

Use `self.family_profile.control_image_kwarg` when constructing control-pipeline
call kwargs. Do not otherwise rewrite SD/SDXL scheduler, style, pipeline, or
generation logic.

- [ ] **Step 4: Run CPU worker/conditioning regressions**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_cuda_worker_base.py \
  tests/test_cuda_worker_capabilities.py \
  tests/test_cuda_worker_controlnet.py \
  tests/test_conditioning_contracts.py \
  tests/test_conditioning_registry.py \
  tests/test_conditioning_compel.py -q
```

- [ ] **Step 5: Commit and update authority**

```bash
git add backends/cuda_worker.py tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py tests/test_cuda_worker_controlnet.py tests/test_conditioning_contracts.py tests/test_conditioning_registry.py tests/test_conditioning_compel.py
git commit -m "refactor(cuda): drive SD workers from family profiles (STABL-ichgkgno) - next: unchanged CUDA gate"
```

Assign the revision, post `STOP/NEXT`, and stop for review.

---

### Task 8: Pass the unchanged SD1.5/SD2.x and SDXL CUDA no-op gate

**Depends on:** Task 7

**Files:**
- Modify only if a behavior regression is found: Phase 2 implementation files
- Do not add Hunyuan worker/configuration in this task

**Produces:** Evidence that Phase 2 is observably a no-op for existing CUDA
families before Phase 3 starts.

- [ ] **Step 1: Build the production test CUDA image**

```bash
docker compose -f docker-compose.test.yml build test-cuda
```

- [ ] **Step 2: Run existing SD/SDXL worker suites unchanged**

```bash
docker compose -f docker-compose.test.yml run --rm test-cuda \
  python -m pytest \
    tests/test_cuda_worker_base.py \
    tests/test_cuda_worker_capabilities.py \
    tests/test_cuda_worker_controlnet.py \
    tests/test_conditioning_compel.py \
    tests/test_worker_factory.py \
    tests/test_worker_pool.py -q
```

Do not loosen assertions or rewrite test fixtures merely to pass the refactor.
Any pipeline class, scheduler fallback, conditioning, img2img, ControlNet,
combined-operation, seed, output, or cleanup delta is a blocker. Diagnose the
first behavioral difference and fix ownership at the profile/hook seam.

- [ ] **Step 3: Run the native contract suite**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests -q
```

- [ ] **Step 4: Record the no-op evidence and commit only necessary fixes**

If no code changes are needed, post the exact image digest and test summaries to
FP and assign the Task 7 revision as Task 8 evidence. If fixes are required:

```bash
git add <only-the-regression-fix-files>
git commit -m "fix(cuda): preserve SD family behavior after profile migration (STABL-ichgkgno) - next: Hunyuan worker"
```

Run `drift check`, post `STOP/NEXT`, and stop for human review. Phase 3 remains
blocked until the reviewer accepts this gate.

---

### Task 9: Add the lazy HunyuanDiT CUDA worker and dependency preflight

**Depends on:** Task 8 accepted

**Files:**
- Modify: `requirements.txt` (`diffusers>=0.39.0`)
- Modify: `backends/family_profiles.py`
- Modify: `backends/platforms/cuda_bindings.py`
- Modify: `backends/cuda_worker.py`
- Create: `tests/test_hunyuandit_worker.py`
- Modify: `tests/test_worker_factory.py`
- Modify: `tests/test_cuda_packaging_contract.py`

**Produces:** Thin native-conditioning Hunyuan worker selected by the existing
binding, with family-specific imports checked before model download.

- [ ] **Step 1: Write dependency/worker RED tests**

Cover:

- selecting SD does not import Hunyuan pipeline classes or tokenizer stack
- Hunyuan construction validates all three Diffusers classes and callable
  `T5Tokenizer.from_pretrained` before any `from_pretrained()` download
- dependency failure includes installed Diffusers, Transformers, and
  SentencePiece versions and raises `HunyuanDiTDependencyError`
- Hunyuan defaults to canonical profile before conditioning setup
- quantization targets `pipe.transformer` only; mT5 is excluded
- ControlNet class is `HunyuanDiT2DControlNetModel`
- worker requires a Diffusers directory and does not inherit SDXL
- base load uses fp16, ControlNet load uses fp16, composition uses
  `HunyuanDiTControlNetPipeline.from_pipe()`, and no post-composition dtype cast
- zero/one ControlNet txt2img uses `control_image` and
  `use_resolution_binning=True`
- every init image fails explicitly in the worker

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_hunyuandit_worker.py tests/test_worker_factory.py tests/test_cuda_packaging_contract.py -q
```

- [ ] **Step 2: Raise the proven dependency floor**

Change only the Diffusers floor to `diffusers>=0.39.0`; retain
`transformers>=4.30.0,<5.0` and `sentencepiece>=0.2.0`. Keep the exact
`rfc8785==0.1.4` pin from Task 3.

- [ ] **Step 3: Add Hunyuan neutral data and its CUDA cell**

Add `HUNYUANDIT_PROFILE` and the transformer-kind predicate to
`FAMILY_REGISTRY`, then add the one CUDA binding:

```python
"hunyuandit": FamilyPlatformBinding(
    "backends.cuda_worker.DiffusersHunyuanDiTCudaWorker",
    ExecutionCapabilities(False, True, False),
)
```

Extend registry tests so the Task 1 fixture now moves from the intentional
zero-match case to exactly one `hunyuandit` match. This is the only expected
family-resolution change after the Phase 2 gate.

- [ ] **Step 4: Implement the lazy dependency preflight**

The Hunyuan worker/module path performs the check at construction, not server
import. Validate:

```python
HunyuanDiTPipeline
HunyuanDiTControlNetPipeline
HunyuanDiT2DControlNetModel
callable(T5Tokenizer.from_pretrained)
```

Collect versions without masking the original missing/placeholder failure.
This check must run before any model download.

- [ ] **Step 5: Implement DiffusersHunyuanDiTCudaWorker**

Subclass `CudaWorkerBase` directly. Preserve the base worker's metadata, cache
release, seed, and cleanup seams, but own Hunyuan pipeline assembly:

- Diffusers directory only
- base `HunyuanDiTPipeline.from_pretrained(..., torch_dtype=self.dtype)`
- native DDPMScheduler unless a tested explicit scheduler is selected
- Hunyuan ControlNet through `_controlnet_model_cls()` and existing cache
- `HunyuanDiTControlNetPipeline.from_pipe(base, controlnet=...)`
- component placement before composition; device-only movement afterward
- native prompt delegation, no SDXL pooled materialization
- `use_resolution_binning=True`
- zero/one Canny ControlNet txt2img and no init image

Allow the known Tencent `learn_sigma`/`norm_type` warning without hiding it;
tests must fail for new missing weights or incompatibility warnings. Preserve
the repository's existing safety-checker posture and visible warning.

- [ ] **Step 6: Verify CPU contracts and lazy imports**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_hunyuandit_worker.py \
  tests/test_worker_factory.py \
  tests/test_cuda_packaging_contract.py \
  tests/test_cuda_worker_controlnet.py -q
python -c 'import sys; from backends.platforms.cuda_bindings import CUDA_FAMILY_BINDINGS; assert "torch" not in sys.modules and "diffusers" not in sys.modules'
```

- [ ] **Step 7: Rebuild CUDA and run preflight before model download**

```bash
docker compose -f docker-compose.test.yml build test-cuda
docker compose -f docker-compose.test.yml run --rm test-cuda \
  python -m pytest tests/test_hunyuandit_worker.py -q -k dependency
```

Record installed Diffusers, Transformers, and SentencePiece versions. The
container must satisfy the repo ranges and prove callable `T5Tokenizer` before
the live model acceptance in Task 10.

- [ ] **Step 8: Commit and update authority**

```bash
git add requirements.txt backends/family_profiles.py backends/platforms/cuda_bindings.py backends/cuda_worker.py tests/test_family_profiles.py tests/test_hunyuandit_worker.py tests/test_worker_factory.py tests/test_cuda_packaging_contract.py
git commit -m "feat(cuda): add native-conditioning HunyuanDiT worker (STABL-ichgkgno) - next: Canny mode acceptance"
```

Assign the revision, post `STOP/NEXT`, and stop for review.

---

### Task 10: Add Canny-first production configuration and live acceptance

**Depends on:** Task 9

**Files:**
- Modify: `conf/modes.yml`
- Modify: `conf/controlnets.yaml`
- Modify: `tests/test_mode_config.py`
- Modify: `tests/test_controlnet_registry.py`
- Modify: `tests/test_backend_runtimes.py`
- Create: `tests/test_hunyuandit_acceptance.py`
- Modify: `docs/TESTING_IN_DOCKER.md` only if the operator command/24 GiB floor is not already documented elsewhere
- Modify: `project-forward-notes.md`
- Modify: `drift.lock` only after bound prose review

**Produces:** One Hunyuan 1024x1024 native-conditioning mode, one Canny entry,
CPU contract proof, and production-path CUDA evidence.

- [ ] **Step 1: Write configuration RED tests**

Assert the first production mode:

- references the local Hunyuan Diffusers base directory
- defaults to 1024x1024
- uses native scheduler and empty/native conditioning service
- enables only Canny with `max_attachments=1`
- does not advertise img2img or combined execution
- resolves the ControlNet entry as compatible only with `hunyuandit`
- startup validation uses detector facts even if `checkpoint_variant` conflicts

Run RED:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_controlnet_registry.py tests/test_model_lifecycle.py tests/test_backend_runtimes.py -q
```

- [ ] **Step 2: Add the mode and registry rows**

Use repository-local model layout conventions and stable IDs:

```yaml
# conf/controlnets.yaml, under models:
hunyuandit-canny:
  path: /models/controlnets/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny
  compatible_with: [hunyuandit]
  control_types: [canny]
```

Add a `HunyuanDiT` mode pointing at
`diffusers/HunyuanDiT-v1.1-Diffusers`, default 1024x1024, native scheduler,
native conditioning (omit the `conditioning` key so
`native_when_unconfigured=True` applies), Canny only, and one attachment. Use
`scheduler_profile: native`; do not invent profile/capability fields in YAML.

- [ ] **Step 3: Run complete CPU contract coverage**

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests -q
```

Fix only failures attributable to this delivery. Record unrelated baseline
failures separately; do not weaken coverage.

- [x] **Step 4: Run live production-path CUDA acceptance**

On the approved NVIDIA host with both model directories mounted under
`/models`:

```bash
docker compose -f docker-compose.test.yml build test-cuda
docker compose -f docker-compose.test.yml run --rm test-cuda \
  python -m pytest tests/test_hunyuandit_acceptance.py -q -m cuda
```

The acceptance must load the configured mode through `WorkerPool` (not the
spike), assert the active `hunyuandit` profile and `(False, True, False)` CUDA
cell, reject img2img/combined before preprocessing, generate one coherent Canny
txt2img result at 1024x1024 with resolution binning, and verify cache release,
mode switch, and stale-epoch behavior.

Coherence needs its own assertion. Size, seed, and PNG text chunks all pass on
pure noise, so the acceptance also captures `diffusers` logger warnings during
generation and fails on any `will be ignored` message — the signature of a
dropped pipeline kwarg. Write the artifact to `ACCEPTANCE_OUT_DIR` when set so
the image survives the container and can be inspected directly.

Record installed dependency versions, generated artifact path, elapsed time,
and `torch.cuda.max_memory_allocated()`. Compare peak allocated VRAM to the
21.37 GiB spike observation; small variance is recorded, while OOM or material
regression on the same host/matrix blocks delivery. The supported non-offloaded
fp16 operator baseline is a tested 24 GiB GPU. Measure with attention slicing
and xformers off, matching how this family actually runs — the worker declines
processor swaps, so any figure taken with them enabled is not comparable.

Measured result:

```text
artifact=/store/hunyuandit-canny-1024-acceptance.png
elapsed_s=113.91  peak_allocated_bytes=20189025280  torch=2.10.0+cu128 cuda=12.8
```

18.80 GiB peak, 2.57 GiB below the 21.37 GiB spike observation and 5.2 GiB
under the 24 GiB operator floor. The byte count is identical across runs taken
with and without xformers and attention slicing, so peak allocation is set by
weights and fixed buffers at load rather than by the attention implementation:
declining processor swaps costs no VRAM. It costs time instead — roughly
3.80 s/it against 3.45 s/it with xformers enabled, about 10%. That trade is not
optional for this family, since the swapped processors silently drop the
rotary positional embeddings.

- [x] **Step 5: Review docs and drift before relinking**

```bash
drift refs conf/modes.yml
drift refs conf/controlnets.yaml
drift refs backends/cuda_worker.py
drift check
```

Update only prose falsified by the implementation. Keep shared build/CI
architecture in `../continuous/docs`; repo docs may state only Stability Toys'
consumption and CUDA verification commands. Then relink each reviewed target
individually and rerun `drift check`.

- [ ] **Step 6: Commit and update authority**

```bash
git add conf/modes.yml conf/controlnets.yaml tests/test_mode_config.py tests/test_controlnet_registry.py tests/test_backend_runtimes.py tests/test_hunyuandit_acceptance.py docs/TESTING_IN_DOCKER.md project-forward-notes.md drift.lock
git commit -m "feat(hunyuandit): ship Canny-first family configuration and acceptance (STABL-ichgkgno) - next: final review"
```

Assign the revision and post a final FP `STOP/NEXT` comment containing CPU/CUDA
commands, dependency matrix, output evidence, VRAM observation, drift result,
and deferred non-goals. Stop for final human review; do not call waveplan `fin`
or open a PR until the review cycle explicitly authorizes it.

## Final Review Checklist

- [ ] `git diff --check` is clean and no unrelated work is included.
- [ ] Every waveplan task has a revision and one FP `STOP/NEXT` comment.
- [ ] `python -m pytest tests -q` passes in Miniforge or documented baseline
  exceptions have independent evidence.
- [ ] Existing SD/SDXL CUDA tests passed unchanged before Hunyuan work began.
- [ ] Live Hunyuan acceptance used `WorkerPool`, not the spike script.
- [ ] Binding capability reads remain Torch/Diffusers import-clean.
- [ ] No request/load path re-detects after `ResolvedModel` emission.
- [ ] Serialized resolution contains no path, binding, epoch, callable, or
  worker reference.
- [ ] Weak artifact fingerprints never authorize cross-node execution.
- [ ] All deferred follow-ups remain deferred.
- [ ] `drift check` is green after prose review and target-specific relinking.
