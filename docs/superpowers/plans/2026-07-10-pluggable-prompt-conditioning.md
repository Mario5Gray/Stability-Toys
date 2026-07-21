# Pluggable Prompt Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project policy forbids subagent-driven development; execute inline and stop for review at each FP task boundary.

**Goal:** Add a source-stable asynchronous conditioning service/filter surface, implement native and Compel terminal services, and route all SD1.5/SDXL CUDA generation branches through live-validated conditioning artifacts.

**Architecture:** `backends/conditioning/` owns framework-neutral contracts, tagged artifacts, invocation handles, registration, and composition. Per-mode configuration builds an immutable chain after worker creation; CUDA supplies a serializable model descriptor plus local encoder capability, invokes the chain once per job, and validates artifacts against the live target pipeline immediately before use. Compel is opt-in per mode; omitted configuration remains native so non-CUDA workers do not regress.

**Tech Stack:** Python 3.10+, dataclasses/Protocols, PyTorch, Diffusers, Compel 2.3.1, pytest, YAML mode configuration, Docker.

**Authority:** `docs/superpowers/specs/2026-07-09-long-prompt-compel-design.md`, FP `STABL-hvalobvn`, brainstorm `mqedifitfpnehpxsuqacxopetpnmonzk` v2.

---

## Global Constraints

- Use Miniforge base for local Python commands:
  `source /Users/darkbit1001/miniforge3/bin/activate base` and then `python -m pytest`.
- Run `drift refs <path>` before editing every bound file. Review prose before any
  `drift link`; do not refresh stale unrelated documents.
- Keep `backends/conditioning/contracts.py`, `artifacts.py`, and `invocation.py`
  importable without Torch, Diffusers, Compel, FastAPI, or server modules.
- Do not add `compel` to `requirements.txt`; its declared `notebook` dependency must
  not enter the production image.
- Do not change HTTP, WebSocket, CLI, or PNG metadata request/response schemas.
- Do not enable Compel in shared `conf/modes.yml` by default. The same file can serve
  RKNN deployments, and non-CUDA materialized consumption remains out of scope.
- Do not implement direct/proxy connectivity, serialization, Redis/Qdrant storage,
  service discovery, or non-CUDA materialized consumption.
- Commit after every task using `STABL-hvalobvn`, assign each revision to FP, and
  leave one milestone comment before continuing.

## File Structure

- **Create `backends/conditioning/contracts.py`:** request, model context,
  requirements, service/filter Protocols, and typed chain configuration.
- **Create `backends/conditioning/artifacts.py`:** delegated/materialized variants and
  plain-data compatibility descriptor.
- **Create `backends/conditioning/invocation.py`:** four-member invocation Protocol,
  completed invocation, transforming wrapper, and native-fallback wrapper.
- **Create `backends/conditioning/registry.py`:** service/filter factories, duplicate
  protection, immutable chain construction, and built-in registration.
- **Create `backends/conditioning/native.py`:** zero-encoding delegated terminal.
- **Create `backends/conditioning/compel_service.py`:** lazy Compel import and
  SD1.5/SDXL materialization.
- **Create `backends/conditioning/__init__.py`:** stable public exports only.
- **Modify `server/mode_config.py`:** strict per-mode conditioning parser.
- **Modify `backends/base.py`:** narrow configurable-worker capability Protocol.
- **Modify `backends/worker_pool.py`:** configure a worker before its thread starts.
- **Modify `backends/cuda_worker.py`:** model context adapter, chain invocation,
  live artifact acceptance, and all eight branch call sites.
- **Create `requirements-conditioning.txt`:** exact Compel pin installed with
  `--no-deps`.
- **Modify `requirements.txt`, `requirements-test.txt`, `Dockerfile`,
  `Dockerfile.test`:** explicit dependency authority and import checks.
- **Create `docs/PROMPT_CONDITIONING.md`:** operator configuration, fallback, and
  chunking caveats.
- **Modify `docs/TESTING_IN_DOCKER.md`, `project-forward-notes.md`:** verification and
  shipped boundary.
- **Create tests:** `tests/test_conditioning_contracts.py`,
  `tests/test_conditioning_registry.py`, `tests/test_conditioning_compel.py`, and
  `tests/test_compel_packaging.py`.
- **Modify tests:** `tests/test_mode_config.py`, `tests/test_worker_pool.py`,
  `tests/test_cuda_worker_base.py`, `tests/test_cuda_worker_capabilities.py`, and
  `tests/test_cuda_worker_controlnet.py`.

---

### Task 1: Framework-neutral contracts, artifacts, and completed invocations

**Files:**
- Create: `backends/conditioning/__init__.py`
- Create: `backends/conditioning/contracts.py`
- Create: `backends/conditioning/artifacts.py`
- Create: `backends/conditioning/invocation.py`
- Create: `tests/test_conditioning_contracts.py`

- [ ] **Step 1: Write failing artifact and invocation tests**

Create `tests/test_conditioning_contracts.py`:

```python
import pytest

from backends.conditioning.artifacts import (
    ConditioningCompatibility,
    DelegatedConditioning,
    MaterializedConditioning,
)
from backends.conditioning.contracts import ConditioningRequest
from backends.conditioning.invocation import CompletedInvocation


def test_conditioning_request_preserves_optional_negative_prompt():
    request = ConditioningRequest(prompt="cat", negative_prompt=None)
    assert request.prompt == "cat"
    assert request.negative_prompt is None


def test_artifact_union_keeps_payload_types_out_of_descriptor():
    marker = object()
    compatibility = ConditioningCompatibility(
        model_family="sd15",
        encoder_identities=("clip-l",),
        hidden_dimensions=(768,),
        pooled_required=False,
        dtype_name="float16",
    )
    artifact = MaterializedConditioning(
        slots={"prompt_embeds": marker, "negative_prompt_embeds": marker},
        compatibility=compatibility,
    )
    assert artifact.kind == "materialized"
    assert artifact.slots["prompt_embeds"] is marker
    assert compatibility.dtype_name == "float16"


def test_completed_invocation_returns_artifact_and_cannot_cancel_completed_work():
    artifact = DelegatedConditioning(prompt="cat", negative_prompt=None)
    invocation = CompletedInvocation.success(artifact)
    assert invocation.done() is True
    assert invocation.cancel() is False
    assert invocation.exception() is None
    assert invocation.result() is artifact


def test_completed_invocation_reraises_stored_exception():
    failure = RuntimeError("encode failed")
    invocation = CompletedInvocation.failure(failure)
    assert invocation.done() is True
    assert invocation.exception() is failure
    with pytest.raises(RuntimeError, match="encode failed"):
        invocation.result()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_conditioning_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named
'backends.conditioning'`.

- [ ] **Step 3: Implement the artifact envelope**

Create `backends/conditioning/artifacts.py`:

```python
from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias


@dataclass(frozen=True)
class ConditioningCompatibility:
    model_family: Literal["sd15", "sdxl"]
    encoder_identities: tuple[str, ...]
    hidden_dimensions: tuple[int, ...]
    pooled_required: bool
    dtype_name: str


@dataclass(frozen=True)
class DelegatedConditioning:
    prompt: str
    negative_prompt: str | None
    kind: Literal["delegated"] = "delegated"


@dataclass(frozen=True)
class MaterializedConditioning:
    slots: Mapping[str, object]
    compatibility: ConditioningCompatibility
    kind: Literal["materialized"] = "materialized"


ConditioningArtifact: TypeAlias = DelegatedConditioning | MaterializedConditioning
```

- [ ] **Step 4: Implement the shared contracts**

Create `backends/conditioning/contracts.py` with no runtime backend imports:

```python
from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol

from .invocation import ConditioningInvocation


@dataclass(frozen=True)
class ConditioningRequest:
    prompt: str
    negative_prompt: str | None


@dataclass(frozen=True)
class ModelContextDescriptor:
    model_family: Literal["sd15", "sdxl"]
    tokenizer_max_length: int
    encoder_identities: tuple[str, ...]
    hidden_dimensions: tuple[int, ...]
    pooled_required: bool
    encode_dtype_name: str
    device: str


class LocalEncoderBundle(Protocol):
    def tokenizers(self) -> tuple[object, ...]: ...
    def text_encoders(self) -> tuple[object, ...]: ...
    def live_dtype(self) -> object: ...


@dataclass(frozen=True)
class ModelContext:
    descriptor: ModelContextDescriptor
    local_encoder_bundle: LocalEncoderBundle | None = None


@dataclass(frozen=True)
class ConditioningServiceRequirements:
    local_encoder_bundle: bool = False


@dataclass(frozen=True)
class ConditioningFallbackConfig:
    native_when_unconfigured: bool = True
    native_on_failure: bool = False


@dataclass(frozen=True)
class ConditioningConfig:
    service: str | None = None
    filters: tuple[str, ...] = ()
    fallback: ConditioningFallbackConfig = field(
        default_factory=ConditioningFallbackConfig
    )

    def requires_configurable_worker(self) -> bool:
        return bool(self.service or self.filters)


class ConditioningService(Protocol):
    requirements: ConditioningServiceRequirements

    def invoke(
        self, request: ConditioningRequest, context: ModelContext
    ) -> ConditioningInvocation: ...


class ConditioningFilter(Protocol):
    def apply(
        self,
        request: ConditioningRequest,
        context: ModelContext,
        next_service: ConditioningService,
    ) -> ConditioningInvocation: ...
```

- [ ] **Step 5: Implement completed invocations**

Create `backends/conditioning/invocation.py`:

```python
from dataclasses import dataclass
from typing import Protocol

from .artifacts import ConditioningArtifact


class ConditioningInvocation(Protocol):
    def result(self, timeout: float | None = None) -> ConditioningArtifact: ...
    def done(self) -> bool: ...
    def cancel(self) -> bool: ...
    def exception(self, timeout: float | None = None) -> BaseException | None: ...


@dataclass(frozen=True)
class CompletedInvocation:
    _artifact: ConditioningArtifact | None = None
    _exception: BaseException | None = None

    @classmethod
    def success(cls, artifact: ConditioningArtifact) -> "CompletedInvocation":
        return cls(_artifact=artifact)

    @classmethod
    def failure(cls, exception: BaseException) -> "CompletedInvocation":
        return cls(_exception=exception)

    def result(self, timeout: float | None = None) -> ConditioningArtifact:
        del timeout
        if self._exception is not None:
            raise self._exception
        if self._artifact is None:
            raise RuntimeError("completed invocation has no result")
        return self._artifact

    def done(self) -> bool:
        return True

    def cancel(self) -> bool:
        return False

    def exception(self, timeout: float | None = None) -> BaseException | None:
        del timeout
        return self._exception
```

- [ ] **Step 6: Export only stable names**

Create `backends/conditioning/__init__.py` exporting the dataclasses and Protocols
from Tasks 1 without importing `compel_service`.

- [ ] **Step 7: Run GREEN and import-isolation checks**

Run:

```bash
python -m pytest tests/test_conditioning_contracts.py -q
python -c 'import sys; import backends.conditioning; assert "compel" not in sys.modules; assert "diffusers" not in sys.modules'
```

Expected: tests pass and the import command exits 0.

- [ ] **Step 8: Commit Task 1**

```bash
git add backends/conditioning tests/test_conditioning_contracts.py
git commit -m "feat(conditioning): add stable artifact and invocation contracts (STABL-hvalobvn)"
```

---

### Task 2: Registry, native terminal, filter composition, and fallback

**Files:**
- Create: `backends/conditioning/native.py`
- Create: `backends/conditioning/registry.py`
- Modify: `backends/conditioning/invocation.py`
- Modify: `backends/conditioning/__init__.py`
- Create: `tests/test_conditioning_registry.py`

- [ ] **Step 1: Write failing registry/composition tests**

Create `tests/test_conditioning_registry.py` covering these exact cases:

```python
import logging
import pytest

from backends.conditioning.artifacts import DelegatedConditioning
from backends.conditioning.contracts import (
    ConditioningConfig,
    ConditioningFallbackConfig,
    ConditioningRequest,
    ConditioningServiceRequirements,
    ModelContext,
    ModelContextDescriptor,
)
from backends.conditioning.invocation import CompletedInvocation, NativeFallbackInvocation
from backends.conditioning.registry import ConditioningRegistry, build_conditioning_chain


def context(local_bundle=None):
    return ModelContext(
        descriptor=ModelContextDescriptor(
            model_family="sd15",
            tokenizer_max_length=77,
            encoder_identities=("clip-l",),
            hidden_dimensions=(768,),
            pooled_required=False,
            encode_dtype_name="float16",
            device="cuda:0",
        ),
        local_encoder_bundle=local_bundle,
    )


def test_duplicate_registration_is_rejected():
    registry = ConditioningRegistry()
    registry.register_service("native", lambda: object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_service("native", lambda: object())


def test_empty_configuration_builds_native_chain():
    chain = build_conditioning_chain(ConditioningConfig(), context())
    artifact = chain.invoke(ConditioningRequest("cat", None), context()).result()
    assert artifact == DelegatedConditioning("cat", None)


def test_unknown_explicit_service_fails_composition():
    with pytest.raises(ValueError, match="unknown conditioning service 'missing'"):
        build_conditioning_chain(
            ConditioningConfig(service="missing"), context()
        )


def test_first_configured_filter_is_outermost():
    events = []
    registry = ConditioningRegistry.with_builtins()
    registry.register_filter("outer", lambda: RecordingFilter("outer", events))
    registry.register_filter("inner", lambda: RecordingFilter("inner", events))
    chain = build_conditioning_chain(
        ConditioningConfig(filters=("outer", "inner")), context(), registry
    )
    chain.invoke(ConditioningRequest("cat", None), context()).result()
    assert events == ["outer:before", "inner:before", "inner:after", "outer:after"]


def test_missing_required_local_bundle_fails_composition():
    registry = ConditioningRegistry.with_builtins()
    registry.register_service("needs-local", NeedsLocalService)
    with pytest.raises(ValueError, match="local encoder bundle"):
        build_conditioning_chain(
            ConditioningConfig(service="needs-local"), context(), registry
        )


def test_native_fallback_handles_service_failure_and_logs(caplog):
    registry = ConditioningRegistry.with_builtins()
    registry.register_service("broken", BrokenService)
    config = ConditioningConfig(
        service="broken",
        fallback=ConditioningFallbackConfig(native_on_failure=True),
    )
    with caplog.at_level(logging.WARNING):
        artifact = build_conditioning_chain(config, context(), registry).invoke(
            ConditioningRequest("cat", None), context()
        ).result()
    assert artifact == DelegatedConditioning("cat", None)
    assert "conditioning fallback" in caplog.text


def test_outer_cancel_never_starts_native_fallback():
    primary = PendingInvocation()
    invocation = NativeFallbackInvocation(
        primary=primary,
        native_service=RecordingNativeService(),
        request=ConditioningRequest("cat", None),
        context=context(),
        service_name="remote",
    )
    assert invocation.cancel() is True
    assert invocation.native_service.calls == 0
```

Define the small `RecordingFilter`, `NeedsLocalService`, `BrokenService`,
`PendingInvocation`, and `RecordingNativeService` fakes in this test file.
`RecordingFilter.apply` records before invocation and returns a wrapper that records
after downstream `result`; `BrokenService.invoke` returns
`CompletedInvocation.failure(RuntimeError("boom"))`; `PendingInvocation.cancel`
records cancellation and returns `True`.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_conditioning_registry.py -q`

Expected: import failure for `backends.conditioning.registry`.

- [ ] **Step 3: Implement the native terminal**

Create `backends/conditioning/native.py`:

```python
from .artifacts import DelegatedConditioning
from .contracts import (
    ConditioningRequest,
    ConditioningServiceRequirements,
    ModelContext,
)
from .invocation import CompletedInvocation


class NativeConditioningService:
    requirements = ConditioningServiceRequirements()

    def invoke(self, request: ConditioningRequest, context: ModelContext):
        del context
        return CompletedInvocation.success(
            DelegatedConditioning(request.prompt, request.negative_prompt)
        )
```

- [ ] **Step 4: Implement transforming and fallback invocations**

Add `TransformingInvocation` and `NativeFallbackInvocation` to `invocation.py`.
`TransformingInvocation` stores a downstream invocation and a pure `on_result`
callable; `result(timeout)` applies the callable after downstream completion,
`done` delegates, `cancel` propagates, and `exception(timeout)` evaluates the final
transformed result so an exception raised by the transform is visible.

`NativeFallbackInvocation` delegates `done`, propagates caller `cancel`, and in
`result(timeout)` catches only the downstream invocation's
failure/timeout/cancel state. When fallback is enabled and the outer invocation was
not cancelled, it logs service name plus `repr(error)` and returns
`native_service.invoke(request, context).result(timeout)`.

Its `exception(timeout)` observes the final result: it returns `None` when native
fallback succeeds and returns the final exception when both primary and fallback
fail. This method may therefore trigger lazy fallback exactly as `result` does.

Do not catch exceptions after `result()` has returned an artifact; compatibility
validation happens later in the CUDA consumer and must never enter this wrapper.

- [ ] **Step 5: Implement registry and immutable composition**

Create `backends/conditioning/registry.py` with:

```python
class ConditioningRegistry:
    def __init__(self):
        self._services = {}
        self._filters = {}

    @classmethod
    def with_builtins(cls):
        registry = cls()
        registry.register_service("native", NativeConditioningService)
        return registry

    def register_service(self, name, factory): ...
    def register_filter(self, name, factory): ...
    def create_service(self, name): ...
    def create_filter(self, name): ...


class ConditioningChain:
    def __init__(self, service):
        self._service = service

    def invoke(self, request, context):
        return self._service.invoke(request, context)
```

`build_conditioning_chain(config, context, registry=None)` must:

1. resolve an omitted service to `native` only when
   `native_when_unconfigured=True`;
2. reject omitted service when that toggle is false;
3. instantiate and requirement-check the terminal;
4. wrap the configured terminal in a fallback service only when
   `native_on_failure=True` and the terminal is not already native;
5. wrap filters around that service in reverse construction order, using a small
   `FilteredConditioningService`, so the first configured filter is outermost and
   filter failures do not silently enter native fallback;
6. return a chain containing no mutable registry reference.

- [ ] **Step 6: Run GREEN and the combined pure-contract suite**

Run:

```bash
python -m pytest tests/test_conditioning_contracts.py tests/test_conditioning_registry.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add backends/conditioning tests/test_conditioning_registry.py
git commit -m "feat(conditioning): compose native services filters and fallback (STABL-hvalobvn)"
```

---

### Task 3: Strict per-mode configuration and worker lifecycle handoff

**Files:**
- Modify: `server/mode_config.py`
- Modify: `backends/base.py`
- Modify: `backends/worker_pool.py`
- Modify: `tests/test_mode_config.py`
- Modify: `tests/test_worker_pool.py`

- [ ] **Step 1: Add failing mode-config tests**

Append tests using the existing `tmp_path/modes.yml` pattern in
`tests/test_mode_config.py`:

```python
def test_mode_config_parses_conditioning_chain(tmp_path):
    write_single_mode_config(tmp_path, """
    conditioning:
      service: compel
      filters: [trace, cache]
      fallback:
        native_when_unconfigured: true
        native_on_failure: false
    """)
    mode = ModeConfigManager(str(tmp_path)).get_mode("test")
    assert mode.conditioning.service == "compel"
    assert mode.conditioning.filters == ("trace", "cache")
    assert mode.conditioning.fallback.native_on_failure is False


@pytest.mark.parametrize("unknown_path", [
    "conditioning.unknown",
    "conditioning.fallback.unknown",
])
def test_mode_config_rejects_unknown_conditioning_keys(tmp_path, unknown_path):
    write_conditioning_config_with_unknown_key(tmp_path, unknown_path)
    with pytest.raises(ValueError, match="unknown conditioning"):
        ModeConfigManager(str(tmp_path))


def test_mode_config_rejects_empty_service_when_native_default_disabled(tmp_path):
    write_single_mode_config(tmp_path, """
    conditioning:
      fallback:
        native_when_unconfigured: false
    """)
    with pytest.raises(ValueError, match="requires a service"):
        ModeConfigManager(str(tmp_path))
```

Implement `write_single_mode_config` in the test file as a helper that writes the
minimal root/model/resolution structure already repeated by current tests. The
unknown-key helper should insert exactly one unknown key at the requested level.

- [ ] **Step 2: Run mode-config tests and verify RED**

Run: `python -m pytest tests/test_mode_config.py -q`

Expected: `ModeConfig` has no `conditioning` attribute and malformed blocks are not
rejected.

- [ ] **Step 3: Parse a typed strict configuration**

In `server/mode_config.py`:

- import `ConditioningConfig` and `ConditioningFallbackConfig` from the shared
  package;
- add `conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)`
  to `ModeConfig`;
- add `_parse_conditioning_config(mode_name, raw)` that accepts only `service`,
  `filters`, and `fallback`; accepts only `native_when_unconfigured` and
  `native_on_failure` under fallback; requires strings in `filters`; and rejects an
  empty service when native default is disabled;
- pass the parsed value into `ModeConfig` construction and include it in `to_dict`.

- [ ] **Step 4: Run mode-config tests and verify GREEN**

Run: `python -m pytest tests/test_mode_config.py -q`

Expected: all existing and new mode-config tests pass.

- [ ] **Step 5: Add failing worker handoff tests**

Append to `tests/test_worker_pool.py`:

```python
def test_load_mode_configures_conditioning_before_worker_thread_starts(
    mock_mode_config, mock_registry
):
    events = []

    class ConfigurableWorker:
        worker_id = 0
        def configure_conditioning(self, config):
            events.append(("configure", config.service))
        def run_job(self, job):
            return b"png", 1

    mode = mock_mode_config.get_mode.return_value
    mode.conditioning = ConditioningConfig(service="compel")
    pool = WorkerPool(
        worker_factory=Mock(return_value=ConfigurableWorker()),
        mode_config=mock_mode_config,
        registry=mock_registry,
    )
    assert events == [("configure", "compel")]
    pool.shutdown()


def test_non_native_config_rejects_worker_without_conditioning_capability(
    mock_mode_config, mock_registry
):
    mode = mock_mode_config.get_mode.return_value
    mode.conditioning = ConditioningConfig(service="compel")
    worker = object()
    with pytest.raises(RuntimeError, match="does not support conditioning"):
        WorkerPool(
            worker_factory=Mock(return_value=worker),
            mode_config=mock_mode_config,
            registry=mock_registry,
        )
```

- [ ] **Step 6: Run worker tests and verify RED**

Run: `python -m pytest tests/test_worker_pool.py -k conditioning -q`

Expected: no configuration call occurs and unsupported workers are accepted.

- [ ] **Step 7: Add the capability and lifecycle handoff**

In `backends/base.py`, add a separate Protocol without changing `PipelineWorker`:

```python
class ConditioningConfigurableWorker(Protocol):
    def configure_conditioning(self, config: "ConditioningConfig") -> None: ...
```

Use `TYPE_CHECKING` for the conditioning config import. In
`WorkerPool._load_mode`, immediately after factory return and before registry/model
registration or `_start_worker_thread`:

```python
configure_conditioning = getattr(self._worker, "configure_conditioning", None)
if callable(configure_conditioning):
    configure_conditioning(mode.conditioning)
elif mode.conditioning.requires_configurable_worker():
    raise RuntimeError(
        f"mode '{mode_name}' configures conditioning but worker "
        f"{type(self._worker).__name__} does not support conditioning"
    )
```

Omitted/native configuration remains accepted for RKNN and placeholders.

- [ ] **Step 8: Run lifecycle regression tests**

Run:

```bash
python -m pytest tests/test_mode_config.py tests/test_worker_pool.py -q
```

Expected: all tests pass and existing worker-factory signatures remain unchanged.

- [ ] **Step 9: Commit Task 3**

```bash
git add server/mode_config.py backends/base.py backends/worker_pool.py tests/test_mode_config.py tests/test_worker_pool.py
git commit -m "feat(conditioning): configure immutable chains per mode (STABL-hvalobvn)"
```

---

### Task 4: CUDA model context and intrinsic artifact acceptance

**Files:**
- Modify: `backends/cuda_worker.py`
- Modify: `tests/test_cuda_worker_base.py`
- Modify: `tests/test_cuda_worker_capabilities.py`

- [ ] **Step 1: Write failing context and acceptance tests**

Add focused tests that construct workers with `__new__` and fake pipeline modules:

```python
def test_sd15_model_context_exposes_plain_descriptor_and_local_bundle():
    worker = make_sd15_worker_with_fake_pipe(dtype=torch.float16)
    context = worker._build_conditioning_context()
    assert context.descriptor.model_family == "sd15"
    assert context.descriptor.hidden_dimensions == (768,)
    assert context.descriptor.pooled_required is False
    assert context.descriptor.encode_dtype_name == "float16"
    assert context.local_encoder_bundle.text_encoders() == (worker.pipe.text_encoder,)


def test_sdxl_model_context_describes_both_encoders_and_pooled_output():
    worker = make_sdxl_worker_with_fake_pipe(dtype=torch.float16)
    context = worker._build_conditioning_context()
    assert context.descriptor.model_family == "sdxl"
    assert context.descriptor.hidden_dimensions == (768, 1280)
    assert context.descriptor.pooled_required is True
    assert len(context.descriptor.encoder_identities) == 2


def test_accept_materialized_rechecks_live_dtype_after_artifact_creation():
    worker = make_sd15_worker_with_fake_pipe(dtype=torch.float16)
    artifact = materialized_sd15(dtype=torch.float16, dtype_name="float16")
    worker.pipe.text_encoder.dtype = torch.float32
    with pytest.raises(ValueError, match="dtype"):
        worker._accept_conditioning_artifact(worker.pipe, artifact)


def test_accept_delegated_returns_only_prompt_kwargs():
    worker = make_sd15_worker_with_fake_pipe(dtype=torch.float16)
    kwargs = worker._accept_conditioning_artifact(
        worker.pipe, DelegatedConditioning("cat", None)
    )
    assert kwargs == {"prompt": "cat", "negative_prompt": None}
```

Also add parameterized materialized rejection cases for unknown/missing slots,
family mismatch, encoder identity mismatch, hidden width mismatch, pooled mismatch,
and tensor dtype mismatch. Assert the target pipeline mock is never called.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
python -m pytest tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py -k conditioning -q
```

Expected: workers lack `_build_conditioning_context` and
`_accept_conditioning_artifact`.

- [ ] **Step 3: Factor one live encode-dtype helper**

In `CudaWorkerBase`, add `_pipeline_encode_dtype(pipe)` using the existing priority
`text_encoder_2`, `text_encoder`, `unet`, then `self.dtype`. Refactor
`_normalize_img2img_modules` to call it so model context, Compel, acceptance, and VAE
normalization cannot diverge on dtype selection.

- [ ] **Step 4: Add the local CUDA encoder bundle and context builder**

Add a private `_CudaEncoderBundle` implementing the shared Protocol with tuples of
tokenizers/encoders and a callable that reads the live encode dtype. Build opaque
encoder identities from each encoder's config `_name_or_path` when present, falling
back to its stable logical role (`text_encoder`, `text_encoder_2`) plus class name.

`_build_conditioning_context` must derive family from the worker class, capture
tokenizer max length and hidden sizes from loaded components, and return plain
descriptor data plus the local bundle. It must never expose `self.pipe` through the
shared contract.

- [ ] **Step 5: Implement intrinsic acceptance**

Add `_accept_conditioning_artifact(pipe, artifact)` to `CudaWorkerBase`:

```python
if isinstance(artifact, DelegatedConditioning):
    return {"prompt": artifact.prompt, "negative_prompt": artifact.negative_prompt}

if not isinstance(artifact, MaterializedConditioning):
    raise ValueError(f"unknown conditioning artifact: {type(artifact).__name__}")

live = self._describe_conditioning_consumer(pipe)
if artifact.compatibility != live.compatibility:
    raise ValueError("conditioning compatibility does not match live pipeline")

required = (
    {"prompt_embeds", "negative_prompt_embeds"}
    if live.model_family == "sd15"
    else {
        "prompt_embeds", "negative_prompt_embeds",
        "pooled_prompt_embeds", "negative_pooled_prompt_embeds",
    }
)
if set(artifact.slots) != required:
    raise ValueError("conditioning slots do not match live pipeline")
```

Then validate all slot values are tensors, prompt/negative batch and sequence lengths
match, hidden widths match SD1.5 width or SDXL concatenated width, pooled tensors have
matching batch and expected second-encoder projection width, and every tensor dtype
equals `_pipeline_encode_dtype(pipe)`.

- [ ] **Step 6: Run CUDA base/capability tests GREEN**

Run:

```bash
python -m pytest tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py -q
```

Expected: all existing dtype normalization tests and new acceptance tests pass.

- [ ] **Step 7: Refresh affected drift prose and commit Task 4**

Run `drift refs backends/cuda_worker.py`. If the 2026-04-18 ControlNet prose remains
accurate, do not silently relink its pre-existing stale anchor; report it separately.
Link the approved conditioning spec to new conditioning files and refreshed symbols.

```bash
git add backends/cuda_worker.py tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py drift.lock docs/superpowers/specs/2026-07-09-long-prompt-compel-design.md
git commit -m "feat(conditioning): validate artifacts against live CUDA pipelines (STABL-hvalobvn)"
```

---

### Task 5: Dependency packaging without the Jupyter runtime stack

**Files:**
- Create: `requirements-conditioning.txt`
- Modify: `requirements.txt`
- Modify: `requirements-test.txt`
- Modify: `Dockerfile`
- Modify: `Dockerfile.test`
- Create: `tests/test_compel_packaging.py`

- [ ] **Step 1: Write failing packaging contract tests**

Create `tests/test_compel_packaging.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compel_pin_is_isolated_from_notebook_dependency_resolution():
    conditioning = (ROOT / "requirements-conditioning.txt").read_text()
    runtime = (ROOT / "requirements.txt").read_text()
    assert "compel==2.3.1" in conditioning
    assert "compel" not in runtime.lower()
    assert "pyparsing~=3.0" in runtime


def test_transformers_major_is_capped_in_runtime_and_test_requirements():
    assert "transformers>=4.30.0,<5.0" in (ROOT / "requirements.txt").read_text()
    assert "transformers>=4.30.0,<5.0" in (ROOT / "requirements-test.txt").read_text()


def test_images_install_compel_without_declared_notebook_dependencies():
    for filename in ("Dockerfile", "Dockerfile.test"):
        text = (ROOT / filename).read_text()
        assert "requirements-conditioning.txt" in text
        assert "--no-deps" in text
        assert "version('compel')" in text
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_compel_packaging.py -q`

Expected: missing requirements file and uncapped test Transformers range.

- [ ] **Step 3: Add explicit dependency authority**

Create `requirements-conditioning.txt`:

```text
# Installed with --no-deps after requirements.txt to exclude compel's unrelated
# notebook>=6.5.7 dependency from runtime images.
compel==2.3.1
```

Add `pyparsing~=3.0` to `requirements.txt` and change the test line to
`transformers>=4.30.0,<5.0`.

- [ ] **Step 4: Install and verify in both image paths**

In `Dockerfile`, copy the conditioning file beside `requirements.txt`; only for
`BACKEND=cuda`, run:

```dockerfile
RUN if [ "$BACKEND" = "cuda" ]; then \
      pip install --no-cache-dir --no-deps -r /app/requirements-conditioning.txt; \
      python -c "from importlib.metadata import version; import compel; assert version('compel') == '2.3.1'"; \
    fi
```

In `Dockerfile.test`, copy the file and install it with `--no-deps` for every test
image so CPU collection can execute Compel unit tests without CUDA.

- [ ] **Step 5: Run packaging tests GREEN**

Run: `python -m pytest tests/test_compel_packaging.py -q`

Expected: 3 passed.

- [ ] **Step 6: Run local dependency resolution smoke**

Run in a disposable virtual environment or the test container, not by mutating the
shared Miniforge environment:

```bash
python -m pip install --dry-run -r requirements.txt
python -m pip install --dry-run --no-deps -r requirements-conditioning.txt
```

Expected: Compel resolves exactly 2.3.1 and the second command lists no Notebook or
Jupyter packages.

- [ ] **Step 7: Commit Task 5**

```bash
git add requirements-conditioning.txt requirements.txt requirements-test.txt Dockerfile Dockerfile.test tests/test_compel_packaging.py
git commit -m "build(conditioning): pin Compel without Jupyter dependencies (STABL-hvalobvn)"
```

---

### Task 6: Local Compel terminal for SD1.5 and SDXL

**Files:**
- Create: `backends/conditioning/compel_service.py`
- Modify: `backends/conditioning/registry.py`
- Modify: `backends/conditioning/__init__.py`
- Create: `tests/test_conditioning_compel.py`

- [ ] **Step 1: Write failing Compel service tests**

Use tiny fake tokenizers/encoders or monkeypatch the lazily imported `Compel` class;
do not load model weights. Cover:

```python
def test_none_negative_prompt_is_encoded_as_empty_string(compel_spy, sd15_context):
    artifact = CompelConditioningService().invoke(
        ConditioningRequest("cat", None), sd15_context
    ).result()
    assert compel_spy.prompts == ["cat", ""]
    assert set(artifact.slots) == {"prompt_embeds", "negative_prompt_embeds"}


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_prompt_and_negative_are_padded_to_same_sequence_length(
    family, context_for_family, compel_spy
):
    compel_spy.return_lengths = (154, 77)
    artifact = CompelConditioningService().invoke(
        ConditioningRequest("long prompt", "short"), context_for_family(family)
    ).result()
    assert artifact.slots["prompt_embeds"].shape[1] == 154
    assert artifact.slots["negative_prompt_embeds"].shape[1] == 154


def test_sdxl_materializes_pooled_pair(sdxl_context):
    artifact = CompelConditioningService().invoke(
        ConditioningRequest("cat", "bad"), sdxl_context
    ).result()
    assert set(artifact.slots) == {
        "prompt_embeds", "negative_prompt_embeds",
        "pooled_prompt_embeds", "negative_pooled_prompt_embeds",
    }


def test_service_uses_live_bundle_dtype_not_snapshot_descriptor(sd15_context):
    stale_context = replace(
        sd15_context,
        descriptor=replace(sd15_context.descriptor, encode_dtype_name="float32"),
    )
    stale_context.local_encoder_bundle.set_live_dtype(torch.float16)
    artifact = CompelConditioningService().invoke(
        ConditioningRequest("cat", None), stale_context
    ).result()
    assert artifact.compatibility.dtype_name == "float16"
    assert all(t.dtype == torch.float16 for t in artifact.slots.values())
```

Also test a >77-token prompt returns sequence length greater than 77 and a short
unweighted prompt is near-equal to direct fake-encoder output.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_conditioning_compel.py -q`

Expected: `CompelConditioningService` is missing.

- [ ] **Step 3: Implement lazy SD1.5 materialization**

In `compel_service.py`, import Compel inside the factory/invocation path. Build the
low-level `Compel` object from `bundle.tokenizers()` and
`bundle.text_encoders()`; do not require a Diffusers pipeline object. For SD1.5,
use `truncate_long_prompts=False`, encode prompt plus `negative_prompt or ""`, and
call Compel's padding helper on both tensors.

- [ ] **Step 4: Implement SDXL dual-encoder and pooled materialization**

For SDXL instantiate low-level Compel with the two tokenizer/encoder tuples,
`ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED`, and
`requires_pooled=[False, True]`. Encode positive and negative strings separately,
pad their sequence tensors to equal length, and preserve both pooled outputs.

Padding SDXL is not symmetric with SD1.5. Live verification found that Compel
2.3.1 routes SDXL through its multi-provider path, whose padding helper does not
expose `empty_z`, so the service must hand
`pad_conditioning_tensors_to_same_length` an explicit empty-string prompt
embedding as `precomputed_padding` rather than letting Compel derive it. See the
spec's materialization section.

Read `live_dtype = bundle.live_dtype()` immediately before normalization. Convert
all four/two tensors to that dtype, then stamp `dtype_name` derived from that same
object into `ConditioningCompatibility`. Copy family, identities, hidden dimensions,
and pooled requirement from the context descriptor.

- [ ] **Step 5: Register Compel without eager import**

Register `"compel"` using a factory function whose module import does not execute
until the service is selected. Re-run the Task 1 import-isolation command and assert
an empty/native chain leaves `compel` absent from `sys.modules`.

- [ ] **Step 6: Run Compel and pure-contract suites GREEN**

Run:

```bash
python -m pytest tests/test_conditioning_contracts.py tests/test_conditioning_registry.py tests/test_conditioning_compel.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add backends/conditioning tests/test_conditioning_compel.py
git commit -m "feat(conditioning): materialize long prompts with Compel (STABL-hvalobvn)"
```

---

### Task 7: Invoke and consume conditioning in all CUDA branches

**Files:**
- Modify: `backends/cuda_worker.py`
- Modify: `tests/test_cuda_worker_capabilities.py`
- Modify: `tests/test_cuda_worker_controlnet.py`

- [ ] **Step 1: Add failing configuration and branch tests**

Add `configure_conditioning` tests proving native default and Compel capability
construction. Then extend existing branch tests so each captured target pipeline
receives either delegated string kwargs or materialized kwargs for all eight targets:

```python
@pytest.mark.parametrize("family,branch", [
    ("sd15", "txt2img"),
    ("sd15", "txt2img_controlnet"),
    ("sd15", "img2img"),
    ("sd15", "img2img_controlnet"),
    ("sdxl", "txt2img"),
    ("sdxl", "txt2img_controlnet"),
    ("sdxl", "img2img"),
    ("sdxl", "img2img_controlnet"),
])
def test_materialized_conditioning_reaches_every_cuda_target(family, branch):
    worker, job, target_pipe = make_branch_case(family, branch)
    artifact = compatible_materialized_artifact(worker, family)
    worker._conditioning_chain = chain_returning(artifact)
    worker.run_job(job)
    kwargs = target_pipe.call_args.kwargs
    assert "prompt" not in kwargs
    assert "negative_prompt" not in kwargs
    assert kwargs["prompt_embeds"] is artifact.slots["prompt_embeds"]
    assert kwargs["negative_prompt_embeds"] is artifact.slots["negative_prompt_embeds"]
    if family == "sdxl":
        assert kwargs["pooled_prompt_embeds"] is artifact.slots["pooled_prompt_embeds"]
```

Implement `make_branch_case` using the existing fake pipeline/from-pipe helpers in
`test_cuda_worker_controlnet.py`; do not duplicate Diffusers module stubs in another
file. Add a separate delegated test asserting the current negative-prompt forwarding
test remains string-based under native configuration.

- [ ] **Step 2: Run branch slice and verify RED**

Run:

```bash
python -m pytest tests/test_cuda_worker_capabilities.py tests/test_cuda_worker_controlnet.py -k conditioning -q
```

Expected: CUDA workers lack chain configuration/invocation and still build prompt
kwargs directly.

- [ ] **Step 3: Configure an immutable chain on CUDA workers**

Add `configure_conditioning(config)` to `CudaWorkerBase`. It builds a fresh model
context from the loaded base pipeline, resolves the built-in registry, and assigns
both context and chain only after composition succeeds. Constructor state uses a
native chain so direct unit construction remains safe before WorkerPool calls the
capability.

- [ ] **Step 4: Invoke once per job and accept per target branch**

At the beginning of each SD1.5/SDXL `run_job`, after request extraction and before
branch dispatch:

```python
conditioning_request = ConditioningRequest(
    prompt=req.prompt,
    negative_prompt=getattr(req, "negative_prompt", None),
)
artifact = self._conditioning_chain.invoke(
    conditioning_request, self._conditioning_context
).result()
```

In each of the eight target branches, remove direct prompt keys and merge:

```python
pipe_kwargs = {
    **self._accept_conditioning_artifact(target_pipe, artifact),
    # existing image/size/steps/guidance/generator/controlnet kwargs remain
}
```

Call acceptance after combined/controlnet pipeline construction and after any
img2img module normalization, immediately before `target_pipe(**pipe_kwargs)`.
Set `artifact = None` in `finally` before `torch.cuda.empty_cache()` to release
embedding references on success and failure.

- [ ] **Step 5: Run all CUDA branch tests GREEN**

Run:

```bash
python -m pytest tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py tests/test_cuda_worker_controlnet.py tests/test_worker_controlnet_metadata.py -q
```

Expected: all tests pass. If the known cross-file Diffusers stub pollution recurs,
also run each affected file independently and report both combined and isolated
results; do not conceal the combined failure.

- [ ] **Step 6: Prove native fallback cannot catch compatibility failure**

Add a worker-level test with `native_on_failure=True` where the service returns an
artifact whose dtype no longer matches the live pipe. Assert
`_accept_conditioning_artifact` raises, the pipeline is not called, and the native
service invocation count remains zero.

Run: `python -m pytest tests/test_cuda_worker_capabilities.py -k compatibility -q`

Expected: pass.

- [ ] **Step 7: Commit Task 7**

```bash
git add backends/cuda_worker.py tests/test_cuda_worker_capabilities.py tests/test_cuda_worker_controlnet.py
git commit -m "feat(conditioning): route every CUDA branch through artifacts (STABL-hvalobvn)"
```

---

### Task 8: Operator docs, container verification, drift, and closeout

**Files:**
- Create: `docs/PROMPT_CONDITIONING.md`
- Modify: `docs/TESTING_IN_DOCKER.md`
- Modify: `project-forward-notes.md`
- Modify: `drift.lock`

- [ ] **Step 1: Write operator documentation**

Create `docs/PROMPT_CONDITIONING.md` with:

- per-mode native and Compel YAML examples;
- the empty-configuration native default;
- `native_on_failure` availability/fidelity warning;
- explicit statement that compatibility failures never fall back;
- Compel chunking caveats: independent chunks, weaker later influence, grammatical
  boundary loss, SDXL pooled first-chunk behavior;
- `negative_prompt=None` empty-string behavior;
- weighting is strategy-specific and not an A1111 compatibility guarantee;
- direct/proxy/Redis/Qdrant connectivity is deferred.

Do not enable Compel in shared `conf/modes.yml`. For live CUDA verification,
document a temporary operator edit under the chosen CUDA-only deployment config.

- [ ] **Step 2: Document deterministic container checks**

Add to `docs/TESTING_IN_DOCKER.md`:

```bash
docker compose -f docker-compose.test.yml build test
docker compose -f docker-compose.test.yml run --rm test \
  python -c "from importlib.metadata import version; import compel; print(version('compel'))"

docker compose -f docker-compose.test.yml build test-cuda
docker compose -f docker-compose.test.yml run --rm test-cuda \
  python -m pytest tests/test_conditioning_compel.py \
    tests/test_cuda_worker_capabilities.py \
    tests/test_cuda_worker_controlnet.py -q
```

State that production/test image package inspection must show Compel 2.3.1 and no
Notebook/Jupyter packages introduced by Compel installation.

- [ ] **Step 3: Run the local full targeted suite**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest \
  tests/test_conditioning_contracts.py \
  tests/test_conditioning_registry.py \
  tests/test_conditioning_compel.py \
  tests/test_compel_packaging.py \
  tests/test_mode_config.py \
  tests/test_worker_pool.py \
  tests/test_cuda_worker_base.py \
  tests/test_cuda_worker_capabilities.py \
  tests/test_cuda_worker_controlnet.py \
  tests/test_worker_controlnet_metadata.py -q
```

Expected: all tests pass, aside from any separately reported pre-existing combined
Diffusers-stub pollution that remains green when files run independently.

- [ ] **Step 4: Build and run the local/native test container**

Run:

```bash
docker compose -f docker-compose.test.yml build test
docker compose -f docker-compose.test.yml run --rm test \
  python -m pytest tests/test_conditioning_contracts.py \
    tests/test_conditioning_registry.py tests/test_conditioning_compel.py \
    tests/test_compel_packaging.py -q
```

Expected: build succeeds and tests pass on the CPU/native image.

- [ ] **Step 5: Run explicit CUDA container verification on an NVIDIA host**

Run:

```bash
docker compose -f docker-compose.test.yml build test-cuda
docker compose -f docker-compose.test.yml run --rm test-cuda \
  python -m pytest tests/test_conditioning_compel.py \
    tests/test_cuda_worker_capabilities.py \
    tests/test_cuda_worker_controlnet.py -q
```

Expected: build succeeds, Compel imports as 2.3.1, and tests pass.

- [ ] **Step 6: Run live long-prompt and dtype verification**

On the CUDA deployment, temporarily set `conditioning.service: compel` on one SD1.5
mode and `SDXL`, reload modes, and run fixed-seed A/B generations where only a strong
tail term after token 77 changes. Confirm the tail changes both families' output and
logs contain no Diffusers truncation warning.

Then run:

```bash
scripts/st-dtype-live-verify.sh --mode lcm-general
```

Expected: `RESULT: PASS (5/5)` and no module dtype poisoning.

- [ ] **Step 7: Refresh drift and project register**

Run `drift refs` for every modified code/doc file. Update the conditioning spec prose
first if implementation differs, then link the spec to all new conditioning modules.
Update `project-forward-notes.md` with the shipped interface, CUDA-only materialized
consumer boundary, Compel opt-in, and direct/proxy deferral.

Run: `drift check`

Expected: the conditioning spec and new docs are `ok`. Record older unrelated stale
anchors separately; do not relink them without prose review.

- [ ] **Step 8: Commit docs and verification closeout**

```bash
git add docs/PROMPT_CONDITIONING.md docs/TESTING_IN_DOCKER.md project-forward-notes.md docs/superpowers/specs/2026-07-09-long-prompt-compel-design.md drift.lock
git commit -m "docs(conditioning): document Compel strategy and verification (STABL-hvalobvn)"
```

- [ ] **Step 9: FP review handoff**

Assign all task revisions to `STABL-hvalobvn` and post one final comment containing:

- commit list;
- targeted/local-container/CUDA/live verification results;
- explicit drift characterization;
- `STOP: implementation complete and ready for review`;
- `NEXT: start review; do not mark done or self-advance`.

Do not call waveplan `fin` or mark the FP issue done before the review cycle completes.

---

## Plan Self-review Checklist

- Every approved spec contract maps to Tasks 1-8.
- The review corrections are explicit: live bundle dtype stamping, empty-string
  negative conditioning, same-length padding for both families, no-deps Compel
  packaging, Transformers `<5`, and eight CUDA targets.
- Native remains the unconfigured default and shared modes remain safe for RKNN.
- Compatibility validation remains intrinsic to the consumer and cannot enter
  native fallback.
- No remote/proxy/storage serialization work is included.
- Every implementation task starts with a failing test, names an exact command,
  and ends with a focused commit.
