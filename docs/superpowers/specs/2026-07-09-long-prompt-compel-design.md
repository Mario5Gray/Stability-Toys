# Pluggable Prompt Conditioning and Compel Long-Prompt Support - Design

**Date:** 2026-07-09
**Status:** Approved
**FP:** STABL-hvalobvn
**Brainstorm:** `fp://brainstorm?id=mqedifitfpnehpxsuqacxopetpnmonzk`

## Problem

CUDA generation currently passes `prompt` and `negative_prompt` strings directly
to Diffusers in every SD1.5 and SDXL pipeline branch. CLIP truncates these strings
to its 77-token input window, leaving roughly 75 content tokens after special
tokens. Content after that boundary has no effect on generation. Diffusers reports
the truncation only in server logs.

Compel implements the established chunk-encode-concatenate technique and handles
SDXL's dual encoders and pooled embeddings. Integrating Compel directly into
`cuda_worker.py`, however, would make the worker depend on one conditioning
implementation and make later local-memory, cache, or remote-conditioning work
another worker rewrite.

This design introduces a Stability-Toys-owned conditioning interface. Compel is
the first materializing implementation. Native prompt delegation remains the
default implementation.

## Goals

1. Provide chunked long-prompt and negative-prompt conditioning through Compel for
   SD1.5 and SDXL CUDA workers.
2. Keep workers dependent on Stability-Toys conditioning contracts rather than
   Compel, Torch, Diffusers, or a future transport protocol.
3. Support independently composed asynchronous terminal services and filters.
4. Select composition per mode and replace it atomically on model or mode reload.
5. Preserve current native prompt behavior when no conditioning service is
   configured.
6. Validate materialized conditioning against the live target pipeline immediately
   before consumption, including the effective encoder dtype.
7. Keep direct and proxy connectivity possible without defining either protocol in
   this slice.

## Non-goals

- Direct remote conditioning connectivity.
- Proxy connectivity, service discovery, or network retry policy.
- Redis, Qdrant, or another artifact store.
- A wire format for materialized slots.
- RKNN, MLX, or CPU execution wiring. Their future consumers constrain the shared
  interface shape, but only CUDA executes materialized conditioning in this slice.
- True long-context attention across chunks.
- A1111 prompt-syntax compatibility guarantees.
- Frontend changes or new CLI flags.
- A failure-class allowlist for native fallback.

## Architectural Model

The conditioning package uses Finagle-style service and filter composition:

```text
ConditioningService:
  (ConditioningRequest, ModelContext) -> ConditioningInvocation

ConditioningFilter:
  (ConditioningRequest, ModelContext, next ConditioningService)
    -> ConditioningInvocation
```

A terminal service produces one conditioning artifact. A filter decorates a
downstream service and may transform the request or result, short-circuit, retry,
or delegate. Prompt enrichment, caching, tracing, timeout, and a future proxy are
filter concerns. Native and Compel are terminal services.

The first implementation creates `backends/conditioning/` with focused modules for
contracts, artifacts, registry/composition, native service, and Compel service.
The shared contracts must not import Torch, Diffusers, or Compel.

## Core Contracts

### ConditioningRequest

`ConditioningRequest` is frozen plain data:

```python
@dataclass(frozen=True)
class ConditioningRequest:
    prompt: str
    negative_prompt: str | None
```

Generation-specific data such as size, steps, ControlNet bindings, init image, and
seed does not belong in this contract. A later enrichment filter can introduce a
separate context field through an additive contract revision; the first version
does not reserve an untyped metadata bag.

### ConditioningInvocation

The framework-neutral invocation Protocol exposes exactly four operations:

```python
class ConditioningInvocation(Protocol):
    def result(self, timeout: float | None = None) -> ConditioningArtifact: ...
    def done(self) -> bool: ...
    def cancel(self) -> bool: ...
    def exception(self, timeout: float | None = None) -> BaseException | None: ...
```

`CompletedInvocation` stores either a completed artifact or an exception and
implements this Protocol without executor or event-loop machinery. Native and local
Compel use it in the first slice. Later connectivity may adapt
`concurrent.futures.Future`, `asyncio`, or another runtime behind the Protocol.

The contract intentionally omits callback registration and mutation methods such
as `set_result`. Those belong to adapters, not consumers.

Asynchronous composition uses invocation wrappers. A filter returns an invocation
whose four operations delegate to or transform the downstream invocation. It does
not call downstream `result()` while applying the filter. Result transformation,
retry, and fallback occur when the composed invocation is observed at the consumer
boundary. This preserves lazy asynchronous behavior without adding executor or
event-loop semantics to the Protocol.

### ConditioningService

Each terminal service declares immutable requirements before chain construction:

```python
@dataclass(frozen=True)
class ConditioningServiceRequirements:
    local_encoder_bundle: bool = False

class ConditioningService(Protocol):
    requirements: ConditioningServiceRequirements

    def invoke(
        self,
        request: ConditioningRequest,
        context: ModelContext,
    ) -> ConditioningInvocation: ...
```

Native requires only the descriptor. Compel requires the local encoder bundle.
The builder rejects a chain during mode load when its context cannot satisfy the
selected service.

### ConditioningFilter

```python
class ConditioningFilter(Protocol):
    def apply(
        self,
        request: ConditioningRequest,
        context: ModelContext,
        next_service: ConditioningService,
    ) -> ConditioningInvocation: ...
```

Filters are ordered as listed in mode configuration. The first listed filter is the
outermost and receives the request first. Filters cannot disable or replace the
consumer's live compatibility check. A filter that transforms a result or handles
an asynchronous failure returns a wrapping `ConditioningInvocation`; it must not
block during `apply`.

## Artifact Envelope

`ConditioningArtifact` is a tagged union with two variants.

### DelegatedConditioning

```python
@dataclass(frozen=True)
class DelegatedConditioning:
    kind: Literal["delegated"]
    prompt: str
    negative_prompt: str | None
```

The native service returns this variant. The CUDA consumer converts it directly to
`prompt=` and `negative_prompt=` kwargs. Native therefore preserves the current
Diffusers path without eager encoding or tensor plumbing.

### MaterializedConditioning

```python
@dataclass(frozen=True)
class MaterializedConditioning:
    kind: Literal["materialized"]
    slots: Mapping[str, object]
    compatibility: ConditioningCompatibility
```

The canonical slot names are:

- `prompt_embeds`
- `negative_prompt_embeds`
- `pooled_prompt_embeds`
- `negative_pooled_prompt_embeds`

SD1.5 materialized artifacts require the first two slots and reject pooled slots.
SDXL materialized artifacts require all four. Unknown slots and missing required
slots are errors.

Slot values are opaque to the shared package. A local Compel service may place
Torch tensors in them, but Torch is not part of the contract signature or
compatibility authority. A future transport can change slot representation without
changing the tagged envelope.

### ConditioningCompatibility

Compatibility is frozen, serializable plain data:

```python
@dataclass(frozen=True)
class ConditioningCompatibility:
    model_family: Literal["sd15", "sdxl"]
    encoder_identities: tuple[str, ...]
    hidden_dimensions: tuple[int, ...]
    pooled_required: bool
    dtype_name: str
```

Encoder identities are opaque exact-match strings produced by the backend context
adapter from the loaded model configuration. The shared package does not parse
them. Canonical cross-host identity negotiation is deferred with the wire protocol.

The descriptor is necessary but not sufficient. The consumer also checks actual
materialized slot shape and dtype against the live target pipeline.

## Model Context

Model context separates remotely safe description from local capabilities.

### ModelContextDescriptor

The worker builds this frozen plain-data descriptor after loading its pipeline:

```python
@dataclass(frozen=True)
class ModelContextDescriptor:
    model_family: Literal["sd15", "sdxl"]
    tokenizer_max_length: int
    encoder_identities: tuple[str, ...]
    hidden_dimensions: tuple[int, ...]
    pooled_required: bool
    encode_dtype_name: str
    device: str
```

This is the only model context a future remote service may require. It represents
chain-construction-time state and is not a substitute for live validation.

### LocalEncoderBundle

The optional local capability exposes tokenizer and encoder access through a narrow
Protocol owned by Stability-Toys. It does not expose the worker or pipeline object.
The CUDA adapter supplies one bundle for SD1.5 and one for SDXL. Compel's
implementation module may translate that bundle into Compel constructor arguments.

`ModelContext` contains the descriptor plus an optional local encoder bundle. A
service with `local_encoder_bundle=True` cannot be composed when that capability is
absent.

## Registry and Composition

`ConditioningRegistry` maps stable names to service and filter factories. Built-in
registration initially contains terminal services `native` and `compel`; no
configurable filters ship in this slice. Registry entries are factories so each
loaded worker receives chain-local component instances.

Unknown names, duplicate names, incompatible requirements, malformed options, and
a missing explicitly requested terminal service fail during mode load. Failure is
reported before the worker thread accepts generation jobs.

The chain builder creates an immutable `ConditioningChain` snapshot. Every job uses
the snapshot associated with its loaded worker. Mode reload constructs a new worker
and chain; it never mutates an in-flight chain.

## Per-mode Configuration

`ModeConfig` gains a typed `ConditioningConfig` parsed from `modes.yml`:

```yaml
conditioning:
  service: compel
  filters: []
  fallback:
    native_when_unconfigured: true
    native_on_failure: false
```

Rules:

- Omitted `conditioning` is equivalent to an empty configuration and resolves to
  native behavior.
- Omitted or empty `service` selects `native` when
  `native_when_unconfigured=true`, which is the default.
- Omitted `filters` is an empty ordered list.
- `native_on_failure` defaults to `false`.
- `native_when_unconfigured=false` with no service is a mode-load error.
- Unknown keys are rejected rather than ignored.

Configuration declares composition; registration remains a runtime operation.

### Passing configuration to workers

The existing worker-factory call signature remains unchanged. After creating a
worker and before starting its execution thread, `WorkerPool._load_mode` checks for
a narrow `ConditioningConfigurableWorker` capability and supplies the parsed
configuration.

- An omitted/native configuration does not require this capability; existing
  non-CUDA workers retain native behavior.
- A non-native service or any filter requires the capability. Mode load fails if
  the selected worker does not implement it.
- CUDA workers implement the capability by creating their descriptor and local
  encoder bundle, resolving the registry, and publishing the immutable chain.

This keeps configuration separate from model-detection metadata and avoids changing
the injected `WorkerFactory` Protocol.

## Native Service

Native is the default only when no terminal service is configured. Its invocation
returns `DelegatedConditioning` with the request strings. It does not tokenize,
encode, truncate, import Compel, or allocate tensors.

Native may also be invoked after an eligible configured-service failure when
`native_on_failure=true`. This fallback is an explicit availability-over-fidelity
choice because long prompts can again truncate.

## Compel Service

The Compel service imports Compel only in its implementation module and uses the
local encoder bundle. It returns `MaterializedConditioning`.

For SD1.5 it materializes:

- chunked `prompt_embeds`;
- chunked `negative_prompt_embeds`.

For SDXL it materializes:

- chunked prompt and negative-prompt embeddings across both encoders;
- `pooled_prompt_embeds` and `negative_pooled_prompt_embeds` using Compel's SDXL
  pooled-output behavior.

For both families, `negative_prompt=None` is encoded as an empty string so every
materialized artifact contains the required negative slots. Prompt and
negative-prompt embeddings are padded to the same sequence length after chunking,
including SDXL when their chunk counts differ. With Compel 2.3.1, SDXL uses
Compel's multi-provider path, whose padding helper does not expose `empty_z`; the
service must pass an explicit empty-string prompt embedding as `precomputed_padding`
for SDXL padding instead of relying on Compel to derive it internally.

The service normalizes returned tensors to the local encoder bundle's live encoder
dtype and stamps that actual dtype into `ConditioningCompatibility.dtype_name`.
The chain-construction descriptor remains snapshot context only. The consumer
independently rechecks the stamped and tensor dtypes against live target-pipeline
modules.

Compel weighting syntax is accepted when the Compel service is selected. Operator
documentation may describe the basic behavior, but Stability-Toys does not promise
complete A1111 syntax compatibility in this version.

## CUDA Consumption

Each CUDA worker invokes its conditioning chain once per generation job. The same
artifact then feeds exactly one target pipeline branch.

Both CUDA worker classes implement one mandatory method:

```python
_accept_conditioning_artifact(pipe, artifact) -> dict[str, Any]
```

Every branch calls this method immediately before pipeline invocation:

| Family | Branch | Target pipeline |
|---|---|---|
| SD1.5 | txt2img | base pipeline |
| SD1.5 | txt2img + ControlNet | ControlNet pipeline |
| SD1.5 | img2img | shared img2img pipeline |
| SD1.5 | img2img + ControlNet | combined pipeline |
| SDXL | txt2img | base pipeline |
| SDXL | txt2img + ControlNet | ControlNet pipeline |
| SDXL | img2img | shared img2img pipeline |
| SDXL | img2img + ControlNet | combined pipeline |

For delegated artifacts, the method returns prompt string kwargs.

For materialized artifacts, it re-reads the target pipeline's live model family,
encoder identities, hidden dimensions, pooled requirement, and encode dtype. It
then validates:

- exact descriptor compatibility;
- exact required and allowed slot names;
- tensor dtype equals the live encode dtype;
- prompt and negative-prompt batch/sequence/hidden dimensions are compatible;
- SDXL pooled tensors exist and match the expected batch and pooled dimensions.

Only then does it return embedding kwargs. The worker never passes both prompt
strings and prompt embeddings to Diffusers.

This live check is structural, not a filter. Shared Diffusers modules and img2img
normalization can alter effective runtime dtype after the chain descriptor was
captured. A snapshot-only check could therefore recreate the float-versus-half
failure addressed by STABL-rrhsmqfc and STABL-crdsypux.

## Failure and Fallback Semantics

Failures are divided by boundary:

1. **Composition failure:** unknown component, malformed configuration, or missing
   local capability. Mode load fails; no worker thread starts.
2. **Invocation failure:** service raise, timeout, or cancellation. The request
   fails unless `native_on_failure=true`, in which case native is invoked and the
   fallback is logged with mode, service identity, and failure representation.
3. **Consumer compatibility failure:** artifact kind, slot, shape, family, encoder,
   pooled requirement, or live dtype mismatch. Always fail closed. The fallback
   toggle never applies.
4. **Pipeline failure:** errors after accepted kwargs reach Diffusers follow the
   existing job-error path.

An optional future filter may perform an earlier descriptor check, but it remains an
optimization and cannot replace the consumer check.

The builder implements native-on-failure behavior as an invocation wrapper around
the configured terminal service. A caller-initiated `cancel()` remains terminal and
never starts native generation. Fallback applies only when the downstream service
itself completes as failed, timed out, or cancelled while the outer invocation has
not been cancelled.

## Packaging and Build Surface

`compel==2.3.1` becomes a pinned backend dependency in a dedicated conditioning
requirements file used by the production and CUDA test images. Version 2.3.1
contains the SDXL 77/78-token boundary fix and declares Transformers 4.x
compatibility, matching this repository's `transformers>=4.30.0,<5.0` range.
Compel 2.4.0 is excluded because it requires Transformers 5.

Compel 2.3.1 also declares `notebook>=6.5.7` as a runtime dependency even though
generation does not require Jupyter. The images must therefore install the
dedicated Compel pin with `--no-deps` after the ordinary runtime requirements,
explicitly add Compel's actual missing leaf requirement `pyparsing~=3.0`, and
import-check Compel. This deliberately excludes `notebook` from production while
keeping Torch, Diffusers, Transformers, and pyparsing under repository authority.
`requirements-test.txt` must align its Transformers range to
`transformers>=4.30.0,<5.0` as part of the same dependency change.

The production `Dockerfile` and `Dockerfile.test` must both install and import-check
the selected Compel version. Local/native CPU tests must continue to collect without
requiring CUDA. Importing shared conditioning contracts or native behavior must not
eagerly import Compel.

`docs/TESTING_IN_DOCKER.md` will record the CUDA verification path. Shared cross-repo
build architecture remains governed by `../continuous/docs`; this repository only
documents its dependency consumption.

## Testing

### Contract tests

- Registry rejects duplicate and unknown component names.
- Filter order is deterministic and first-listed is outermost.
- Empty configuration selects native.
- Disabled unconfigured fallback with no service fails composition.
- Service requirements reject missing local capabilities.
- `CompletedInvocation` covers success, exception, completion, and cancellation
  semantics.
- Native returns delegated strings without importing Compel.
- Config parser rejects unknown keys and applies documented defaults.

### Artifact and safety tests

- Delegated artifacts become prompt kwargs only.
- SD1.5 materialized artifacts accept exactly two embedding slots.
- SDXL materialized artifacts require all four embedding and pooled slots.
- Unknown/missing slots, family mismatch, encoder mismatch, hidden-size mismatch,
  pooled mismatch, and dtype mismatch fail before pipeline invocation.
- Compatibility failure never invokes native even when `native_on_failure=true`.
- Invocation failure does invoke native when the toggle is enabled and logs it.
- A regression test mutates live encoder dtype after artifact materialization but
  before consumption and proves the consumer rejects the now-stale artifact.

### Compel tests

- A prompt longer than 77 tokens produces a sequence longer than one CLIP window.
- Negative prompts receive the same chunking behavior.
- `negative_prompt=None` materializes the empty-string negative slots.
- Prompt and negative embeddings are padded to equal sequence length for both SD1.5
  and SDXL.
- SDXL unequal-length padding passes explicit precomputed empty-string padding so
  Compel's multi-provider path does not require `empty_z`.
- Short unweighted prompts are numerically equivalent or near-equivalent to direct
  pipeline encoding on controlled stub encoders.
- SD1.5 outputs use live encoder dtype and expected hidden width.
- SDXL outputs include both pooled tensors and use live encoder dtype.
- Compel-specific imports remain isolated from native contract tests.

### Worker branch tests

Both delegated and materialized artifacts are exercised across all eight CUDA rows in
the branch table. Existing ControlNet, img2img, metadata, scheduler, and dtype tests
must remain green. The combined-path tests must continue proving preprocessing and
aspect-ratio validation behavior.

### Live verification

- Run a fixed-seed SD1.5 A/B pair whose only meaningful difference is a strong term
  after the first CLIP window.
- Repeat for SDXL.
- Confirm Compel mode emits no Diffusers truncation warning.
- Confirm native mode retains current behavior.
- Run `scripts/st-dtype-live-verify.sh` and require PASS 5/5.

## Documentation

Operator documentation must state:

- Compel chunking is not true long-context attention.
- Chunks encode independently and text spanning a boundary can lose coherence.
- Later chunks may have weaker influence.
- SDXL pooled conditioning follows Compel's first-chunk behavior.
- Tag-style prompts are generally better suited than long grammatical prose.
- `native_on_failure` can silently restore native truncation and is disabled by
  default.
- Weighting is strategy-specific and does not establish a broad A1111 compatibility
  guarantee.

No HTTP, WebSocket, CLI, or PNG metadata schema changes are required. Prompt strings
remain the request and provenance authority; materialized embeddings are runtime
artifacts.

## Acceptance Criteria

The design is implemented when:

1. Omitted conditioning configuration preserves existing native generation.
2. Per-mode `service: compel` enables long prompt and negative-prompt conditioning
   on SD1.5 and SDXL CUDA workers.
3. All txt2img, img2img, ControlNet, and combined CUDA branches consume artifacts
   through `_accept_conditioning_artifact`.
4. Materialized artifacts are checked against live modules immediately before use.
5. Compatibility failures always fail closed; invocation failures follow the
   explicit fallback toggle.
6. Shared interfaces import without Torch, Diffusers, or Compel.
7. Direct/proxy connectivity can later implement or wrap the same service/filter
   contracts without changing worker-facing interfaces.
8. Targeted unit tests, existing CUDA worker regression tests, container dependency
   checks, and live dtype verification pass.
