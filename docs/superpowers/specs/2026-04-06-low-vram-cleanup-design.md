# Low-VRAM Checkpoint Cleanup Design

## Summary

This design captures the cleanup pass for the current low-VRAM SDXL checkpoint. The core low-VRAM path is already working well enough to treat as a stable development checkpoint: SDXL single-file FP8 loading succeeds, idle VRAM sits around 3 GB, and generation spikes have stayed around 4.9 GB instead of immediately OOMing.

The remaining work is cleanup, not core enablement. Three issues remain:

1. The CUDA image can start with a broken `xformers` binary mismatch even though the runtime otherwise works.
2. VRAM logs and registry accounting mix different CUDA memory metrics, which produces misleading output such as `0.00 GB VRAM` for a model that is clearly resident.
3. SDXL single-file loading can reach out to Hugging Face during local model loads instead of behaving as a local-first path.

This design closes those gaps without reopening the larger low-VRAM feature effort or the deferred strict-6GB acceptance requirement.

## Goals

- Make the CUDA image install a Torch/xFormers stack that is internally compatible at runtime
- Fail the CUDA image build early if the packaged Torch/xFormers stack is broken
- Standardize VRAM reporting on a single allocator-based metric model
- Eliminate misleading `ModelRegistry` and `WorkerPool` load-time VRAM logs
- Make SDXL single-file loading local-first and prevent silent Hugging Face fetches during normal local loads
- Preserve the current working SDXL FP8 runtime policy and mode configuration

## Non-Goals

- Re-open the broader low-VRAM architecture work
- Revisit the current `runtime_offload` and `runtime_quantize` mode policy
- Guarantee strict acceptance on a physical 6 GB card in this pass
- Add frontend/backend version surfacing
- Perform a full dependency lockfile or package-management redesign

## Current Problems

### xFormers packaging is not deterministic

The current Docker build installs Python dependencies from `requirements.txt`, which includes bare `torch` and `xformers`, and later installs pinned CUDA wheels in a separate Docker layer. That allows the runtime stack to drift or be partially overwritten in ways that only show up at startup.

Observed symptom:

- startup warns that `xformers` was built for a different Torch, CUDA, or Python version
- `enable_xformers_memory_efficient_attention()` fails even though the rest of the pipeline loads

This means the image can appear healthy while silently missing a memory optimization that the current runtime policy expects to use when available.

### VRAM accounting mixes incompatible metrics

The current code mixes `torch.cuda.device_memory_used(...)`, `torch.cuda.memory_allocated(...)`, and `torch.cuda.memory_reserved(...)` in different places while labeling them as if they describe the same thing.

Observed symptom:

- `WorkerPool` can log meaningful allocated/reserved numbers after model load
- `ModelRegistry` can still register the same model as `0.00 GB VRAM`

This makes the current logs untrustworthy for acceptance and cleanup work because the same runtime state is described differently depending on which code path emitted the log.

### SDXL single-file load is not local-first

The current SDXL single-file code path calls `StableDiffusionXLPipeline.from_single_file(...)` without enforcing local-only behavior or supplying a strictly local resolution strategy.

Observed symptom:

- switching to the local SDXL checkpoint can trigger Hugging Face metadata and asset fetches

That is undesirable for a local checkpoint workflow. It hides missing local prerequisites behind network behavior and makes startup nondeterministic.

## Proposed Approach

Keep the current feature behavior and tighten the supporting systems around it:

- make the CUDA image own the Torch/xFormers install path explicitly
- standardize runtime VRAM reporting on allocator metrics that already reflect this process's memory behavior
- make the SDXL single-file loader fail locally and clearly instead of resolving remotely

This is intentionally a bounded cleanup pass. It does not change the current working SDXL mode semantics beyond making their supporting behavior more deterministic and easier to trust.

## Design

### 1. Deterministic CUDA Torch/xFormers packaging

Files in scope:

- `Dockerfile`
- `requirements.txt`
- optional helper verification script if the build step needs to stay readable

Design:

- Treat the CUDA image as the authoritative place that installs Torch, torchvision, torchaudio, and xFormers for CUDA builds.
- Remove or gate unconstrained `torch` and `xformers` installation from `requirements.txt` so the general pip install step does not compete with the CUDA-specific wheel install.
- Keep the CUDA-specific pinned install in `Dockerfile`, but make it the only place that selects these packages for the CUDA image.
- Add a build-time verification step that imports `torch` and `xformers`, verifies CUDA availability metadata, and fails the image build if the binary stack is incompatible.

Expected outcome:

- the current startup mismatch warning disappears
- `enable_xformers_memory_efficient_attention()` can succeed when runtime policy enables it
- CUDA image failures move to build time instead of surfacing only after deployment

### 2. Consistent VRAM reporting semantics

Files in scope:

- `backends/model_registry.py`
- `backends/worker_pool.py`
- tests covering registry and load-time reporting behavior

Design:

- Standardize process-level reporting on allocator metrics from the same family:
  - `memory_allocated` for live tensor allocations
  - `memory_reserved` for allocator-held memory
- Stop treating `device_memory_used` as interchangeable with allocator metrics in user-facing logs and model registration math.
- Rename or clarify helper methods if their current names imply different semantics than the values they return.
- Register model VRAM using a delta derived from the same allocator metric that the surrounding load log uses, so the registry entry and the worker-pool log describe the same event.

Expected outcome:

- model registration no longer reports `0.00 GB VRAM` for clearly loaded models
- logs become usable for manual validation and issue close-out
- the remaining difference between idle, reserved, and peak generation VRAM is explicit instead of accidental

### 3. Local-first SDXL single-file loading

Files in scope:

- `backends/cuda_worker.py`
- tests for SDXL single-file loader arguments and failure behavior

Design:

- Update the SDXL single-file path to prefer strictly local resolution.
- Pass local-only loader arguments where Diffusers supports them.
- If the pipeline requires local auxiliary files or config that are absent, raise a clear local error instead of silently falling back to Hugging Face.
- Leave diffusers-directory model loading behavior unchanged because that path is already local by construction.

Expected outcome:

- local SDXL checkpoints stop emitting Hugging Face fetch traffic during normal loads
- missing local prerequisites fail deterministically and are easier to diagnose

## Testing Strategy

### Unit tests

- Add or update CUDA worker tests to assert the SDXL single-file loader uses the intended local-only argument set.
- Add or update registry/worker-pool tests so allocator-based load reporting and model registration stay aligned.

### Build verification

- Extend the CUDA Docker build with an import check for `torch` and `xformers`.
- The build should fail if the binary versions are incompatible.

### Manual acceptance

Validate the following in a CUDA environment:

1. Container startup no longer shows the current xFormers binary mismatch warning.
2. SDXL mode load does not emit Hugging Face fetch traffic for the local single-file checkpoint.
3. Load-time VRAM log lines and model registration figures are internally consistent.
4. The current low-VRAM SDXL generation behavior remains intact.

## Risks And Tradeoffs

### xFormers availability may still be wheel-constrained

If the desired Torch/Python/CUDA combination does not have a compatible published `xformers` wheel, the cleanup may require adjusting the pinned stack rather than merely reordering installs. The design still holds; the image should fail early instead of shipping a broken combination.

### Offline single-file loading may expose hidden local prerequisites

That is a feature, not a regression. The current network fallback hides dependency gaps that should be explicit in a local checkpoint workflow.

### VRAM numbers will still vary by phase

Even after cleanup, idle, reserved, and peak generation memory will differ. The goal is not to collapse those values into one number; it is to ensure each reported number is clearly defined and internally consistent.

## Rollout

Implement the cleanup in this order:

1. Fix CUDA packaging and add build-time verification
2. Fix VRAM reporting semantics and tests
3. Make SDXL single-file loading local-first and update tests

This order keeps the most deployment-breaking issue first, then restores trustworthy observability, then removes the remaining nondeterministic loader behavior.

## Acceptance

This cleanup pass is complete when:

- the CUDA image builds with a verified working Torch/xFormers stack
- SDXL single-file local loads no longer hit Hugging Face
- VRAM logging and registry output are internally consistent
- the currently working low-VRAM SDXL behavior is preserved

The strict 6 GB hardware acceptance issue remains intentionally deferred.
