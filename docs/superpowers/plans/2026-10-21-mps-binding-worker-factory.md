# Binding-Driven Worker Factory Plan

**Goal:** Create workers from the binding table selected by one backend provider.

**Architecture:** Keep neutral family authority in `ResolvedModel`. Pass the backend name and family binding table into one generic factory.

**Scope:** Preserve CUDA behavior. Add no MPS worker or feature support.

## Task 1: Generalize worker creation

**Files:** `backends/worker_factory.py`, `backends/platforms/cuda.py`, `tests/test_worker_factory.py`, `docs/WORKER_SELECTION.md`

1. Add failing tests for supplied binding tables and explicit MPS errors.
2. Run `python -m pytest tests/test_worker_factory.py -q --no-cov` and confirm failure.
3. Update worker-selection prose before changing drift-bound code.
4. Add a generic binding-driven factory without eager worker imports.
5. Bind CUDA provider creation to `CUDA_FAMILY_BINDINGS`.
6. Keep `create_cuda_worker` as a compatibility wrapper.
7. Run focused factory and provider tests.
8. Run drift checks and inspect the diff.
9. Commit named files and report review readiness.
