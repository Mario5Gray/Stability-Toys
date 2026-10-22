# Torch Execution Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`. Do not use subagent-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate shared Diffusers worker behavior from CUDA device actions behind a small Torch execution policy.

**Architecture:** `backends/torch_policy.py` defines the contract. `backends/platforms/cuda_policy.py` holds the CUDA policy. `CudaWorkerBase.execution_policy` is a class attribute that defaults to the CUDA policy. A later MPS worker overrides that attribute.

**Tech Stack:** Python 3.12, pytest, Diffusers, optimum-quanto.

**Spec:** FP `STABL-xkiwjvrm` description and its sigma HANDOFF comment (target `e355ab9`).

## Global Constraints

- Preserve every `CUDA_*` environment variable, default, and `model_info.runtime_*` override.
- Preserve current CUDA worker class names, attributes, log text, and output contracts.
- Add no MPS policy, MPS binding, or MPS feature.
- Do not move provider or subprocess factory wiring. `STABL-lnlreqeh` owns it.
- The policy modules must not import Torch at module import time.

## Policy Seam

| Concern | Owner after this task |
|---|---|
| `CUDA_*` env parse and runtime overrides | Policy `resolve_settings` |
| fp8 runtime quantization (optimum.quanto) | Policy `quantize_modules` |
| xformers attention backend | Policy `enable_memory_efficient_attention` |
| Offload or `.to(device)` placement, `gpu_id` | Policy `place_pipeline` |
| Allocator cache release after jobs | Policy `release_cache` |
| VAE tiling/slicing, attention slicing, processor-swap gate, schedulers, styles, conditioning, ControlNet | Shared worker (unchanged) |
| `torch.Generator(device=...)`, generic `.to(self.device)` moves | Shared worker (device string from settings) |

## Review Focus

- `CUDA_DEVICE=cuda:bad` must still give `gpu_id=0`. Pinned by existing `_device_index` tests.
- `runtime_*` overrides must still beat env values. Pinned by existing base test.
- An fp8 checkpoint with `CUDA_QUANTIZE=fp8` must still skip runtime quanto. Pinned by existing capability test.
- HunyuanDiT must still skip xformers and attention slicing. Pinned by existing Hunyuan tests.
- Tests that patch `backends.cuda_worker.torch.cuda.empty_cache` must still intercept the release. The policy imports Torch at call time from `sys.modules`.

---

### Task 1: Extract the CUDA Torch execution policy

**Files:**
- Create: `backends/torch_policy.py`
- Create: `backends/platforms/cuda_policy.py`
- Modify: `backends/cuda_worker.py` (`_parse_env`, `_setup_pipe_memory_opts`, `_device_index`, five `torch.cuda.empty_cache()` sites)
- Test: `tests/test_torch_policy.py`

**Interfaces:**
- Produces: `TorchExecutionSettings(device: str, dtype: Any, dtype_str: str, enable_xformers: bool, attention_slicing: bool, quantize: str, offload: str)` frozen dataclass.
- Produces: `TorchExecutionPolicy` runtime-checkable Protocol with `backend_id: str`, `resolve_settings(model_info) -> TorchExecutionSettings`, `quantize_modules(modules: tuple) -> None`, `enable_memory_efficient_attention(pipe) -> None`, `place_pipeline(pipe, *, device: str, offload: str) -> Any`, `release_cache() -> None`.
- Produces: `CudaTorchPolicy` with static `device_index(device: str) -> int`, and module singleton `CUDA_TORCH_POLICY`.
- Produces: `CudaWorkerBase.execution_policy: TorchExecutionPolicy = CUDA_TORCH_POLICY`.

- [x] **Step 1: Write failing contract tests** in `tests/test_torch_policy.py`:
  - `test_policy_modules_import_without_torch` — subprocess imports both modules; asserts `torch` not in `sys.modules`.
  - `test_cuda_policy_satisfies_contract` — `isinstance(CUDA_TORCH_POLICY, TorchExecutionPolicy)` and `backend_id == "cuda"`.
  - `test_cuda_policy_resolves_env_defaults` — empty env gives `device="cuda:0"`, `dtype_str="fp16"`, `quantize="none"`, `offload="none"`, both flags False.
  - `test_cuda_policy_runtime_overrides_beat_env` — `runtime_offload="model"`, `runtime_quantize="FP8"` beat env; result `offload="model"`, `quantize="fp8"`.
  - `test_cuda_policy_places_with_device_index` — `offload="model"`, `device="cuda:2"` calls `enable_model_cpu_offload(gpu_id=2)`.
  - `test_cuda_policy_release_cache_calls_cuda_empty_cache` — patched `torch.cuda.empty_cache` called once.
  - `test_worker_delegates_device_actions_to_policy` — a recording policy on a `CudaWorkerBase` subclass receives `resolve_settings`, `place_pipeline`, and `quantize_modules` (fp8), and the worker keeps its `device`, `dtype`, `_offload`, `_quantize` attributes from the settings.

- [x] **Step 2: Run RED**

Run: `python -m tests.run tests/test_torch_policy.py -- -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: backends.torch_policy`.

- [x] **Step 3: Add `backends/torch_policy.py` and `backends/platforms/cuda_policy.py`.** Move the `_parse_env` env logic verbatim into `CudaTorchPolicy.resolve_settings`. Move `_device_index` logic into `CudaTorchPolicy.device_index`. Import Torch and `optimum.quanto` inside methods only.

- [x] **Step 4: Route `CudaWorkerBase` through `self.execution_policy`.** `_parse_env` copies settings onto the existing attributes. `_setup_pipe_memory_opts` keeps its order: quantize, VAE, attention, place. `_device_index` delegates. Replace the five `torch.cuda.empty_cache()` calls with `self.execution_policy.release_cache()`.

- [x] **Step 5: Run GREEN**

Run: `python -m tests.run tests/test_torch_policy.py -- -q --no-cov`
Expected: PASS.

- [x] **Step 6: Run focused CUDA worker suites**

Run: `python -m tests.run tests/test_torch_policy.py tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py tests/test_cuda_worker_controlnet.py tests/test_hunyuandit_worker.py tests/test_sdxl_worker.py tests/test_progress_bar_suppression.py tests/test_worker_controlnet_metadata.py tests/test_cuda_packaging_contract.py tests/test_worker_factory.py tests/test_backend_runtimes.py -- -q --no-cov`
Expected: all pass. Baseline at `e355ab9`: 41 passed + 8 skipped, 129 passed.
Result: real cohort 74 passed + 8 skipped. Stub cohort 225 passed, 1 failed: `test_model_lifecycle.py::TestEdgeCases::test_unload_triggers_gc_and_cache_clear`. The same test fails with the `e355ab9` worker file. It is order-dependent and predates this task.

- [x] **Step 7: Check drift.** Run `drift refs backends/cuda_worker.py`. Review bound prose that names `_parse_env`, `_setup_pipe_memory_opts`, or `empty_cache`. Do not relink older stale `cuda_worker.py` anchors without prose review. Run `drift check`.

- [x] **Step 8: Commit** with issue id and next step.
