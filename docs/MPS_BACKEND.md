# MPS Backend (Apple Silicon)

`BACKEND=mps` runs the worker pool on a Mac with Torch's Metal (MPS) device.
It binds one family: **`sd15` LCM text-to-image**. No other family loads on MPS,
and `sd15` accepts no img2img, no ControlNet, and no scheduler other than LCM.

Docker cannot reach Metal. Do not use [`TESTING_IN_DOCKER.md`](TESTING_IN_DOCKER.md)
for MPS. Run everything in this guide natively on the Mac.

Code:

- [`backends/platforms/mps.py`](../backends/platforms/mps.py) — `MPSProvider`
- [`backends/platforms/mps_policy.py`](../backends/platforms/mps_policy.py) — `MPS_TORCH_POLICY`
- [`backends/platforms/mps_bindings.py`](../backends/platforms/mps_bindings.py) — `MPS_FAMILY_BINDINGS` (`sd15`)
- [`backends/mps_worker.py`](../backends/mps_worker.py) — `DiffusersMpsWorker`
- [`scripts/mps_host_probe.py`](../scripts/mps_host_probe.py) — live host probe

For provider, factory, and family-binding flow, see
[`WORKER_SELECTION.md`](WORKER_SELECTION.md#mps-framework).

## Setup

1. Create the `stability-toys` conda env with `scripts/local-host.sh`. Choose
   backend `cpu` at the prompt. On macOS the script installs the PyPI Torch
   wheel, which includes MPS. The script offers no `mps` choice.
2. Activate it. Do not use the Miniforge base env.

   ```bash
   conda activate stability-toys
   ```

3. Select the backend explicitly. The server never infers MPS from hardware.

   ```bash
   export BACKEND=mps
   export MODE_CONFIG_PATH=conf
   python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200
   ```

   `server/lcm_sr_server.py` uses a relative import, so running it as a script
   raises `ImportError`. The command above is the container entrypoint, and it
   is the one the live acceptance below used.

4. Choose worker isolation. Both modes read the same binding table.

   | `WORKER_ISOLATION` | Worker factory | Parent memory consumer |
   |---|---|---|
   | `inproc` (default) | `MPSProvider.create_worker_factory()` | none |
   | `subprocess` | `backends.worker_factory.create_mps_worker` in the child | `MPS_TORCH_POLICY` |

The server starts without a model when the default mode has no MPS binding.
That is the current state for every family except `sd15`.

## Limits

- **One model family.** `MPS_FAMILY_BINDINGS` binds `sd15` to
  `backends.mps_worker.DiffusersMpsWorker`. `sdxl` and `hunyuandit` raise
  `UnsupportedFamilyError`: `family 'sdxl' has no MPS platform binding`.
- **Text-to-image only.** The `sd15` cell carries
  `ExecutionCapabilities(False, False, False)`, so `admit_generation_operation`
  rejects img2img, ControlNet, and the combined operation on both HTTP and
  WebSocket. The worker repeats both checks as defense in depth.
- **LCM scheduler only.** `lcm-general` permits `euler` in configuration. The
  worker raises `ValueError("MPS SD1.5 supports LCM scheduler only")` before the
  pipeline call. HTTP returns 500 and WebSocket returns a job error. There is no
  admission-time 400 for this case.
- **No `run_job_with_latents`.** Production generation calls `run_job` only.
- **Mode switch returns HTTP 422** for an unbound family, with that message
  as the detail.
- **float16 only.** The policy always resolves `device="mps"` and `torch.float16`.
- **The MPS policy reads the `CUDA_*` env names.** There are no `MPS_*` names.
  It raises on:
  - `CUDA_OFFLOAD` other than `none` (also a mode's `runtime_offload`)
  - `CUDA_QUANTIZE` other than `none`, so no fp8 quantization (also `runtime_quantize`)
  - `CUDA_ENABLE_XFORMERS` true (also `runtime_enable_xformers`)
  - `CUDA_DTYPE` other than `fp16` (case-insensitive)
  - an empty `CUDA_DTYPE`, `CUDA_OFFLOAD`, or `CUDA_QUANTIZE`. A set but
    empty value is rejected, not defaulted. Only an unset name takes the
    default (`fp16`, `none`, `none`)

  An empty `CUDA_ENABLE_XFORMERS` or `CUDA_ATTENTION_SLICING` reads as false.
  It ignores `CUDA_DEVICE`: the device is always `mps`. It honors
  `CUDA_ATTENTION_SLICING` and `runtime_attention_slicing`.
- **Same seed, different image.** The MPS generator does not match the CUDA
  generator. A seed reproduces on the same MPS host only.
- **`/status` reports 0 total memory.** MPS uses `PlaceholderModelRegistry`.
  Its totals read 0 and its `vram.device` reads `MPS placeholder`. Process
  memory accounting itself is real; see the probe's `memory` check.
- **Super-resolution is a placeholder.** `supports_superres` is false.
- **`supports_generation` reads true, and it reaches only
  `/api/models/status`.** The flag gates nothing. The real generation gate is
  the `sd15` cell. The flag turned true after the live render below passed the
  validity checks in both isolation modes.
  `tests/test_mps_registration.py::test_mps_reports_generation_after_live_acceptance`
  pins it, together with the three flags that stay false.
- **MLX is unchanged.** `BACKEND=mlx` is a separate backend.

## Host probe

`scripts/mps_host_probe.py` prints one JSON report (`schema: mps_host_probe/v1`):

| Check | Proves |
|---|---|
| `torch` | Torch version, `mps.is_built()`, `mps.is_available()` |
| `policy` | `MPS_TORCH_POLICY.resolve_settings` gives `mps` and `torch.float16` under the current env |
| `seeded_tensor` | `torch.Generator(device="mps")` with seed 7 gives a float16 `(4,)` tensor on `mps`; a second draw in the same process matches |
| `memory` | `memory_stats()` before, after a 4 MiB float32 allocation, and after `release_cache()` |
| `provider` | `BACKEND=mps` selects `MPSProvider`; the env and provider singleton are restored after |
| `unsupported_family` | `create_mps_worker` on `sdxl` raises the exact unsupported-family message |

The report also records `commit`, `dirty`, and the host (`system`, `release`,
`mac_ver`, `machine`, `python`).

Exit codes:

| Code | Status | Meaning |
|---|---|---|
| 0 | `pass` | Every check passed on a live MPS runtime |
| 1 | `failed` | At least one check failed. Outranks `unavailable` |
| 3 | `unavailable` | MPS is not built or not available. MPS checks are `skipped` |

An unavailable runtime is **never** a pass. A restricted sandbox hides Metal,
so `is_available()` is false there and the probe exits 3. Never record an
exit-3 report as live evidence.

### Repeatability procedure

Run the probe twice on the same host. The second run must reproduce the
first run's tensor values.

```bash
conda activate stability-toys
python scripts/mps_host_probe.py > /tmp/mps_probe_1.json; echo "exit=$?"
python scripts/mps_host_probe.py --expect /tmp/mps_probe_1.json > /tmp/mps_probe_2.json; echo "exit=$?"
```

Both exits must be 0. Run 2 must show `seeded_tensor.matches_expected: true`.
A mismatch fails the check and exits 1.

## Evidence

Keep the two kinds of evidence apart. Each carries its own commit.

### Contract tests (patched Torch, any host)

These tests patch `torch.backends.mps` and `torch.mps`. They prove the
contract, not live Metal behavior.

| Concern | Test file |
|---|---|
| Unavailable runtime errors | `tests/test_mps_policy.py::test_mps_policy_requires_runtime` |
| Rejected `CUDA_*` options, including empty values | `tests/test_mps_policy.py::test_mps_policy_rejects_cuda_env_option` |
| Unsupported-family errors, `sd15` binding, factory selection | `tests/test_mps_registration.py` |
| Worker load, LCM scheduler, PNG metadata, seeded MPS generator, cancellation, cache release | `tests/test_mps_worker.py` |
| Operation gate on the MPS `sd15` cell | `tests/test_controlnet_execution.py` |
| HTTP `/generate` rejects ControlNet before preprocessing | `tests/test_controlnet_dispatch.py` |
| WebSocket rejects an init image and ControlNet after `job:ack` | `tests/test_ws_routes.py` |
| Progress-bar failure line names the backend | `tests/test_progress_bar_suppression.py` |
| Probe: unavailable is code 3 and never a pass, failed check is code 1, stable report schema, torch-free import | `tests/test_mps_host_probe.py` |

Run `tests/test_progress_bar_suppression.py` in its own `tests.run` call. It
imports `backends.cuda_worker` at module scope, and
`tests/test_mps_registration.py` asserts that module is absent from
`sys.modules`. One pytest process cannot hold both. The collision predates the
`sd15` binding.

Probe tests at commit `3a073d7`: RED before the probe existed (collection
error: `scripts/mps_host_probe.py` not found), then GREEN, 12 passed.
The empty-value cases in `test_mps_policy.py` (commit `a53bb88`) pin existing
behavior, so they passed on first run. No RED step applies.

Cohort runs on the Mac at commit `2b58c23` plus the one-line
`supports_generation` change of this commit (2026-10-22, Python 3.12.13,
torch 2.10.0). The flag change moved no cohort number. The earlier
framework-only run at `cb02402` gave 199 passed (real, 10 files) and
78 passed (stub, 2 files).

```bash
python -m tests.run \
  tests/test_mps_worker.py tests/test_mps_policy.py tests/test_mps_registration.py \
  tests/test_mps_host_probe.py \
  tests/test_controlnet_execution.py tests/test_controlnet_dispatch.py \
  tests/test_ws_routes.py tests/test_torch_policy.py \
  tests/test_worker_factory.py tests/test_backend_runtimes.py \
  tests/test_device_memory.py tests/test_worker_isolation.py \
  tests/test_subprocess_worker_handle.py tests/test_governor.py \
  tests/test_worker_pool.py tests/test_worker_handle.py \
  -- -q -p no:cacheprovider --no-cov

python -m tests.run tests/test_progress_bar_suppression.py \
  -- -q -p no:cacheprovider --no-cov
```

| MPS cohort | Files | Result |
|---|---|---|
| real | 14 | 289 passed |
| stub | 2 | 78 passed |
| progress bar (own call) | 1 | 10 passed |

```bash
python -m tests.run \
  tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py \
  tests/test_cuda_worker_controlnet.py tests/test_hunyuandit_worker.py \
  tests/test_sdxl_worker.py tests/test_cuda_packaging_contract.py \
  tests/test_platform_registry.py tests/test_superres_http.py \
  -- -q -p no:cacheprovider --no-cov -rs
```

| CUDA regression cohort | Files | Result |
|---|---|---|
| real | 4 | 46 passed, 8 skipped |
| stub | 4 | 124 passed |

All 8 skips are in `tests/test_sdxl_worker.py`, with the reason "CUDA not
functional - SDXL worker requires working GPU". This is contract-level CUDA
evidence only. There is no live CUDA-host run for this commit.
`scripts/enigma-dev-verify.sh` runs that check on a CUDA host. The CUDA cohort
result is unchanged from `cb02402`, which is the point: the `sd15` MPS binding
and the backend-aware progress-bar label leave CUDA behavior alone.

Failures that predate this work, outside these cohorts:

- `tests/test_model_lifecycle.py::TestEdgeCases::test_unload_triggers_gc_and_cache_clear` (order-dependent)
- `tests/test_entry_spans.py::test_an_UNHASHABLE_type_does_not_break_the_span` (failing since `491a91c`)

### Live host checks (Mac only)

#### `sd15` binding, probe run at `2b58c23` (2026-10-22)

Two-probe repeatability sequence, clean tree, outside a restricted sandbox.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64, Python 3.12.13
- Torch: 2.10.0, `mps_built: true`, `mps_available: true`
- Run 1 `python scripts/mps_host_probe.py` exit 0; run 2 `--expect <run 1>` exit 0
- `seeded_tensor.matches_expected: true`, values
  `[0.673828125, 0.6142578125, 0.052734375, 0.568359375]` — unchanged from `cb02402`
- `unsupported_family`: `family 'sdxl' has no MPS platform binding`, status `pass`.
  This is the moved negative check. `sd15` is bound and no longer carries it.

#### Live generation: PASS, both isolation modes (2026-10-22)

Code at `2b58c23`, clean tree. The model root is a host path outside the repo.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64, Python 3.12.13, Torch 2.10.0
- Model: `<model root>/diffusers/lcm-dreamshaper-v7`,
  `_class_name: LatentConsistencyModelPipeline`
- Mode `lcm-general`, Compel service, prompt
  "a bright red sailboat on a blue lake beneath a green mountain", seed 7
- Server: `python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200`
  with `BACKEND=mps`, `MODE_CONFIG_PATH=<temporary conf copy>`,
  `CONTROLNET_REGISTRY_VALIDATION=off`
- `DEFAULT_SIZE`, `DEFAULT_STEPS`, and `DEFAULT_GUIDANCE` unset in both shells

| Isolation | HTTP | Wall time | Bytes | Result |
|---|---|---|---|---|
| `inproc` | 200 | 17.2 s | 955786 | pass |
| `subprocess` | 200 | 10.3 s | 955786 | pass |

Both PNGs are byte-identical, `sha256` prefix `e7907d8e79439ca2`. Seed 7 gives
the same image across the spawn boundary, so the generator is seeded inside the
worker and not in the parent.

Validity, per image: format PNG, size `768x768`, 3561 distinct RGB colors in a
64x64 resample (limit 32), grayscale standard deviation 38.43 (limit 5). The
`lcm` text chunk reads seed 7, steps 8, cfg 3.0, size `768x768`,
`scheduler_id: lcm`. Visual check: a red sailboat on a blue lake below a gray
mountain. Neither image is black, white, or corrupt.

`/api/models/status` during both runs, before the flag change: `backend: mps`,
`current_mode: lcm-general`, `is_loaded: true`, `models_loaded: 1`,
`device: MPS placeholder`, no numeric device total, and
`supports_generation: false` — the deliberate understatement this slice removes.

After the flag change, the same check on a reloaded server reads
`supports_generation: true` with every other field unchanged.

Two deviations from Step 5 of the plan, both forced by the host:

1. The plan starts the server with `python server/lcm_sr_server.py`. That file
   uses a relative import at line 111, so script mode raises `ImportError`.
   The container entrypoint is `uvicorn server.lcm_sr_server:app`, which is
   what ran.
2. The plan sets `lora_root` to `<model root>/loras`. That directory does not
   exist on this host; the LoRA files are in `<model root>/diffusers/loras`,
   and `ModeConfigManager` reports a missing `lora_root` as a path error. The
   temporary config points at the directory that holds the files. The MPS
   worker loads no LoRA, so this changes nothing in the render.

#### Framework-only probe at `cb02402`

The probe ran on a clean checkout of commit
`cb0240287723c693ea421cc03fe417728d47dd93` (`dirty: false`). The shell
reached Metal (`mps_available: true`). The docs commit that adds this guide
changes no code. An earlier clean run at `3a073d7`, before Black formatting,
gave the same values and bytes.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64, Python 3.12.13
- Torch: 2.10.0, `mps_built: true`, `mps_available: true`
- Date: 2026-10-22
- Run 1: `python scripts/mps_host_probe.py`, exit 0
- Run 2: `python scripts/mps_host_probe.py --expect <run 1 report>`, exit 0, `matches_expected: true`

This run names `sd15` as the unsupported family, because `cb02402` bound no
family at all. `STABL-hvvogszy` bound `sd15` and moved the negative check to
`sdxl`. Both results are correct for their own commit. Do not edit this record
to match current behavior.

The two reports are identical except for `matches_expected`. Run 2 output:

```json
{
  "checks": {
    "memory": {
      "after_alloc": {"current_allocated": 4194304, "driver_allocated": 42352640},
      "after_release": {"current_allocated": 0, "driver_allocated": 409600},
      "alloc_bytes": 4194304,
      "before": {"current_allocated": 0, "driver_allocated": 8798208},
      "error": null,
      "status": "pass"
    },
    "policy": {
      "attention_slicing": false,
      "device": "mps",
      "dtype": "torch.float16",
      "error": null,
      "status": "pass"
    },
    "provider": {"backend": "mps", "error": null, "provider": "MPSProvider", "status": "pass"},
    "seeded_tensor": {
      "device": "mps",
      "dtype": "torch.float16",
      "error": null,
      "matches_expected": true,
      "repeatable_in_process": true,
      "seed": 7,
      "shape": [4],
      "status": "pass",
      "values": [0.673828125, 0.6142578125, 0.052734375, 0.568359375]
    },
    "unsupported_family": {
      "error": null,
      "family": "sd15",
      "message": "family 'sd15' has no MPS platform binding",
      "status": "pass"
    }
  },
  "commit": "cb0240287723c693ea421cc03fe417728d47dd93",
  "dirty": false,
  "exit_code": 0,
  "host": {"mac_ver": "26.0.1", "machine": "arm64", "python": "3.12.13", "release": "25.0.0", "system": "Darwin"},
  "schema": "mps_host_probe/v1",
  "status": "pass",
  "torch": {"mps_available": true, "mps_built": true, "version": "2.10.0"}
}
```

Earlier live numbers came from one-off inline commands, not from this probe.
Each belongs to its own commit:

- `337233b` and `6979b16` (STABL-lhpndzru): seed 7 with shape `(1,)` gave
  the tensor value `[0.673828125]`. The probe's first value matches.
- `491a91c` (STABL-qpumvysu): 4,194,304 allocated bytes and 42,352,640
  driver bytes. The probe's `after_alloc` matches.
