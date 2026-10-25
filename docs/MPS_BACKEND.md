# MPS Backend (Apple Silicon)

`BACKEND=mps` runs the worker pool on a Mac with Torch's Metal (MPS) device.
It binds two families:

- **`sd15` LCM**, for text-to-image, img2img, one Canny ControlNet attachment,
  and the combined img2img + ControlNet operation with that one attachment.
  `sd15` accepts no Depth or Pose ControlNet, no second attachment, and no
  scheduler other than LCM.
- **`hunyuandit`**, for text-to-image, img2img, and one Canny, Depth or Pose
  ControlNet attachment. The combined img2img + ControlNet operation stays
  unbound on MPS. `hunyuandit` accepts no second attachment and no timing
  window.

No other family loads on MPS. `sdxl` is unbound.

Docker cannot reach Metal. Do not use [`TESTING_IN_DOCKER.md`](TESTING_IN_DOCKER.md)
for MPS. Run everything in this guide natively on the Mac.

Code:

- [`backends/platforms/mps.py`](../backends/platforms/mps.py) — `MPSProvider`
- [`backends/platforms/mps_policy.py`](../backends/platforms/mps_policy.py) — `MPS_TORCH_POLICY`
- [`backends/platforms/mps_bindings.py`](../backends/platforms/mps_bindings.py) — `MPS_FAMILY_BINDINGS` (`sd15`, `hunyuandit`)
- [`backends/mps_worker.py`](../backends/mps_worker.py) — `DiffusersMpsWorker`
- [`backends/hunyuan_mps_worker.py`](../backends/hunyuan_mps_worker.py) — `DiffusersHunyuanDiTMpsWorker`
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
`lcm-general` is the default mode, and it binds `sd15`.

## Limits

- **Two model families.** `MPS_FAMILY_BINDINGS` binds `sd15` to
  `backends.mps_worker.DiffusersMpsWorker` and `hunyuandit` to
  `backends.hunyuan_mps_worker.DiffusersHunyuanDiTMpsWorker`. `sdxl` raises
  `UnsupportedFamilyError`: `family 'sdxl' has no MPS platform binding`.
- **HunyuanDiT is text-to-image, img2img, and plain Canny, Depth or Pose
  ControlNet.** The `hunyuandit` cell carries
  `ExecutionCapabilities(True, True, False)`. Admission rejects the combined
  img2img + ControlNet operation for the family before preprocessing. The
  worker repeats every operation and scope check for a direct `run_job` call.
  The worker subclasses the CUDA HunyuanDiT worker and
  swaps only the execution policy, so the load, conditioning, scheduler, and
  both job paths are shared code. img2img runs the repo-owned
  `backends/hunyuandit_img2img.py` loop, because Diffusers has no HunyuanDiT
  img2img pipeline. The PNG `lcm` chunk records `denoise_strength`.
- **HunyuanDiT ControlNet is one Canny, Depth or Pose attachment,
  text-to-image only.** The cell carries
  `ControlNetScope(allowed_control_types=("canny", "depth", "pose"),
  max_attachments=1)`. Admission enforces the scope before any preprocessing,
  and the worker repeats it. The live messages are `backend 'mps' admits at most
  1 controlnet attachment(s) for family 'hunyuandit', got 2` and `family
  'hunyuandit' does not support the img2img+controlnet operation`. Plain
  text-to-image ControlNet runs the shared
  `HunyuanDiTControlNetPipeline` branch of the CUDA worker: it composes the
  pipeline with `from_pipe(..., torch_dtype=None)` and passes `control_image`
  and `controlnet_conditioning_scale`. It does not use the img2img loop.
- **HunyuanDiT ControlNet accepts no timing window.**
  `HunyuanDiTControlNetPipeline` has no `control_guidance_start` or
  `control_guidance_end` argument. The shared kwarg builder drops both keys for
  this family, so a nondefault window would be silently ignored. The MPS worker
  refuses a nondefault window with `MPS HunyuanDiT does not support control
  timing windows`, before any ControlNet load. The refusal is worker-side
  because admission reads the mode policy only, and a platform test there would
  change CUDA behavior (`STABL-xvnjfuae`).
- **The worker-side gate is shared with `sd15`.** Both MPS workers call
  `backends/platforms/operation_guard.py::check_platform_operation`. It reads
  the same platform-family cell that admission reads, and it repeats the
  operation capability and the ControlNet scope.
- **Text-to-image, img2img, one Canny ControlNet, and the combined operation.**
  The `sd15` cell carries `ExecutionCapabilities(True, True, True)`, so
  `admit_generation_operation` admits all four. The worker repeats each check
  against `cell.execution_capabilities`, so a cell that forbids an operation
  still refuses it after admission.
- **ControlNet is narrowed to one Canny attachment.** The `sd15` cell also
  carries `ControlNetScope(allowed_control_types=("canny",),
  max_attachments=1)`. `conf/modes.yml` gives `lcm-general` canny, depth, pose,
  and `max_attachments: 2`, so the narrowing cannot come from the mode policy.
  Admission enforces the scope before any preprocessing, and the worker repeats
  it. Depth and Pose are still registered in `conf/controlnets.yaml` for
  `sd15`; they are refused on this platform only. The live messages are
  `backend 'mps' does not support control_type 'depth' for family 'sd15';
  allowed: canny` and `backend 'mps' admits at most 1 controlnet
  attachment(s) for family 'sd15', got 2`. The scope is a platform field, so
  CUDA keeps every control type the mode permits (`STABL-xomvolgw`).
- **img2img goes over the WebSocket only.** `POST /generate` carries no init
  image, so its admission always reads `has_init_image=False`. An img2img job
  needs `init_image_ref` on a `job:submit` frame. HTTP admission with a
  ControlNet reaches the ControlNet branch. HTTP admission without one reaches
  text-to-image. HTTP admission never reaches the img2img branch. The combined
  operation carries the same limit, because it needs the init image too.
- **img2img shares the loaded weights.** The worker builds one
  `StableDiffusionImg2ImgPipeline` from `self.pipe.components` on the first
  img2img job and keeps it. The pipeline reads the size from the init image, so
  the worker resizes the init image to the requested size and sends no width or
  height. `denoise_strength` becomes the pipeline `strength`, default 0.75.
  The output PNG records that value in its `lcm` metadata for
  `st gen --recreate`. This mirrors the CUDA SD1.5 worker.
- **The combined operation builds its pipe per job.** It calls
  `StableDiffusionControlNetImg2ImgPipeline.from_pipe`, which shares every
  loaded component, so no weights are copied. The init image goes to `image`
  and the control map goes to `control_image`, keeping the two inputs separate.
  ControlNet strength becomes `controlnet_conditioning_scale`. A combined
  render costs more wall time than a plain img2img render because the ControlNet
  runs at every step; the live figures are below.
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
- **`/api/models/status` reported `supports_img2img` as false — FIXED on
  `main`.** Measured live on 2026-10-23 with the ControlNet
  acceptance below: `supports_img2img: true`, `supports_controlnet: true`,
  `supports_img2img_and_controlnet: false`. The route reports the cell. It does
  NOT report the platform ControlNet scope, so a client reading it learns that
  ControlNet is admitted but not that only Canny is. The route read the execution flags from `BackendCapabilities`,
  which does not carry them, so the default false reached every backend,
  including CUDA. Measured during the img2img live acceptance below, which
  still records that false. `STABL-xuzguqyx` fixed it in `e89856e`: the route
  now reads the active snapshot's family cell, the same source admission reads.
  The response also gained `supports_controlnet`. Not an MPS defect.
  **At the combined acceptance on 2026-10-24 the flag reads true**, because the
  `sd15` cell now admits the combined operation. Measured in both isolation
  modes.
- **Super-resolution is a placeholder.** `supports_superres` is false.
- **`supports_generation` reads true, and it reaches only
  `/api/models/status`.** The flag gates nothing. The real generation gate is
  the `sd15` cell. The flag turned true after the live render below passed the
  validity checks in both isolation modes.
  `tests/test_mps_registration.py::test_mps_reports_generation_after_live_acceptance`
  pins it. `test_mps_provider_binds_sd15_and_shares_runtime` pins the cell:
  text-to-image true, img2img true, ControlNet true, combined operation true.
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
| img2img pipeline reuse, init-image resize, denoise strength, VAE alignment, combined pipeline, init and control images kept separate | `tests/test_mps_worker.py` |
| HunyuanDiT load on MPS, text-to-image PNG metadata, seeded MPS generator, step-progress wiring, img2img through the repo-owned loop with strength in metadata, one Canny, one Depth and one Pose binding through the shared ControlNet pipeline, refusal of the combined operation, a second attachment, and a timing window | `tests/test_hunyuan_mps_worker.py` |
| `hunyuandit` cell: worker ref, text-to-image, img2img, and ControlNet capabilities, one Canny, Depth or Pose scope, `sdxl` still unbound | `tests/test_mps_registration.py` |
| Operation gate on the MPS `sd15` and `hunyuandit` cells, through the real provider | `tests/test_controlnet_execution.py` |
| HTTP `/generate` rejects a Depth ControlNet on the MPS cell before preprocessing, because the platform scope allows Canny only | `tests/test_controlnet_dispatch.py::test_http_generate_rejects_a_depth_controlnet_on_mps_sd15_cell` |
| WebSocket admits an init image, a Canny ControlNet, and the combined operation; refuses Depth and a second attachment on the MPS cell | `tests/test_ws_routes.py` |
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

Cohort runs on the Mac at commit `39d4c6c` (img2img, 2026-10-23, Python
3.12.13, torch 2.10.0). The real cohort reads 297. The img2img slice added
10 tests and removed the 3 that pinned the img2img rejection, for a net 7. The
other 1 of the 8 came from `07692a2`, after this guide last recorded 289.

Earlier runs, each for its own commit. Do not read them as current:
`2b58c23` plus the one-line `supports_generation` change (2026-10-22) gave
289 passed (real) and 78 passed (stub). The framework-only run at `cb02402`
gave 199 passed (real, 10 files) and 78 passed (stub, 2 files).

```bash
python -m tests.run \
  tests/test_mps_worker.py tests/test_mps_policy.py tests/test_mps_registration.py \
  tests/test_mps_host_probe.py tests/test_hunyuan_mps_worker.py \
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
| real | 14 | 297 passed (img2img slice, `39d4c6c`) |
| stub | 2 | 78 passed |
| progress bar (own call) | 1 | 10 passed |

The combined slice ran the same commands at `8fb8dbb` (2026-10-24, same host and
interpreter): real **330 passed**, stub **78 passed**, progress bar **10
passed**, with no failure and no skip. The 33 new real tests cover combined
admission, the combined worker path, and the MPS WebSocket cases.

The HunyuanDiT text-to-image slice ran them at `135e86c` (2026-10-25, same
host and interpreter, plus `tests/test_hunyuan_mps_worker.py`): real **336
passed** (15 files), stub **78 passed**, progress bar **10 passed**, no failure
and no skip. The 6 new real tests cover the MPS load, text-to-image metadata,
the seeded MPS generator, step-progress wiring, and both refusals.

The HunyuanDiT img2img slice re-ran them at `1a99ae6` with the same counts:
real **336 passed**, stub **78 passed**, progress bar **10 passed**. One test
changed rather than grew: the img2img refusal became an img2img render test.

The HunyuanDiT Canny ControlNet slice ran them on 2026-10-25 (same host and
interpreter): real **345 passed** (15 files), stub **78 passed**, progress bar
**10 passed**, no failure and no skip. The net change is +9 real tests. One
ControlNet refusal test became five (one render through the shared ControlNet
pipeline, and four refusals), and five admission tests read the live
`hunyuandit` cell through the real `MPSProvider`.

The HunyuanDiT Depth ControlNet slice ran them on 2026-10-25 (same host and
interpreter): real **347 passed** (15 files), stub **78 passed**, progress bar
**10 passed**, no failure and no skip. The net change is +2 real tests. The
worker's Depth rejection became a Depth render through the shared ControlNet
pipeline, and a Pose refusal was added to the worker file and to the admission
file.

The HunyuanDiT Pose ControlNet slice ran them on 2026-10-25 (same host and
interpreter): real **349 passed** (15 files), stub **78 passed**, progress bar
**10 passed**, no failure and no skip. The net change is +2 real tests. The
worker's Pose rejection became a Pose render through the shared ControlNet
pipeline. The admission file gained a Pose admission test, a combined-operation
Pose refusal, and a control type outside the scope, `lineart`. The first full
run before that change showed 1 failure, the stale Pose refusal in
`tests/test_hunyuan_mps_worker.py`, at 348 passed.

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
| stub | 4 | 128 passed |

The HunyuanDiT text-to-image slice re-ran this cohort at `135e86c` with the
same real result: 46 passed, 8 skipped, and 128 stub tests. The stub count was
124 at `8fb8dbb`; the difference came from commits between the two records. The
img2img slice re-ran it at `1a99ae6` with identical counts: 46 passed, 8
skipped, 128 stub. No CUDA file changed. The Canny ControlNet slice re-ran it on
2026-10-25 with the same counts: 46 passed, 8 skipped, 128 stub. The
`_check_platform_operation` extraction changed `backends/mps_worker.py` only,
and its behavior is identical. The Depth ControlNet slice re-ran it on
2026-10-25 with the same counts again: 46 passed, 8 skipped, 128 stub. The
slice changed `backends/platforms/mps_bindings.py`,
`backends/hunyuan_mps_worker.py` (docstrings only), and three test files. It
touched no CUDA file. The Pose ControlNet slice re-ran it on 2026-10-25 with the
same counts again: 46 passed, 8 skipped, 128 stub. It changed
`backends/platforms/mps_bindings.py` (one scope tuple),
`backends/hunyuan_mps_worker.py` (docstrings only), and three test files. It
touched no CUDA file.

The CUDA cohort is unchanged from `2b58c23`, which is the point: the img2img
cell and the MPS worker leave CUDA behavior alone. The combined slice re-ran it
at `8fb8dbb` on 2026-10-24 with the same counts: real 46 passed and 8 skipped,
stub 124 passed.

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

#### Live HunyuanDiT Pose ControlNet: PASS (2026-10-25)

Code at the `codex/stabl-ziorpomq` branch head. Client:
`spikes/mps_hunyuan_pose_acceptance.py`. It starts no server. Ten signals, all
required. The scope signal carries two refusals.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64 (M1 Max, 64 GiB), Python 3.12.13,
  Torch 2.10.0
- Base model: `/Users/darkbit1001/models/diffusers/HunyuanDiT-v1.1-Diffusers`,
  fp16
- ControlNet: `hunyuandit-pose` at
  `/Users/darkbit1001/models/controlnets/HunyuanDiT-v1.1-ControlNet-Pose`
- Registry: `CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml`, the same
  host prefix swap of `conf/controlnets.yaml`. Strict validation ran and
  accepted every entry.
- Conf: the same trimmed copy, holding `lcm-general` and `HunyuanDiT` only, with
  host model paths
- Server: `python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200`
  with `BACKEND=mps`, `STORAGE_PROVIDER=MEMORY`, `WORKER_ISOLATION=inproc`
- Mode `HunyuanDiT`, prompt "a person standing in a sunlit room, photo, full
  body", negative prompt "blurry, watermark, text, low quality", 1024x1024,
  6 steps, guidance 6.0, seed 4242
- Source image: one generated 1024x1024 PNG, `30b22f8f...`, uploaded with
  `type=image` into the `ref_image` bucket. It is a HunyuanDiT render. It holds
  two figures, a large head at the top and a full standing body below.
- Preprocess: the repository Pose preprocessor, `preprocessor_id: pose`,
  `{"max_res": 1024}`, through `source_asset_ref`. It runs on CPU, with the
  `openpose` model and `parts: ["body", "face"]`.
- Artifacts: `/tmp/mps_hunyuan_pose_*.png` holds the four renders, the source
  image, its mirror, and the local map re-run

| Render | Wall time | Bytes | `sha256` prefix | Colors | Grey stddev | Progress frames |
|---|---|---|---|---|---|---|
| text-to-image | 60.13 s | 689737 | `09f2510c1449bb30` | 3332 | 68.31 | 6 |
| preprocessed Pose | 98.20 s | 841156 | `926f20b5d451179f` | 3575 | 67.02 | 6 |
| mirrored source | 97.95 s | 811142 | `dd850637973566be` | 3572 | 63.98 | 6 |
| strength 0.0 | 97.02 s | 689933 | `f393ead26d8d6516` | 3332 | 68.31 | 6 |

This record uses its own prompt. Its text-to-image bytes therefore do not
compare to the Canny and Depth records, which read `746a55f3...` under the
apple prompt. No byte claim crosses these records.

| Check | Result |
|---|---|
| admitted | all four `job:complete`, backend `mps`, seed 4242 |
| preprocess | artifact `preprocessor_id: pose`, `control_type: pose`, source ref `d4c4d41e...`, new map ref `af3ce22e...`, `metadata.device: cpu` |
| not text-to-image | 38.68, mean per-channel at 128x128 |
| map content steers | 16.64 between the source and its mirror at one strength and one seed |
| strength reaches the pipeline | 38.68 between strength 1.0 and 0.0 |
| valid renders | each 1024x1024 PNG, more than 32 distinct colors, grey stddev above 5 |
| `controlnet` chunk | control type `pose`, generation `model_id hunyuandit-pose`, strength 1.0, window 0.0 to 1.0 |
| map is a pose map | 0.9802 black fraction, 17992 colored pixels, not grayscale |
| scope refuses two | `job:error`, `backend 'mps' admits at most 1 controlnet attachment(s) for family 'hunyuandit', got 2` |
| scope refuses img2img plus Pose | `job:error`, `family 'hunyuandit' does not support the img2img+controlnet operation` |

Cross-check: `distances.zero_vs_txt2img` is 0.0. A strength of 0.0 therefore
removes the ControlNet contribution entirely. The two files differ in size,
689933 against 689737 bytes. Their mean per-channel distance at 128x128 is 0.0.

The map recorded here is a local re-run, because the delivered map is not
retrievable. It read 10340 bytes, `sha256 e39a52d0...`, 1024x1024, 0.9802 black
fraction, 17992 colored pixels, and not grayscale. Two runs on the same source
gave the same bytes, in 5.31 s and 5.38 s. It draws one body skeleton with
colored limbs and a face at the head. It also draws a second face at mid-torso,
because the source holds two face-like regions.

Observed conditioning, against the source, the map, and the four renders:

- The preprocessed render holds one standing figure. The head sits near the top
  of the frame. The arms hang at the sides, with the hands at hip level. The
  legs reach the bottom edge, and the feet stand on the floor. That layout
  follows the skeleton.
- The map's second face sits at the abdomen, and the render carries a dark oval
  mass at that spot. Map content beyond the first face reaches the pipeline, and
  the preprocessor does not reduce the map to one person.
- The mirrored-source render shows a different body, and it holds a separate
  head floating high on the wall. Same prompt, same strength, same seed. That
  is the content signal.
- The strength-0.0 render reproduces the text-to-image scene.
- These renders do not reproduce the source's person. This record claims no
  pose fidelity and no image-quality parity. It claims that one Pose attachment
  reaches the MPS HunyuanDiT ControlNet pipeline, that the server Pose
  preprocessor builds the map on CPU from the uploaded source, that the map's
  content steers the render, and that the platform scope refuses what it must.

Limitations, stated rather than glossed:

- The server-side control-map bytes are not retrievable. The asset store has no
  HTTP surface. `/storage/af3ce22e...` returned 404. The delivered map was
  therefore re-run locally. The local bytes are not a copy of the server
  artifact.
- The source image is itself a HunyuanDiT render with two figures. The map
  draws two faces. This record infers that the second face follows source
  content. It ran no control: it recorded no map for the mirrored source. A
  mirrored source renders differently, so the second face also moved. Part of
  the 16.64 distance can come from that face and not from the body pose.
- Two checks in the client are weaker than their names suggest. `admitted`
  tests `sha256 is not None`, and `describe()` always fills that field, so the
  check passes whenever the four renders complete. The record row above
  therefore rests on the terminal frames read during the run, not on this
  check. `map_is_a_pose_map` reads the operator-supplied `--map`, because the
  delivered map is not retrievable. It proves what the preprocessor builds on
  that source, not what the server wrote.
- The Pose map came from `controlnet-aux==0.0.10` and its OpenPose annotator
  weights. The record does not pin the annotator weight files. A different
  weight set draws a different skeleton from the same source.
- This record does not reproduce `STABL-tqpfnwoj`. All four jobs streamed 6
  progress frames. The first render ran after an earlier render had already
  loaded `HunyuanDiT` in the same server, so no job performed the mode switch.
- The ControlNet path is not bit-stable on MPS. This record makes no
  bit-stability claim. The Canny record measures the same limit at 0.92.
- `/api/models/status` reported `controlnet_scope` as
  `["canny", "depth", "pose"]` with `max_attachments: 1`, and
  `supports_img2img_and_controlnet: false`. The route reports the scope of the
  active family only.
- Only `inproc` isolation ran. The acceptance does not separate the two
  isolation modes.

```bash
conda activate stability-toys
python spikes/mps_hunyuan_pose_acceptance.py \
  --source /tmp/mps_hunyuan_pose_source.png \
  --map /tmp/mps_hunyuan_pose_map_local.png \
  --out /tmp/mps_hunyuan_pose
```

#### Live HunyuanDiT Canny ControlNet: PASS (2026-10-25)

Code at the `STABL-xvnjfuae` branch head. Client:
`spikes/mps_hunyuan_canny_acceptance.py`. It starts no server. Eight signals,
all required, plus two cross-checks. The scope signal carries two refusals.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64 (M1 Max, 64 GiB), Python 3.12.13,
  Torch 2.10.0
- Base model: `/Users/darkbit1001/models/diffusers/HunyuanDiT-v1.1-Diffusers`,
  fp16
- ControlNet: `hunyuandit-canny` at
  `/Users/darkbit1001/models/controlnets/HunyuanDiT-v1.1-ControlNet-Canny`
- Registry: `CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml`, a prefix swap
  of `conf/controlnets.yaml` from `/models/controlnets` to the host path. Strict
  validation ran and accepted all nine entries.
- Conf: a trimmed copy holding `lcm-general` and `HunyuanDiT` only, with host
  model paths. Server startup resolves every mode model path, so an untrimmed
  copy fails on a missing host checkpoint.
- Server: `python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200`
  with `BACKEND=mps`, `STORAGE_PROVIDER=MEMORY`, `WORKER_ISOLATION=inproc`
- Mode `HunyuanDiT`, prompt "a red apple on a wooden table, soft light, photo",
  negative prompt "blurry, watermark, text, low quality", 1024x1024, 6 steps,
  guidance 6.0, seed 4242
- Source images: two generated 1024x1024 PNGs, uploaded with `type=image`. The
  first holds a green field, a blue band, a red ellipse, and a yellow triangle.
  The second holds an 8x8 dark grid on a light field.
- Preprocess: the repository Canny preprocessor, `low_threshold 100`,
  `high_threshold 200`, through `source_asset_ref` and a `preprocess` block
- Artifacts: `/tmp/mps_hunyuan_canny/` holds the five renders, both source
  images, both local maps, the stdout report (`report.log`), and the
  determinism probe (`determinism-probe.json`)

| Render | Wall time | Bytes | `sha256` prefix | Colors | Grey stddev | Progress frames |
|---|---|---|---|---|---|---|
| text-to-image | 61.09 s | 511783 | `746a55f314b355f2` | 2304 | 50.77 | 6 |
| preprocessed Canny | 93.97 s | 660756 | `9c9ea768b3a5904f` | 2994 | 54.77 | 6 |
| second source | 93.70 s | 916332 | `e2b9d311388881e1` | 2956 | 27.40 | 6 |
| strength 0.0 | 92.46 s | 511981 | `42000f03a80ee91a` | 2304 | 50.77 | 6 |
| explicit map | 91.62 s | 661987 | `922101345d9a3492` | 2983 | 54.57 | 6 |

The text-to-image render is byte-identical to the text-to-image acceptance
record (`746a55f3...`), so the Canny slice moved no text-to-image byte. The
strength 0.0 render is byte-different only in its PNG metadata: its pixels are
identical to the text-to-image render.

| Check | Result |
|---|---|
| admitted | all five `job:complete`, backend `mps`, seed 4242 |
| preprocess | artifact `preprocessor_id: canny`, `control_type: canny`, source ref recorded, new map ref different from the source ref |
| not text-to-image | 26.44, mean per-channel at 128x128 |
| map content steers | 32.17 between the two source images at one strength and one seed |
| strength reaches the pipeline | 26.44 between strength 1.0 and 0.0 |
| valid renders | each 1024x1024 PNG, more than 32 distinct colors, grey stddev above 5 |
| `controlnet` chunk | control type `canny`, generation `model_id hunyuandit-canny`, strength 1.0, window 0.0 to 1.0 |
| scope refuses Depth | `job:error`, `backend 'mps' does not support control_type 'depth' for family 'hunyuandit'; allowed: canny` |
| scope refuses two | `job:error`, `backend 'mps' admits at most 1 controlnet attachment(s) for family 'hunyuandit', got 2` |

The `scope refuses Depth` row holds for `58084f6`, the commit this record
measured. The Depth slice (`ef13200`) and the Pose slice (`3542bd6`) widened
the same scope, so that message no longer exists. `mps_hunyuan_canny_acceptance.py`
fails on that one signal at the current head. The other seven signals stay
valid, including the two-attachment refusal. Run that spike against `58084f6`
only. This record is not restated.

Cross-check: `distances.zero_vs_txt2img` is 0.0, so a strength of 0.0 removes
the ControlNet contribution entirely.

Cross-check, and its limit: the preprocessed render and the explicit-map render
differ by 1.67. That comparison does not prove the two maps match, because the
ControlNet path is not bit-stable on MPS. Two identical explicit-map jobs at the
same seed differ by 0.92 (90.04 s and 92.40 s). The text-to-image path is
bit-stable across runs; adding the ControlNet residual is what loses it. Map
identity rests on the artifact record and the map-content signal, not on this
distance.

Observed conditioning, against the source and the map:

- The preprocessed render shows an apple on a wooden table with a dark triangle
  at the right edge, in the position of the triangle in the Canny map. The
  circle, the band, and their edges are not reproduced. The map holds those
  edges weakly at these thresholds: the grayscale Canny at 100/200 finds the
  triangle outline and little else.
- The second source render shows the 8x8 grid as a clear dark lattice across the
  same scene, with the apple reduced and displaced. The map's content reaches
  the pipeline and shapes the output.
- Neither render shows the apple's structure following the map. This record
  claims no general image-quality parity. It claims that one Canny attachment
  reaches the MPS HunyuanDiT ControlNet pipeline, that the repository
  preprocessor builds the map from a source image, that the map's content steers
  the render, and that the platform scope refuses what it must.

Only `inproc` isolation was run. The acceptance does not separate the two
isolation modes.

```bash
conda activate stability-toys
python spikes/mps_hunyuan_canny_acceptance.py --out /tmp/mps_hunyuan_canny
```

#### Live HunyuanDiT Depth ControlNet: PASS (2026-10-25)

Code at the `codex/stabl-xvnjfuae` working tree, on top of `58084f6`. Client:
`spikes/mps_hunyuan_depth_acceptance.py`. It starts no server. Nine signals,
all required, plus one cross-check. The scope signal carries two refusals.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64 (M1 Max, 64 GiB), Python 3.12.13,
  Torch 2.10.0
- Base model: `/Users/darkbit1001/models/diffusers/HunyuanDiT-v1.1-Diffusers`,
  fp16
- ControlNet: `hunyuandit-depth` at
  `/Users/darkbit1001/models/controlnets/HunyuanDiT-v1.1-ControlNet-Depth`
- Registry: `CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml`, the same
  host prefix swap of `conf/controlnets.yaml`. Strict validation accepted all
  nine entries.
- Conf: the same trimmed copy, holding `lcm-general` and `HunyuanDiT` only, with
  host model paths
- Server: `python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200`
  with `BACKEND=mps`, `STORAGE_PROVIDER=MEMORY`, `WORKER_ISOLATION=inproc`
- Mode `HunyuanDiT`, the same prompt, negative prompt, size, steps, guidance,
  and seed as the Canny record
- Source images: two generated 1024x1024 PNGs, uploaded with `type=image`. The
  first holds a light wall, a brown table band, a red ellipse, and a blue
  rectangle. The second holds a dark top band, a light mid panel, and a centred
  bright ellipse.
- Preprocess: the repository Depth preprocessor, `preprocessor_id: depth`, no
  options, through `source_asset_ref`
- Artifacts: `/tmp/mps_hunyuan_depth/` holds the four renders, both source
  images, the local map re-run, and the stdout report (`report.log`)

| Render | Wall time | Bytes | `sha256` prefix | Colors | Grey stddev | Progress frames |
|---|---|---|---|---|---|---|
| text-to-image | 73.05 s | 511783 | `746a55f314b355f2` | 2304 | 50.77 | 0 |
| preprocessed Depth | 102.39 s | 579610 | `6588c5c5960f9872` | 2898 | 51.20 | 6 |
| second source | 97.64 s | 672039 | `c191097f6af40520` | 2281 | 50.78 | 6 |
| strength 0.0 | 94.00 s | 511981 | `df43b01924191e00` | 2304 | 50.77 | 6 |

The text-to-image render is byte-identical to the text-to-image acceptance
record (`746a55f3...`), so the Depth slice moved no text-to-image byte.

| Check | Result |
|---|---|
| admitted | all four `job:complete`, backend `mps`, seed 4242 |
| preprocess | artifact `preprocessor_id: depth`, `control_type: depth`, source ref `40a7a5a7...`, new map ref `206c11f7...` |
| not text-to-image | 40.1, mean per-channel at 128x128 |
| map content steers | 46.03 between the two source images at one strength and one seed |
| strength reaches the pipeline | 40.1 between strength 1.0 and 0.0 |
| valid renders | each 1024x1024 PNG, more than 32 distinct colors, grey stddev above 5 |
| `controlnet` chunk | control type `depth`, generation `model_id hunyuandit-depth`, strength 1.0, window 0.0 to 1.0 |
| scope refuses Pose | `job:error`, `backend 'mps' does not support control_type 'pose' for family 'hunyuandit'; allowed: canny, depth` |
| scope refuses two | `job:error`, `backend 'mps' admits at most 1 controlnet attachment(s) for family 'hunyuandit', got 2` |

Cross-check: `distances.zero_vs_txt2img` is 0.0. A strength of 0.0 therefore
removes the ControlNet contribution entirely.

The map-format prerequisite, settled on four axes before the live run:

- Dimensions. The preprocessor preserves the source size. The worker
  `_decode_control_image` resizes the map to the request size. No constraint
  survives.
- Channels. The checkpoint declares `conditioning_channels: 3`. The
  preprocessor emits three-channel RGB with R = G = B. `channels_equal` is true.
- Range. The preprocessor output and the checkpoint author's shipped sample map
  both span 0 to 255.
- Polarity. Brighter means nearer in both. Measured on a real photo, the
  preprocessor gives 169.7 in the foreground, 37.6 on the columns, and 2.6 on
  the sky. The shipped sample map reads 214.81 at the near band and 26.88 at
  the far band.

No official document states that polarity. The checkpoint card returns 401. The
checkpoint README passes a raw `control_image`. Tencent's `controlnet/README.md`
and `hydit/inference_controlnet.py` run no preprocessing. The polarity claim
rests on the two measurements above, not on documentation.

Observed conditioning, against the two sources and the four renders:

- The control map is a smooth 1024x1024 grey gradient. It is dark at the top
  and brighter toward the bottom. The ellipse and the rectangle read as lighter
  mid-grey masses.
- The preprocessed render shows a red apple on a wooden table. A dark mass
  fills the upper left. A blurred brown-orange block sits on the right at
  mid-height. The foreground-to-background split and the right-hand object both
  echo the source layout.
- The second-source render shows a flat orange panel across the middle. The
  apple sits centred on it, with a hard shadow to its left. Dark regions fill
  the top and the bottom. That band structure tracks the second source almost
  directly.
- Two different maps at one strength and one seed produce two structurally
  different scenes. That is the claim. The map's content steers the
  composition. A successful load and a valid PNG would not have shown this.

Limitations, stated rather than glossed:

- The server-side control-map bytes are not retrievable. The asset store has no
  HTTP surface. `/storage/{ref}` returns 404 for a bare 32-hex asset key. This
  client reported that 404. The delivered map was therefore re-run locally.
- The local re-run is not a byte copy of the server artifact. It reported a
  0 to 255 range, a 105.89 mean, 92273 bytes, and `sha256 b28c3164...` in
  4.82 s. The transformers pipeline logged `Device set to use mps:0`. MPS
  non-determinism may make the re-run differ from the server map.
- The text-to-image job streams 0 progress frames. The three ControlNet jobs
  stream 6 each. That loss is `STABL-tqpfnwoj`.
- The ControlNet path is not bit-stable on MPS. This record makes no
  bit-stability claim. The Canny record measures the same limit at 0.92.
- `/api/models/status` reports the scope of the active family only. Startup
  leaves `lcm-general` active, so its `controlnet_scope` reads `["canny"]`,
  from the `sd15` cell. The endpoint reports the `hunyuandit` scope only while
  a `hunyuandit` mode is active.
- Only `inproc` isolation ran. The acceptance does not separate the two
  isolation modes.
- That spike asserts the Pose refusal above, and it now fails on that signal.
  The Pose slice added `pose` to the MPS `hunyuandit` scope on 2026-10-25. The
  assertion belongs to `58084f6`, where this record was taken. Re-run the Depth
  spike only against that commit.

```bash
conda activate stability-toys
python spikes/mps_hunyuan_depth_acceptance.py --out /tmp/mps_hunyuan_depth
```

#### Live HunyuanDiT img2img: PASS (2026-10-25)

Code at `1a99ae6`. Client: `spikes/mps_hunyuan_img2img_acceptance.py`. It starts
no server. Four signals, all required.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64 (M1 Max, 64 GiB), Python 3.12.13,
  Torch 2.10.0
- Model: `/Users/darkbit1001/models/diffusers/HunyuanDiT-v1.1-Diffusers`, fp16
- Mode `HunyuanDiT`, same prompt, size, guidance, and seed as the text-to-image
  record, 6 steps, `WORKER_ISOLATION=inproc`
- Init image: one generated 1024x1024 PNG, green field, blue band, red ellipse,
  yellow triangle, uploaded with `type=image`
- Three jobs in one server: text-to-image, img2img at strength 0.35, img2img at
  strength 0.9

| Render | Steps run | Wall time | Bytes | `sha256` prefix | Colors | Grey stddev | Progress frames |
|---|---|---|---|---|---|---|---|
| text-to-image | 6 | 68.34 s | 511783 | `746a55f314b355f2` | 91115 | 50.77 | 0 |
| img2img 0.35 | 2 | 25.20 s | 701599 | `9957a28e40d8eea3` | 148108 | 18.70 | 2 |
| img2img 0.90 | 5 | 55.08 s | 904641 | `e1439840cc40c7c8` | 291673 | 35.01 | 5 |

The text-to-image render is byte-identical to the text-to-image acceptance
record above (`746a55f3...`), so the img2img slice moved no text-to-image byte.
Step counts are `int(steps * strength)`, read from the progress frames.

| Check | Result |
|---|---|
| admitted | all three `job:complete` |
| differs from text-to-image | 65.36 at 0.35 and 59.89 at 0.90, mean per-channel at 128x128 |
| strength reaches the loop | distance to the init image is 32.21 at 0.35 and 42.99 at 0.90 |
| metadata | `denoise_strength` 0.35 and 0.9, seed 4242 in both PNGs |

Progress frames are 0 on the text-to-image job because it performs the mode
switch. The two img2img jobs streamed 2 and 5 frames. That loss is
`STABL-tqpfnwoj`.

#### Live HunyuanDiT text-to-image: PASS (2026-10-25)

Code at `135e86c`. Client: `spikes/mps_hunyuan_t2i_acceptance.py`. It starts no
server.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64 (M1 Max, 64 GiB), Python 3.12.13,
  Torch 2.10.0
- Model: `/Users/darkbit1001/models/diffusers/HunyuanDiT-v1.1-Diffusers`, fp16
- Mode `HunyuanDiT`, prompt "a red apple on a wooden table, soft light, photo",
  negative prompt "blurry, watermark, text, low quality", 1024x1024, 6 steps,
  guidance 6.0, seed 4242
- Server: `python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200`
  with `BACKEND=mps`, `MODE_CONFIG_PATH=<temporary conf copy with host model
  paths>`, `STORAGE_PROVIDER=MEMORY`, `CONTROLNET_REGISTRY_VALIDATION=off`,
  `WORKER_ISOLATION=inproc`
- The first attempt asked for 4 steps and rendered 50.
  `server/generation_constraints.py` replaces a request whose step count equals
  the unset `DEFAULT_STEPS` default (4) with the mode default. Use a different
  count in an acceptance.

| Signal | Result |
|---|---|
| admitted | `job:complete` |
| PNG | 1024x1024, 511783 bytes, `sha256 746a55f314b355f2d03a90f2fcabc24fa4b5c094151df574bcb92a7acbe497d6` |
| not flat | 91115 distinct colors, greyscale standard deviation 50.77 |
| metadata | seed 4242, steps 6, cfg 6.0, negative prompt recorded, `scheduler_id` null |
| backend | `mps` |
| wall time | 76.17 s, including the demand mode switch and the model load |
| progress frames | 0 |

`progress_frames` is 0 because this job performed the mode switch. A second job
on the already-loaded mode streamed 3 of 3 frames. The loss is not specific to
HunyuanDiT or MPS. It is tracked as `STABL-tqpfnwoj`. The pipeline callback
itself works: a standalone two-step call on this host fired it for both steps,
and `tests/test_hunyuan_mps_worker.py` proves the worker wires it.

Only `inproc` isolation was run. The acceptance does not separate the two
isolation modes.

```bash
conda activate stability-toys
python spikes/mps_hunyuan_t2i_acceptance.py --out /tmp/mps_hunyuan
```

#### `sd15` binding, probe run at `2b58c23` (2026-10-22)

Two-probe repeatability sequence, clean tree, outside a restricted sandbox.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64, Python 3.12.13
- Torch: 2.10.0, `mps_built: true`, `mps_available: true`
- Run 1 `python scripts/mps_host_probe.py` exit 0; run 2 `--expect <run 1>` exit 0
- `seeded_tensor.matches_expected: true`, values
  `[0.673828125, 0.6142578125, 0.052734375, 0.568359375]` — unchanged from `cb02402`
- `unsupported_family`: `family 'sdxl' has no MPS platform binding`, status `pass`.
  This is the moved negative check. `sd15` is bound and no longer carries it.

#### Live ControlNet: PASS, both isolation modes (2026-10-23)

Code at `10871a2`, clean tree except one untracked JPEG. Spike:
`spikes/mps_controlnet_acceptance.py`, seven signals.

This record is the re-run of 2026-10-23, after a PR review strengthened two
signals: map content against map strength, and an exact scope message against
any failure. Every DETERMINISTIC figure the two runs share is identical - byte
counts, `sha256` prefixes, colour counts, standard deviations, and every mean
distance. Wall times differ, because they measure the machine rather than the
render: the first run reported 10.31 s, 14.50 s, and 13.20 s against this run's
10.17 s, 13.44 s, and 13.51 s for the same three jobs in `inproc`.

- Host: macOS 26.0.1, arm64, Python 3.12.13, Torch 2.10.0, diffusers 0.39.0,
  OpenCV 5.0.0
- ControlNet: `sd15-canny` at `<model root>/controlnets/sd15-canny`
- Registry: `CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml`, a prefix swap
  of `conf/controlnets.yaml` from `/models/controlnets` to the host path, and
  nothing else. `CONTROLNET_REGISTRY_VALIDATION` unset, so strict path
  validation ran and accepted all nine entries. `conf/controlnets.yaml` is
  unchanged; its container paths are rejected by the same strict check.
- Modes: a temporary conf copy holding `lcm-general` only. The startup
  reference check resolves every mode's `model_path`, and this host carries one
  model.
- Control map: Canny from `cv2.Canny(array, 100, 200)` over a deterministic
  source image, uploaded with `type=canny` and routed to the `control_map`
  bucket, 768x768. The attachment carries no `preprocess` block, so the render
  needs no preprocessing dependency.
- Prompt and seed as the slice 1 record, seed 7, 8 steps, cfg 3.0

Two control maps are used, both Canny, from two deterministic source images.
The second map exists because of a review finding: comparing ONE map at two
strengths proves that `strength` reaches the pipeline, not that the map's
CONTENT steers the render. A worker that passed a fixed or wrong image while
honouring `strength` would pass that comparison. The two maps sit at mean
distance 8.26 from each other.

| Render | Isolation | Wall time | Bytes | sha256 prefix | 64x64 colors | Grey stddev |
|---|---|---|---|---|---|---|
| text-to-image | `inproc` | 10.17 s | 955786 | `e7907d8e79439ca2` | 3561 | 38.43 |
| text-to-image | `subprocess` | 10.17 s | 955786 | `e7907d8e79439ca2` | 3561 | 38.43 |
| map A, strength 1.0 | `inproc` | 13.44 s | 651606 | `76a6052b60b2962f` | 2091 | 21.22 |
| map A, strength 1.0 | `subprocess` | 17.18 s | 651606 | `76a6052b60b2962f` | 2091 | 21.22 |
| map B, strength 1.0 | `inproc` | 12.60 s | 604027 | `ca43874398d927b3` | — | — |
| map B, strength 1.0 | `subprocess` | 13.57 s | 604027 | `ca43874398d927b3` | — | — |
| map A, strength 0.0 | `inproc` | 13.51 s | 955978 | `9b9c921bc594d38a` | 3561 | 38.43 |
| map A, strength 0.0 | `subprocess` | 13.74 s | 955978 | `9b9c921bc594d38a` | 3561 | 38.43 |

Every render is byte-identical across the two isolation modes. The
text-to-image bytes still match the slice 1 record, `sha256` prefix
`e7907d8e79439ca2`, so ControlNet changed nothing on the text-to-image path.

Mean per-channel distances, 0 to 255:

| Pair | Distance | What it proves |
|---|---|---|
| map A 1.0 against map B 1.0 | 26.78 | the map's CONTENT steers the render |
| map A 1.0 against text-to-image | 32.31 | the map reaches the pipeline |
| map B 1.0 against text-to-image | 39.05 | the same, for the second map |
| map A 1.0 against map A 0.0 | 32.31 | `strength` reaches the pipeline |
| map A 0.0 against text-to-image | 0.00 | at strength 0 the ControlNet is inert |

**Row one is the load-bearing one.** Same strength, same prompt, same seed, two
different maps, two different renders. Row five is the complement: at strength
0.0 the render returns to the text-to-image image, so nothing but the map
accounts for rows two and three. The strength-0.0 files are not byte-identical
to text-to-image, 955978 against 955786 bytes, because the ControlNet
pipeline's numerics differ slightly; the images are the same.

**The float16 question is settled by this run.** The ControlNet weights load at
`self.dtype`, which `MPS_TORCH_POLICY` fixes at float16, and the renders are
valid: 2091 distinct colors in a 64x64 resample against a limit of 32, and a
greyscale standard deviation of 21.22 against a limit of 5. A NaN pipeline
returns a black image and fails both checks. No float32 fallback is needed, and
none was added.

Platform scope, refused live in both modes:

| Request | Terminal | Message |
|---|---|---|
| one Depth attachment | `job:error` | `backend 'mps' does not support control_type 'depth' for family 'sd15'; allowed: canny` |
| two Canny attachments | `job:error` | `backend 'mps' admits at most 1 controlnet attachment(s) for family 'sd15', got 2` |

The mode permits both requests. `conf/modes.yml` gives `lcm-general` depth and
`max_attachments: 2`, so each refusal is the platform's.

The spike requires `job:error` carrying each message above, not merely a
failure. This too came from the review: the first version accepted any failure,
so a timeout, a dropped socket, or a mode-policy rejection would have counted
as a platform refusal.

The generation PNG carries the `controlnet` text chunk:
`[{"attachment_id": "cn_1", "control_type": "canny", "generation":
{"model_id": "sd15-canny", "strength": 1.0, "start_percent": 0.0,
"end_percent": 1.0}, "source": null}]`

Eight `job:progress` frames arrived for each ControlNet render, so step
progress injects on this path.

`/api/models/status` during both runs: `backend: mps`, `current_mode:
lcm-general`, `is_loaded: true`, `supports_generation: true`,
`supports_img2img: true`, `supports_controlnet: true`,
`supports_img2img_and_controlnet: false`.

#### Live img2img: PASS, both isolation modes (2026-10-23)

Code at `39d4c6c`. Only the two documentation files were uncommitted.
`spikes/mps_img2img_acceptance.py` ran the whole check. It is a client. It
starts no server.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64, Python 3.12.13, Torch 2.10.0
- Model: `/Users/darkbit1001/models/diffusers/lcm-dreamshaper-v7`
- Mode `lcm-general`, prompt
  "a bright red sailboat on a blue lake beneath a green mountain", seed 7,
  size 768x768, 8 steps, guidance 3.0
- Init image: one generated 768x768 PNG, 4919 bytes,
  `sha256 b6723b3d2774dc5b`. Green field, blue band, red circle, yellow
  triangle. Hard shapes, so a surviving structure is visible by eye.
- Server: `python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200`
  with `BACKEND=mps`, `MODE_CONFIG_PATH=<temporary conf copy>`,
  `STORAGE_PROVIDER=MEMORY`, `CONTROLNET_REGISTRY_VALIDATION=off`
- `DEFAULT_SIZE`, `DEFAULT_STEPS`, and `DEFAULT_GUIDANCE` unset in both shells
- Upload: `POST /v1/upload` with `type=image` put the init image in the
  `ref_image` bucket. The job went over the WebSocket, because `POST /generate`
  cannot carry an init image.

```bash
conda activate stability-toys
python spikes/mps_img2img_acceptance.py --out /tmp/mps_img2img_inproc
# restart the server with WORKER_ISOLATION=subprocess, then:
python spikes/mps_img2img_acceptance.py --out /tmp/mps_img2img_subprocess
```

| Run | Strength | Steps run | Bytes | `sha256` prefix | inproc | subprocess |
|---|---|---|---|---|---|---|
| txt2img | n/a | 8 | 955786 | `e7907d8e79439ca2` | 9.17 s | 9.89 s |
| img2img | 0.35 | 2 | 170285 | `0fb7e55e29b92317` | 4.49 s | 4.58 s |
| img2img | 0.90 | 7 | 724745 | `dea2a76281224a42` | 9.60 s | 9.85 s |

All three renders are byte-identical across the two isolation modes. The
text-to-image `sha256` prefix also matches the slice 1 record at `2b58c23`, so
the img2img change moved no text-to-image byte.

Four checks, all required. A valid PNG alone proves nothing, because
text-to-image returns one too:

| Check | Result |
|---|---|
| admitted | `job:complete`, never `job:error`. The cell admits img2img |
| differs from txt2img | `0fb7e55e` is not `e7907d8e` |
| valid | each render is a 768x768 PNG, over 32 distinct colors in a 64x64 resample, grayscale standard deviation over 5 |
| strength reaches the pipeline | mean distance to the init image is 3.49 at strength 0.35 and 43.03 at strength 0.90 |

Mean distance is the mean per-channel difference at 128x128, 0 to 255. The
distance from the 0.35 render to the text-to-image render is 64.70.

The step counts are the strength at work: the pipeline runs
`int(steps * strength)` steps, so 0.35 ran 2 of 8 and 0.90 ran 7 of 8. The
counts are the WebSocket progress frames, one per step.

Visual check of the 0.90 render: the init image's red circle is a red sun in
the same place, the blue band is a lake, the green field is sky and mountains,
and the yellow triangle is a sail. The structure survives. The init image
reached the pipeline.

#### Live combined img2img + ControlNet: PASS, both isolation modes (2026-10-24)

Code at `8fb8dbb`, plus this documentation commit. Spike:
`spikes/mps_combined_controlnet_acceptance.py`. It is a client. It starts no
server. Eight signals, all required. The combined-versus-plain-img2img distance
is reported and does not gate the result; the acceptance rests on the
within-pipeline comparisons.

- Host: Darwin 25.0.0, macOS 26.0.1, arm64, Python 3.12.13, Torch 2.10.0
- Model: `/Users/darkbit1001/models/diffusers/lcm-dreamshaper-v7`
- ControlNet: `sd15-canny` at `<model root>/controlnets/sd15-canny`
- Configs: the same two temporary files as the ControlNet acceptance above.
  Strict registry path validation ran, with `CONTROLNET_REGISTRY_VALIDATION`
  unset.
- Mode `lcm-general`, prompt
  "a bright red sailboat on a blue lake beneath a green mountain", seed 7,
  768x768, 8 steps, guidance 3.0, denoise strength 0.6 in every row
- Init image: the same 768x768 source as the ControlNet acceptance, uploaded
  with `type=image` into the `ref_image` bucket. Each Canny map was uploaded
  with `type=canny` into the `control_map` bucket.
- Two Canny maps come from two deterministic source images. They sit at mean
  distance 8.26 from each other.

| Render | Isolation | Wall time | Bytes | `sha256` prefix | 64x64 colors | Grey stddev | Progress frames |
|---|---|---|---|---|---|---|---|
| plain img2img | `inproc` | 6.55 s | 225907 | `fd7fa8d33e86db91` | 497 | 30.21 | 4 |
| plain img2img | `subprocess` | 6.67 s | 225907 | `fd7fa8d33e86db91` | 497 | 30.21 | 4 |
| map A, strength 1.0 | `inproc` | 15.47 s | 434546 | `f6f7df8955966d23` | 1688 | 32.70 | 4 |
| map A, strength 1.0 | `subprocess` | 15.36 s | 434546 | `f6f7df8955966d23` | 1688 | 32.70 | 4 |
| map B, strength 1.0 | `inproc` | 13.29 s | 392215 | `22c9163ad14af79b` | 1612 | 32.82 | 4 |
| map B, strength 1.0 | `subprocess` | 13.11 s | 392215 | `22c9163ad14af79b` | 1612 | 32.82 | 4 |
| map A, strength 0.0 | `inproc` | 13.71 s | 480803 | `d060d7bfe67053e9` | 1913 | 32.85 | 4 |
| map A, strength 0.0 | `subprocess` | 14.13 s | 480803 | `d060d7bfe67053e9` | 1913 | 32.85 | 4 |

Every render is byte-identical across the two isolation modes. An earlier run on
the same host produced the same four hashes in both modes; that session then
stopped on a usage limit. Repeatability therefore holds across sessions, not
only within one.

Mean per-channel distances, 0 to 255:

| Pair | Distance | What it proves |
|---|---|---|
| map A 1.0 against map B 1.0 | 4.77 | the map's CONTENT steers the render |
| map A 1.0 against map A 0.0 | 3.81 | `strength` reaches the pipeline |
| map A 1.0 against plain img2img | 10.86 | reported only, not a signal |
| map A 0.0 against plain img2img | 13.57 | reported only; the ControlNet UNet branch is not bit-identical |
| map A 0.0 against map B 0.0 | 0.00 | at strength 0 the map has no effect, byte for byte |

**Control is exactly inert at strength 0.0.** Map A and map B give the same
`sha256` `d060d7bfe67053e9` at that strength. A separate diagnostic sent one
map twice and confirmed the run was deterministic.

**The last two rows need care.** At strength 0.0 the combined render is not the
plain img2img render. A diagnostic measured the reason at `8fb8dbb`:

- Both pipelines resolve the same four timesteps, 499, 379, 259, and 139.
- Both start from the same init latents.
- The combined run passes exactly-zero ControlNet residuals, with maximum
  absolute value 0.0 in every step.
- The UNet still returns a different prediction, so the difference is inside
  the ControlNet UNet branch, not in the worker's inputs.

The load-bearing evidence is therefore the first two rows. They compare renders
made by the SAME pipeline. The cross-pipeline row is reported because it is
measured, not because it is a signal.

Each combined PNG carries the `controlnet` text chunk:
`[{"attachment_id": "cn_1", "control_type": "canny", "generation":
{"model_id": "sd15-canny", "strength": 1.0, "start_percent": 0.0,
"end_percent": 1.0}, "source": null}]`

Four `job:progress` frames arrived for each render. A strength-0.6 job runs 4 of
8 steps, so the counts match the steps.

Platform scope, refused live in both modes:

| Request | Terminal | Message |
|---|---|---|
| combined Depth attachment | `job:error` | `backend 'mps' does not support control_type 'depth' for family 'sd15'; allowed: canny` |
| combined two Canny attachments | `job:error` | `backend 'mps' admits at most 1 controlnet attachment(s) for family 'sd15', got 2` |

The mode permits both requests. The spike requires `job:error` whose message
EQUALS the one above, so a timeout or a mode-policy rejection cannot pass.

`/api/models/status` during both runs: `backend: mps`, `current_mode:
lcm-general`, `is_loaded: true`, `supports_img2img: true`,
`supports_controlnet: true`, `supports_img2img_and_controlnet: true`.

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
