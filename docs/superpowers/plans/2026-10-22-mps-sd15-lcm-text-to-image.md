# MPS SD1.5 LCM Text-to-Image Implementation Plan

> Execute inline with `superpowers:executing-plans`. Claim only a human-popped waveplan task before code. Do not dispatch subagents.

**Goal:** Render `lcm-general` text-to-image on MPS in both worker isolation modes.

**Architecture:** `DiffusersMpsWorker` subclasses `CudaWorkerBase` and selects `MPS_TORCH_POLICY`.
`MPS_FAMILY_BINDINGS["sd15"]` selects that worker. Existing operation admission rejects img2img and ControlNet.

**Tech stack:** Python 3.12, Torch MPS, Diffusers, Compel, Pillow, pytest through `tests.run`, FP, drift.

**Authority:** Approved spec is `docs/superpowers/specs/2026-10-22-mps-sd15-lcm-text-to-image-design.md` at `099a9a4`.
This plan proposes steps. A human-popped waveplan row defines the work allowed in each execution turn.
Do not self-pop, self-finish, deploy, or merge. Stop if the row or its path bundle is absent.

## File map

| Path | Responsibility |
| --- | --- |
| `backends/mps_worker.py` | MPS SD1.5 pipeline, LCM text-to-image job, PNG result |
| `backends/platforms/mps_bindings.py` | Lazy `sd15` binding and false optional operation flags |
| `backends/platforms/mps.py` | Generation status flag after live acceptance |
| `backends/cuda_worker.py` | Keep rare progress-bar error log backend-aware |
| `scripts/mps_host_probe.py` | Move unsupported-family probe from `sd15` to `sdxl` |
| `tests/test_mps_worker.py` | Patched-Torch worker contract tests |
| `tests/test_mps_registration.py` | Binding, factory, isolation, and status tests |
| `tests/test_mps_host_probe.py` | Revised probe expectation and schema tests |
| `tests/test_progress_bar_suppression.py` | Preserve CUDA log and test MPS log |
| `tests/test_controlnet_execution.py` | Assert MPS cell rejects unsupported operations |
| `tests/test_controlnet_dispatch.py` | Prove HTTP generation uses the MPS operation gate |
| `tests/test_ws_routes.py` | Prove WebSocket init-image and ControlNet rejection |
| `docs/MPS_BACKEND.md` | Limits, commands, contract results, and live evidence |
| `docs/WORKER_SELECTION.md` | MPS binding and worker selection |
| `drift.lock` | Reviewed provenance for changed bound files |

Do not edit `server/lcm_sr_server.py` or `server/ws_routes.py` for backend selection.
Do not add `run_job_with_latents`. Production generation calls `run_job`.

## Step 0: Claim and inspect

1. Read the human-popped waveplan row. Confirm its `fp_refs` includes `STABL-hvvogszy`.
2. Run `waveplan-cli swim local status` with the complete matching `WAVEPLAN_*` path bundle.
3. Claim an `agent_dispatch` row with `waveplan-cli swim local claim --agent sigma` only when assigned.
4. Run `FP_AGENT_NAME='sigma' fp context STABL-hvvogszy` and `git log --oneline -8`.
5. Confirm branch `codex/mps-sd15-lcm-t2i`, ancestor `099a9a4`, and clean status.
6. Activate `stability-toys`. Confirm `python` resolves under `/Users/darkbit1001/miniforge3/envs/stability-toys`.
7. Run `drift refs` on each planned code path before edits.

Expected: matching row claimed, no unexpected worktree changes, and no code touched outside that row.
Use `waveplan-cli swim local log` for factual progress. Leave FP comments at milestones.

## Step 1: Build worker through red and green tests

**Files:** `tests/test_mps_worker.py`, `backends/mps_worker.py`.

1. Add `test_mps_worker_loads_lcm_on_mps` using a fake SD1.5 pipeline.
2. Patch `torch.backends.mps.is_built` and `is_available` to return true.
3. Patch `backends.mps_worker._sd_pipeline_cls` and `_lcm_scheduler_cls` with recording fakes.
4. Assert `from_pretrained` receives the local path and `torch.float16` for a Diffusers directory.
5. Assert `pipe.to("mps")`, LCM scheduler installation, and `worker.execution_policy is MPS_TORCH_POLICY`.
6. Add the matching `from_single_file` case for `loader_format="single_file"`.
7. Run the worker file through `tests.run`. Expected RED: `backends.mps_worker` import fails.

Use these exact worker assertions with recording fakes:

```python
assert worker.device == "mps"
assert worker.dtype is torch.float16
assert worker.execution_policy is MPS_TORCH_POLICY
pipeline_class.from_pretrained.assert_called_once_with(
    "/node/models/diffusers/lcm-dreamshaper-v7",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.to.assert_called_once_with("mps")
assert worker.pipe is pipe
```

```bash
python -m tests.run tests/test_mps_worker.py -- -q -p no:cacheprovider --no-cov
```

8. Add `DiffusersMpsWorker(CudaWorkerBase)` in `backends/mps_worker.py`.
9. Call `super().__init__(worker_id, model_info=model_info, family_profile=family_profile)` first.
10. Load `StableDiffusionPipeline` through `_sd_pipeline_cls()` by `loader_format`.
11. Set `pipe.scheduler = _lcm_scheduler_cls().from_config(pipe.scheduler.config)`.
12. Call `self._setup_pipe_memory_opts(pipe)`, then store `self.pipe` and capture its scheduler.
13. Reuse SD1.5 style loading from `DiffusersCudaWorker.__init__` without a new base class.
14. Keep MPS worker logs prefixed `[mps]`. Do not change CUDA environment values.
15. Run the worker file again. Expected GREEN: all constructor tests pass.

The constructor must accept `worker_id`, `model_path`, `model_info`, and `family_profile`.
That signature matches `create_bound_worker` in `backends/worker_factory.py`.

## Step 2: Add text-to-image job behavior

**Files:** `tests/test_mps_worker.py`, `backends/mps_worker.py`.

1. Add a fake pipeline that returns a Pillow RGB image.
2. Give its `__call__` a `callback_on_step_end` parameter.
3. Add a test for seed 7, mode defaults, PNG bytes, and seven `lcm` metadata keys.
4. Assert an MPS generator receives seed 7. Assert inference runs under `torch.inference_mode()`.
5. Add a test that requests `scheduler_id="euler"`. Expect the exact worker `ValueError`.
6. Add tests for `job.init_image` and `job.controlnet_bindings`. Expect rejection before pipeline call.
7. Add a cancellation test. Make `should_cancel` return true at the first step.
8. Expect `concurrent.futures.CancelledError("job cancelled at step boundary")`.
9. Assert `MPS_TORCH_POLICY.release_cache()` runs after success and pipeline failure.
   Early operation rejection allocates no pipeline result and needs no cache release.
10. Run the worker file. Expected RED: missing `run_job` or failed assertions.

Use one request and job shape for positive and negative cases:

```python
req = SimpleNamespace(
    prompt="a bright red sailboat", negative_prompt=None,
    scheduler_id=None, size="768x768", num_inference_steps=8,
    guidance_scale=3.0, seed=7, style_lora=None,
)
job = SimpleNamespace(req=req, init_image=None, controlnet_bindings=[])
png_bytes, seed = worker.run_job(job)
assert seed == 7
assert Image.open(io.BytesIO(png_bytes)).format == "PNG"
assert json.loads(Image.open(io.BytesIO(png_bytes)).info["lcm"])["scheduler_id"] == "lcm"
```

Set `worker.model_info.default_scheduler_id="lcm"` in the fake.
Give the fake pipeline a real `callback_on_step_end` parameter for cancellation tests.

```bash
python -m tests.run tests/test_mps_worker.py -- -q -p no:cacheprovider --no-cov
```

11. Implement `run_job(job, progress=None, should_cancel=None)` with this order:

```python
if getattr(job, "init_image", None) is not None:
    raise ValueError("MPS SD1.5 does not support img2img")
if getattr(job, "controlnet_bindings", None):
    raise NotImplementedError("MPS SD1.5 does not support ControlNet")
req = job.req
if self._resolve_scheduler_id(req) != "lcm":
    raise ValueError("MPS SD1.5 supports LCM scheduler only")
```

12. Parse `req.size` as width and height. Use `torch.Generator(device="mps")` with the selected seed.
13. Apply style and LCM scheduler through base helpers.
14. Get conditioning through `_conditioning_artifact_for_request` and `_accept_conditioning_artifact`.
15. Pass `width`, `height`, steps, guidance, generator, and conditioning to the pipeline.
16. Pass `should_cancel` to `inject_step_progress`. Run the pipeline inside `torch.inference_mode()`.
17. Encode `out.images[0]` as PNG. Put these exact keys in the `lcm` text chunk:

```python
{
    "prompt": req.prompt,
    "seed": seed,
    "size": req.size,
    "steps": int(req.num_inference_steps),
    "cfg": float(req.guidance_scale),
    "negative_prompt": getattr(req, "negative_prompt", None),
    "scheduler_id": "lcm",
}
```

18. In `finally`, release result references, disable the active style, and call `release_cache()`.
19. Run the worker file. Expected GREEN: positive and negative worker tests pass.
20. Commit named paths with `STABL-hvvogszy` and the next waveplan step in the message.

The scheduler error runs inside the worker. HTTP returns 500 through existing generation handling.
WebSocket returns a job error. Do not claim a 400 admission response.

## Step 3: Bind SD1.5 and update probe contracts

**Files:** `backends/platforms/mps_bindings.py`, `scripts/mps_host_probe.py`, `tests/test_mps_registration.py`, `tests/test_mps_host_probe.py`, `tests/test_controlnet_execution.py`, `tests/test_controlnet_dispatch.py`, `tests/test_ws_routes.py`.

1. Add failing binding tests in `tests/test_mps_registration.py`.
2. Assert the `sd15` worker ref is `backends.mps_worker.DiffusersMpsWorker`.
3. Assert all three optional execution flags are false.
4. Assert `sdxl` and `hunyuandit` still raise `UnsupportedFamilyError`.
5. Replace `test_mps_provider_has_empty_bindings_and_shared_runtime` with a bound-SD1.5 assertion.
6. Change `test_mps_subprocess_entry_rejects_every_family` to check `sdxl` and `hunyuandit`.
7. Change both negative pool and child tests to resolve `sdxl` instead of `sd15`.
8. Change the no-model lifespan test to use still-unbound `sdxl`.
9. Change the mocked mode-switch error message to name `sdxl`.
10. Use a fake worker for positive factory selection. Keep provider and binding imports free of Torch.
11. Add MPS admission tests in `tests/test_controlnet_execution.py`.
12. Assert text-to-image admits, img2img raises `ValueError`, and ControlNet raises `NotImplementedError`.
13. Add an HTTP test in `tests/test_controlnet_dispatch.py` with a bound MPS `sd15` snapshot.
    Send ControlNet through `generate`. Expect HTTP 501 before `submit_generate`.
    HTTP passes `has_init_image=False` because `/generate` cannot express img2img.
14. Add WebSocket tests in `tests/test_ws_routes.py` with the same MPS cell.
    Send `init_image_ref`, then ControlNet, as separate requests to `/v1/ws`.
    Expect `job:ack`, then `job:error`. Assert no job submission or preprocessing.
15. Run focused tests. Expected RED: no `sd15` MPS binding.

```bash
python -m tests.run tests/test_mps_registration.py tests/test_controlnet_execution.py tests/test_controlnet_dispatch.py tests/test_ws_routes.py -- -q -p no:cacheprovider --no-cov
```

16. Review `docs/MPS_BACKEND.md` and `docs/WORKER_SELECTION.md`. Replace obsolete empty-binding prose before changing bound code.
17. Add the binding without importing the worker module:

```python
from backends.platforms.base import ExecutionCapabilities, FamilyPlatformBinding

MPS_FAMILY_BINDINGS: dict[str, FamilyPlatformBinding] = {
    "sd15": FamilyPlatformBinding(
        "backends.mps_worker.DiffusersMpsWorker",
        ExecutionCapabilities(False, False, False),
    ),
}
```

18. Change `UNSUPPORTED_FAMILY` in `scripts/mps_host_probe.py` from `sd15` to `sdxl`.
19. Change the matching message in `tests/test_mps_host_probe.py`.
20. Change its failure-injection key from `sd15` to `sdxl`.
21. Keep probe report keys and exit codes unchanged.
22. Run focused tests. Expected GREEN: binding, probe, and admission tests pass.

```bash
python -m tests.run tests/test_mps_registration.py tests/test_mps_host_probe.py tests/test_controlnet_execution.py tests/test_controlnet_dispatch.py tests/test_ws_routes.py tests/test_mps_worker.py -- -q -p no:cacheprovider --no-cov
```

23. Confirm `/api/models/status` still reports `supports_generation=false` at this intermediate commit.
24. Commit named paths. Do not deploy or merge this temporary status mismatch.

## Step 4: Preserve log and status contracts

**Files:** `backends/cuda_worker.py`, `tests/test_progress_bar_suppression.py`, `tests/test_mps_registration.py`.

1. Add a test for the rare progress-bar suppression failure under MPS.
2. Assert it logs `[mps]`. Assert the existing CUDA path still logs `[cuda]`.
3. Run `tests/test_progress_bar_suppression.py`. Expected RED: MPS case logs `[cuda]`.
4. Add an optional `backend_id="cuda"` argument to static `_maybe_disable_progress_bar`.
5. Pass `self.execution_policy.backend_id` from `_setup_pipe_memory_opts`.
6. Use `backend_id` only in that debug message. Preserve direct static calls and CUDA text.

```python
@staticmethod
def _maybe_disable_progress_bar(pipe, backend_id="cuda"):
    try:
        if resolve_log_format() != JSON:
            return pipe
        configure = getattr(pipe, "set_progress_bar_config", None)
        if callable(configure):
            configure(disable=True)
    except Exception:
        logger.debug("[%s] could not disable the progress bar", backend_id, exc_info=True)
    return pipe

# In _setup_pipe_memory_opts:
self._maybe_disable_progress_bar(pipe, backend_id=self.execution_policy.backend_id)
```

7. Add a status test for loaded model name and count in `PlaceholderModelRegistry`.
8. Assert its total remains zero and `supports_model_registry_stats` remains false.
9. Run focused tests. Expected GREEN: both log labels and placeholder status pass.

```bash
python -m tests.run tests/test_progress_bar_suppression.py tests/test_mps_registration.py tests/test_model_routes.py -- -q -p no:cacheprovider --no-cov
```

10. Review drift-bound prose before any `drift link`. Commit named paths and reviewed docs.

## Step 5: Prove live MPS before the status flag changes

**Files:** `docs/MPS_BACKEND.md` after evidence collection. Keep host config and images outside the repo.

1. Stop if no local `lcm-dreamshaper-v7` Diffusers directory exists.
2. Export `MPS_SLICE_MODEL_ROOT` as the absolute parent model root.
3. Require `$MPS_SLICE_MODEL_ROOT/diffusers/lcm-dreamshaper-v7/model_index.json`.
4. Copy `conf/modes.yml` to `/tmp/stabl-hvvogszy-conf/modes.yml`.
5. Set `model_root` to the model root. Set `lora_root` to its `loras` directory.

```bash
printf 'Absolute model root: '
read -r MPS_SLICE_MODEL_ROOT
export MPS_SLICE_MODEL_ROOT
test -f "$MPS_SLICE_MODEL_ROOT/diffusers/lcm-dreamshaper-v7/model_index.json"
mkdir -p /tmp/stabl-hvvogszy-conf
cp conf/modes.yml /tmp/stabl-hvvogszy-conf/modes.yml
python - <<'PY'
import os
from pathlib import Path
import yaml

path = Path('/tmp/stabl-hvvogszy-conf/modes.yml')
data = yaml.safe_load(path.read_text())
data['model_root'] = os.environ['MPS_SLICE_MODEL_ROOT']
data['lora_root'] = str(Path(os.environ['MPS_SLICE_MODEL_ROOT']) / 'loras')
path.write_text(yaml.safe_dump(data, sort_keys=False))
PY
```

6. Keep mode `lcm-general`, its LCM default, and its Compel service unchanged.
7. Confirm `python -c 'import compel'` succeeds in `stability-toys`.
8. Confirm `conditioning.service == "compel"` and `fallback.native_on_failure is False` in the served config.

```bash
python -c 'import compel; from server.mode_config import ModeConfigManager; mode = ModeConfigManager("/tmp/stabl-hvvogszy-conf").get_mode("lcm-general"); assert mode.conditioning.service == "compel"; assert mode.conditioning.fallback.native_on_failure is False'
```
9. Commit implementation first. Require `git status --porcelain` empty before the probe.
10. Run the two-probe repeatability sequence from `docs/MPS_BACKEND.md` outside a restricted sandbox.
11. Require both probe exits 0 and `seeded_tensor.matches_expected: true` on run two.
12. Start the server with `BACKEND=mps`, `MODE_CONFIG_PATH=/tmp/stabl-hvvogszy-conf`, and `WORKER_ISOLATION=inproc`.
13. Switch to `lcm-general`. Confirm the loaded mode through `/api/models/status`.

```bash
BACKEND=mps WORKER_ISOLATION=inproc MODE_CONFIG_PATH=/tmp/stabl-hvvogszy-conf CONTROLNET_REGISTRY_VALIDATION=off python server/lcm_sr_server.py
```

Use a second terminal for the request. Save the server log and commit SHA with the PNG.

```bash
curl --fail-with-body --silent --show-error -H 'Content-Type: application/json' \
  -d '{"mode":"lcm-general"}' http://127.0.0.1:4200/api/modes/switch
curl --fail-with-body --silent --show-error http://127.0.0.1:4200/api/models/status
curl --fail-with-body --silent --show-error --max-time 900 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a bright red sailboat on a blue lake beneath a green mountain","mode":"lcm-general","seed":7}' \
  http://127.0.0.1:4200/generate \
  --output /tmp/stabl-hvvogszy-inproc.png
```

14. Stop that server. Repeat the same request under subprocess isolation:

```bash
BACKEND=mps WORKER_ISOLATION=subprocess MODE_CONFIG_PATH=/tmp/stabl-hvvogszy-conf CONTROLNET_REGISTRY_VALIDATION=off python server/lcm_sr_server.py
```

Use a second terminal for the request:

```bash
curl --fail-with-body --silent --show-error -H 'Content-Type: application/json' \
  -d '{"mode":"lcm-general"}' http://127.0.0.1:4200/api/modes/switch
curl --fail-with-body --silent --show-error http://127.0.0.1:4200/api/models/status
curl --fail-with-body --silent --show-error --max-time 900 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a bright red sailboat on a blue lake beneath a green mountain","mode":"lcm-general","seed":7}' \
  http://127.0.0.1:4200/generate \
  --output /tmp/stabl-hvvogszy-subprocess.png
```
15. For each PNG, require size `768x768`, more than 32 RGB colors, and grayscale standard deviation above 5.
16. Check PNG `lcm` metadata for seed 7, steps 8, cfg 3, size `768x768`, and scheduler `lcm`.
17. Inspect both images visually for black, white, or corrupt output.

```python
from pathlib import Path
from PIL import Image, ImageStat
import json

for name in ('inproc', 'subprocess'):
    path = Path(f'/tmp/stabl-hvvogszy-{name}.png')
    with Image.open(path) as source:
        assert source.format == 'PNG'
        assert source.size == (768, 768)
        meta = json.loads(source.info['lcm'])
        assert (meta['seed'], meta['steps'], meta['cfg']) == (7, 8, 3.0)
        assert (meta['size'], meta['scheduler_id']) == ('768x768', 'lcm')
        rgb = source.convert('RGB')
        assert len(set(rgb.resize((64, 64)).getdata())) > 32
        assert ImageStat.Stat(rgb.convert('L')).stddev[0] > 5
```
18. Record command, seed, path, SHA, host, Torch version, isolation mode, and served Compel service.
19. Check `/api/models/status` during each run. Record the temporary false flag and loaded mode.

If probe exits 3, record unavailable. If Compel fails, record failure. Neither result proves acceptance.
If a PNG fails validity, keep `supports_generation=false`. Revise the design before any VAE precision change.

## Step 6: Flip status flag and finish evidence

**Files:** `backends/platforms/mps.py`, `tests/test_mps_registration.py`, `docs/MPS_BACKEND.md`, `docs/WORKER_SELECTION.md`, `drift.lock`.

1. Add a test that expects `MPSProvider.capabilities().supports_generation is True`.
2. Run it alone. Expected RED: value remains false.
3. Change only the first `BackendCapabilities` value in `MPSProvider.capabilities()` to true.
   The resulting return is `BackendCapabilities(True, True, False, False)`.
4. Run the test. Expected GREEN: value true. Keep the registry flag false.
5. Check `/api/models/status` with MPS and a loaded `lcm-general` mode.
6. Require `supports_generation=true`, `current_mode="lcm-general"`, loaded count 1, and no numeric MPS total claim.
7. Run the MPS contract cohort and the CUDA regression cohort from `docs/MPS_BACKEND.md`.

```bash
python -m tests.run \
  tests/test_mps_worker.py tests/test_mps_policy.py tests/test_mps_registration.py \
  tests/test_mps_host_probe.py tests/test_progress_bar_suppression.py \
  tests/test_controlnet_execution.py tests/test_controlnet_dispatch.py \
  tests/test_ws_routes.py tests/test_torch_policy.py \
  tests/test_worker_factory.py tests/test_backend_runtimes.py \
  tests/test_device_memory.py tests/test_worker_isolation.py \
  tests/test_subprocess_worker_handle.py tests/test_governor.py \
  tests/test_worker_pool.py tests/test_worker_handle.py \
  -- -q -p no:cacheprovider --no-cov

python -m tests.run \
  tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py \
  tests/test_cuda_worker_controlnet.py tests/test_hunyuandit_worker.py \
  tests/test_sdxl_worker.py tests/test_cuda_packaging_contract.py \
  tests/test_platform_registry.py tests/test_superres_http.py \
  -- -q -p no:cacheprovider --no-cov -rs
```

Expected: zero failures. CUDA-only tests may skip on a Mac. Report skip count and reason.
8. Read final summaries and failures first. Expand logs only when needed.
9. Record exact pass, skip, fail, commit, and host details in `docs/MPS_BACKEND.md`.
10. Update `docs/WORKER_SELECTION.md` to name the MPS `sd15` binding and unsupported families.
11. Review every changed doc claim against final code before `drift link`.
    Inspect all docs reported by `drift refs backends/cuda_worker.py` before any link.
    Link only reviewed MPS docs and any task-stale CUDA-worker docs.
12. Run `drift check --changed backends/` and `drift check`.
13. Separate old, unrelated drift failures from task-caused failures. Never relink stale prose.
14. Run `git diff --check` and inspect named-file diff.
15. Commit named paths with `STABL-hvvogszy` and next review step.
16. Attach revisions to FP. Add one actionable `STOP/NEXT` comment.
17. Complete only the claimed local waveplan row with test and artifact evidence.
18. Advance SWIM to the next dispatch or review boundary. Report ready for review.

Do not call waveplan `fin`. Do not start another task without human direction.
