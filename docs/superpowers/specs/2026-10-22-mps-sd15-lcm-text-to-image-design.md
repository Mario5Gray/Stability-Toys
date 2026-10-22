# MPS SD1.5 LCM Text-to-Image Design

Issue: `STABL-hvvogszy`. Base: `a260f9a` on `codex/mps-sd15-lcm-t2i`.

## Scope

Bind family `sd15` to a separate MPS worker. Support LCM text-to-image generation for `lcm-general`.
Keep img2img, ControlNet, combined generation, SDXL, and super-resolution unsupported.
Keep HTTP, WebSocket, CLI, asset, and mode contracts stable.

`lcm-general` permits both LCM and Euler in configuration. The MPS worker accepts only scheduler ID `lcm`.
The worker raises `ValueError("MPS SD1.5 supports LCM scheduler only")` before pipeline execution for another scheduler.
This error occurs in the worker. Existing HTTP generation handling returns 500, and WebSocket returns a job error.
Do not claim an admission-time 400 until a generic scheduler capability gate exists.

`ExecutionCapabilities(False, False, False)` supplies the primary operation gate.
`admit_generation_operation` rejects ControlNet on HTTP and WebSocket. It also rejects init images on WebSocket.
It preserves `ValueError` for img2img or combined operations and `NotImplementedError` for ControlNet.
The MPS worker also rejects any init image or ControlNet binding as defense in depth.

## Worker and binding

Add `DiffusersMpsWorker` in `backends/mps_worker.py`. Bind it through `MPS_FAMILY_BINDINGS["sd15"]` with all optional execution capabilities false.
Keep the binding module free of Torch and Diffusers imports. Resolve the worker through the existing lazy factory.
Use `MPS_TORCH_POLICY` for device, float16, placement, and cache release.
Do not change CUDA policy or CUDA environment variable behavior.

`DiffusersMpsWorker` subclasses `CudaWorkerBase`, not `DiffusersCudaWorker`.
It sets `execution_policy = MPS_TORCH_POLICY` and owns its SD1.5 load and text-to-image methods.
The base already delegates settings, placement, and cache release through that policy.
The base supplies conditioning, scheduler, and style helpers. `inject_step_progress` supplies cancellation.
The MPS worker owns PNG encoding and metadata because the base has no PNG helper.
Do not extract a new shared base in this slice.
MPS-specific logs use `[mps]`. Any reachable base log must use the policy backend ID.
CUDA logs must retain `[cuda]` and CUDA behavior.

The worker loads the SD1.5 Diffusers pipeline from the resolved local model path. It selects the LCM scheduler.
It uses an MPS generator, an explicit seed, and `torch.inference_mode()` during inference.
It returns PNG bytes and the seed through the existing worker contract.
It uses `inject_step_progress` for step progress and cancellation.
The callback checks `should_cancel` at each denoise step boundary, outside the progress emitter that swallows consumer errors.
Cancellation raises `concurrent.futures.CancelledError` for the existing terminal `CANCELLED` path.
The worker preserves text conditioning and style selection for supported text-to-image requests.
PNG `lcm` metadata keeps `prompt`, `seed`, `size`, `steps`, `cfg`, `negative_prompt`, and `scheduler_id`.
Do not implement `run_job_with_latents` in this slice. Production generation calls `run_job` only.
HunyuanDiT already uses this narrower worker contract. Add the latent method only when a caller needs it.

Do not add MPS branches to transport routes.

## Probe and contract tests

Change `scripts/mps_host_probe.py` and `tests/test_mps_host_probe.py` when the SD1.5 binding lands.
Keep the negative `UnsupportedFamilyError` check, but use still-unbound family `sdxl`.
The expected error becomes `family 'sdxl' has no MPS platform binding`.
The probe retains its report schema and exit codes. Exit 0 remains required on a live MPS host.

Change `tests/test_mps_registration.py`. Replace assertions that `sd15` is unbound with binding and worker-selection assertions.
Keep negative tests for `sdxl` and `hunyuandit`.
Test the SD1.5 worker with patched Torch on any host.
Cover LCM selection, policy use, unsupported operations, progress, cancellation, and PNG result fields.
Test the operation gate on both HTTP and WebSocket paths.

## Status and memory

Keep `supports_model_registry_stats=false` in the first slice. Keep the placeholder registry and its zero total-memory value.
The zero value means that the registry has no device-total measurement. It does not mean that MPS uses zero memory.
The registry continues to report the loaded model count and names. Document these meanings in `docs/MPS_BACKEND.md`.
The existing MPS execution policy and host probe provide process memory measurements outside `/api/models/status`.
Do not add CUDA-style VRAM totals or a route-specific MPS branch.

## Acceptance sequence

`supports_generation` only changes `/api/models/status`. It does not block generation.
The real generation gate is the presence of an `sd15` cell in `MPS_FAMILY_BINDINGS`.
Land the binding and worker on this review branch with `supports_generation=false` temporarily.
Do not deploy or merge that intermediate commit. Its status flag understates available behavior.
Run contract tests and `scripts/mps_host_probe.py` on the Mac outside a restricted sandbox. Require probe exit code 0.
Render `lcm-general` on live MPS under both `WORKER_ISOLATION=inproc` and `WORKER_ISOLATION=subprocess`.
Use seed 7 and prompt `a bright red sailboat on a blue lake beneath a green mountain`.
Use the mode defaults for size, steps, guidance, and LCM scheduler.
Record each command, seed, output path, commit, host, Torch version, and isolation mode.
Inspect each PNG with Pillow. Require expected dimensions, more than 32 distinct RGB colors, and grayscale standard deviation above 5.
Inspect the image visually for black, white, or corrupted output.
If float16 VAE output fails these checks, stop acceptance and keep `supports_generation=false`.
Revise the MPS policy and design for a VAE precision fix before another live test.
Do not silently change the policy dtype, which this slice fixes at float16.
After both live renders pass, set `supports_generation=true` and test `/api/models/status`.
Merge only the final, consistent branch after review.
Run the CUDA regression cohort from `docs/MPS_BACKEND.md`. Update the guide with results and limits.

The current worktree does not contain the configured `/models/diffusers/lcm-dreamshaper-v7` asset.
The live test requires an accessible model at that path or an explicit, documented host model-root mapping.
An unavailable model or restricted MPS runtime cannot satisfy live acceptance.
