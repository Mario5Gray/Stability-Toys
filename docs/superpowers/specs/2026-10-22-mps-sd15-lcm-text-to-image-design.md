# MPS SD1.5 LCM Text-to-Image Design

Issue: `STABL-hvvogszy`. Base: `a260f9a` on `codex/mps-sd15-lcm-t2i`.

## Scope

Bind family `sd15` to a separate MPS worker. Support LCM text-to-image generation for `lcm-general`.
Keep img2img, ControlNet, combined generation, SDXL, and super-resolution unsupported.
Keep HTTP, WebSocket, CLI, asset, and mode contracts stable.

`lcm-general` permits both LCM and Euler in configuration. The MPS worker must reject a non-LCM request.
The MPS worker must reject an init image and all ControlNet bindings before pipeline execution.
These checks prevent the family-level binding from implying support for every SD1.5 operation.

## Worker and binding

Add `DiffusersMpsWorker` in `backends/mps_worker.py`. Bind it through `MPS_FAMILY_BINDINGS["sd15"]` with all optional execution capabilities false.
Keep the binding module free of Torch and Diffusers imports. Resolve the worker through the existing lazy factory.
Use `MPS_TORCH_POLICY` for device, float16, placement, and cache release.
Do not change CUDA policy or CUDA environment variable behavior.

The worker loads the SD1.5 Diffusers pipeline from the resolved local model path. It selects the LCM scheduler.
It uses an MPS generator, an explicit seed, and `torch.inference_mode()` during inference.
It returns PNG bytes and the seed through the existing worker contract.
It preserves required progress, cancellation, conditioning, and PNG metadata behavior for text-to-image jobs.
It supports the text-to-image latent result method if a caller uses that worker contract.

Extract shared Torch worker behavior only when the MPS worker needs it. Keep the new MPS component distinct from `DiffusersCudaWorker`.
Do not add MPS branches to transport routes.

## Status and memory

Keep `supports_model_registry_stats=false` in the first slice. Keep the placeholder registry and its zero total-memory value.
The zero value means that the registry has no device-total measurement. It does not mean that MPS uses zero memory.
The registry continues to report the loaded model count and names. Document these meanings in `docs/MPS_BACKEND.md`.
The existing MPS execution policy and host probe provide process memory measurements outside `/api/models/status`.
Do not add CUDA-style VRAM totals or a route-specific MPS branch.

## Acceptance gate

Keep `supports_generation=false` while writing and testing the worker. Contract tests use patched Torch on any host.
Run `scripts/mps_host_probe.py` on the Mac outside a restricted sandbox. Require exit code 0.
Render `lcm-general` on live MPS. Record the command, seed, output, commit, host, and Torch version.
Enable `supports_generation` only after that live render passes. Record a second status check after the flag changes.
Run the CUDA regression cohort from `docs/MPS_BACKEND.md`. Update the guide with results and limits.

The current worktree does not contain the configured `/models/diffusers/lcm-dreamshaper-v7` asset.
The live test requires an accessible model at that path or an explicit, documented host model-root mapping.
An unavailable model or restricted MPS runtime cannot satisfy live acceptance.
