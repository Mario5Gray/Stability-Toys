# HunyuanDiT img2img + ControlNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. AGENTS.md in the execution environment takes precedence over any skill suggestion to delegate.

**Goal:** Run HunyuanDiT img2img with one Canny ControlNet attachment, preserving strength-selected denoising and reproducible seeded control latents.

**Architecture:** Extend the existing Hunyuan img2img component with optional ControlNet latent preparation and the installed Diffusers 0.39 residual handoff. The Hunyuan worker reuses its cache, control-map decoding, metadata, and cleanup; admission enables the combined operation only with the worker path. Existing plain img2img and ControlNet text-to-image calls remain separate.

**Tech Stack:** Python 3.12, PyTorch, Diffusers 0.39, Pillow, pytest through `tests.run`, CUDA `test-cuda` Compose service.

**Spec:** `docs/superpowers/specs/2026-10-20-hunyuandit-img2img-controlnet-design.md` at `2bec745`.

---

## File map and execution constraints

| File | Responsibility |
| --- | --- |
| `server/controlnet_execution.py` | Reject Hunyuan combined requests with non-Canny controls or nondefault control windows before preprocessing. |
| `backends/hunyuandit_img2img.py` | Prepare seeded Canny latent and pass ControlNet residuals to the transformer on selected steps. |
| `backends/cuda_worker.py` | Route combined jobs, validate image alignment, reuse model cache, write both PNG metadata chunks, release pins. |
| `backends/platforms/cuda_bindings.py` | Enable the Hunyuan combined capability in the worker-route task. |
| `tests/test_controlnet_execution.py`, `tests/test_ws_routes.py` | Admission and WebSocket ordering. |
| `tests/test_hunyuandit_img2img.py` | Real-tensor control latent, generator, and denoising contract. |
| `tests/test_hunyuandit_worker.py`, `tests/test_backend_runtimes.py` | Worker routing, cache cleanup, metadata, capability and import-clean checks. |
| `tests/test_hunyuandit_acceptance.py` | Production-path CUDA acceptance and artifact capture. |
| `docs/TESTING_IN_DOCKER.md` | CUDA command and review checklist. |

Keep `cuda_bindings.py` import-clean. Do not load a second base model, use an SD/SDXL pipeline, cast shared components to fp32, enable Depth/Pose or multiple Hunyuan attachments, or change other families. The local worktree is a planning worktree; execution should use its own branch/worktree. Stage named paths only. Use `python -m tests.run` for mixed cohorts because direct pytest collection is guarded. A local CUDA skip is not acceptance.

## Task 1: Hunyuan combined admission

**Files:** Modify `server/controlnet_execution.py`, `tests/test_controlnet_execution.py`, `tests/test_ws_routes.py`.

- [ ] **Step 1: Write RED admission tests.** Use a Hunyuan snapshot and a fake family binding with `ExecutionCapabilities(True, True, True)` plus `ControlNetAttachment(control_type="canny", map_asset_ref="map", attachment_id="cn")`. Assert init plus attachment returns `"img2img+controlnet"`; `control_type="depth"` raises `ValueError` naming Canny; `start_percent=0.1` or `end_percent=0.9` raises `ValueError` naming the unsupported window. Keep the existing SD family combined expectations. In the WebSocket test, use a fake provider advertising combined support and assert a rejected Hunyuan window does not call `preprocess_controlnet_attachments` or submit a job. Production Hunyuan capability remains false until Task 4.
- [ ] **Step 2: Run RED:** `python -m tests.run tests/test_controlnet_execution.py tests/test_ws_routes.py -- -q --no-cov`. Expected: new Hunyuan invalid-control/window cases fail.
- [ ] **Step 3: Implement the family-specific check** inside the `has_init_image and has_controlnet` branch of `admit_generation_operation`, after the capability check and before returning. Use the already parsed attachment defaults; no schema change is needed:

```python
if family_id == "hunyuandit":
    for attachment in req.controlnets:
        if attachment.control_type != "canny":
            raise ValueError("HunyuanDiT combined img2img+controlnet supports Canny only")
        if attachment.start_percent != 0.0 or attachment.end_percent != 1.0:
            raise ValueError("HunyuanDiT combined img2img+controlnet does not support control timing windows")
```

  Keep the mode policy's one-attachment limit and existing family registry validation; do not duplicate them. Leave production Hunyuan combined capability false until Task 4.
- [ ] **Step 4: Run GREEN:** same command; expected all selected tests pass. Confirm the Hunyuan rejected-window WebSocket test observes zero preprocessing and zero submissions.
- [ ] **Step 5: Commit** only the three named files: `feat: validate Hunyuan combined Canny admission`.

## Task 2: Seeded control-image latent preparation

**Files:** Modify `backends/hunyuandit_img2img.py`, `tests/test_hunyuandit_img2img.py`.

- [ ] **Step 1: Write RED tests** for `prepare_control_latents(pipe, control_image, *, width, height, generator, dtype, device, do_cfg)`. Use a fake VAE latent distribution whose `sample(generator=...)` records the generator and draws one value from it. Assert: the image processor sees the requested size; non-CFG VAE input batch is 1; CFG input batch is 2, matching Diffusers 0.39's `prepare_image(..., do_classifier_free_guidance=True)` before VAE encode; output is multiplied by `vae.config.scaling_factor` and moved to transformer dtype/device; the exact request generator reaches `sample`. Same seed and input must yield equal prepared latents; a changed seed must change them. Invalid dimensions fail before VAE encode.
- [ ] **Step 2: Run RED:** `python -m tests.run tests/test_hunyuandit_img2img.py -- -q --no-cov`. Expected: missing helper/new tests fail.
- [ ] **Step 3: Implement** a focused helper following the installed `HunyuanDiTControlNetPipeline.prepare_image` and `__call__` steps 4/9. For this delivery, batch size and images per prompt are both 1. Normalize with `pipe.image_processor.preprocess(control_image, height=height, width=width)`, move to VAE device/dtype, duplicate the image tensor before VAE encode if CFG is enabled, then use:

```python
control_dist = pipe.vae.encode(control_tensor).latent_dist
control_latents = control_dist.sample(generator=generator)
control_latents = control_latents * pipe.vae.config.scaling_factor
return control_latents.to(device=device, dtype=dtype)
```

  Reuse the existing positive, divisible-by-8 dimension rule. Keep imports lazy. This `generator=` argument intentionally differs from the installed ControlNet pipeline's unseeded `sample()` call.
- [ ] **Step 4: Run GREEN:** same command; expected all selected tests pass.
- [ ] **Step 5: Commit** only the two named files: `feat: seed Hunyuan Canny control latents`.

## Task 3: ControlNet residuals in the selected denoising steps

**Files:** Modify `backends/hunyuandit_img2img.py`, `tests/test_hunyuandit_img2img.py`.

- [ ] **Step 1: Write RED component tests.** Extend the existing fake Hunyuan pipe/transformer with a fake ControlNet that records `hidden_states`, `timestep`, prompt embeddings and masks, `image_meta_size`, `style`, `image_rotary_emb`, `controlnet_cond`, and `conditioning_scale`, returning a sentinel block tuple. Run 30 steps at strength 0.25 and assert exactly 7 ControlNet calls, one per transformer call, and the sentinel is passed as `controlnet_block_samples`. Assert `conditioning_scale=0.8`; CFG doubles the control latent batch; progress total is 7. Assert no ControlNet call and no changed output contract when `controlnet=None`. Assert cancellation after the first step prevents further ControlNet calls. With a recording VAE distribution and generator, assert draw order is source sample, source noise, then control sample; two same-seed runs prepare equal source/control latents, while a changed seed changes at least one. Run the combined path with a real `DDPMScheduler` timestep tensor and assert a scheduler without `add_noise` fails before any ControlNet call.
- [ ] **Step 2: Run RED:** `python -m tests.run tests/test_hunyuandit_img2img.py -- -q --no-cov`. Expected: new control arguments/residual assertions fail.
- [ ] **Step 3: Extend** `run_hunyuan_img2img` with keyword-only `controlnet=None`, `control_image=None`, and `controlnet_conditioning_scale=1.0`. Reject a model without an image or an image without a model. Keep the source-sample then source-noise generator order in `prepare_source_latents`; afterward prepare the Canny latent using Task 2 and the same generator. Within each selected timestep, before `pipe.transformer`, call the ControlNet with the same `latent_model_input`, `t_expand`, two text embeddings/masks, metadata, style, and rotary embeddings as the transformer. Pass `controlnet_cond=control_latents`, `conditioning_scale=controlnet_conditioning_scale`, `return_dict=False`; take `[0]` as `control_block_samples` and pass it to `pipe.transformer(controlnet_block_samples=...)`. Omit that transformer keyword entirely on the plain path. Keep the existing scheduler step, progress, cancellation, safety check, and decode sequence.
- [ ] **Step 4: Run GREEN:** same command; expected all selected tests pass. Compare the argument names against the installed `diffusers/pipelines/controlnet_hunyuandit/pipeline_hunyuandit_controlnet.py` in the test environment; do not use SDXL's ControlNet call shape.
- [ ] **Step 5: Commit** only the two named files: `feat: apply Hunyuan Canny residuals during img2img`.

## Task 4: Combined worker route and cleanup

**Files:** Modify `backends/cuda_worker.py`, `backends/platforms/cuda_bindings.py`, `tests/test_hunyuandit_worker.py`, `tests/test_backend_runtimes.py`.

- [ ] **Step 1: Write RED worker tests.** Replace the Hunyuan combined rejection test with one resolved Canny binding and a patched `run_hunyuan_img2img` that returns a PIL image. Assert `_validate_control_image_aspect_ratio` is invoked before resize; `_build_controlnet_kwargs` loads the cached model and decodes `control_image`; the component receives that model, image, and scale; execution is inside `torch.inference_mode`; PNG `lcm` includes `denoise_strength`, and `controlnet` contains the existing provenance. Test a 3% native aspect mismatch, bad init/map bytes, a request with `controlnets` but no resolved binding, non-Canny or two direct bindings, and a nondefault timing window as worker defense. Assert none reaches the component. For a component exception or cancellation after acquisition, assert every acquired model ID is released exactly once; also assert normal release. Update the capability test to expect Hunyuan `(True, True, True)` while retaining the import-clean assertion. Keep plain img2img and ControlNet txt2img tests unchanged and passing.
- [ ] **Step 2: Run RED:** `python -m tests.run tests/test_hunyuandit_worker.py -- -q --no-cov`. Expected: new combined route tests fail at the old guard.
- [ ] **Step 3: Replace** the old Hunyuan `init_image and bindings` rejection with validation for exactly one Canny binding and default timing. If `req.controlnets` is nonempty but resolved bindings are absent, fail instead of falling into plain img2img. Route combined jobs through the existing init-image branch. Before resizing, call `_validate_control_image_aspect_ratio(init_image, bindings)`. Wrap invalid init/map image decoding errors with a label that names the failing input. Reuse `_build_controlnet_kwargs(bindings, (width, height), loaded_ids)` and pop its `controlnet`, `control_image`, and `controlnet_conditioning_scale` values to pass into Task 3's component. Do not call `_build_controlnet_pipe`; the component uses the already loaded base and cached control model. Preserve the `finally` cache release even when map decoding or generation raises. Add the existing `_controlnet_metadata(bindings)` PNG chunk when bindings are present. Set only Hunyuan's third `ExecutionCapabilities` boolean to `True` in `cuda_bindings.py` in this same change. Do not change the text-to-image branch or its resolution binning.
- [ ] **Step 4: Run GREEN:** same command; expected all selected tests pass. Then run `python -m tests.run tests/test_hunyuandit_img2img.py tests/test_controlnet_execution.py tests/test_backend_runtimes.py tests/test_ws_routes.py tests/test_hunyuandit_worker.py -- -q --no-cov`; expected all selected tests pass across isolated cohorts.
- [ ] **Step 5: Commit** only the four named files: `feat: route Hunyuan combined img2img Canny jobs`.

## Task 5: Live CUDA acceptance and documentation

**Files:** Modify `tests/test_hunyuandit_acceptance.py`, `docs/TESTING_IN_DOCKER.md`.

- [ ] **Step 1: Add a separate combined acceptance case** to `tests/test_hunyuandit_acceptance.py`. In the production `HunyuanDiT` mode, use a 1024x1024 source image plus an aligned Canny map, one `hunyuandit-canny` binding, 30 steps, CFG 5, seed 4242, and strengths 0.25 and 0.75. Assert admission returns `img2img+controlnet`; save source, map, and both output PNGs under `ACCEPTANCE_OUT_DIR`; assert dimensions, seed, `denoise_strength`, and `controlnet` provenance. Capture Diffusers warnings through the existing helper and reject blocking ignored-input warnings. Print each case's elapsed seconds and peak allocated VRAM before assertions. Assert `free_vram` unloads the worker. Run this test alone through `tests.run` on CPU; expected CUDA skip, which verifies collection only.
- [ ] **Step 2: Run the live test on enigma.** SSH to `enigma.lan` (fallback `enigma` or `100.96.199.17`) and use an isolated worktree from `~/workspace/Stability-Toys` at the exact implementation commit; preserve the remote main checkout and its local files. Mount `/media/cold1/ComfyUI/models` as `/models` with `TEST_MODELS_HOST_PATH`. In the isolated worktree, create `hunyuan-combined.override.yml` with `services: {test-cuda: {image: stability-toys-hunyuan-combined-test, container_name: stability-toys-hunyuan-combined-test-cuda}}` so it cannot collide with another test run. Build the `test-cuda` image from that commit and run:

```bash
TEST_MODELS_HOST_PATH=/media/cold1/ComfyUI/models docker compose -f docker-compose.test.yml -f hunyuan-combined.override.yml build test-cuda
TEST_MODELS_HOST_PATH=/media/cold1/ComfyUI/models docker compose -f docker-compose.test.yml -f hunyuan-combined.override.yml run --rm \
  -e ACCEPTANCE_OUT_DIR=/app/logs/hunyuan-combined test-cuda \
  python -m tests.run tests/test_hunyuandit_acceptance.py -- -q -s --no-cov
```

  Expected: existing ControlNet text-to-image, plain img2img, and new combined acceptance all pass with zero blocked warnings. A skip, missing checkpoint, OOM, or timeout leaves live acceptance open; report it rather than claiming support.
- [ ] **Step 3: Review artifacts** directly. Compare source and both combined outputs for source preservation at low strength and visible Canny influence, and inspect saved PNG metadata. Record exact commit, Diffusers/Torch/CUDA versions, command, pass/fail/skip counts, elapsed times, peak allocated bytes, artifact paths, and any limitation. A fixed seed is required, but do not require byte-identical PNG output across differing GPU loads or library versions.
- [ ] **Step 4: Update** `docs/TESTING_IN_DOCKER.md` with the combined command, expected three-case count, artifacts, and evidence boundary. Add or update user-facing Hunyuan capability text only where existing docs claim combined support is unavailable; keep Depth/Pose unverified.
- [ ] **Step 5: Run focused verification** on the implementation commit: `python -m tests.run tests/test_hunyuandit_img2img.py tests/test_hunyuandit_worker.py tests/test_controlnet_execution.py tests/test_ws_routes.py tests/test_backend_runtimes.py -- -q --no-cov`. Expected all selected tests pass. Inspect `git diff --check`, then commit only the acceptance test and documentation: `test: accept Hunyuan combined img2img Canny on CUDA`.

## Final review gate

- [ ] Recheck the spec line by line against the diff: Canny only, one binding, fixed generator order, full control window, no second base load or fp32 recast, per-step residuals, metadata, cleanup, and untouched plain paths.
- [ ] Verify the exact branch and commit in every test report. Keep the PR open for review and merge decision; do not merge as part of this plan.
