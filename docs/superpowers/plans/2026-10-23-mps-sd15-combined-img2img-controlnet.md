# MPS SD1.5 LCM Combined img2img and ControlNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans after human plan review. Work task by task. Do not dispatch subagents for this issue. Steps use checkboxes for tracking.

**Goal:** Admit and render SD1.5 LCM img2img with one Canny ControlNet attachment on MPS.

**Architecture:** The existing MPS family binding remains the source of the one-Canny limit. Admission applies that scope to combined requests before preprocessing. The MPS worker builds a Diffusers combined pipeline from the loaded SD1.5 pipe and the inherited ControlNet loader, then uses the current render, progress, metadata, and cache-release paths.

**Tech Stack:** Python 3.12, Torch 2.10 MPS, Diffusers 0.39, OpenCV 5, `python -m tests.run`.

**Source:** `fp context STABL-vkluiibn` at `45a8f9e40c702fbd3967fc8d524046e2dd8a1c36`. Architecture and task order were approved in the 2026-10-23 plan review. This revision incorporates its two blockers and three follow-ups. The issue stays `todo` while the tasks run.

## Decisions and boundaries

- **Control window:** Allow non-default `start_percent` and `end_percent` when `start_percent < end_percent`. At equality, Diffusers 0.39 sets `controlnet_keep=0.0` for every step, so the render completes while silently ignoring the requested ControlNet. `ControlNetAttachment` now rejects equality for plain and combined requests on every backend; this intentionally changes that invalid CUDA input from a render to request validation error. The SD combined pipeline takes `control_guidance_start` and `control_guidance_end`; `CudaWorkerBase._build_controlnet_kwargs` forwards both. HunyuanDiT still rejects non-default windows because its combined pipeline lacks those arguments.
- Keep one Canny attachment on MPS. Do not change CUDA or HunyuanDiT limits. Apply the MPS binding's scope to combined admission. The worker repeats the check before loading weights.
- Pass `image` as the init PIL image and `control_image` as the Canny map. Validate the control map aspect ratio against the native init image before resizing, as the CUDA combined path does.
- Run `_normalize_img2img_modules()` after `from_pipe` and before the combined call. Inject step progress on the combined pipe through the same call as the other MPS paths.
- Retain strict registry validation. Use a temporary host-path registry and a temporary `lcm-general`-only mode configuration for native acceptance. Do not edit `conf/controlnets.yaml`, the existing JPEG, or the unrelated worktree.
- The final report separates patched contract tests, native MPS runs, and CUDA contract tests. Record wall time per render. Record process memory if a repeatable measurement is available; do not claim numeric VRAM from the placeholder MPS registry.

## Review Focus

1. Combined admission must apply the MPS binding scope before preprocessing. A Depth attachment and two Canny attachments must produce the platform messages, while CUDA's unnarrowed binding still admits its existing scope.
2. A zero-width window silently disables ControlNet in Diffusers 0.39. Pin `ControlNetAttachment` validation for equality, including `GenerateRequest` parsing. This rule applies to plain and combined requests on every backend. Keep a non-default positive-width window admitted.
3. `_validate_control_image_aspect_ratio` checks native **content alignment**, not the resized dimensions. Test mismatched native aspect ratios and assert the combined pipe and cache loader were not reached.
4. The combined pipeline takes separate `image` and `control_image` kwargs. The class helper belongs beside the other SD helpers in `cuda_worker.py`, imported into `mps_worker.py` and called unqualified there.
5. A full-suite run after the MPS cell flip must distinguish the eight known baseline failures from any new failure. Cohort runs alone cannot establish this.

## Task 1: Close the combined admission gap before enabling the cell

**Files:** `server/controlnet_execution.py`; `server/controlnet_models.py`; `tests/test_controlnet_execution.py`; `tests/test_controlnet_models.py`; `CONTROLNET.md`; `tests/test_ws_routes.py` (combined refusal coverage is completed in Task 3).

- [x] Add tests named `test_combined_scope_admits_canny`, `test_combined_scope_rejects_depth`, `test_combined_scope_rejects_two_attachments`, `test_combined_scope_allows_sd15_window`. Use `make_family_provider(family_id="sd15", supports_combined=True, controlnet_scope=ControlNetScope(("canny",), 1), backend_id="mps")` and `make_active_snapshot(mode, family_id="sd15")`. Assert one Canny returns `"img2img+controlnet"`. Assert a Depth attachment raises `ValueError` with `backend 'mps' does not support control_type 'depth' for family 'sd15'; allowed: canny`. Assert two Canny attachments raise `ValueError` with `backend 'mps' admits at most 1 controlnet attachment(s) for family 'sd15', got 2`. Assert a Canny attachment with `start_percent=0.2, end_percent=0.8` is admitted. Use `ControlNetAttachment(attachment_id="cn_1", control_type="canny", map_asset_ref="map")` as the base fixture; no asset resolution is needed. Add a CUDA `sd15` combined provider test with Depth and two attachments to prove its unnarrowed scope remains admitted.
- [x] Run `python -m tests.run tests/test_controlnet_execution.py -- -q --no-cov -k combined_scope`. Expected: Depth and two-attachment tests fail at the missing scope check; the Canny, positive-width window, and CUDA scope cases pass.
- [x] In `admit_generation_operation`, after the combined capability check and before the HunyuanDiT restriction, add `_admit_controlnet_scope(req, binding, provider, family_id)`. Do not move or weaken the HunyuanDiT checks.
- [x] In `tests/test_controlnet_models.py`, write a failing `ControlNetAttachment` and `GenerateRequest` test for `start_percent=end_percent=0.5`; both must raise `ValidationError` with `start_percent must be < end_percent`. Change the validator in `server/controlnet_models.py` from `>` to `>=`, and document the all-backend request rule in `CONTROLNET.md`. Run `python -m tests.run tests/test_controlnet_models.py tests/test_controlnet_execution.py -- -q --no-cov`; expect both new tests and the existing admission tests to pass.
- [x] Run `python -m tests.run tests/test_controlnet_execution.py -- -q --no-cov`. Expected: all selected tests pass. The new scope tests pass while the MPS cell is still false because they use the explicit capable binding.

## Task 2: Build and test the MPS combined worker path

**Files:** `backends/cuda_worker.py`; `backends/mps_worker.py`; `tests/test_mps_worker.py`; `tests/test_cuda_worker_controlnet.py`.

- [x] Extend the `fake_pipeline` fixture with a dedicated combined-pipe list and a patched module-global `_sd_controlnet_img2img_pipeline_cls` class helper. The fake `from_pipe(base_pipe, *, controlnet, torch_dtype)` must record the base pipe, ControlNet object, and `torch_dtype`. Patch the helper on `backends.mps_worker`, as the existing fixture patches `_sd_controlnet_pipeline_cls`; this prevents a test from importing real weights.
- [x] Replace `test_mps_worker_rejects_an_init_image_with_controlnet_bindings` with tests that render `_job(init_image=_control_png(), controlnet_bindings=[_binding()])`. Assert the combined fake is called once; `image` and `control_image` are distinct PIL images sized 768x768; `strength` is the request's `denoise_strength`; `controlnet_conditioning_scale`, `control_guidance_start`, and `control_guidance_end` carry the attachment values; no `width` or `height` is passed; the seed and PNG `controlnet` chunk are present; and the cache pin is released. Include a non-default control window (`0.2`, `0.8`).
- [x] Add worker tests for Depth and two attachments with an init image. Each must raise the existing MPS scope `ValueError` before a ControlNet load or pipeline build. Add a native-aspect-ratio mismatch test (for example, 768x768 init and 768x512 control map): expect `_validate_control_image_aspect_ratio`'s `ValueError` before a ControlNet load or combined-pipe build. Add one failing-pipeline test to prove the pin releases, one callback/cancellation test, and one VAE alignment test after a prior upcast. Reuse the existing `fake_pipeline`, `_binding`, `recording_generator`, and cache fixtures; test the call records and cache lists rather than comparing fake image pixels.
- [x] Run `python -m tests.run tests/test_mps_worker.py -- -q --no-cov -k combined`. Expected: the new render tests fail on the current `NotImplementedError`.
- [x] Add the lazy helper beside `_sd_controlnet_pipeline_cls` in `backends/cuda_worker.py`:

  ```python
  def _sd_controlnet_img2img_pipeline_cls() -> Any:
      return _import_attr(
          "diffusers.pipelines.controlnet.pipeline_controlnet_img2img",
          "StableDiffusionControlNetImg2ImgPipeline",
      )
  ```

  Import the helper at module scope into `backends/mps_worker.py`, then call it **unqualified** from `_build_controlnet_img2img_pipe`, which calls `.from_pipe(self.pipe, controlnet=controlnet_obj, torch_dtype=None)`. Follow the existing `_build_controlnet_pipe` shape. Replace the inline SD combined `_import_attr` call in `backends/cuda_worker.py` with this helper call; pin that CUDA path in the existing CUDA regression cohort.
- [x] Import `_validate_control_image_aspect_ratio` from `backends.cuda_worker`. Replace the combined `NotImplementedError` with `_check_controlnet_scope(bindings)` for every request with bindings. In the init-image branch, call `_validate_control_image_aspect_ratio(init_image, bindings)` and `_build_controlnet_kwargs(bindings, (width, height), loaded_ids, image_kwarg="control_image")`. Pop the `controlnet` object, build the combined pipe, resize the init image as the plain img2img path does, call `_normalize_img2img_modules()`, and build `pipe_kwargs` with `image`, `strength`, steps, guidance, generator, and remaining ControlNet kwargs. Let the existing conditioning, `torch.inference_mode()`, `inject_step_progress`, PNG metadata, and `finally` release logic execute once for all paths.
- [x] Run `python -m tests.run tests/test_mps_worker.py -- -q --no-cov`. Expected: all selected tests pass; no Diffusers model is loaded by tests.

## Task 3: Enable the cell and prove admission through real MPSProvider

**Files:** `backends/platforms/mps_bindings.py`; `backends/mps_worker.py`; `tests/test_mps_registration.py`; `tests/test_mps_worker.py`; `tests/test_controlnet_execution.py`; `tests/test_ws_routes.py`.

- [x] Before the cell flip, write a worker test that temporarily changes each execution capability on the real `MPS_FAMILY_BINDINGS["sd15"]` cell. Combined, plain ControlNet, and plain img2img requests must each be refused when their operation flag is false, before a pipeline or cache load. The test failed in all three cases before `_check_platform_operation` and passed afterward. Give all worker test instances `SD15_PROFILE`, as live workers do. With the combined flag still false, the worker's existing combined render tests remain blocked until the cell flip.
- [x] Before the cell change, replace the old combined-rejection test in `tests/test_controlnet_execution.py` with real-`MPSProvider` tests for one Canny admission, Depth refusal, two-attachment refusal, and a non-default Canny window. Extend WebSocket tests at the existing `_mps_ws_submission` and `_mps_ws_rejection` helpers: an uploaded init PNG plus one Canny reaches the dispatched job; combined Depth and two attachments each produce `job:error` with the **exact** platform scope message, and preprocessing is not called. For the admitted case, patch `resolve_controlnet_bindings` and preprocessing as the plain Canny test does; the rejection tests need neither patch. Keep the real `lcm-general` policy stub, which permits Depth and two attachments.
- [x] Update `tests/test_mps_registration.py::test_mps_provider_binds_sd15_and_shares_runtime` to expect `ExecutionCapabilities(True, True, True)`. Run `python -m tests.run tests/test_mps_registration.py tests/test_controlnet_execution.py tests/test_ws_routes.py -- -q --no-cov -k 'mps_sd15 or binds_sd15'`. Expected: combined Canny admission fails until the cell flips; combined refusal tests expose the current capability gate.
- [x] Set the `sd15` binding to `ExecutionCapabilities(True, True, True)` and update the binding/worker comments. Run the same command. Expected: all selected tests pass and the Depth/two-attachment errors come from the platform scope.
- [x] Run `python -m tests.run tests/test_mps_worker.py tests/test_mps_registration.py tests/test_controlnet_execution.py tests/test_ws_routes.py -- -q -p no:cacheprovider --no-cov`. Expected: zero failures in the selected cohorts.
- [x] Run `python -m tests.run tests/ -- -q --no-cov` after the cell flip. Expected baseline: eight existing failures, one in `tests/test_entry_spans.py` and seven in vector-map CAD mesh tests. Record their exact names; investigate any additional failure before Task 4.

## Task 4: Native MPS acceptance and regression evidence

**Files:** Create `spikes/mps_combined_controlnet_acceptance.py`; modify `docs/MPS_BACKEND.md` after live evidence exists.

- [x] Copy the client-only structure of `spikes/mps_controlnet_acceptance.py`. Upload one deterministic init image and two **different** deterministic Canny maps. Submit plain img2img plus combined map A at strength 1.0, combined map B at strength 1.0, and combined map A at strength 0.0. Fix prompt, seed, init image, denoise strength, steps, guidance, and control window across the comparison. Require `job:complete`, valid nonflat 768x768 PNGs, a `controlnet` metadata chunk, and progress frames for combined jobs. Require map A versus B mean pixel distance above `MIN_RENDER_DISTANCE=1.0`, combined A versus plain img2img distance above it, and combined A strength 1 versus 0 distance above it. A valid PNG alone does not pass.
- [x] Submit combined Depth and two-attachment requests with the same uploaded init image. Require `job:error` and equality to the two `SCOPE_MESSAGES` strings in the slice 3 spike. A timeout, generic error, or mode-policy message fails. Emit JSON with hashes, distance metrics, progress-frame counts, wall seconds, platform messages, backend, and isolation mode. Use the prior spike's `render`, `describe`, and upload helpers as a reference; keep this as a separate script.
- [x] On a native Mac with the model and ControlNet assets, set `MPS_MODELS_ROOT` to the absolute host directory containing `diffusers/lcm-dreamshaper-v7`, `diffusers/loras`, and `controlnets/`. Verify each with `test -d`. Create `/tmp/mps-slice4-conf/modes.yml` from `conf/modes.yml` with Python/YAML: retain only `modes['lcm-general']`, set `model_root` to `$MPS_MODELS_ROOT` and `lora_root` to `$MPS_MODELS_ROOT/diffusers/loras`, and leave the policy intact. Create `/tmp/controlnets-host.yaml` from `conf/controlnets.yaml` by replacing the `/models/` prefix on every registry path with `$MPS_MODELS_ROOT/`. Verify strict loading with `CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml python -c "from server.controlnet_registry import load_controlnet_registry as l; r=l(config_path='/tmp/controlnets-host.yaml'); print(sorted(r._specs))"`. Expected: all model IDs print with no missing-path error. Leave `CONTROLNET_REGISTRY_VALIDATION` unset.
- [x] In the `stability-toys` environment, run `BACKEND=mps MODE_CONFIG_PATH=/tmp/mps-slice4-conf STORAGE_PROVIDER=MEMORY CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml WORKER_ISOLATION=inproc python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200`; in another shell run `python spikes/mps_combined_controlnet_acceptance.py --out /tmp/mps_combined_inproc`. Stop the server and repeat both commands with `WORKER_ISOLATION=subprocess` and `--out /tmp/mps_combined_subprocess`. Expected: the script exits 0 in each mode, every signal passes, and the server reports MPS with `lcm-general` loaded.
- [x] Run the MPS contract cohort from `docs/MPS_BACKEND.md` through `python -m tests.run` and run `tests/test_progress_bar_suppression.py` separately as documented. Run the unchanged CUDA cohort documented there through `python -m tests.run`. Expected: no new failures; report exact pass, skip, failure, and environment-limited counts. Do not turn contract evidence into a CUDA live claim.
- [x] Update `docs/MPS_BACKEND.md` with the capability, limits, accepted control-window rule, contract-test result, both live JSON summaries, exact scope messages, map-content comparison, progress counts, and wall times. Remove the outdated combined-rejection claim. Record the commit and environment. Stage named paths only.

**Task 4 verification (2026-10-24):** Both isolation modes passed all nine
signals. In `inproc` the wall times were 6.55 s, 15.47 s, 13.29 s, and 13.71 s.
In `subprocess` they were 6.67 s, 15.36 s, 13.11 s, and 14.13 s. The four PNG
hashes match across both modes and match an earlier interrupted run. The
subprocess server was confirmed by a `multiprocessing.spawn` child process.
Contract evidence at `8fb8dbb`: MPS real 330 passed, stub 78 passed, progress
bar 10 passed, CUDA real 46 passed and 8 skipped, CUDA stub 124 passed.

**Open question carried forward:** at ControlNet strength 0.0 the combined
render is byte-identical across two different maps, so control is inert. It is
not byte-identical to a plain img2img render; the mean pixel distance is 13.57.
A diagnostic showed both pipelines use the same timesteps and the same init
latents, and the ControlNet residuals are exactly zero. The difference is
inside the ControlNet UNet branch. The acceptance rests on the within-pipeline
comparisons, which are clean. `docs/MPS_BACKEND.md` states the measured
behaviour and does not claim the two pipelines are equivalent.

## Review gate

The 2026-10-23 review approved architecture and task order. Execute Task 1 after this plan correction is committed on the slice 4 branch. Keep the issue `todo` until the implementation and live evidence satisfy the issue's Done criteria. Require both native isolation modes and CUDA regression evidence before calling slice 4 done.

**Task 3 verification (2026-10-24):** The selected real-provider and WebSocket tests were red before the cell flip (8 failed, 10 passed) and green after it (18 passed). The four-file focused cohort passed 140 tests. The full suite in the restricted sandbox had 64 real-cohort failures, mostly `/psm_*` permission errors and an OpenSCAD Qt processor error. A focused unsandboxed probe passed 6 tests. The required full suite was then rerun outside the sandbox: 8 known baseline failures, 3,283 real-cohort passes, 12 skips, 226 stub-cohort passes. No additional failure was observed. This is contract evidence; native combined MPS acceptance remains Task 4.
