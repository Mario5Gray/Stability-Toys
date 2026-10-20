# HunyuanDiT img2img Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate HunyuanDiT images from an init image and prompt, with working denoise strength, while continuing to reject init image plus ControlNet.

**Architecture:** A focused Hunyuan img2img module encodes the source with the loaded VAE and runs the Hunyuan denoising loop over the strength-selected suffix of scheduler timesteps. The existing Hunyuan worker supplies request data and conditioning, then writes the PNG. The family binding enables only plain img2img after the path is implemented.

**Tech Stack:** Python 3.12, Diffusers `>=0.39.0`, PyTorch, Pillow, pytest, CUDA `test-cuda` Compose service.

**Spec:** `docs/superpowers/specs/2026-10-20-hunyuandit-img2img-design.md` at `8368c3b`.

---

## File map and constraints

| File | Responsibility |
| --- | --- |
| `backends/hunyuandit_img2img.py` | Strength selection, source VAE encoding/noising, Hunyuan-only prompt and denoising execution; import heavy dependencies only when called. |
| `backends/cuda_worker.py` | Route init-image-only jobs into the new component and stamp PNG metadata; preserve txt2img and ControlNet behavior. |
| `backends/platforms/cuda_bindings.py` | Advertise Hunyuan plain img2img; keep combined capability false. |
| `tests/test_hunyuandit_img2img.py` | Focused CPU tests of timestep/noise math and Hunyuan call contract. |
| `tests/test_hunyuandit_worker.py`, `tests/test_backend_runtimes.py` | Worker routing and capability regression tests. |
| `tests/test_hunyuandit_acceptance.py` | Real CUDA `WorkerPool` img2img artifact and resource evidence. |
| `docs/TESTING_IN_DOCKER.md` | Exact CUDA acceptance command and output evidence. |

Keep `cuda_bindings.py` import-clean. Never construct an SD/SDXL img2img pipeline
for Hunyuan. Reuse loaded modules and preserve fp16; no model reload or implicit
`from_pipe` fp32 recast. Do not add combined img2img+ControlNet, new UI controls,
or an HTTP img2img route. Do not treat CPU tests as image-quality evidence.

Local focused tests use the project Python environment. This machine's base
Diffusers is 0.37, below the project floor; if its Hunyuan import is unusable,
use the test stubs for CPU contract tests and run the real pipeline only in the
CUDA container. `tests.run` isolates the live acceptance module from test stubs.

### Task 1: Strength and scheduler contract

**Files:** Create `backends/hunyuandit_img2img.py`; create
`tests/test_hunyuandit_img2img.py`.

- [ ] **Step 1: Write RED tests** for `select_timesteps(scheduler, steps,
  strength, device)`: 30 steps at strength 0.25 select 7 steps, strength 1.0
  selects all 30, strength 0.01 selects at least one, and zero/negative steps
  fail. A scheduler missing callable `set_timesteps` or `add_noise` must raise
  `ValueError` naming the unsupported scheduler. Test with a small fake whose
  `set_timesteps` records `steps` and populates `timesteps`.

```python
class FakeScheduler:
    order = 1
    def set_timesteps(self, steps, device=None):
        self.timesteps = list(range(steps - 1, -1, -1))
    def add_noise(self, latent, noise, timestep):
        return latent + noise

timesteps, effective_steps = select_timesteps(FakeScheduler(), 30, 0.25, "cpu")
assert timesteps == list(range(6, -1, -1))
assert effective_steps == 7
```

- [ ] **Step 2: Run RED:** `python -m pytest tests/test_hunyuandit_img2img.py -q`.
  Expected: import or missing-function failure.
- [ ] **Step 3: Implement** `select_timesteps`. Validate `0 < strength <= 1`,
  positive steps, callable scheduler methods, and nonempty timesteps. Call
  `scheduler.set_timesteps(steps, device=device)` once. Calculate
  `effective_steps = min(steps, max(1, int(steps * strength)))`, then select
  `scheduler.timesteps[(steps - effective_steps) * scheduler.order:]` and
  reject an
  unexpected or empty schedule. Return the selected timesteps and effective
  step count; avoid mutating `scheduler.timesteps` outside `set_timesteps`.
- [ ] **Step 4: Run GREEN:** same command; expected all new tests pass.
- [ ] **Step 5: Commit:** stage only the two named files and commit
  `feat: select Hunyuan img2img scheduler steps`.

### Task 2: Encode and noise the source image

**Files:** Modify `backends/hunyuandit_img2img.py`; modify
`tests/test_hunyuandit_img2img.py`.

- [ ] **Step 1: Write RED tests** for `prepare_source_latents(pipe, image,
  *, width, height, timestep, generator, dtype, device)`. Use a tiny fake VAE and image
  processor. Assert RGB preprocessing at the requested dimensions, VAE encode
  exactly once, multiplication by `vae.config.scaling_factor`, one seeded
  `randn_tensor` call, `scheduler.add_noise` at the first selected timestep,
  and target device/dtype. Repeat with the same generator seed and assert the
  noisy latent is equal; change the seed and assert it differs. Reject invalid
  dimensions before VAE encode.
- [ ] **Step 2: Run RED:** `python -m pytest tests/test_hunyuandit_img2img.py -q`.
  Expected: new source-latent test fails.
- [ ] **Step 3: Implement** source preprocessing with
  `pipe.image_processor.preprocess(image, height=height, width=width)`, move
  the image tensor to the loaded VAE's device/dtype, encode and sample the
  latent distribution using the request generator, multiply by the VAE scale,
  create matching noise with Diffusers `randn_tensor`, and call
  `scheduler.add_noise(source_latent, noise, timestep_batch)`. Keep all tensor
  imports inside this module's called functions so other backends stay lazy.
- [ ] **Step 4: Run GREEN:** same command; expected the source-latent tests pass.
- [ ] **Step 5: Commit:** stage only the two named files and commit
  `feat: encode and noise Hunyuan img2img source`.

### Task 3: Hunyuan denoising component

**Files:** Modify `backends/hunyuandit_img2img.py`; modify
`tests/test_hunyuandit_img2img.py`.

- [ ] **Step 1: Write RED tests** for `run_hunyuan_img2img(pipe, image,
  conditioning, *, width, height, steps, strength, guidance_scale, generator,
  progress, should_cancel)`. With fake BERT/mT5 encoders, transformer,
  scheduler, and VAE, assert that both prompt encodings and their attention
  masks reach the transformer; rotary embeddings, `image_meta_size`, and
  `style` match the Diffusers 0.39 Hunyuan text-to-image call; only selected
  timesteps execute; classifier-free guidance splits the predicted variance
  channels before scheduler step; the VAE decodes scaled latents; and the
  return value contains one PIL image. Assert cancellation raises
  `concurrent.futures.CancelledError` at a step boundary and no later step runs.
- [ ] **Step 2: Run RED:** `python -m pytest tests/test_hunyuandit_img2img.py -q`.
  Expected: the new denoising test fails.
- [ ] **Step 3: Implement** a standalone Hunyuan-only loop using the installed
  `HunyuanDiTPipeline.__call__` in Diffusers 0.39 as the exact call-contract
  reference. Use `pipe.encode_prompt` twice (encoder indexes 0 and 1,
  sequence lengths 77 and 256), then Task 1 timesteps and Task 2 source
  latent. Build Hunyuan rotary embeddings with the same grid crop helper and
  DiT patch dimensions as the reference. For each selected timestep:
  scale the latent input, call `pipe.transformer` with the two embeddings,
  masks, image metadata, style, and rotary embedding; split predicted noise
  from variance; apply guidance; call `scheduler.step`; check cancellation and
  emit progress using the selected step count. Decode with
  `pipe.vae.decode(latents / pipe.vae.config.scaling_factor)`, use the
  pipeline's safety check and image postprocessor, and free hooks as the
  reference does. Do not substitute a generic SD pipeline or call Hunyuan
  text-to-image with noisy latents for a full schedule.
- [ ] **Step 4: Run GREEN:** same command; expected all focused tests pass.
- [ ] **Step 5: Commit:** stage only the two named files and commit
  `feat: run Hunyuan img2img denoising`.

### Task 4: Worker route and admission

**Files:** Modify `backends/cuda_worker.py`,
`backends/platforms/cuda_bindings.py`, `tests/test_hunyuandit_worker.py`,
`tests/test_backend_runtimes.py`.

- [ ] **Step 1: Write RED tests.** Replace the old Hunyuan init-image
  rejection assertion with a plain img2img call assertion using a patched
  `run_hunyuan_img2img` that returns one PIL image. Verify decoded RGB input and requested size,
  strength default 0.75, selected scheduler, generator seed, delegated
  conditioning, PNG `lcm` metadata including strength, and no ControlNet
  composition. Assert combined init image and bindings fail in the worker as
  defense in depth. Update capability tests to expect `(True, True, False)`
  and admission to return `img2img` for a Hunyuan init-only request.
- [ ] **Step 2: Run RED:**
  `python -m pytest tests/test_hunyuandit_worker.py tests/test_backend_runtimes.py -q`.
  Expected: new route/capability assertions fail.
- [ ] **Step 3: Implement** the init-only branch in Hunyuan `run_job` after
  scheduler selection. Decode bytes via Pillow with an explicit invalid-image
  error, convert to RGB, resize to request size, obtain the existing
  conditioning artifact, call `_accept_conditioning_artifact` on the loaded
  Hunyuan pipeline, and pass those values to `run_hunyuan_img2img`. Use the
  same seed/generator and PNG writer as text-to-image. Record
  `denoise_strength` in `lcm` metadata for this branch. Reject combined jobs
  before model or image work. Set only Hunyuan plain img2img capability true.
- [ ] **Step 4: Run GREEN:** same command plus
  `python -m pytest tests/test_hunyuandit_img2img.py tests/test_model_routes.py -q`.
  Expected: all focused tests pass.
- [ ] **Step 5: Commit:** stage only the four named files and commit
  `feat: route Hunyuan init image jobs`.

### Task 5: Live CUDA acceptance and user instructions

**Files:** Modify `tests/test_hunyuandit_acceptance.py` and
`docs/TESTING_IN_DOCKER.md`.

- [ ] **Step 1: Add a separate live img2img acceptance test.** Follow the
  existing `WorkerPool` setup and collection isolation. Use a committed small
  generated PNG source fixture or construct a simple RGB source in the test;
  keep it distinct from the Canny map. Generate a 1024x1024 Hunyuan img2img
  output at strengths 0.25 and 0.75 with the same prompt and seed. Assert
  capability `(True, True, False)`, plain admission succeeds, combined
  admission still fails, output dimensions/seed/strength metadata, and no
  blocked ignored-input warning. Save source and both outputs in
  `ACCEPTANCE_OUT_DIR`, print runtime and peak allocated VRAM. After the run,
  free VRAM and verify unload as the existing test does.
- [ ] **Step 2: Run the collection-safe local check:**
  `python -m tests.run tests/test_hunyuandit_acceptance.py -- -q`.
  Expected on non-CUDA hosts: clean skip, not an import-stub failure.
- [ ] **Step 3: Document and run the real CUDA command** on the NVIDIA host:

```bash
docker compose -f docker-compose.test.yml build test-cuda
docker compose -f docker-compose.test.yml run --rm test-cuda \
  python -m tests.run tests/test_hunyuandit_acceptance.py -- -q
```

  Expected: both existing ControlNet and new img2img acceptance pass with
  source/output paths, elapsed seconds, peak allocated bytes, and no ignored
  Hunyuan input warnings. Review both outputs against the source for visible
  source influence, especially at low strength. Record exact versions,
  command, artifacts, and any unavailable live gate in the implementation
  report; a local skip is not a pass.
- [ ] **Step 4: Commit:** stage only these two files and commit
  `test: accept Hunyuan img2img on CUDA`.

### Task 6: Final regression and report

**Files:** No additional files unless a concrete regression requires a fix.

- [ ] **Step 1: Run focused regression:**

```bash
python -m pytest tests/test_hunyuandit_img2img.py \
  tests/test_hunyuandit_worker.py tests/test_backend_runtimes.py \
  tests/test_model_routes.py tests/test_controlnet_execution.py -q
git diff --check
git status --short
```

  Expected: focused tests pass, no whitespace errors, only intended files
  changed.
- [ ] **Step 2: Report** which tests ran locally, whether the live CUDA gate
  ran, artifact locations, VRAM, visual observations, and remaining limitations.
  Keep an unrun or skipped CUDA gate explicit; do not claim generated-image
  quality from unit tests.
