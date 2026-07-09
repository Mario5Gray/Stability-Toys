# img2img + ControlNet — Pipeline Wiring (Parallel Group B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project policy forbids subagent-driven development — execute inline.

**Goal:** Replace the `NotImplementedError` guards in `backends/cuda_worker.py` with real combined img2img+ControlNet execution for both SD1.5 and SDXL, using the design decisions from Group A, and make Group A's WS fail-fast guard capability-gated so the new execution is actually reachable once it ships.

**Architecture:** Mirror the existing txt2img+ControlNet pattern (`_build_controlnet_pipe` + `.from_pipe(self.pipe, controlnet=...)`, built fresh per request) rather than the separately-cached plain-img2img pipeline (`self._img2img_pipe`) — this keeps the "zero extra base-model VRAM" property the txt2img ControlNet path already has, and avoids extending `_normalize_img2img_modules()` (which exists specifically to fix dtype/VAE drift on the long-lived cached `_img2img_pipe`, not needed for a freshly-wrapped pipe). The critical fix is keeping the init image and the ControlNet conditioning map as distinct diffusers kwargs (`image=` vs `control_image=`) instead of letting a naive `**controlnet_kwargs` merge collide with the img2img branch's own `image=` key.

**Tech Stack:** Python (diffusers, PyTorch), pytest with fully-stubbed diffusers modules (matches `tests/test_cuda_worker_controlnet.py`'s existing pattern).

**Prerequisite:** Both tasks depend on `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md` (Group A, Tasks 3-4) and the `supports_img2img_and_controlnet` capability field (Group A, Task 2) already being merged.

**FP tree:** `STABL-ztaxgbhv` (parent). This plan covers `STABL-vgbxamoz` (SD1.5) and `STABL-umvdwgsm` (SDXL), plus a Task 4 fix (no dedicated FP issue) closing a blocker found in human review of this plan and the groundwork plan together: without it, Group A's fail-fast guard keeps rejecting combined requests forever, even after this plan ships CUDA execution.

## Global Constraints

- `_CONTROL_IMAGE_KWARG` (`backends/cuda_worker.py:87`, `"image"`) stays unchanged for the *txt2img* ControlNet path. The combined path must NOT reuse it as-is — it must emit `control_image` for the conditioning map(s) so it doesn't collide with the combined pipeline's `image=` (the init image). Confirmed real collision risk, not speculative — see FP comments on `STABL-vgbxamoz`/`STABL-umvdwgsm`.
- Aspect-ratio validation (2% tolerance, per `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md` Decision 2) runs before either image is resized, and raises `ValueError` naming the offending `attachment_id`.
- `start_percent`/`end_percent` pass straight through to `control_guidance_start`/`control_guidance_end` unmodified by `denoise_strength` (Decision 1, same doc) — no renormalization logic.
- No new backend support, cache architecture, asset-store, or public schema changes (FP guardrails on both issues).
- Once combined execution works end-to-end for both families, flip `backends/platforms/cuda.py:67`'s `BackendCapabilities` call to report `supports_img2img_and_controlnet=True`.
- Run tests from repo root with the Miniforge base env active: `pytest tests/test_cuda_worker_controlnet.py -v`.

## File Structure

- **Modify `backends/cuda_worker.py`:**
  - `_build_controlnet_kwargs` gains an `image_kwarg` override parameter.
  - New shared helper `_validate_control_image_aspect_ratio` (module-level function).
  - SD1.5 `DiffusersCudaWorker.run_job` (~line 511-613): replace the `NotImplementedError` branch with real combined execution.
  - SDXL `SDXLCudaWorker.run_job` (~line 835-945): same, mirrored.
  - `backends/platforms/cuda.py:67`: flip the capability flag once both branches pass.
- **Modify `server/ws_routes.py`:** add `_supports_img2img_and_controlnet` helper; wire it into the `reject_combined_img2img_controlnet` call site so the guard added in Group A stops being hardcoded to `False` (Task 4).
- **Test:** `tests/test_cuda_worker_controlnet.py` (extend the existing diffusers-stub scaffolding with combined-pipeline fakes), `tests/test_ws_routes.py` (Task 4).

---

### Task 1: Shared helpers — `image_kwarg` override + aspect-ratio validation

**Files:**
- Modify: `backends/cuda_worker.py`
- Test: `tests/test_cuda_worker_controlnet.py`

**Interfaces:**
- Produces:
  - `_build_controlnet_kwargs(self, bindings, size, loaded_ids, image_kwarg: str | None = None) -> dict[str, Any]` — when `image_kwarg` is given, uses it instead of `self._CONTROL_IMAGE_KWARG` as the dict key for the control map(s).
  - `_validate_control_image_aspect_ratio(init_image_bytes: bytes, bindings: list[Any], *, tolerance: float = 0.02) -> None` (module-level function) — raises `ValueError` naming the first offending `attachment_id` if any binding's native aspect ratio diverges from the init image's native aspect ratio by more than `tolerance`.
- Consumes: `ControlNetBinding.attachment_id`, `.control_image_bytes` (`server/controlnet_execution.py:18-26`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cuda_worker_controlnet.py` (after the existing imports/stubs, alongside the other test functions):

```python
def _make_png_bytes(width: int, height: int) -> bytes:
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_build_controlnet_kwargs_uses_image_kwarg_override():
    from backends.cuda_worker import DiffusersCudaWorker

    worker = DiffusersCudaWorker.__new__(DiffusersCudaWorker)
    binding = SimpleNamespace(
        model_id="sdxl-canny",
        model_path="/models/controlnets/sdxl-canny",
        control_image_bytes=_make_png_bytes(64, 64),
        strength=0.8,
        start_percent=0.0,
        end_percent=1.0,
    )
    with patch.object(DiffusersCudaWorker, "_load_controlnet_model", return_value=Mock()):
        kwargs = worker._build_controlnet_kwargs(
            [binding], (64, 64), [], image_kwarg="control_image"
        )
    assert "control_image" in kwargs
    assert "image" not in kwargs


def test_build_controlnet_kwargs_defaults_to_class_kwarg_when_no_override():
    from backends.cuda_worker import DiffusersCudaWorker

    worker = DiffusersCudaWorker.__new__(DiffusersCudaWorker)
    binding = SimpleNamespace(
        model_id="sdxl-canny",
        model_path="/models/controlnets/sdxl-canny",
        control_image_bytes=_make_png_bytes(64, 64),
        strength=0.8,
        start_percent=0.0,
        end_percent=1.0,
    )
    with patch.object(DiffusersCudaWorker, "_load_controlnet_model", return_value=Mock()):
        kwargs = worker._build_controlnet_kwargs([binding], (64, 64), [])
    assert "image" in kwargs
    assert "control_image" not in kwargs


def test_validate_control_image_aspect_ratio_passes_within_tolerance():
    from backends.cuda_worker import _validate_control_image_aspect_ratio

    init_bytes = _make_png_bytes(1024, 768)  # ratio 1.333
    binding = SimpleNamespace(attachment_id="cn_1", control_image_bytes=_make_png_bytes(1000, 750))  # ratio 1.333
    _validate_control_image_aspect_ratio(init_bytes, [binding])  # must not raise


def test_validate_control_image_aspect_ratio_rejects_beyond_tolerance():
    from backends.cuda_worker import _validate_control_image_aspect_ratio

    init_bytes = _make_png_bytes(1024, 768)  # ratio 1.333
    binding = SimpleNamespace(attachment_id="cn_1", control_image_bytes=_make_png_bytes(512, 512))  # ratio 1.0
    with pytest.raises(ValueError, match="cn_1"):
        _validate_control_image_aspect_ratio(init_bytes, [binding])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cuda_worker_controlnet.py -k "image_kwarg or aspect_ratio" -v`
Expected: FAIL — `TypeError: _build_controlnet_kwargs() got an unexpected keyword argument 'image_kwarg'` and `ImportError: cannot import name '_validate_control_image_aspect_ratio'`.

- [ ] **Step 3: Implement `image_kwarg` override**

In `backends/cuda_worker.py`, replace:

```python
    def _build_controlnet_kwargs(
        self, bindings: list[Any], size: tuple[int, int], loaded_ids: list[str]
    ) -> dict[str, Any]:
        """Assemble the ControlNet pipeline kwargs from resolved bindings.

        Shared across families: the only per-family variance is the control-map
        kwarg name (self._CONTROL_IMAGE_KWARG). Each value is single-or-list to
        match diffusers' single-vs-multi-ControlNet signature.

        Appends each loaded model_id to the caller's `loaded_ids` *as it pins*,
        so a mid-loop load failure still leaves the already-pinned models visible
        to the caller's finally-block cleanup (no cache/VRAM leak on partial load).
        """
        controlnets: list[Any] = []
        images: list[Image.Image] = []
        scales: list[float] = []
        starts: list[float] = []
        ends: list[float] = []
        for binding in bindings:
            controlnets.append(self._load_controlnet_model(binding))
            loaded_ids.append(binding.model_id)
            images.append(_decode_control_image(binding.control_image_bytes, size))
            scales.append(binding.strength)
            starts.append(binding.start_percent)
            ends.append(binding.end_percent)
        return {
            "controlnet": controlnets[0] if len(controlnets) == 1 else controlnets,
            self._CONTROL_IMAGE_KWARG: images[0] if len(images) == 1 else images,
            "controlnet_conditioning_scale": scales[0] if len(scales) == 1 else scales,
            "control_guidance_start": starts[0] if len(starts) == 1 else starts,
            "control_guidance_end": ends[0] if len(ends) == 1 else ends,
        }
```

with:

```python
    def _build_controlnet_kwargs(
        self,
        bindings: list[Any],
        size: tuple[int, int],
        loaded_ids: list[str],
        image_kwarg: str | None = None,
    ) -> dict[str, Any]:
        """Assemble the ControlNet pipeline kwargs from resolved bindings.

        Shared across families: the only per-family variance is the control-map
        kwarg name (self._CONTROL_IMAGE_KWARG, or the `image_kwarg` override below).
        Each value is single-or-list to match diffusers' single-vs-multi-ControlNet
        signature.

        `image_kwarg` overrides `self._CONTROL_IMAGE_KWARG` for the control-map dict
        key. The combined img2img+ControlNet path passes `image_kwarg="control_image"`
        because the combined pipeline's `image=` kwarg is the init image, not the
        control map — reusing `_CONTROL_IMAGE_KWARG` unchanged there would silently
        overwrite the init image with the control map (or vice versa, depending on
        kwarg merge order).

        Appends each loaded model_id to the caller's `loaded_ids` *as it pins*,
        so a mid-loop load failure still leaves the already-pinned models visible
        to the caller's finally-block cleanup (no cache/VRAM leak on partial load).
        """
        controlnets: list[Any] = []
        images: list[Image.Image] = []
        scales: list[float] = []
        starts: list[float] = []
        ends: list[float] = []
        for binding in bindings:
            controlnets.append(self._load_controlnet_model(binding))
            loaded_ids.append(binding.model_id)
            images.append(_decode_control_image(binding.control_image_bytes, size))
            scales.append(binding.strength)
            starts.append(binding.start_percent)
            ends.append(binding.end_percent)
        key = image_kwarg if image_kwarg is not None else self._CONTROL_IMAGE_KWARG
        return {
            "controlnet": controlnets[0] if len(controlnets) == 1 else controlnets,
            key: images[0] if len(images) == 1 else images,
            "controlnet_conditioning_scale": scales[0] if len(scales) == 1 else scales,
            "control_guidance_start": starts[0] if len(starts) == 1 else starts,
            "control_guidance_end": ends[0] if len(ends) == 1 else ends,
        }
```

- [ ] **Step 4: Implement the aspect-ratio validator**

In `backends/cuda_worker.py`, add after `_decode_control_image` (module-level, before `class CudaWorkerBase`):

```python
def _native_aspect_ratio(data: bytes) -> float:
    image = Image.open(io.BytesIO(data))
    width, height = image.size
    return width / height


def _validate_control_image_aspect_ratio(
    init_image_bytes: bytes, bindings: list[Any], *, tolerance: float = 0.02
) -> None:
    """Reject a combined request when a control map's native aspect ratio diverges
    from the init image's native aspect ratio by more than `tolerance`.

    Both images get force-resized to the request size regardless (see
    _decode_control_image / the img2img resize in run_job), so this isn't a
    dimension-mismatch check — it's a content-alignment check: a control map with a
    different native aspect ratio than the init image gets stretched differently
    than the init image, so ControlNet's spatial conditioning (e.g. canny edges) no
    longer lines up with the init image's content once both are force-resized.
    See docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md
    Decision 2.
    """
    init_ratio = _native_aspect_ratio(init_image_bytes)
    for binding in bindings:
        control_ratio = _native_aspect_ratio(binding.control_image_bytes)
        if abs(control_ratio - init_ratio) / init_ratio > tolerance:
            raise ValueError(
                f"controlnet attachment '{binding.attachment_id}' aspect ratio "
                f"{control_ratio:.2f} diverges from init image aspect ratio "
                f"{init_ratio:.2f} by more than {tolerance:.0%}"
            )
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_cuda_worker_controlnet.py -v`
Expected: PASS — all tests including the four new ones. (This step needs a real `PIL.Image` — confirm `Pillow` is on the test path; it already is, since `backends/cuda_worker.py` imports `from PIL import Image, PngImagePlugin` unstubbed at module scope today.)

- [ ] **Step 6: Commit**

```bash
git add backends/cuda_worker.py tests/test_cuda_worker_controlnet.py
git commit -m "feat(controlnet): add image_kwarg override + aspect-ratio validation shared by SD1.5/SDXL combined path (STABL-vgbxamoz, STABL-umvdwgsm) — next: SD1.5 combined execution"
```

---

### Task 2: SD1.5 combined execution

**Files:**
- Modify: `backends/cuda_worker.py:511-579` (`DiffusersCudaWorker.run_job`)
- Test: `tests/test_cuda_worker_controlnet.py`

**Interfaces:**
- Consumes: `_build_controlnet_kwargs(..., image_kwarg="control_image")`, `_validate_control_image_aspect_ratio` (Task 1).
- Produces: combined execution — no new public interface, `run_job`'s existing `(bytes, int)` return contract is unchanged.

- [ ] **Step 1: Add the combined-pipeline fake to the test stub scaffolding**

In `tests/test_cuda_worker_controlnet.py`, near the other fake pipeline classes, add. `.from_pipe` returns a *new* instance each call, so the fake appends each one to a module-level list the test clears at its own start — instances must not leak across tests regardless of run order:

```python
_COMBINED_PIPE_INSTANCES: list = []


class _FakeStableDiffusionControlNetImg2ImgPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet):
        instance = cls()
        _COMBINED_PIPE_INSTANCES.append(instance)
        return instance


sys.modules["diffusers"].StableDiffusionControlNetImg2ImgPipeline = _FakeStableDiffusionControlNetImg2ImgPipeline
```

(add the `class` definition alongside `_FakeStableDiffusionControlNetPipeline` and the `sys.modules[...] = ...` line alongside the other `sys.modules["diffusers"].StableDiffusion...` assignments.)

- [ ] **Step 2: Write the failing test**

Find the existing test that exercises the SD1.5 txt2img+ControlNet path (mirror its binding/mode fixtures) and add:

```python
def test_sd15_combined_img2img_controlnet_keeps_init_image_and_control_map_distinct():
    from backends.cuda_worker import DiffusersCudaWorker

    _COMBINED_PIPE_INSTANCES.clear()  # tests run in the same process; don't inherit state

    worker = DiffusersCudaWorker.__new__(DiffusersCudaWorker)
    worker.device = "cuda:0"
    worker.dtype = "fp16_sentinel"
    worker._img2img_pipe = None
    worker._style_loaded = {}
    worker.pipe = _FakeStableDiffusionPipeline()

    binding = SimpleNamespace(
        attachment_id="cn_1",
        model_id="sd15-canny",
        model_path="/models/controlnets/sd15-canny",
        control_image_bytes=_make_png_bytes(512, 512),
        strength=0.8,
        start_percent=0.0,
        end_percent=1.0,
    )
    job = SimpleNamespace(
        req=SimpleNamespace(
            prompt="a cat",
            negative_prompt=None,
            size="512x512",
            seed=42,
            style_lora=None,
            scheduler_id=None,
            num_inference_steps=20,
            guidance_scale=7.5,
            denoise_strength=0.6,
        ),
        init_image=_make_png_bytes(512, 512),
        controlnet_bindings=[binding],
    )

    with patch.object(DiffusersCudaWorker, "_apply_style", return_value=None), \
            patch.object(DiffusersCudaWorker, "_apply_request_scheduler", return_value=None), \
            patch.object(DiffusersCudaWorker, "_load_controlnet_model", return_value=Mock()):
        png_bytes, seed = worker.run_job(job)

    assert seed == 42
    assert len(png_bytes) > 0
    # The combined pipeline was invoked exactly once, with distinct init-image and
    # control-image kwargs — this is the regression guard for the kwarg collision.
    assert len(_COMBINED_PIPE_INSTANCES) == 1
    call_kwargs = _COMBINED_PIPE_INSTANCES[0].calls[0]
    assert call_kwargs["image"] is not call_kwargs["control_image"]
    assert call_kwargs["strength"] == 0.6
    assert call_kwargs["control_guidance_start"] == 0.0
    assert call_kwargs["control_guidance_end"] == 1.0
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_cuda_worker_controlnet.py -k sd15_combined -v`
Expected: FAIL — `NotImplementedError: ControlNet bindings on the img2img path are not supported in v1.`

- [ ] **Step 4: Implement the combined branch**

In `backends/cuda_worker.py`, in `DiffusersCudaWorker.run_job` (SD1.5), replace:

```python
        bindings = getattr(job, "controlnet_bindings", []) or []
        loaded_ids: list[str] = []
        controlnet_kwargs: dict[str, Any] = {}

        out = None
        try:
            if init_image is not None and bindings:
                raise NotImplementedError(
                    "ControlNet bindings on the img2img path are not supported in v1."
                )

            if bindings:
                controlnet_kwargs = self._build_controlnet_kwargs(
                    bindings, (width, height), loaded_ids
                )

            if init_image is not None:
                # img2img path: reuse loaded weights at zero extra VRAM cost
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize((width, height))
                if self._img2img_pipe is None:
                    self._img2img_pipe = _sd_img2img_pipeline_cls()(**self.pipe.components)
                self._normalize_img2img_modules()
                denoise_strength = float(getattr(req, 'denoise_strength', 0.75))
                pipe_kwargs = {
                    "prompt": req.prompt,
                    "negative_prompt": getattr(req, "negative_prompt", None),
                    "image": init_pil,
                    "strength": denoise_strength,
                    "num_inference_steps": int(req.num_inference_steps),
                    "guidance_scale": float(req.guidance_scale),
                    "generator": gen,
                    **controlnet_kwargs,
                }
                with torch.inference_mode():
                    out = self._img2img_pipe(**pipe_kwargs)
            else:
                pipe_kwargs = {
```

with:

```python
        bindings = getattr(job, "controlnet_bindings", []) or []
        loaded_ids: list[str] = []
        controlnet_kwargs: dict[str, Any] = {}

        out = None
        try:
            if init_image is not None and bindings:
                _validate_control_image_aspect_ratio(init_image, bindings)
                controlnet_kwargs = self._build_controlnet_kwargs(
                    bindings, (width, height), loaded_ids, image_kwarg="control_image"
                )
                controlnet_obj = controlnet_kwargs.pop("controlnet")
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize((width, height))
                combined_pipe = _import_attr(
                    "diffusers", "StableDiffusionControlNetImg2ImgPipeline"
                ).from_pipe(self.pipe, controlnet=controlnet_obj)
                denoise_strength = float(getattr(req, 'denoise_strength', 0.75))
                pipe_kwargs = {
                    "prompt": req.prompt,
                    "negative_prompt": getattr(req, "negative_prompt", None),
                    "image": init_pil,
                    "strength": denoise_strength,
                    "num_inference_steps": int(req.num_inference_steps),
                    "guidance_scale": float(req.guidance_scale),
                    "generator": gen,
                    **controlnet_kwargs,
                }
                with torch.inference_mode():
                    out = combined_pipe(**pipe_kwargs)
            elif init_image is not None:
                # img2img path: reuse loaded weights at zero extra VRAM cost
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize((width, height))
                if self._img2img_pipe is None:
                    self._img2img_pipe = _sd_img2img_pipeline_cls()(**self.pipe.components)
                self._normalize_img2img_modules()
                denoise_strength = float(getattr(req, 'denoise_strength', 0.75))
                pipe_kwargs = {
                    "prompt": req.prompt,
                    "negative_prompt": getattr(req, "negative_prompt", None),
                    "image": init_pil,
                    "strength": denoise_strength,
                    "num_inference_steps": int(req.num_inference_steps),
                    "guidance_scale": float(req.guidance_scale),
                    "generator": gen,
                }
                with torch.inference_mode():
                    out = self._img2img_pipe(**pipe_kwargs)
            else:
                if bindings:
                    controlnet_kwargs = self._build_controlnet_kwargs(
                        bindings, (width, height), loaded_ids
                    )
                pipe_kwargs = {
```

Note: the plain img2img branch (`elif init_image is not None:`) no longer merges `**controlnet_kwargs` — `bindings` is guaranteed empty there (the combined case is handled above), so `controlnet_kwargs` would always be `{}` anyway; dropping the merge just makes that explicit instead of accidental.

The `else:` branch (pure txt2img, possibly with ControlNet) now builds `controlnet_kwargs` itself instead of relying on the unconditional `if bindings:` that used to run before the three-way split — everything downstream of `pipe_kwargs = {` (the existing `**controlnet_kwargs` merge, the `pipe_kwargs.pop("controlnet", None)` / `_build_controlnet_pipe` swap) is unchanged.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_cuda_worker_controlnet.py -v`
Expected: PASS — all tests, including the new combined-path test and every existing SD1.5 txt2img/img2img/ControlNet test (the three-way branch preserves their exact code paths).

- [ ] **Step 6: Commit**

```bash
git add backends/cuda_worker.py tests/test_cuda_worker_controlnet.py
git commit -m "feat(controlnet): implement SD1.5 combined img2img+ControlNet execution via from_pipe (STABL-vgbxamoz) — next: SDXL combined execution"
```

---

### Task 3: SDXL combined execution

**Files:**
- Modify: `backends/cuda_worker.py:835-901` (`SDXLCudaWorker.run_job`)
- Test: `tests/test_cuda_worker_controlnet.py`

**Interfaces:**
- Consumes: same as Task 2, SDXL variant.
- Produces: same contract, SDXL worker.

- [ ] **Step 1: Add the SDXL combined-pipeline fake**

In `tests/test_cuda_worker_controlnet.py`, alongside `_FakeStableDiffusionControlNetImg2ImgPipeline` from Task 2:

```python
_COMBINED_XL_PIPE_INSTANCES: list = []


class _FakeStableDiffusionXLControlNetImg2ImgPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet):
        instance = cls()
        _COMBINED_XL_PIPE_INSTANCES.append(instance)
        return instance


sys.modules["diffusers"].StableDiffusionXLControlNetImg2ImgPipeline = _FakeStableDiffusionXLControlNetImg2ImgPipeline
```

- [ ] **Step 2: Write the failing test**

Add, mirroring Task 2's SD1.5 test but for `SDXLCudaWorker`:

```python
def test_sdxl_combined_img2img_controlnet_keeps_init_image_and_control_map_distinct():
    from backends.cuda_worker import SDXLCudaWorker

    _COMBINED_XL_PIPE_INSTANCES.clear()  # tests run in the same process; don't inherit state

    worker = SDXLCudaWorker.__new__(SDXLCudaWorker)
    worker.device = "cuda:0"
    worker.dtype = "fp16_sentinel"
    worker._img2img_pipe = None
    worker._style_loaded = {}
    worker.pipe = _FakeStableDiffusionXLPipeline()

    binding = SimpleNamespace(
        attachment_id="cn_1",
        model_id="sdxl-canny",
        model_path="/models/controlnets/sdxl-canny",
        control_image_bytes=_make_png_bytes(1024, 1024),
        strength=0.8,
        start_percent=0.0,
        end_percent=1.0,
    )
    job = SimpleNamespace(
        req=SimpleNamespace(
            prompt="a cat",
            negative_prompt=None,
            size="1024x1024",
            seed=42,
            style_lora=None,
            scheduler_id=None,
            num_inference_steps=20,
            guidance_scale=7.5,
            denoise_strength=0.6,
        ),
        init_image=_make_png_bytes(1024, 1024),
        controlnet_bindings=[binding],
    )

    with patch.object(SDXLCudaWorker, "_apply_style", return_value=None), \
            patch.object(SDXLCudaWorker, "_apply_request_scheduler", return_value=None), \
            patch.object(SDXLCudaWorker, "_load_controlnet_model", return_value=Mock()):
        png_bytes, seed = worker.run_job(job)

    assert seed == 42
    assert len(png_bytes) > 0
    assert len(_COMBINED_XL_PIPE_INSTANCES) == 1
    call_kwargs = _COMBINED_XL_PIPE_INSTANCES[0].calls[0]
    assert call_kwargs["image"] is not call_kwargs["control_image"]
    assert call_kwargs["strength"] == 0.6
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_cuda_worker_controlnet.py -k sdxl_combined -v`
Expected: FAIL — `NotImplementedError: ControlNet bindings on the img2img path are not supported in v1.`

- [ ] **Step 4: Implement the combined branch**

In `backends/cuda_worker.py`, in `SDXLCudaWorker.run_job`, replace:

```python
        bindings = getattr(job, "controlnet_bindings", []) or []
        loaded_ids: list[str] = []
        controlnet_kwargs: dict[str, Any] = {}

        out = None
        try:
            if init_image is not None and bindings:
                raise NotImplementedError(
                    "ControlNet bindings on the img2img path are not supported in v1."
                )

            if bindings:
                controlnet_kwargs = self._build_controlnet_kwargs(
                    bindings, (width, height), loaded_ids
                )

            if init_image is not None:
                # img2img path: reuse loaded weights at zero extra VRAM cost
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize((width, height))
                if self._img2img_pipe is None:
                    self._img2img_pipe = _sdxl_img2img_pipeline_cls()(**self.pipe.components)
                self._normalize_img2img_modules()
                denoise_strength = float(getattr(req, 'denoise_strength', 0.75))
                pipe_kwargs = {
                    "prompt": req.prompt,
                    "negative_prompt": getattr(req, "negative_prompt", None),
                    "image": init_pil,
                    "strength": denoise_strength,
                    "num_inference_steps": int(req.num_inference_steps),
                    "guidance_scale": float(req.guidance_scale),
                    "generator": gen,
                    **controlnet_kwargs,
                }
                with torch.inference_mode():
                    out = self._img2img_pipe(**pipe_kwargs)
            else:
                pipe_kwargs = {
```

with:

```python
        bindings = getattr(job, "controlnet_bindings", []) or []
        loaded_ids: list[str] = []
        controlnet_kwargs: dict[str, Any] = {}

        out = None
        try:
            if init_image is not None and bindings:
                _validate_control_image_aspect_ratio(init_image, bindings)
                controlnet_kwargs = self._build_controlnet_kwargs(
                    bindings, (width, height), loaded_ids, image_kwarg="control_image"
                )
                controlnet_obj = controlnet_kwargs.pop("controlnet")
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize((width, height))
                combined_pipe = _import_attr(
                    "diffusers", "StableDiffusionXLControlNetImg2ImgPipeline"
                ).from_pipe(self.pipe, controlnet=controlnet_obj)
                denoise_strength = float(getattr(req, 'denoise_strength', 0.75))
                pipe_kwargs = {
                    "prompt": req.prompt,
                    "negative_prompt": getattr(req, "negative_prompt", None),
                    "image": init_pil,
                    "strength": denoise_strength,
                    "num_inference_steps": int(req.num_inference_steps),
                    "guidance_scale": float(req.guidance_scale),
                    "generator": gen,
                    **controlnet_kwargs,
                }
                with torch.inference_mode():
                    out = combined_pipe(**pipe_kwargs)
            elif init_image is not None:
                # img2img path: reuse loaded weights at zero extra VRAM cost
                init_pil = Image.open(io.BytesIO(init_image)).convert("RGB").resize((width, height))
                if self._img2img_pipe is None:
                    self._img2img_pipe = _sdxl_img2img_pipeline_cls()(**self.pipe.components)
                self._normalize_img2img_modules()
                denoise_strength = float(getattr(req, 'denoise_strength', 0.75))
                pipe_kwargs = {
                    "prompt": req.prompt,
                    "negative_prompt": getattr(req, "negative_prompt", None),
                    "image": init_pil,
                    "strength": denoise_strength,
                    "num_inference_steps": int(req.num_inference_steps),
                    "guidance_scale": float(req.guidance_scale),
                    "generator": gen,
                }
                with torch.inference_mode():
                    out = self._img2img_pipe(**pipe_kwargs)
            else:
                if bindings:
                    controlnet_kwargs = self._build_controlnet_kwargs(
                        bindings, (width, height), loaded_ids
                    )
                pipe_kwargs = {
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_cuda_worker_controlnet.py -v`
Expected: PASS — all tests.

- [ ] **Step 6: Flip the capability flag**

In `backends/platforms/cuda.py`, replace:

```python
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, True, True, True, True, True)
```

with:

```python
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, True, True, True, True, True, True)
```

`tests/test_backend_runtimes.py` already covers `CUDAProvider` (e.g.
`test_cuda_provider_creates_runtime_without_server_branching`) but has no existing
test on `.capabilities()`. Add one there:

```python
def test_cuda_provider_reports_supports_img2img_and_controlnet():
    from backends.platforms.cuda import CUDAProvider

    caps = CUDAProvider().capabilities()
    assert caps.supports_img2img_and_controlnet is True
```

- [ ] **Step 7: Run full suite for touched files**

Run: `pytest tests/test_cuda_worker_controlnet.py tests/test_backend_runtimes.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backends/cuda_worker.py backends/platforms/cuda.py tests/test_cuda_worker_controlnet.py tests/test_backend_runtimes.py
git commit -m "feat(controlnet): implement SDXL combined img2img+ControlNet execution, flip supports_img2img_and_controlnet capability (STABL-umvdwgsm)"
```

---

### Task 4: Capability-gate the WS fail-fast guard

**Blocker fix (human review, 2026-07-08):** Group A's fail-fast guard
(`docs/superpowers/plans/2026-07-08-img2img-controlnet-groundwork.md` Task 1) calls
`reject_combined_img2img_controlnet(..., supports_combined=False)` with a hardcoded
`False` — correct at the time Group A lands (nothing supports combined execution
yet), but wrong once this plan's Tasks 2-3 ship real execution and flip
`supports_img2img_and_controlnet=True` on the CUDA provider. Without this task, a
combined WS/CLI request would still be rejected before it ever reaches the new
worker code, even on a fully capable CUDA backend. This task replaces the hardcoded
`False` with a live capability check.

**Files:**
- Modify: `server/ws_routes.py`
- Test: `tests/test_ws_routes.py`

**Interfaces:**
- Produces: `_supports_img2img_and_controlnet(provider: Any) -> bool` in
  `server/ws_routes.py`, mirroring the existing `_supports_controlnet` helper
  (`server/ws_routes.py:108-115`).
- Consumes: `reject_combined_img2img_controlnet(..., supports_combined=...)`
  (Group A Task 1), `BackendCapabilities.supports_img2img_and_controlnet` (Group A
  Task 2, flipped `True` on CUDA by this plan's Task 3 Step 6).

- [ ] **Step 1: Write the failing "supported proceeds" test**

Append to the `TestJobSubmit` class in `tests/test_ws_routes.py`. This mirrors
`test_generate_mode_system_forwards_negative_prompt_and_scheduler` (full flow through
to a dispatch attempt) but stops short of a real worker call — it asserts the
fail-fast guard is passed and preprocessing runs, not that a job completes:

```python
    def test_generate_mode_system_proceeds_past_combined_guard_when_backend_capability_set(self):
        """Once the backend reports supports_img2img_and_controlnet=True, a combined
        init_image + controlnets request must reach ControlNet preprocessing instead
        of being rejected by reject_combined_img2img_controlnet."""
        from server.mode_config import ControlNetControlTypePolicy, ControlNetPolicy

        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "SDXL"
        app.state.worker_pool = pool
        app.state.storage = None
        app.state.backend_provider = SimpleNamespace(
            capabilities=lambda: SimpleNamespace(
                supports_controlnet=True, supports_img2img_and_controlnet=True
            )
        )

        # handle_job_submit resolves params["init_image_ref"] via resolve_file_ref
        # unconditionally, before job:ack is sent — an unknown ref raises KeyError
        # and short-circuits into a bare `_error()` frame with no job:ack at all,
        # which would break this test's ack-then-job:error expectation below. Use a
        # real ref from the upload store (get_store()/_solid_png_bytes() are already
        # imported/defined in this test file).
        init_ref = get_store().write("upload", _solid_png_bytes())

        fake_lcm_module = types.ModuleType("server.lcm_sr_server")

        class _FakeGenerateRequest:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def _fake_store_image_blob(*args, **kwargs):
            return None

        fake_lcm_module.GenerateRequest = _FakeGenerateRequest
        fake_lcm_module._store_image_blob = _fake_store_image_blob
        original_lcm_module = sys.modules.get("server.lcm_sr_server")
        sys.modules["server.lcm_sr_server"] = fake_lcm_module

        controlnet_policy = ControlNetPolicy(
            enabled=True,
            max_attachments=1,
            allow_reuse_emitted_maps=True,
            allowed_control_types={
                "canny": ControlNetControlTypePolicy(
                    default_model_id="sdxl-canny",
                    allowed_model_ids=["sdxl-canny"],
                    allow_preprocess=False,
                    default_strength=0.8,
                    min_strength=0.0,
                    max_strength=2.0,
                )
            },
        )

        try:
            with patch("server.ws_routes.get_mode_config") as get_mode_config:
                get_mode_config.return_value = SimpleNamespace(
                    get_mode=lambda name: SimpleNamespace(
                        name=name,
                        default_size="1024x1024",
                        default_steps=4,
                        default_guidance=1.0,
                        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
                        controlnet_policy=controlnet_policy,
                        model_path=None,  # forces a later, unrelated failure — see assertion below
                    )
                )
                with patch("server.controlnet_preprocessing.preprocess_controlnet_attachments") as preprocess_mock:
                    preprocess_mock.return_value = []
                    with client.websocket_connect("/v1/ws") as ws:
                        ws.receive_json()  # consume status
                        ws.send_json({
                            "type": "job:submit",
                            "id": "t-combined-proceeds",
                            "jobType": "generate",
                            "params": {
                                "prompt": "a cat",
                                "size": "1024x1024",
                                "init_image_ref": init_ref,
                                "controlnets": [
                                    {"attachment_id": "cn_1", "control_type": "canny", "map_asset_ref": "ref1"}
                                ],
                            },
                        })

                        ack = ws.receive_json()
                        assert ack["type"] == "job:ack"

                        err = ws.receive_json()
                        # Fails later — mode.model_path is None, so resolving
                        # ControlNet bindings raises — but crucially NOT on our
                        # fail-fast guard, which is the thing this test verifies.
                        assert err["type"] == "job:error"
                        assert "img2img" not in err["error"]

                    preprocess_mock.assert_called_once()
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module
            app.state.use_mode_system = False
            app.state.worker_pool = None
            app.state.backend_provider = None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ws_routes.py -k proceeds_past_combined_guard -v`
Expected: FAIL — the guard still hardcodes `supports_combined=False`, so `err["error"]`
contains "img2img" and `preprocess_mock` is never called.

- [ ] **Step 3: Add the capability helper and wire it into the guard call**

In `server/ws_routes.py`, add next to `_supports_controlnet`:

```python
def _supports_img2img_and_controlnet(provider: Any) -> bool:
    if provider is None:
        return False
    try:
        capabilities = provider.capabilities()
    except Exception:
        return False
    return getattr(capabilities, "supports_img2img_and_controlnet", False) is True
```

Then, in `handle_job_submit`, replace:

```python
            req = _build_generate_request(params)
            from server.controlnet_constraints import reject_combined_img2img_controlnet
            reject_combined_img2img_controlnet(
                has_init_image=bool(params.get("init_image_ref")),
                controlnets=req.controlnets,
                # No backend supports combined execution yet. Group B
                # (docs/superpowers/plans/2026-07-08-img2img-controlnet-pipeline-wiring.md
                # Task 4) replaces this hardcoded False with a live capability check
                # once CUDA execution exists — until then this guard must stay
                # unconditional or it would silently let unsupported requests through.
                supports_combined=False,
            )
```

with:

```python
            req = _build_generate_request(params)
            from server.controlnet_constraints import reject_combined_img2img_controlnet
            reject_combined_img2img_controlnet(
                has_init_image=bool(params.get("init_image_ref")),
                controlnets=req.controlnets,
                supports_combined=_supports_img2img_and_controlnet(
                    getattr(state, "backend_provider", None)
                ),
            )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ws_routes.py -v`
Expected: PASS — all `TestJobSubmit` tests, including:
- `test_generate_mode_system_rejects_combined_img2img_controlnet_before_preprocessing`
  (Group A, unchanged) — still passes, because that test never sets
  `app.state.backend_provider`, so `_supports_img2img_and_controlnet(None)` returns
  `False` and the guard still rejects. This is the "unsupported still rejects" half
  of the fix, already covered without modification.
- `test_generate_mode_system_proceeds_past_combined_guard_when_backend_capability_set`
  (this task) — the "supported proceeds" half.

- [ ] **Step 5: Commit**

```bash
git add server/ws_routes.py tests/test_ws_routes.py
git commit -m "fix(controlnet): capability-gate the img2img+controlnet fail-fast guard so CUDA combined execution is reachable (STABL-kjkrmrlk follow-up, STABL-vgbxamoz, STABL-umvdwgsm)"
```

---

## Self-Review

**FP issue coverage:**
- `STABL-vgbxamoz` (SD1.5 wiring) → Tasks 1-2. ✓
- `STABL-umvdwgsm` (SDXL wiring) → Tasks 1, 3. ✓
- Fail-fast guard correctness (human review blocker, no single FP issue — closes a gap between `STABL-kjkrmrlk`'s Group A guard and this plan's execution) → Task 4. ✓

**Guardrail coverage:** the `image`/`control_image` kwarg collision is the explicit subject of Task 1's `image_kwarg` override and both Task 2/3 regression tests (`call_kwargs["image"] is not call_kwargs["control_image"]`) — matches the FP comments on both issues verbatim, and was independently verified against installed diffusers source during human review. No new backend/cache/asset-store/schema work introduced. Task 4 closes the reviewer-flagged blocker: without it, `reject_combined_img2img_controlnet`'s hardcoded `supports_combined=False` (Group A) would keep rejecting every combined WS/CLI request even after this plan ships real CUDA execution — Task 4 makes the guard read the live capability flag Task 3 flips.

**Placeholder scan:** no TBD/TODO. The `_COMBINED_PIPE_INSTANCES`/`_COMBINED_XL_PIPE_INSTANCES` module-level lists are cleared at the start of their respective tests (Task 2 Step 2, Task 3 Step 2) so assertion counts don't depend on test execution order within the file.

**Type consistency:** `_build_controlnet_kwargs(self, bindings, size, loaded_ids, image_kwarg=None)` (Task 1) is called identically in both Task 2 (SD1.5) and Task 3 (SDXL) with `image_kwarg="control_image"`. `_validate_control_image_aspect_ratio(init_image_bytes, bindings, *, tolerance=0.02)` (Task 1) is called identically in both. `BackendCapabilities.supports_img2img_and_controlnet` (Group A Task 2) is flipped to `True` only in Task 3 Step 6, once both families are proven working. `_supports_img2img_and_controlnet(provider)` (Task 4) mirrors `_supports_controlnet`'s exact shape and is the single call site `reject_combined_img2img_controlnet`'s `supports_combined` argument now reads from.
