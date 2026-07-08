# img2img + ControlNet — Groundwork (Parallel Group A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project policy forbids subagent-driven development — execute inline.

**Goal:** Land the five FP subissues that have no dependencies on each other — the WS-side fail-fast rejection of combined img2img+ControlNet requests, the combined-capability signal, and three written design decisions (strength interaction, sizing reconciliation, cache-key behavior) plus one non-goal doc — so the two pipeline-wiring tasks (Group B, separate plan) have a fixed contract to build against.

**Architecture:** No new subsystems. This plan touches existing validation seams (`server/controlnet_constraints.py`, `server/ws_routes.py`), an existing capability struct (`backends/platforms/base.py`, `server/model_routes.py`), and produces/extends three docs (`docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md`, `notes/2026-02-22-img2img-data-flow.md`, `CONTROLNET.md` + `project-forward-notes.md`).

**Tech Stack:** Python (FastAPI/pydantic), pytest. No new dependencies.

**FP tree:** `STABL-ztaxgbhv` (parent). This plan covers `STABL-kjkrmrlk`, `STABL-uiwneiqf`, `STABL-bwkjcbwc`, `STABL-dghgcuzy`, `STABL-dxaheihz`.

## Global Constraints

- Combined support is CLI/WS-only. HTTP `/generate`'s `GenerateRequest` (`server/lcm_sr_server.py:128`) has no `init_image_ref` field and must not gain one in this plan — confirmed via FP guardrail comments on `STABL-kjkrmrlk`/`STABL-ztaxgbhv`.
- No new result cache, schema, asset-store, frontend, gallery, or non-CUDA backend work in this plan (FP guardrail on `STABL-dghgcuzy`, `STABL-dxaheihz`).
- Do not remove the `NotImplementedError` guards in `backends/cuda_worker.py:539-542`/`:875-878` — they stay as defensive backstops until Group B replaces them with real execution.
- Run Python tests with the Miniforge base env active: `source /Users/darkbit1001/miniforge3/bin/activate base` then `pytest <path> -v`.

## File Structure

- **Modify `server/controlnet_constraints.py`:** add `reject_combined_img2img_controlnet(*, has_init_image, controlnets)`.
- **Modify `server/ws_routes.py`:** call the new guard in `handle_job_submit`, before ControlNet policy/preprocessing runs.
- **Modify `backends/platforms/base.py`:** add `supports_img2img_and_controlnet: bool = False` to `BackendCapabilities`.
- **Modify `server/model_routes.py`:** expose the new field in `GET /models/status`.
- **Create `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md`:** strength-interaction and sizing-reconciliation decisions (consumed by Group B).
- **Modify `notes/2026-02-22-img2img-data-flow.md`:** explicit cache-key decision for combined runs.
- **Modify `CONTROLNET.md`, `project-forward-notes.md`:** explicit non-CUDA non-goal statement for the combined path.
- **Test:** `tests/test_controlnet_constraints.py`, `tests/test_ws_routes.py`, `tests/test_model_routes.py`.

---

### Task 1: Fail-fast rejection of combined img2img+ControlNet requests

**Files:**
- Modify: `server/controlnet_constraints.py`
- Modify: `server/ws_routes.py:169-179` (inside `handle_job_submit`)
- Test: `tests/test_controlnet_constraints.py`
- Test: `tests/test_ws_routes.py`

**Interfaces:**
- Produces: `reject_combined_img2img_controlnet(*, has_init_image: bool, controlnets: Any) -> None` in `server/controlnet_constraints.py` — raises `ValueError` iff both are truthy.
- Consumes: nothing new. Called from `server/ws_routes.py` `handle_job_submit`, using raw `params.get("init_image_ref")` (WS-only field, never reaches `GenerateRequest`) and the already-built `req.controlnets`.

- [ ] **Step 1: Write the failing unit tests**

Add to the top imports and end of `tests/test_controlnet_constraints.py`:

```python
from server.controlnet_constraints import enforce_controlnet_policy, reject_combined_img2img_controlnet
```

(replace the existing single-name import line with this two-name import), then append:

```python
def test_reject_combined_img2img_controlnet_raises_when_both_present():
    with pytest.raises(ValueError, match="img2img"):
        reject_combined_img2img_controlnet(has_init_image=True, controlnets=[object()])


def test_reject_combined_img2img_controlnet_allows_img2img_alone():
    reject_combined_img2img_controlnet(has_init_image=True, controlnets=None)
    reject_combined_img2img_controlnet(has_init_image=True, controlnets=[])


def test_reject_combined_img2img_controlnet_allows_controlnet_alone():
    reject_combined_img2img_controlnet(has_init_image=False, controlnets=[object()])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_controlnet_constraints.py -v`
Expected: FAIL — `ImportError: cannot import name 'reject_combined_img2img_controlnet'`

- [ ] **Step 3: Implement the guard**

Append to `server/controlnet_constraints.py`:

```python
def reject_combined_img2img_controlnet(*, has_init_image: bool, controlnets: Any) -> None:
    """Fail fast when a request carries both an init image and ControlNet attachments.

    img2img + ControlNet in the same request is not implemented yet (see
    docs/superpowers/plans/2026-07-08-img2img-controlnet-pipeline-wiring.md). Callers
    must invoke this before any ControlNet preprocessing or worker dispatch so the
    combination is rejected before assets get written or a job is queued.
    """
    if has_init_image and controlnets:
        raise ValueError(
            "img2img (init_image) combined with ControlNet attachments in the same "
            "request is not supported"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_controlnet_constraints.py -v`
Expected: PASS — all tests including the three new ones.

- [ ] **Step 5: Write the failing WS integration test**

Append to the `TestJobSubmit` class in `tests/test_ws_routes.py` (mirror `test_generate_mode_system_rejects_invalid_size_before_submit` at line 613, but the rejection here fires before mode lookup, so no `get_mode_config` patch is needed):

```python
    def test_generate_mode_system_rejects_combined_img2img_controlnet_before_preprocessing(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "SDXL"
        app.state.worker_pool = pool
        app.state.storage = None

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

        try:
            with patch("server.controlnet_preprocessing.preprocess_controlnet_attachments") as preprocess_mock:
                with client.websocket_connect("/v1/ws") as ws:
                    ws.receive_json()  # consume status
                    ws.send_json({
                        "type": "job:submit",
                        "id": "t-combined-reject",
                        "jobType": "generate",
                        "params": {
                            "prompt": "a cat",
                            "init_image_ref": "abc123",
                            "controlnets": [
                                {"attachment_id": "cn_1", "control_type": "canny", "map_asset_ref": "ref1"}
                            ],
                        },
                    })

                    ack = ws.receive_json()
                    assert ack["type"] == "job:ack"
                    assert ack["id"] == "t-combined-reject"

                    err = ws.receive_json()
                    assert err["type"] == "job:error"
                    assert err["jobId"] == ack["jobId"]
                    assert "img2img" in err["error"]

                pool.submit_job.assert_not_called()
                preprocess_mock.assert_not_called()
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module
            app.state.use_mode_system = False
            app.state.worker_pool = None
```

- [ ] **Step 6: Run to verify failure**

Run: `pytest tests/test_ws_routes.py -k combined_img2img_controlnet -v`
Expected: FAIL — no rejection happens today; `preprocess_mock.assert_not_called()` fails (preprocessing runs), and the `job:error` frame (if any) won't contain "img2img".

- [ ] **Step 7: Wire the guard into `handle_job_submit`**

In `server/ws_routes.py`, in `handle_job_submit`, find:

```python
            req = _build_generate_request(params)
            if current_mode:
                mode = get_mode_config().get_mode(current_mode)
                finalize_mode_generate_request(
```

Replace with:

```python
            req = _build_generate_request(params)
            from server.controlnet_constraints import reject_combined_img2img_controlnet
            reject_combined_img2img_controlnet(
                has_init_image=bool(params.get("init_image_ref")),
                controlnets=req.controlnets,
            )
            if current_mode:
                mode = get_mode_config().get_mode(current_mode)
                finalize_mode_generate_request(
```

- [ ] **Step 8: Run to verify pass**

Run: `pytest tests/test_ws_routes.py -v`
Expected: PASS — the new test plus all existing `TestJobSubmit` tests (the guard call sits before any behavior those tests depend on, since it's a no-op unless both `init_image_ref` and `controlnets` are present).

- [ ] **Step 9: Commit**

```bash
git add server/controlnet_constraints.py server/ws_routes.py tests/test_controlnet_constraints.py tests/test_ws_routes.py
git commit -m "feat(controlnet): reject combined img2img+ControlNet requests before preprocessing (STABL-kjkrmrlk) — next: combined capability signal"
```

---

### Task 2: Combined capability signal

**Files:**
- Modify: `backends/platforms/base.py`
- Modify: `server/model_routes.py:126-134`
- Test: `tests/test_model_routes.py`

**Interfaces:**
- Produces: `BackendCapabilities.supports_img2img_and_controlnet: bool = False` (default keeps every existing 5-arg/6-arg positional construction site — `backends/platforms/cuda.py:67`, `rknn.py:60`, `cpu.py:47`, `mlx.py:14` — unchanged and valid).
- Consumes: `caps` from `provider.capabilities()` in the `GET /models/status` handler (`server/model_routes.py`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_routes.py`:

```python
async def test_models_status_reports_supports_img2img_and_controlnet_when_true():
    runtime = Mock()
    runtime.get_current_mode.return_value = None
    runtime.is_model_loaded.return_value = False
    runtime.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {"backend": "cuda", "device": "cuda:0", "models_loaded": 1}

    provider = Mock()
    provider.backend_id = "cuda"
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=True,
        supports_modes=True,
        supports_superres=True,
        supports_model_registry_stats=True,
        supports_img2img=True,
        supports_img2img_and_controlnet=True,
    )

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry):
        data = await model_routes.get_models_status(_status_request())

    assert data["capabilities"]["supports_img2img_and_controlnet"] is True


async def test_models_status_defaults_supports_img2img_and_controlnet_false_when_capability_omitted():
    runtime = Mock()
    runtime.get_current_mode.return_value = None
    runtime.is_model_loaded.return_value = False
    runtime.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {"backend": "cpu", "device": "CPU placeholder", "models_loaded": 0}

    provider = Mock()
    provider.backend_id = "cpu"
    # SimpleNamespace without supports_img2img_and_controlnet — mirrors an older
    # BackendCapabilities instance built before this field existed.
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=False,
        supports_modes=True,
        supports_superres=False,
        supports_model_registry_stats=False,
        supports_img2img=False,
    )

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry):
        data = await model_routes.get_models_status(_status_request())

    assert data["capabilities"]["supports_img2img_and_controlnet"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_model_routes.py -k supports_img2img_and_controlnet -v`
Expected: FAIL — `KeyError: 'supports_img2img_and_controlnet'` (key not yet in the response dict).

- [ ] **Step 3: Add the field to `BackendCapabilities`**

In `backends/platforms/base.py`, replace:

```python
@dataclass(frozen=True)
class BackendCapabilities:
    supports_generation: bool
    supports_modes: bool
    supports_superres: bool
    supports_model_registry_stats: bool
    supports_img2img: bool
    supports_controlnet: bool = False
```

with:

```python
@dataclass(frozen=True)
class BackendCapabilities:
    supports_generation: bool
    supports_modes: bool
    supports_superres: bool
    supports_model_registry_stats: bool
    supports_img2img: bool
    supports_controlnet: bool = False
    supports_img2img_and_controlnet: bool = False
```

- [ ] **Step 4: Expose it in `GET /models/status`**

In `server/model_routes.py`, replace:

```python
        "capabilities": {
            "supports_generation": caps.supports_generation,
            "supports_modes": caps.supports_modes,
            "supports_superres": caps.supports_superres,
            "supports_model_registry_stats": caps.supports_model_registry_stats,
            "supports_img2img": caps.supports_img2img,
        },
```

with:

```python
        "capabilities": {
            "supports_generation": caps.supports_generation,
            "supports_modes": caps.supports_modes,
            "supports_superres": caps.supports_superres,
            "supports_model_registry_stats": caps.supports_model_registry_stats,
            "supports_img2img": caps.supports_img2img,
            "supports_img2img_and_controlnet": getattr(caps, "supports_img2img_and_controlnet", False),
        },
```

Use `getattr(..., False)` (not `caps.supports_img2img_and_controlnet`) so any test double or older capabilities object built without the field still resolves to the correct "unsupported" default instead of raising `AttributeError` — mirrors the existing `getattr(capabilities, "supports_controlnet", False)` pattern already used in `server/ws_routes.py`'s `_supports_controlnet`.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_model_routes.py -v`
Expected: PASS — all tests, including the two new ones.

- [ ] **Step 6: Commit**

```bash
git add backends/platforms/base.py server/model_routes.py tests/test_model_routes.py
git commit -m "feat(backend): add supports_img2img_and_controlnet capability signal, default false everywhere (STABL-kjkrmrlk) — next: strength-interaction design decision"
```

---

### Task 3: Design decision — denoise_strength × ControlNet strength/start/end interaction

**Files:**
- Create: `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md`

**Interfaces:**
- Produces: a written decision Group B (pipeline-wiring plan) treats as a hard contract for kwargs shape and behavior — no code interface.

- [ ] **Step 1: Write the decision doc**

Create `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md`:

```markdown
# img2img + ControlNet Combined Path — Design Decisions

**Date:** 2026-07-08
**Status:** Decided
**FP:** STABL-ztaxgbhv (parent), STABL-uiwneiqf, STABL-bwkjcbwc

Two decisions needed before the pipeline-wiring tasks (`STABL-vgbxamoz` SD1.5,
`STABL-umvdwgsm` SDXL) can implement the combined execution branch in
`backends/cuda_worker.py`. Both `StableDiffusionControlNetImg2ImgPipeline` and its
SDXL counterpart are thin wrappers diffusers exposes on top of the same base
components already loaded by this worker — this doc governs how we drive them, not
how diffusers implements them internally.

## Decision 1: denoise_strength × strength/start_percent/end_percent interaction

`start_percent`/`end_percent` on each ControlNet attachment are passed straight
through to the combined pipeline's `control_guidance_start`/`control_guidance_end`
kwargs, unmodified by `denoise_strength`. We do not attempt to renormalize them
against the nominal (pre-strength) step count — diffusers' combined pipeline already
applies `strength` to compute its own effective step schedule internally, and
`control_guidance_start`/`control_guidance_end` are diffusers' contract against
whatever schedule it derives. Re-deriving that math in our wrapper would duplicate
diffusers internals and drift the first time the installed diffusers version changes
its slicing behavior.

Concretely: `denoise_strength` flows into `strength=` (already the case for the
plain img2img path today), and each attachment's existing `strength` (ControlNet
conditioning scale — not to be confused with `denoise_strength`) flows into
`controlnet_conditioning_scale=` as it does on the txt2img ControlNet path. No new
plumbing beyond what the txt2img ControlNet branch already does for
`controlnet_conditioning_scale`/`control_guidance_start`/`control_guidance_end`.

**At strength=1.0** (full regenerate): behaves identically in spirit to the existing
txt2img + ControlNet path — the full nominal `num_inference_steps` schedule runs
and `control_guidance_start`/`end` apply across all of it. This must be covered by a
Group B test asserting the combined-path call captures the same
`controlnet_conditioning_scale`/`control_guidance_start`/`control_guidance_end`
values the txt2img branch would for equivalent attachment strength/start/end inputs.

**At low strength** (e.g. `denoise_strength=0.05`): diffusers computes very few
effective denoising steps from a low strength. If `end_percent` is also small (e.g.
`0.3`), the ControlNet conditioning may end up applied to zero of the few remaining
effective steps — the generation looks like an almost-unconditioned img2img pass.
This is **accepted v1 behavior, not a bug**: no auto-clamping, no validation error.
It is a documented operator caveat (added to `CONTROLNET.md`'s "Not supported in v1"
section update, see `STABL-dxaheihz`) so users understand very low `denoise_strength`
combined with a narrow `start_percent`/`end_percent` window may produce
ControlNet-invisible results. Group B does not need special-case code for this.

## Decision 2: control-map vs init-image size reconciliation

See the sizing-reconciliation section appended by the follow-up task (`STABL-bwkjcbwc`)
below.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md
git commit -m "docs(controlnet): decide denoise_strength x controlnet strength/start/end interaction for combined path (STABL-uiwneiqf) — next: sizing reconciliation decision"
```

---

### Task 4: Design decision — control-map vs init-image size reconciliation

**Files:**
- Modify: `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md` (created by Task 3)

**Interfaces:**
- Produces: the exact validation contract (tolerance, error shape, where it runs) Group B implements in `backends/cuda_worker.py`.

- [ ] **Step 1: Append the sizing decision**

In `docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md`, replace the placeholder line:

```markdown
See the sizing-reconciliation section appended by the follow-up task (`STABL-bwkjcbwc`)
below.
```

with:

```markdown
Today, both consumers independently force-resize to the request's `(width, height)`
regardless of source: `_decode_control_image` resizes each control map
(`backends/cuda_worker.py:62-66`), and the img2img branch resizes the init image
the same way (`backends/cuda_worker.py:551` SD1.5, `:885` SDXL). So there is no
dimension *mismatch* possible post-resize — both always land on the request size.
The real risk is **content misalignment**: if a control map's native aspect ratio
differs from the init image's native aspect ratio, forcing both to the same target
size stretches one or both non-uniformly, so ControlNet's spatial conditioning
(e.g. canny edges) no longer lines up with the init image's content.

**Decision:** reject the combined request when a binding's native aspect ratio
diverges from the init image's native aspect ratio by more than 2% (relative
difference of `width/height`). Validation:

- Runs in `backends/cuda_worker.py`, in the combined-path branch of `run_job`, before
  either image is opened for resizing (Group B implements this — see
  `docs/superpowers/plans/2026-07-08-img2img-controlnet-pipeline-wiring.md`).
- Reads native dimensions via `PIL.Image.open(...).size` on the raw bytes (no
  decode-then-resize needed just to compare ratios).
- Tolerance: `abs(control_ratio - init_ratio) / init_ratio > 0.02` triggers rejection.
- On mismatch: raise `ValueError` naming the offending `attachment_id` and both
  ratios, e.g. `"controlnet attachment 'cn_1' aspect ratio 1.78 diverges from init "
  "image aspect ratio 1.33 by more than 2%"` — caught by the same worker error path
  that already surfaces other `run_job` exceptions as `job:error`.
- Within tolerance: both images are resized independently to `(width, height)`
  exactly as today — no new cross-scaling or letterboxing logic. This keeps the
  fix a validation gate, not a new image-processing pipeline.
- Applies per-binding when a request has multiple ControlNet attachments; the first
  binding whose ratio diverges from the init image's ratio fails the request.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-07-08-img2img-controlnet-combined-design.md
git commit -m "docs(controlnet): decide control-map vs init-image aspect-ratio reconciliation for combined path (STABL-bwkjcbwc) — next: cache-key decision"
```

---

### Task 5: Cache-key decision for combined runs

**Files:**
- Modify: `notes/2026-02-22-img2img-data-flow.md`

**Interfaces:**
- None (docs only — no code today implements result caching for either img2img or ControlNet, so there is nothing to wire against).

- [ ] **Step 1: State the decision explicitly**

In `notes/2026-02-22-img2img-data-flow.md`, replace the "Cache Behavior" section:

```markdown
## Cache Behavior

img2img results are **not cached** (unlike txt2img). The cache key is based on
`{ prompt, size, steps, cfg, seed, superres, superresLevel }` — it does not include the init
image content. Caching img2img results would risk serving stale results for a different init
image with the same params.
```

with:

```markdown
## Cache Behavior

img2img results are **not cached** (unlike txt2img). The cache key is based on
`{ prompt, size, steps, cfg, seed, superres, superresLevel }` — it does not include the init
image content. Caching img2img results would risk serving stale results for a different init
image with the same params.

**Combined img2img + ControlNet runs (STABL-dghgcuzy):** the same rule extends
unchanged. ControlNet attachments add another content dimension (`map_asset_ref`/
`source_asset_ref` bytes) that is likewise excluded from the existing cache key, so
combined-path results stay uncached for the same reason plain img2img results do.
This is a decision, not an oversight: no new result-cache key, no image hashing, no
schema change. The separate ControlNet *model* cache (`backends/controlnet_cache.py`,
keyed by `model_id`/`model_path`) is unaffected — it caches loaded model weights, not
generation results, and needs no change for the combined path.
```

- [ ] **Step 2: Commit**

```bash
git add notes/2026-02-22-img2img-data-flow.md
git commit -m "docs(controlnet): state cache-key decision for combined img2img+controlnet runs explicitly (STABL-dghgcuzy) — next: non-CUDA non-goal doc"
```

---

### Task 6: Non-goal doc — non-CUDA backends for the combined path

**Files:**
- Modify: `CONTROLNET.md`
- Modify: `project-forward-notes.md`

**Interfaces:**
- None (docs only).

- [ ] **Step 1: Update `CONTROLNET.md`'s deferred list**

In `CONTROLNET.md`, replace the "Current v1 Objectives and Limits" deferred bullet list:

```markdown
Still-open or intentionally deferred as built-in/default surfaces:

- non-CUDA backends
- img2img + ControlNet
- more built-in server preprocessors like pose / normal / segmentation
- richer model-registry metadata beyond ControlNet-specific fields
- MLX runtime wiring; see [`docs/CONTROLNET_MLX_CONVERSION.md`](docs/CONTROLNET_MLX_CONVERSION.md)
```

with:

```markdown
Still-open or intentionally deferred as built-in/default surfaces:

- non-CUDA backends
- img2img + ControlNet (CUDA-only combined-path work is tracked under `STABL-ztaxgbhv`;
  RKNN/MLX/CPU execution of the combined path remains out of scope even once the
  CUDA combined path lands — it compounds onto the existing non-CUDA ControlNet
  deferral above, not a separate gap)
- more built-in server preprocessors like pose / normal / segmentation
- richer model-registry metadata beyond ControlNet-specific fields
- MLX runtime wiring; see [`docs/CONTROLNET_MLX_CONVERSION.md`](docs/CONTROLNET_MLX_CONVERSION.md)
```

Also update the "Not supported in v1" list near the top:

```markdown
Not supported in v1:

- CPU backend ControlNet execution
- RKNN backend ControlNet execution
- MLX backend ControlNet execution
- img2img + ControlNet in same request
- preprocessor families beyond `canny` and `depth`
```

with:

```markdown
Not supported in v1:

- CPU backend ControlNet execution
- RKNN backend ControlNet execution
- MLX backend ControlNet execution
- img2img + ControlNet in same request (CUDA-only support in progress, `STABL-ztaxgbhv`;
  non-CUDA execution of this combination is an explicit non-goal, not a future v1.x item)
- preprocessor families beyond `canny` and `depth`
```

- [ ] **Step 2: Update `project-forward-notes.md`'s deferred tracks table**

In `project-forward-notes.md`, find the "Deferred tracks (explicit, with rationale)" table and add a row after the existing table rows:

```markdown
| Non-CUDA img2img+ControlNet execution | Compounds onto the existing non-CUDA ControlNet deferral; explicit non-goal even after CUDA combined path (`STABL-ztaxgbhv`) ships |
```

- [ ] **Step 3: Check drift bindings**

Run: `drift refs CONTROLNET.md` and `drift refs project-forward-notes.md`
If either file is drift-bound, follow the Drift Discipline in `AGENTS.md`: update prose first (already done in Steps 1-2), then run `drift check` to confirm no stale anchors before relinking.

- [ ] **Step 4: Commit**

```bash
git add CONTROLNET.md project-forward-notes.md
git commit -m "docs(controlnet): state non-CUDA execution of combined img2img+controlnet path as explicit non-goal (STABL-dxaheihz)"
```

---

## Self-Review

**FP issue coverage:**
- `STABL-kjkrmrlk` (fail-fast + capability signal) → Tasks 1-2. ✓
- `STABL-uiwneiqf` (strength-interaction decision) → Task 3. ✓
- `STABL-bwkjcbwc` (sizing-reconciliation decision) → Task 4. ✓
- `STABL-dghgcuzy` (cache-key decision) → Task 5. ✓
- `STABL-dxaheihz` (non-CUDA non-goal doc) → Task 6. ✓

**Guardrail coverage:** HTTP `/generate` untouched (Task 1 only wires the WS path, matching the `STABL-kjkrmrlk`/`STABL-anhahetw` guardrail comments); no new cache/schema/asset-store work (Task 5 is decision-doc-only, matching the `STABL-dghgcuzy` guardrail); no non-CUDA execution work (Task 6 is a non-goal statement, not implementation).

**Placeholder scan:** no TBD/TODO; every doc task shows the exact markdown to write, every code task shows exact diffs and exact test code.

**Type consistency:** `reject_combined_img2img_controlnet(*, has_init_image: bool, controlnets: Any) -> None` (Task 1) and `BackendCapabilities.supports_img2img_and_controlnet: bool = False` (Task 2) are each defined once and used only at their own call sites in this plan — Group B (separate plan) is the next consumer of the capability flag (flips it to `True` on the CUDA provider once execution lands) and of the two design decisions in Task 3/4's doc.
