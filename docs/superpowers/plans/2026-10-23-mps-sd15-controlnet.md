# MPS SD1.5 LCM ControlNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render SD1.5 LCM text-to-image with one Canny ControlNet attachment on the MPS backend, and reject Depth, Pose, and multiple attachments on that platform only.

**Architecture:** The scope limit lives on the platform-family binding as a defaulted optional field, so CUDA keeps its current behavior and no CUDA constructor changes. `admit_generation_operation` enforces the limit before preprocessing. The worker repeats it as defense in depth. Execution reuses `CudaWorkerBase._load_controlnet_model` and `_build_controlnet_kwargs`; the MPS worker only overrides `_build_controlnet_pipe`.

**Tech Stack:** Python 3.12, torch 2.10 MPS, diffusers 0.39, OpenCV 5.0 (`cv2`), pytest through `python -m tests.run`.

**Spec:** `STABL-xomvolgw` — its description plus two comments: the VERIFIED prerequisite note and the DESIGN NOTE on the guard seam. Sigma was offline when this plan was written, so no separate design document exists. Read `fp context STABL-xomvolgw` first.

## Global Constraints

- The `sd15` MPS cell is the admission gate. The worker repeats each check as defense in depth.
- MPS policy fixes device `mps` and dtype float16 (`backends/platforms/mps_policy.py`). Do not add a device or dtype branch without a measurement.
- LCM scheduler only. `_resolve_scheduler_id(req) != "lcm"` raises `ValueError("MPS SD1.5 supports LCM scheduler only")`.
- One Canny attachment. Depth, Pose, and two or more attachments are rejected on MPS.
- The combined img2img plus ControlNet operation stays rejected. That is slice 4.
- Do not change `conf/controlnets.yaml`. Its `/models/controlnets` paths are container paths.
- Native macOS has no `/models` mount. Live acceptance uses a temporary host-path registry through `CONTROLNET_REGISTRY_PATH`, read at singleton construction in `server/controlnet_registry.py:77`.
- `CONTROLNET_REGISTRY_VALIDATION` defaults to `strict`. Keep it strict for acceptance.
- Run tests with `python -m tests.run <files> -- <pytest options>`. Mixed direct collection is rejected.
- Prose follows AGENTS.md controlled English.

## Review Focus

1. **Two attachments on a mode that permits two.** `conf/modes.yml:133` sets `max_attachments: 2`, so the request is legal at the mode layer. MPS must reject it at admission. Test in Task 2.
2. **A Depth or Pose attachment.** The registry registers `sd15-depth` and `sd15-openpose` for family `sd15`, so binding resolution succeeds. MPS must reject at admission with a message naming the platform limit. Test in Task 2.
3. **A CUDA `sd15` job with Depth.** The limit must not reach CUDA. This is the regression a family-keyed guard would cause. Test in Task 2.
4. **A ControlNet render that raises mid-pipeline.** The cache pin must be released, or the weights leak. Test in Task 3.
5. **An unnarrowed binding built with two positional arguments.** Every existing CUDA construction site and test helper passes two arguments. The new field must default. Test in Task 1.

---

### Task 1: The platform ControlNet scope field

**Files:**
- Modify: `backends/platforms/base.py:29-33`
- Modify: `backends/platforms/mps_bindings.py:12-20`
- Test: `tests/test_backend_runtimes.py`, `tests/test_mps_registration.py`

**Interfaces:**
- Produces: `ControlNetScope(allowed_control_types: tuple[str, ...] | None = None, max_attachments: int | None = None)`, frozen dataclass in `backends/platforms/base.py`. `None` on either field means the platform adds no narrowing.
- Produces: `FamilyPlatformBinding.controlnet_scope: ControlNetScope = ControlNetScope()`, the third field, defaulted.

- [ ] **Step 1: Write the failing tests**

In `tests/test_mps_registration.py`, inside `test_mps_provider_binds_sd15_and_shares_runtime`:

```python
    from backends.platforms.base import ControlNetScope
    assert cell.controlnet_scope == ControlNetScope(
        allowed_control_types=("canny",), max_attachments=1
    )
```

In `tests/test_backend_runtimes.py`, a new test:

```python
def test_cuda_bindings_carry_an_unnarrowed_controlnet_scope():
    """The platform scope is an MPS limit. CUDA must keep every control type
    the mode permits, and the two-argument construction must stay valid."""
    from backends.platforms.base import ControlNetScope
    from backends.platforms.cuda import CUDAProvider

    provider = CUDAProvider()
    for family_id in family_ids():
        assert provider.family_binding(family_id).controlnet_scope == ControlNetScope()
    assert ControlNetScope().allowed_control_types is None
    assert ControlNetScope().max_attachments is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m tests.run tests/test_mps_registration.py tests/test_backend_runtimes.py -- -q --no-cov -k "controlnet_scope or binds_sd15"`
Expected: FAIL with `AttributeError` or `ImportError` on `ControlNetScope`.

- [ ] **Step 3: Add `ControlNetScope` and the defaulted field in `backends/platforms/base.py`**

Place `ControlNetScope` above `FamilyPlatformBinding`. Both stay `@dataclass(frozen=True)`. Document that `None` means no platform narrowing, and that the field is defaulted so existing two-argument construction stays valid.

- [ ] **Step 4: Set the MPS `sd15` scope in `backends/platforms/mps_bindings.py`**

Keep `ExecutionCapabilities(True, False, False)` in this task. Pass `ControlNetScope(allowed_control_types=("canny",), max_attachments=1)`. State in the comment that the cell flips in Task 4, after admission and execution exist.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m tests.run tests/test_mps_registration.py tests/test_backend_runtimes.py tests/test_worker_factory.py -- -q --no-cov`
Expected: PASS. `test_worker_factory.py` proves the default keeps two-argument construction valid.

- [ ] **Step 6: Commit**

```bash
git add backends/platforms/base.py backends/platforms/mps_bindings.py tests/test_backend_runtimes.py tests/test_mps_registration.py
git commit -m "feat(mps): carry a platform ControlNet scope on the family binding (STABL-xomvolgw) - next: Task 2 admission enforcement"
```

---

### Task 2: Admission enforces the scope before preprocessing

**Files:**
- Modify: `server/controlnet_execution.py:17-66`
- Modify: `tests/snapshot_test_helpers.py:36-61`
- Test: `tests/test_controlnet_execution.py`

**Interfaces:**
- Consumes: `FamilyPlatformBinding.controlnet_scope` from Task 1.
- Produces: `admit_generation_operation` raises `ValueError` when the attachment count exceeds `max_attachments`, or when an attachment's `control_type` is outside `allowed_control_types`. Each message names the backend id and the limit.
- Produces: `make_family_provider(..., controlnet_scope: ControlNetScope | None = None)` in the test helper. `None` keeps today's unnarrowed binding.

- [ ] **Step 1: Write the failing tests**

In `tests/test_controlnet_execution.py`, beside the existing `test_admit_generation_operation_mps_sd15_*` tests:

```python
def test_admit_generation_operation_rejects_a_control_type_outside_the_platform_scope():
    provider = make_family_provider(
        family_id="sd15", supports_controlnet=True, backend_id="mps",
        controlnet_scope=ControlNetScope(allowed_control_types=("canny",), max_attachments=1),
    )
    req = SimpleNamespace(controlnets=[_attachment("depth")])
    with pytest.raises(ValueError, match="depth"):
        admit_generation_operation(req, snapshot=snapshot, provider=provider, has_init_image=False)


def test_admit_generation_operation_rejects_more_attachments_than_the_platform_scope_allows():
    """conf/modes.yml permits two attachments. The platform cap is one."""
    ...
    with pytest.raises(ValueError, match="1"):
        ...


def test_admit_generation_operation_admits_one_canny_attachment_within_the_platform_scope():
    ...
    assert admit_generation_operation(...) == "controlnet"


def test_admit_generation_operation_leaves_cuda_depth_admitted():
    """The scope is an MPS limit. An unnarrowed CUDA binding admits depth."""
    provider = make_family_provider(family_id="sd15", supports_controlnet=True)
    ...
    assert admit_generation_operation(...) == "controlnet"
```

Use the existing fixtures and the `make_active_snapshot` helper. `_attachment(control_type)` is a local factory returning a `SimpleNamespace` with `control_type`, `start_percent=0.0`, `end_percent=1.0`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m tests.run tests/test_controlnet_execution.py -- -q --no-cov -k platform_scope`
Expected: FAIL. The two rejection tests fail because admission returns `"controlnet"`.

- [ ] **Step 3: Enforce the scope in `admit_generation_operation`**

Add the check in the `has_controlnet` branch, after `caps.supports_controlnet` passes and before the function returns `"controlnet"`. Read `binding.controlnet_scope`. Skip each half when its field is `None`. Raise `ValueError`, matching the HunyuanDiT Canny-only precedent at `server/controlnet_execution.py:40`; `NotImplementedError` stays reserved for a family that supports no ControlNet at all. Do not test `provider.backend_id` for the decision; read it only to name the platform in the message.

- [ ] **Step 4: Add the helper parameter in `tests/snapshot_test_helpers.py`**

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m tests.run tests/test_controlnet_execution.py tests/test_controlnet_http_contract.py tests/test_controlnet_success_contract.py -- -q --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/controlnet_execution.py tests/snapshot_test_helpers.py tests/test_controlnet_execution.py
git commit -m "feat(controlnet): enforce the platform ControlNet scope at admission (STABL-xomvolgw) - next: Task 3 worker execution"
```

---

### Task 3: The MPS worker renders one Canny ControlNet

**Files:**
- Modify: `backends/mps_worker.py:119-250`
- Test: `tests/test_mps_worker.py`

**Interfaces:**
- Consumes: `CudaWorkerBase._load_controlnet_model(binding)`, `_build_controlnet_kwargs(bindings, size, loaded_ids, image_kwarg=None)`, `_controlnet_model_cls()`, and `_controlnet_metadata(bindings)`. All are inherited; do not reimplement them.
- Produces: `DiffusersMpsWorker._build_controlnet_pipe(controlnet_obj)`, returning `StableDiffusionControlNetPipeline.from_pipe(self.pipe, controlnet=controlnet_obj, torch_dtype=None)`, mirroring `backends/cuda_worker.py:1042`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_mps_worker.py`, following the existing fake-pipeline fixtures:

```python
def test_mps_worker_renders_one_canny_controlnet(fake_pipeline, ...):
    """The ControlNet pipe comes from self.pipe components, so the render costs
    no second copy of the weights."""

def test_mps_worker_rejects_a_depth_controlnet_binding(...):
    """Defense in depth. Admission rejects it first."""
    with pytest.raises(ValueError, match="depth"):

def test_mps_worker_rejects_two_controlnet_bindings(...):
    with pytest.raises(ValueError, match="1"):

def test_mps_worker_releases_the_controlnet_pin_after_a_failed_render(...):
    """Review Focus 4. A mid-pipeline raise must not leak the weights."""

def test_mps_worker_cancels_a_controlnet_render_at_a_step_boundary(...):
    """Mirrors test_mps_worker_cancels_img2img_at_step_boundary."""

def test_mps_worker_writes_the_controlnet_png_chunk(...):
```

Keep `test_mps_worker_rejects_an_init_image_with_controlnet_bindings` passing, with `NotImplementedError` unchanged. Replace `test_mps_worker_rejects_controlnet_bindings`, whose premise this task removes.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m tests.run tests/test_mps_worker.py -- -q --no-cov -k controlnet`
Expected: FAIL with `NotImplementedError: MPS SD1.5 does not support ControlNet` from `backends/mps_worker.py:132`.

- [ ] **Step 3: Split the defense-in-depth check in `run_job`**

The current single check rejects any `controlnet_bindings`. Replace it with: the combined operation (an init image together with bindings) raises `NotImplementedError`; a binding outside the platform scope raises `ValueError`. Read the scope from the worker's own family binding: `get_backend_provider().family_binding(self.family_profile.family_id).controlnet_scope`, importing `get_backend_provider` from `backends.platform_registry` inside the method, as the worker's other local imports do. One source then states the limit. State in the comment that admission rejects all three cases first.

- [ ] **Step 4: Add the ControlNet execution branch in `run_job`**

Mirror `backends/cuda_worker.py:1130-1150`: build the kwargs with `_build_controlnet_kwargs(bindings, (width, height), loaded_ids)`, pop `controlnet`, build the pipe with `_build_controlnet_pipe`, and inject step progress exactly as the text-to-image branch does. Load at `self.dtype`, which the inherited loader already does — add no dtype branch here. Add the `controlnet` PNG text chunk from `_controlnet_metadata(bindings)`, as the CUDA worker does at `cuda_worker.py:1174`. Release each pin in the existing `finally` block, following `cuda_worker.py:1183-1189`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m tests.run tests/test_mps_worker.py -- -q --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backends/mps_worker.py tests/test_mps_worker.py
git commit -m "feat(mps): render one Canny ControlNet on MPS SD1.5 (STABL-xomvolgw) - next: Task 4 flip the sd15 cell"
```

---

### Task 4: Flip the `sd15` cell and prove admission end to end

**Files:**
- Modify: `backends/platforms/mps_bindings.py`
- Test: `tests/test_mps_registration.py`, `tests/test_ws_routes.py`

**Interfaces:**
- Consumes: Task 2's admission guard and Task 3's execution path. Both must be merged into the branch before this task, or admission admits an operation the worker cannot run.
- Produces: `ExecutionCapabilities(True, True, False)` on the MPS `sd15` cell.

- [ ] **Step 1: Write the failing tests**

Update `tests/test_mps_registration.py` to assert `ExecutionCapabilities(True, True, False)`. In `tests/test_ws_routes.py`, replace `test_ws_generate_rejects_controlnet_on_mps_sd15_cell` with:

```python
def test_ws_generate_admits_one_canny_controlnet_on_mps_sd15_cell():
def test_ws_generate_rejects_a_depth_controlnet_on_mps_sd15_cell():
def test_ws_generate_rejects_two_controlnet_attachments_on_mps_sd15_cell():
```

Keep `test_ws_generate_rejects_the_combined_operation_on_mps_sd15_cell` unchanged. Follow the fixtures already used at `tests/test_ws_routes.py:2114-2160`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m tests.run tests/test_mps_registration.py tests/test_ws_routes.py -- -q --no-cov -k "mps_sd15_cell or binds_sd15"`
Expected: FAIL. The admit test fails on the cell's `supports_controlnet` being false.

- [ ] **Step 3: Flip the cell**

- [ ] **Step 4: Prove the cell is load-bearing**

Set the cell back to `ExecutionCapabilities(True, False, False)` by hand. Run the same command. The admit test must fail. Restore the flip. Record both outcomes in the FP comment; slice 2 required this proof.

- [ ] **Step 5: Run the full cohorts**

Run: `python -m tests.run tests/ -- -q --no-cov`
Expected: no new failure. Eight failures pre-exist on `main` — one in `tests/test_entry_spans.py` and seven vector-map CAD mesh tests. Name any other failure in the report.

- [ ] **Step 6: Commit**

```bash
git add backends/platforms/mps_bindings.py tests/test_mps_registration.py tests/test_ws_routes.py
git commit -m "feat(mps): admit one Canny ControlNet on the sd15 cell (STABL-xomvolgw) - next: Task 5 live acceptance"
```

---

### Task 5: Live MPS acceptance and recorded evidence

**Files:**
- Create: `spikes/mps_controlnet_acceptance.py`
- Create: a temporary host-path registry config outside the repo, for example `/tmp/controlnets-host.yaml`
- Modify: `docs/MPS_BACKEND.md`

**Interfaces:**
- Consumes: the merged branch through Task 4.
- Produces: recorded live evidence under the existing evidence headings in `docs/MPS_BACKEND.md`.

- [ ] **Step 1: Write the temporary registry config**

Copy `conf/controlnets.yaml` and swap the prefix `/models/controlnets` for `/Users/darkbit1001/models/controlnets`. Change nothing else. Keep it outside the repo. Do not edit `conf/controlnets.yaml`.

- [ ] **Step 2: Prove strict validation accepts it**

Run: `CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml python -c "from server.controlnet_registry import load_controlnet_registry as l; r=l(config_path='/tmp/controlnets-host.yaml'); print(sorted(r.specs))"`
Expected: the nine model ids print. A missing directory raises `ValueError: ControlNet model path does not exist`.

- [ ] **Step 3: Write the acceptance spike**

Model it on `spikes/mps_img2img_acceptance.py`: reuse that file's `post_multipart`, `render`, `describe`, and `mean_abs_difference` shapes. Build the source image with `build_init_image()`, derive the Canny map with `cv2.Canny`, upload it with `{"type": "canny"}` so the route stores it in the `control_map` bucket, and submit one attachment:

```python
{"attachment_id": "a1", "control_type": "canny", "model_id": "sd15-canny",
 "map_asset_ref": canny_ref, "strength": 1.0}
```

Render three jobs: text-to-image for the slice 1 byte comparison, one Canny render at strength 1.0, and one at strength 0.0. Assert the strength-0.0 render differs from the Canny render, so the control map is proven load-bearing rather than decorative.

- [ ] **Step 4: Run the acceptance in both isolation modes**

Start the server with `CONTROLNET_REGISTRY_PATH` set and `CONTROLNET_REGISTRY_VALIDATION` unset, once with `WORKER_ISOLATION=inproc` and once with `subprocess`. Use `PYTHONPATH=/app` or the spike's own bootstrap; `python spikes/x.py` puts the spike directory on `sys.path`, not the repo root.
Expected: every render passes the validity checks in `docs/MPS_BACKEND.md`, and each render matches across the two modes.

- [ ] **Step 5: Settle the dtype question with the measurement**

The inherited loader uses `self.dtype`, which MPS policy fixes at float16. If a render fails a validity check — a black image, or a near-zero standard deviation — load the ControlNet in float32 instead, re-run, and record both results. Do not change the dtype without a failed check to cite.

- [ ] **Step 6: Record the evidence and update the Limits section**

Add the live record under the existing evidence headings: model path, seed, both isolation modes, sha256 prefixes, and the strength comparison. Update the Limits bullets: the cell now admits ControlNet; Depth, Pose, and a second attachment are rejected on MPS while the mode permits them; the combined operation stays rejected. State that the acceptance used a temporary host-path registry, and that `conf/controlnets.yaml` is unchanged.

- [ ] **Step 7: Verify provenance**

Run: `drift refs docs/MPS_BACKEND.md` then `drift check --changed docs/MPS_BACKEND.md`
Expected: ok. Update prose before any relink.

- [ ] **Step 8: Commit**

```bash
git add spikes/mps_controlnet_acceptance.py docs/MPS_BACKEND.md
git commit -m "docs(mps): record the live Canny ControlNet acceptance (STABL-xomvolgw) - next: human review, then open the pull request"
```

---

## Open questions this plan answers, and how

- **Which control types?** Canny only, by scope, enforced at admission and repeated in the worker. Depth and Pose stay registered and stay rejected on MPS. The slice adds no `mediapipe` dependency.
- **Does the ControlNet need float32 on MPS?** Unanswered by design. Task 3 keeps `self.dtype`; Task 5 Step 5 measures, and only a failed validity check justifies float32.
- **Does step progress inject on the ControlNet path?** Yes. Task 3 Step 4 injects it, and Task 3 Step 1 pins cancellation at a step boundary, following slice 2.
