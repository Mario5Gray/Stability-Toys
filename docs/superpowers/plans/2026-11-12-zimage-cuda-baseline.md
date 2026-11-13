# Z-Image CUDA Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. This repo forbids sub-agent driven development. Steps use checkbox (`- [ ]`) syntax. Check each step off immediately when it is complete.

**Goal:** Add the `zimage` family to the CUDA worker with txt2img, img2img, and a stored int8 Qwen text encoder.

**Architecture:** One neutral family serves Z-Image-Turbo and Z-Image base. A new worker module loads a Diffusers folder with bfloat16 weights. A new store module quantizes the text encoder to int8 one time and reloads it by content digest. The HunyuanDiT family is the template for each seam.

**Tech Stack:** Python 3.12, `diffusers 0.39.0`, `transformers 4.57.6`, `optimum-quanto 0.2.7`, `torch 2.10.0+cu128`, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-zimage-family-design.md` (revision 3). Read it before Task 1.

**Status:** plan revision 3. Revision 2 was approved 2026-11-12. Revision 3 adds four implementation requirements from that approval. Task 8 needs a separate go.

**FP:** umbrella `STABL-kbzlfbts`. Each task below has one child issue. Its id is in the task heading.

## Global Constraints

- Family id is `zimage`. Scheduler id is `flow_match_euler`. Detector kind is `zimage`.
- No dependency change. Do not edit `requirements*.txt`.
- The worker accepts a Diffusers folder only.
- Transformer and VAE dtype is bfloat16. The family sets it. `CUDA_DTYPE` does not change it.
- Text encoder default is int8 (`optimum.quanto.qint8`). `none` gives the bf16 encoder.
- The worker must not download a model file. Set `local_files_only=True` on each `from_pretrained` call.
- CUDA capabilities for `zimage`: img2img true, ControlNet false, combined false. MPS has no `zimage` binding.
- Admitted runtime settings: offload `none`, runtime quantize `none`. Other values are refused before a weight file loads.
- Do not edit `backends/cuda_worker.py`. The new worker imports `CudaWorkerBase` from it.
- No unit test loads real weights. Unit tests use fakes, as in `tests/test_hunyuandit_worker.py`.
- Python env: `conda activate stability-toys`. Use `python`, not `python3`.
- Each commit message names the FP issue, the change, and the exact next step.
- Link each commit to its FP issue when you make it: `fp issue assign <id> --rev <sha>`.
- Prose in code comments, logs, errors, commits, and FP comments follows the Controlled English rule in `AGENTS.md`.
- Use semantic tools for symbol lookup. Do not use `grep` to find a definition or a caller.
- Before each edit of a file with a drift binding, do the review in Task 0.

## Plan refinements of the spec

The spec is aligned with these three points.

- The worker is in a new module, `backends/zimage_cuda_worker.py`. `backends/cuda_worker.py` has 2064 lines and stale drift anchors.
- The store default root is `<parent of the model folder>/.cache/stability-toys/zimage-text-encoder`. The worker gets `model_path`, not `model_root`.
- Mode config validates the value of `text_encoder_quantize`. The worker factory refuses the field for other families, because the family is known only after resolution.

## Review Focus

These conditions are implied by the spec. Each has a test in the named task.

1. A model folder with an absent or empty `text_encoder` shard list. Expect `ZImageEncoderStoreError`, not an empty-digest key. Task 4.
2. A store root that is not writable. Expect `ZImageEncoderStoreError` that names the path, before generation. Task 4.
3. An img2img request with invalid init-image bytes. Expect `RuntimeError("invalid init image bytes: ...")`. Task 6.
4. A Turbo request that carries a negative prompt from an old client. Expect no `negative_prompt` kwarg at the pipeline call. Task 6.
5. A request that carries ControlNet bindings and reaches `run_job` directly. Expect `RuntimeError`, not a silent txt2img. Task 6.

Added after the plan review of 2026-11-12, each with tests in its task: two loaders that miss one store key at the same time (Task 4), an explicit zero strength (Task 6), a host that lacks the Z-Image folders (Task 7).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `utils/model_detector.py` | modify | Set `transformer_kind = "zimage"`. Add `ModelInfo.text_encoder_quantize`. |
| `backends/family_profiles.py` | modify | `ZIMAGE_PROFILE`, predicate, registry row. |
| `backends/scheduler_registry.py` | modify | `flow_match_euler` spec. |
| `server/mode_config.py` | modify | `ModeConfig.text_encoder_quantize`: parse, validate value, serialize. |
| `backends/model_resolution.py` | modify | Carry the field through the overlay and the snapshot. |
| `backends/worker_factory.py` | modify | Refuse the field for a family other than `zimage`. |
| `backends/zimage_text_encoder.py` | create | Store key, manifest, atomic write, load or build. |
| `backends/zimage_cuda_worker.py` | create | Admission, load, txt2img, img2img. |
| `backends/platforms/cuda_bindings.py` | modify | `zimage` binding row. |
| `conf/modes.yml` | modify | `ZImage-Turbo`, `ZImage-Base`. |
| `docs/WORKER_SELECTION.md` | modify | Name the new family and worker. |
| `tests/test_zimage_family.py` | create | Detector, family, bindings. |
| `tests/test_zimage_text_encoder.py` | create | Store tests. |
| `tests/test_zimage_worker.py` | create | Worker tests. |
| `tests/test_scheduler_registry.py`, `tests/test_mode_config.py`, `tests/test_model_resolution.py`, `tests/test_worker_factory.py` | modify | New cases. |

---

### Task 0: Branch and stale-binding review (STABL-kbzlfbts)

`drift check` exits 1 on `main` with 37 stale anchors in existing docs. This task records which of them touch this plan. It changes no code.

**Files with drift bindings that this plan edits:**

| File | Bound docs |
|---|---|
| `utils/model_detector.py` | `docs/superpowers/specs/2026-07-16-hunyuandit-family-profile-design.md` |
| `backends/worker_factory.py` | `docs/WORKER_SELECTION.md`, `docs/superpowers/specs/2026-04-13-explicit-backend-provider-design.md`, the 2026-07-16 spec |
| `server/mode_config.py` | six docs. Run `drift refs server/mode_config.py`. |
| `conf/modes.yml` | `docs/superpowers/specs/2026-04-16-chat-connection-decoupling-design.md`, the 2026-07-16 spec |

`backends/family_profiles.py`, `backends/scheduler_registry.py`, `backends/model_resolution.py`, and `backends/platforms/cuda_bindings.py` have no binding.

- [x] **Step 1: Create the worktree**

```bash
git worktree add .worktrees/zimage-cuda -b feat/stabl-kbzlfbts-zimage-cuda main
cd .worktrees/zimage-cuda
```

`conf/modes.yml` on `main` has user changes that are not committed. The worktree does not have them. Tell the user before the merge.

- [x] **Step 2: Commit the spec, this plan, and the spike script on the branch**

Copy the three untracked files from the main checkout: the spec, this plan, `spikes/zimage_encoder_placement.py`. Add one line for the spike to `spikes/README.md`.

```bash
git add docs/superpowers/specs/2026-11-12-zimage-family-design.md \
        docs/superpowers/plans/2026-11-12-zimage-cuda-baseline.md \
        spikes/zimage_encoder_placement.py spikes/README.md
git commit -m "docs(zimage): add family design, slice 1 plan, encoder spike (STABL-kbzlfbts) — next: Task 0 step 3 drift baseline"
```

- [ ] **Step 3: Record the drift baseline**

Stale anchors are compared as (document, target) pairs. A bare `STALE` line does not identify the document.

```bash
export LC_ALL=C   # byte order for sort and comm. The committed pair files use it.
drift_pairs() {  # $1 = raw log, prints "document<TAB>target and reason", sorted
  awk '/^[^ \t]/ {doc=$0; next} /^[ \t]+STALE/ {sub(/^[ \t]+STALE[ \t]+/, ""); print doc "\t" $0}' "$1" | LC_ALL=C sort -u
}
drift check > /tmp/zimage-drift-baseline.log 2>&1; echo "exit=$?"
drift_pairs /tmp/zimage-drift-baseline.log > /tmp/zimage-drift-baseline.pairs
wc -l < /tmp/zimage-drift-baseline.pairs; grep -c "^[^ ]" /tmp/zimage-drift-baseline.log
```

Expected: `exit=1`, a pair count of 37 or more, and a document count above 0. Record both counts in the FP comment of step 5.

If the target text carries an anchor or symbol id, the pair keeps it, because the pair holds the whole line.

Stop and report when: the exit code is not 0 or 1, the log has no document line, or the log has a line that starts with `error`. Such a result is a command failure, not a baseline.

Result on 2026-11-12, branch base `81cd3df`: `exit=1`, 34 pairs, 23 documents. The 37 pairs above came from the `main` checkout at a later state. That set is a superset: two `Makefile` pairs from `main` commit `c43ccce` and one `conf/controlnets.yaml` pair from uncommitted `main` state. The branch baseline is 34. It is stored in `docs/superpowers/plans/2026-11-12-zimage-cuda-drift/baseline.pairs`. Later steps read that file, not `/tmp`. Both committed pair files are sorted with `LC_ALL=C sort -u`. Each `sort` and `comm` on them must run under `LC_ALL=C`, or `comm` reports false pairs.

- [ ] **Step 4: Review each bound doc for the four files in the table**

For each file: run `drift refs <path>`. For each doc it prints, read the section that names the file. Sort the doc into one class:

- **Historical record** (a dated spec or plan that describes past work): do not edit prose. Do not relink.
- **Living doc that this plan makes wrong**: update the prose in the task that edits the code. Relink after the prose is correct.
- **Living doc that is already wrong for another reason**: do not relink. File one FP issue that names the doc and the stale statement.

`docs/WORKER_SELECTION.md` is a living doc. Task 7 updates it.

- [ ] **Step 5: Record the review in FP**

`fp comment STABL-kbzlfbts "<one line for each bound doc: path, class, action>"`. The comment must also state the baseline count and the next step, Task 1.

**Pass rule for the whole plan:** at the end, `drift check` does not fail. Each (document, target) pair that is absent from the baseline must be in the historical exception list.

**Historical exceptions (added after the Task 3 review).** A code edit makes each historical doc bound to that file stale. This plan does not edit or relink historical docs. Thus the original "no new pair" rule cannot pass. The seven accepted pairs are in `docs/superpowers/plans/2026-11-12-zimage-cuda-drift/historical-exceptions.pairs`:

- Six historical docs against `server/mode_config.py` (Task 3): the 2026-04-11, 2026-04-16, and 2026-07-10 plans, the 2026-04-16, 2026-07-09, and 2026-07-16 specs.
- The 2026-07-16 HunyuanDiT spec against `utils/model_detector.py` (Task 1).

The list is closed. A new pair that is not in it fails the gate. Do not add a pair to the list without a review.

`docs/WORKER_SELECTION.md` against `backends/worker_factory.py` (Task 3) is not an exception. It is a living doc. Task 7 updates its prose and relinks it, so the pair must be absent at the end.

---

### Task 1: Detection and family profile (STABL-mioumhii)

**Files:**
- Modify: `utils/model_detector.py` (transformer block near line 370)
- Modify: `backends/family_profiles.py`
- Create: `tests/test_zimage_family.py`
- Modify: `tests/test_family_profiles.py` only if a test there asserts the exact registry length

**Interfaces:**
- Produces: `info.transformer_kind == "zimage"`. `backends.family_profiles.ZIMAGE_PROFILE`. `resolve_family(info).profile.family_id == "zimage"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_zimage_family.py
import json
import pytest
from utils.model_detector import ModelInfo, detect_model
from backends.family_profiles import (
    FAMILY_REGISTRY, ZIMAGE_PROFILE, family_ids, resolve_family,
)

def _zimage_dir(tmp_path, transformer_class="ZImageTransformer2DModel"):
    root = tmp_path / "Z-Image-Turbo-BF16"
    (root / "transformer").mkdir(parents=True)
    (root / "text_encoder").mkdir()
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "ZImagePipeline",
        "transformer": ["diffusers", transformer_class],
        "text_encoder": ["transformers", "Qwen3Model"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": transformer_class, "dim": 3840}))
    (root / "text_encoder" / "config.json").write_text(json.dumps({"hidden_size": 2560}))
    return root

def test_detector_marks_zimage_folder(tmp_path):
    info = detect_model(str(_zimage_dir(tmp_path)))
    assert info.base_arch == "transformer"
    assert info.transformer_kind == "zimage"
    assert info.loader_format == "diffusers_dir"
    assert info.cross_attention_dim is None

def test_unknown_transformer_class_is_not_zimage(tmp_path):
    info = detect_model(str(_zimage_dir(tmp_path, "SomeOtherTransformer")))
    assert info.transformer_kind is None

def test_profile_values():
    assert ZIMAGE_PROFILE.family_id == "zimage"
    assert ZIMAGE_PROFILE.encoder_roles == ("text_encoder",)
    assert ZIMAGE_PROFILE.pooled_required is False
    assert ZIMAGE_PROFILE.pooled_projection_role is None
    assert ZIMAGE_PROFILE.control_image_kwarg == "control_image"

def test_resolve_family_selects_exactly_zimage(tmp_path):
    info = detect_model(str(_zimage_dir(tmp_path)))
    assert resolve_family(info).profile is ZIMAGE_PROFILE
    assert "zimage" in family_ids()

def test_zimage_predicate_rejects_other_families():
    detect = next(r.detect for r in FAMILY_REGISTRY if r.profile is ZIMAGE_PROFILE)
    assert not detect(ModelInfo(path="x", base_arch="transformer", transformer_kind="hunyuandit"))
    assert not detect(ModelInfo(path="x", base_arch="unet", cross_attention_dim=2048))
```

Check the `ModelInfo` constructor and the `resolve_family` signature with the language server before you run the tests. Adapt the two calls if a required argument differs.

- [ ] **Step 2: Run the tests and confirm that they fail**

Run: `python -m pytest tests/test_zimage_family.py -q`
Expected: FAIL with `ImportError: cannot import name 'ZIMAGE_PROFILE'`.

- [ ] **Step 3: Implement**

- In `utils/model_detector.py`: where `_class_name == "HunyuanDiT2DModel"` sets the kind, add the `ZImageTransformer2DModel` case that sets `"zimage"`.
- In `backends/family_profiles.py`: add `ZIMAGE_PROFILE` with the values in the test, `_is_zimage(info)` that reads only `info.transformer_kind == "zimage"`, and one `FamilyRegistration` row at the end of `FAMILY_REGISTRY`.

- [ ] **Step 4: Run the tests and confirm that they pass**

Run: `python -m pytest tests/test_zimage_family.py tests/test_family_profiles.py tests/test_model_detector.py tests/test_hunyuandit_detector.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add utils/model_detector.py backends/family_profiles.py tests/test_zimage_family.py
git commit -m "feat(zimage): detect Z-Image folders and add the zimage family profile (STABL-mioumhii) — next: Task 2 step 1 scheduler tests"
```

---

### Task 2: Flow-match scheduler id (STABL-zqonvkok)

**Files:**
- Modify: `backends/scheduler_registry.py`
- Modify: `tests/test_scheduler_registry.py`

**Interfaces:**
- Produces: scheduler id `flow_match_euler`. `build_scheduler("flow_match_euler", config)` returns a `FlowMatchEulerDiscreteScheduler`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from backends.scheduler_registry import build_scheduler, get_scheduler_class, list_scheduler_ids

def test_flow_match_euler_is_registered():
    assert "flow_match_euler" in list_scheduler_ids()
    assert get_scheduler_class("flow_match_euler").__name__ == "FlowMatchEulerDiscreteScheduler"

@pytest.mark.parametrize("shift", [3.0, 6.0])
def test_flow_match_euler_keeps_checkpoint_shift(shift):
    from diffusers import FlowMatchEulerDiscreteScheduler
    baseline = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    rebuilt = build_scheduler("flow_match_euler", baseline.config)
    assert rebuilt.config.shift == shift
    assert rebuilt.config.use_dynamic_shifting is False
```

If `tests/test_scheduler_registry.py` stubs `diffusers`, put the `shift` test behind `pytest.importorskip("diffusers")` in a way that still runs in the conda env.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_scheduler_registry.py -q -k flow_match`
Expected: FAIL with `Unknown scheduler_id 'flow_match_euler'`.

- [ ] **Step 3: Implement**

Add to `SCHEDULER_SPECS`:

```python
"flow_match_euler": SchedulerSpec(
    "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler"
),
```

- [ ] **Step 4: Run and confirm pass**

Run: `python -m pytest tests/test_scheduler_registry.py -q`
Expected: all pass. A test that asserts the exact id list needs the new id.

- [ ] **Step 5: Commit**

```bash
git add backends/scheduler_registry.py tests/test_scheduler_registry.py
git commit -m "feat(scheduler): add flow_match_euler id for Z-Image (STABL-zqonvkok) — next: Task 3 step 1 mode field tests"
```

---

### Task 3: Mode field `text_encoder_quantize` (STABL-breblkmz)

**Files:**
- Modify: `server/mode_config.py` (`ModeConfig`, the parse call near line 388, the capability list near line 846, the dict near line 1105)
- Modify: `utils/model_detector.py` (`ModelInfo`, capability block)
- Modify: `backends/model_resolution.py` (`ModelInfoSnapshot`, `_SCALAR_SNAPSHOT_FIELDS`, `merge_mode_capabilities`)
- Modify: `backends/worker_factory.py` (`create_bound_worker`)
- Modify: `tests/test_mode_config.py`, `tests/test_model_resolution.py`, `tests/test_worker_factory.py`

Do the Task 0 review result for `server/mode_config.py` and `backends/worker_factory.py` before the edit.

**Interfaces:**
- Produces: `ModeConfig.text_encoder_quantize: Optional[str]`, values `"int8"`, `"none"`, or `None`.
- Produces: `ModelInfo.text_encoder_quantize: Optional[str] = None`, carried through the snapshot and the wire form.
- Produces: `backends.worker_factory.UnsupportedModeFieldError(ValueError)`.

- [ ] **Step 1: Write the failing tests**

Follow the fixture style of each test file. The assertions:

```python
# tests/test_mode_config.py
def test_text_encoder_quantize_parses():           # mode yaml has text_encoder_quantize: int8
    assert mode.text_encoder_quantize == "int8"
def test_text_encoder_quantize_defaults_to_none(): # field absent
    assert mode.text_encoder_quantize is None
def test_text_encoder_quantize_rejects_unknown():  # value "fp8"
    with pytest.raises(ValueError, match="text_encoder_quantize"): load(...)
def test_text_encoder_quantize_round_trips():      # the save/to-dict path near line 1105
    assert dumped["modes"]["Z"]["text_encoder_quantize"] == "none"

# tests/test_model_resolution.py
def test_mode_overlay_carries_text_encoder_quantize():
    assert merge_mode_capabilities(info, mode).text_encoder_quantize == "int8"
def test_snapshot_wire_round_trip_keeps_text_encoder_quantize():
    wire = resolved_model_to_json_dict(resolved)
    assert resolved_model_from_json_dict(wire).info.text_encoder_quantize == "int8"

# tests/test_worker_factory.py
def test_factory_refuses_text_encoder_quantize_for_other_family():
    # resolved family sd15, info.text_encoder_quantize == "int8"
    with pytest.raises(UnsupportedModeFieldError, match="text_encoder_quantize.*sd15"):
        create_bound_worker(...)
def test_factory_admits_text_encoder_quantize_for_zimage():
    # resolved family zimage, a fake worker class in the bindings
    assert create_bound_worker(...) is not None
```

The analysis-test fixture in this repo is `dedent`-ed. A `.replace()` on a base YAML needs 4-space targets. Assert that the replacement landed before you trust a RED result.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_mode_config.py tests/test_model_resolution.py tests/test_worker_factory.py -q -k "text_encoder_quantize"`
Expected: FAIL with `AttributeError` or `TypeError` on the new field.

- [ ] **Step 3: Implement**

- Add the field to `ModeConfig`, `ModelInfo`, and `ModelInfoSnapshot`. Add its name to each of the four field lists.
- In mode parse: accept `None`, `"int8"`, `"none"`. Raise `ValueError(f"Mode '{name}' has invalid text_encoder_quantize '{value}'. Use int8 or none.")` for other values.
- In `create_bound_worker`, before the worker class resolves:

```python
if resolved.info.text_encoder_quantize is not None and family_id != "zimage":
    raise UnsupportedModeFieldError(
        f"text_encoder_quantize is valid only for family 'zimage'; got family {family_id!r}"
    )
```

- [ ] **Step 4: Run and confirm pass**

Run: `python -m pytest tests/test_mode_config.py tests/test_model_resolution.py tests/test_worker_factory.py tests/test_analysis_mode_config.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add server/mode_config.py utils/model_detector.py backends/model_resolution.py backends/worker_factory.py tests/
git commit -m "feat(modes): add text_encoder_quantize mode field, zimage only (STABL-breblkmz) — next: Task 4 step 1 encoder store tests"
```

---

### Task 4: int8 text-encoder store (STABL-ocodiumx)

**Files:**
- Create: `backends/zimage_text_encoder.py`
- Create: `tests/test_zimage_text_encoder.py`

The module imports `torch`, `transformers`, `optimum.quanto`, `safetensors`, and `accelerate` only inside the four private functions. It must import clean without them. `spikes/zimage_encoder_placement.py` (`arm_int8`, `arm_reload`) has the proven quantize, save, and reload calls.

**Interfaces:**
- Produces:

```python
class ZImageEncoderStoreError(RuntimeError): ...

STORE_ENV = "ZIMAGE_ENCODER_STORE"
QUANT_TAG = "qint8"
ENTRY_FILES = ("model.safetensors", "quanto_qmap.json", "config.json", "store_manifest.json")

def encoder_store_root(model_path: str) -> Path
    # env STORE_ENV when set, else <parent of model_path>/.cache/stability-toys/zimage-text-encoder

def compute_store_manifest(encoder_dir: Path, quanto_version: str) -> dict
    # {"key_inputs": {"config_sha256": str,
    #                 "shard_sha256": [str, ...],      # sorted by digest value
    #                 "quantization": "qint8",
    #                 "quanto_version": str},
    #  "diagnostic": {"shards": [{"name": str, "bytes": int, "sha256": str}, ...],
    #                 "encoder_dir": str}}

def store_key(manifest: dict) -> str
    # SHA-256 hex of json.dumps(manifest["key_inputs"], sort_keys=True, separators=(",", ":"))
    # "diagnostic" is never a key input.

def load_int8_text_encoder(model_path: str, *, device: str, dtype: Any,
                           store_root: Path | None = None) -> Any

# Private seams. Tests patch these four.
def _load_bf16_encoder(encoder_dir: Path, dtype: Any) -> Any
def _quantize_and_freeze(encoder: Any, device: str) -> Any
def _save_entry(encoder: Any, entry_dir: Path) -> None      # writes the first three ENTRY_FILES
def _load_entry(entry_dir: Path, device: str) -> Any
```

Key contract:

- Shards are `text_encoder/*.safetensors`. Hash each file in full, in 8 MiB chunks.
- The key binds to content only. Shard digests are ordered by digest value. Name, size, time, and folder path are not key inputs.
- A renamed shard with equal content gives the same key. Each shard holds named tensors, so shard order has no meaning.
- Names, sizes, and the folder path go in `diagnostic` for a person to read.

`load_int8_text_encoder` flow:

1. Compute the manifest and the key. Log the key and the seconds used.
2. Create the store root. A failure raises `ZImageEncoderStoreError` that names the path.
3. Take an exclusive lock for the key: `fcntl.flock` on `root / f"{key}.lock"`. Hold it until step 6 completes. The lock serializes threads and processes.
4. Check for `entry = root / key` after the lock is held. This recheck is the only hit test.
5. Hit: read `store_manifest.json`. `key_inputs` that differ from the computed `key_inputs` raise `ZImageEncoderStoreError`. Then return `_load_entry(entry, device)`. An exception from `_load_entry` is raised again as `ZImageEncoderStoreError`. The function does not rebuild.
6. Miss: remove each stale `root / ".tmp-{key}-*"` directory. That is safe, because the lock is held. Call `_load_bf16_encoder`, then `_quantize_and_freeze`. Write into `tempfile.mkdtemp(dir=root, prefix=f".tmp-{key}-")`. Write the manifest last. Then `os.replace` the directory to `entry`. Remove the temporary directory on a failure. Return the quantized encoder.
7. Log `store hit` or `store miss` with the key. Release the lock in `finally`.

Lock file rules:

- The lock path is stable: `root / f"{key}.lock"`. It does not change between calls.
- Each call of `load_int8_text_encoder` opens its own file descriptor for the lock and closes it at the end.
- No code path unlinks a lock file. Stale-directory cleanup removes only `.tmp-{key}-*` directories.
- The lock protects the construction of one store entry. It does not limit total GPU memory across different models.

A second loader that waits on the lock sees a hit at step 4. One key is quantized one time, so two loaders do not hold two bf16 encoders on the GPU.

- [ ] **Step 1: Write the failing tests**

Each test uses `tmp_path`, small fake shard files, and `monkeypatch` on the four private seams. `quanto_version` is the literal `"0.2.7"` in key tests.

```python
def test_key_is_stable_for_equal_content()
    # two folders, same bytes -> same store_key
def test_key_changes_when_content_changes_and_size_name_time_stay_equal()
    # rewrite one shard with different bytes of the same length, os.utime back to the old times
    # -> store_key differs
def test_key_ignores_shard_time()
    # os.utime only -> same store_key
def test_key_changes_with_quanto_version()
def test_key_is_equal_after_shard_rename()
    # rename model-00001-of-00002.safetensors -> a.safetensors, same bytes -> same store_key
def test_key_ignores_folder_path()
    # same content under two different parent folders -> same store_key
def test_shard_names_are_diagnostic_only()
    # manifest["diagnostic"]["shards"] has the names; "name" appears nowhere in manifest["key_inputs"]
def test_shard_digests_are_sorted_by_digest_value()
def test_no_shards_raises()                       # Review Focus 1
    # text_encoder/ has config.json only -> ZImageEncoderStoreError, match "no weight shards"
def test_store_root_uses_env(monkeypatch)
def test_store_root_default_is_beside_model_folder()
    # encoder_store_root("/models/repos/X") == Path("/models/repos/.cache/stability-toys/zimage-text-encoder")
def test_miss_builds_and_stores()
    # _load_bf16_encoder called once, entry dir has all four ENTRY_FILES, no ".tmp-" dir remains
def test_hit_does_not_load_bf16_encoder()
    # second call: _load_bf16_encoder not called, _load_entry called once
def test_interrupted_write_leaves_no_entry()
    # _save_entry raises -> ZImageEncoderStoreError or the original error, entry dir absent, no ".tmp-" dir
def test_simultaneous_misses_quantize_once()
    # two threads call load_int8_text_encoder for one key; _quantize_and_freeze blocks on an Event
    # until both threads have started -> _load_bf16_encoder call count is 1, both calls return,
    # one entry dir, no ".tmp-" dir
def test_simultaneous_misses_in_two_processes_quantize_once()
    # two multiprocessing "spawn" children with fake seams that append one line to a shared file
    # on each build -> the file has 1 line, both children exit 0
def test_temp_dirs_are_unique_per_writer()
    # patch tempfile.mkdtemp to record its results over two forced builds of different keys
    # in one process -> the recorded paths differ and none ends in the bare pid
def test_stale_temp_dir_from_dead_writer_is_removed()
    # create root/".tmp-<key>-dead" with a partial file, then call on a miss ->
    # build succeeds, the stale dir is gone, the entry has all four ENTRY_FILES
def test_lock_file_is_kept_after_build_and_cleanup()
    # after a miss with a stale ".tmp-<key>-dead" dir present: root/"<key>.lock" still exists
def test_each_call_opens_its_own_lock_descriptor()
    # record os.open / open calls on the lock path over two calls -> two opens, two closes,
    # and the path is equal in both calls
def test_lock_is_released_after_failure()
    # first call: _save_entry raises; second call with working seams succeeds without a block
def test_manifest_mismatch_fails_load()
    # edit key_inputs in store_manifest.json in the entry -> ZImageEncoderStoreError, match "manifest"
def test_corrupt_entry_is_an_error_not_a_rebuild()
    # _load_entry raises -> ZImageEncoderStoreError, _load_bf16_encoder not called
def test_unwritable_store_root_raises()           # Review Focus 2
    # store_root is a path under a regular file -> ZImageEncoderStoreError that names the path
def test_module_imports_without_torch()
    # subprocess: python -c "import sys; sys.modules['torch']=None; import backends.zimage_text_encoder"
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_zimage_text_encoder.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.zimage_text_encoder'`.

- [ ] **Step 3: Implement the module to the interface above**

- [ ] **Step 4: Run and confirm pass**

Run: `python -m pytest tests/test_zimage_text_encoder.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backends/zimage_text_encoder.py tests/test_zimage_text_encoder.py
git commit -m "feat(zimage): add content-keyed int8 text-encoder store (STABL-ocodiumx) — next: Task 5 step 1 worker admission tests"
```

---

### Task 5: Worker admission, load, and binding (STABL-ezdrotph)

**Files:**
- Create: `backends/zimage_cuda_worker.py`
- Modify: `backends/platforms/cuda_bindings.py`
- Create: `tests/test_zimage_worker.py`
- Modify: `tests/test_zimage_family.py` (binding tests)

**Interfaces:**
- Consumes: `load_int8_text_encoder(model_path, *, device, dtype)` from Task 4. `ZIMAGE_PROFILE` from Task 1. `model_info.text_encoder_quantize` from Task 3.
- Produces:

```python
class ZImageDependencyError(RuntimeError): ...
class ZImageUnsupportedRuntimeError(RuntimeError): ...

class DiffusersZImageCudaWorker(CudaWorkerBase):
    family_profile = ZIMAGE_PROFILE
    supports_attention_processor_swap = False
    log_tag = "zimage-cuda"
    def __init__(self, worker_id: int, model_path: str,
                 model_info: Any | None = None, family_profile: FamilyProfile | None = None)
    def _quantization_targets(self, pipe) -> tuple   # raises ZImageUnsupportedRuntimeError
```

Constructor order. Each item must come before the next:

1. `super().__init__(worker_id, model_info=model_info, family_profile=family_profile)`.
2. Dependency preflight: resolve `diffusers.ZImagePipeline` and `diffusers.ZImageImg2ImgPipeline` lazily. A failure raises `ZImageDependencyError` with the installed `diffusers` and `transformers` versions.
3. Admission. Raise `ZImageUnsupportedRuntimeError` when:
   - the path is not a Diffusers folder (the same test as the HunyuanDiT worker),
   - `self._offload != "none"`,
   - `self._quantize != "none"`,
   - `text_encoder_quantize` is not `None`, `"int8"`, or `"none"`.

   Message format: `Z-Image does not support <setting>=<value>. Admitted: <values>.`
4. Set `self.dtype = torch.bfloat16` and `self.dtype_str = "bf16"`. Log a warning when `CUDA_DTYPE` asked for a different value.
5. Encoder: when the mode value is `None` or `"int8"`, call `load_int8_text_encoder(model_path, device=self.device, dtype=self.dtype)` and give the result as `text_encoder=`. When it is `"none"`, give no `text_encoder` argument.
6. `ZImagePipeline.from_pretrained(model_path, torch_dtype=self.dtype, local_files_only=True, **encoder_kw)`.
7. `self._setup_pipe_memory_opts(pipe)`, `self._capture_baseline_scheduler(self.pipe)`, `self._style_api = "none"`.
8. One info log line: worker id, folder name, device, dtype, encoder mode.

Binding row in `backends/platforms/cuda_bindings.py`:

```python
# Z-Image: txt2img and plain img2img. ControlNet is slice 2.
"zimage": FamilyPlatformBinding(
    "backends.zimage_cuda_worker.DiffusersZImageCudaWorker",
    ExecutionCapabilities(True, False, False),
),
```

- [ ] **Step 1: Write the failing tests**

Copy the stub preamble and `_FakePipelineBase` pattern from `tests/test_hunyuandit_worker.py` (lines 22 to 100). Patch the lazy resolvers and `load_int8_text_encoder` in `backends.zimage_cuda_worker`, where the call happens.

```python
def test_selecting_sd_family_does_not_import_zimage_worker()
    # creating the sd15 worker leaves "backends.zimage_cuda_worker" out of sys.modules
def test_preflight_failure_reports_versions()
    # resolver raises ImportError -> ZImageDependencyError, message has "diffusers" and "transformers"
def test_non_folder_path_is_refused()
    # "/x/model.safetensors" -> ZImageUnsupportedRuntimeError; from_pretrained not called
@pytest.mark.parametrize("offload", ["model", "sequential"])
def test_offload_is_refused_before_load(offload)
    # model_info.runtime_offload=offload -> error names "offload" and the value;
    # from_pretrained and load_int8_text_encoder not called
def test_env_offload_is_refused_before_load(monkeypatch)       # CUDA_OFFLOAD=model
def test_runtime_quantize_is_refused_before_load()             # runtime_quantize="fp8"
def test_env_quantize_is_refused_before_load(monkeypatch)      # CUDA_QUANTIZE=fp8
def test_dtype_is_bf16_even_when_env_asks_fp16(monkeypatch)
    # CUDA_DTYPE=fp16 -> from_pretrained kwargs["torch_dtype"] is torch.bfloat16
def test_default_uses_int8_store()
    # text_encoder_quantize=None -> load_int8_text_encoder called once;
    # from_pretrained kwargs["text_encoder"] is its return value
def test_none_loads_native_encoder()
    # text_encoder_quantize="none" -> load_int8_text_encoder not called; no "text_encoder" kwarg
def test_from_pretrained_is_offline()
    # kwargs["local_files_only"] is True
def test_attention_swaps_are_skipped()
    # CUDA_ATTENTION_SLICING=1 -> pipe.enable_attention_slicing not called
def test_quantization_targets_raises()

# tests/test_zimage_family.py
def test_cuda_binding_for_zimage():
    cell = CUDA_FAMILY_BINDINGS["zimage"]
    assert cell.worker_ref == "backends.zimage_cuda_worker.DiffusersZImageCudaWorker"
    assert cell.capabilities == ExecutionCapabilities(True, False, False)
def test_mps_has_no_zimage_binding():
    assert "zimage" not in MPS_FAMILY_BINDINGS
```

Check the attribute name of the capabilities on `FamilyPlatformBinding` with the language server.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_zimage_worker.py tests/test_zimage_family.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.zimage_cuda_worker'`.

- [ ] **Step 3: Implement the worker constructor and the binding row**

- [ ] **Step 4: Run and confirm pass**

Run: `python -m pytest tests/test_zimage_worker.py tests/test_zimage_family.py tests/test_hunyuandit_worker.py tests/test_platform_registry.py -q`
Expected: all pass. Run the two worker test files together and in both orders. An order-dependent failure means a split `torch` binding: patch where the call happens.

- [ ] **Step 5: Commit**

```bash
git add backends/zimage_cuda_worker.py backends/platforms/cuda_bindings.py tests/test_zimage_worker.py tests/test_zimage_family.py
git commit -m "feat(zimage): add CUDA worker load path with runtime admission (STABL-ezdrotph) — next: Task 6 step 1 run_job tests"
```

---

### Task 6: txt2img and img2img execution (STABL-wpnyattk)

**Files:**
- Modify: `backends/zimage_cuda_worker.py`
- Modify: `tests/test_zimage_worker.py`

**Interfaces:**
- Consumes: the worker from Task 5. `inject_step_progress` from `backends.step_progress`. `DelegatedConditioning` from `backends.conditioning.artifacts`.
- Produces: `run_job(self, job, progress=None, should_cancel=None) -> tuple[bytes, int]`, the same contract as `DiffusersHunyuanDiTCudaWorker.run_job`.

Rules for `run_job`:

- Non-empty `job.controlnet_bindings`, or a truthy `req.controlnets`, raises `RuntimeError("Z-Image does not support ControlNet")`.
- Size parse, seed choice, and generator: the same as the HunyuanDiT worker.
- `scheduler_id = self._apply_request_scheduler(req)`.
- Conditioning: `self._conditioning_artifact_for_request(req)` must return a `DelegatedConditioning`. Another type raises `ValueError("Z-Image supports native conditioning only")`.
- Negative prompt: send `negative_prompt` only when it is non-empty and the mode permits one. The mode permits one when `model_info.allow_custom_negative_prompt` is true or `model_info.negative_prompt_templates` is non-empty.
- txt2img kwargs: `prompt`, optional `negative_prompt`, `width`, `height`, `num_inference_steps`, `guidance_scale`, `generator`. No `use_resolution_binning`.
- img2img: decode the init image as RGB and resize it to the request size. Invalid bytes raise `RuntimeError(f"invalid init image bytes: {e!r}")`. Build the pipe one time and keep it in `self._img2img_pipe`: `ZImageImg2ImgPipeline.from_pipe(self.pipe, torch_dtype=None)`. Use that exact attribute. The base-class scheduler helpers update `self._img2img_pipe.scheduler` with the base pipe.
- `strength`: read `getattr(req, "denoise_strength", None)`. Use 0.75 only when the value is `None`. Do not use `or`.
- A strength that is not in the range `0 < strength <= 1` raises `ValueError(f"Z-Image img2img needs 0 < denoise_strength <= 1; got {strength}")`. Zero is refused, not replaced.
- Effective steps are `steps - int(max(steps - min(steps * strength, steps), 0))`. This is the formula of `ZImageImg2ImgPipeline.get_timesteps`. A result below 1 raises `ValueError` that names the steps, the strength, and the result. The pipe is not called.
- img2img logs the effective step count.
- Add kwargs `image` and `strength`.
- Peak memory: the worker has a method `_reset_peak_memory()` and a method `_read_peak_memory() -> tuple[int, int]` (allocated bytes, reserved bytes). On CUDA they call `torch.cuda.synchronize`, `torch.cuda.reset_peak_memory_stats`, `torch.cuda.max_memory_allocated`, and `torch.cuda.max_memory_reserved` for the worker device. `run_job` resets immediately before the pipe call and reads after it, in `finally`, before `release_cache`.
- Each job logs one info line: `[zimage-cuda] job=<job_id> op=<txt2img|img2img> size=<WxH> steps=<n> peak_allocated_bytes=<int> peak_reserved_bytes=<int> peak_allocated_gib=<x.xx> peak_reserved_gib=<x.xx>`. Task 8 reads this line. The byte values are exact. The GiB values are for a person to read.
- Memory instrumentation must not change the result of a job. `_reset_peak_memory` and `_read_peak_memory` calls are each in their own `try`. A failure logs one warning, and the job line shows `peak_allocated_bytes=unavailable`.
- A failed memory read must not replace a generation error, and must not stop `release_cache`.
- A failed reset makes the counters stale. The job line then shows `unavailable` for all four peak values, and the read is skipped.
- Call `inject_step_progress(pipe, pipe_kwargs, progress, req.num_inference_steps, should_cancel=should_cancel)` before the pipe call, inside `torch.inference_mode()`.
- PNG `lcm` text chunk: the same keys as the HunyuanDiT worker. img2img adds `denoise_strength`.
- `finally`: clear local references and call `self.execution_policy.release_cache()`.

- [x] **Step 1: Write the failing tests**

```python
def test_txt2img_kwargs()
    # 1024x768, steps 9, guidance 0.0, seed 42 -> the ZImagePipeline fake gets exactly:
    # prompt, width=1024, height=768, num_inference_steps=9, guidance_scale=0.0, generator
    # and no "use_resolution_binning", no "negative_prompt"
def test_turbo_mode_drops_negative_prompt()               # Review Focus 4
    # allow_custom_negative_prompt False, templates {} , req.negative_prompt="blurry"
    # -> "negative_prompt" not in call kwargs
def test_base_mode_sends_negative_prompt()
    # templates non-empty, req.negative_prompt="blurry, low quality, watermark" -> kwarg equals it
def test_img2img_uses_img2img_pipeline_without_recast()
    # from_pipe called with (worker.pipe, torch_dtype=None); call kwargs have "image" and strength 0.6
def test_img2img_pipeline_is_built_once()
    # two img2img jobs -> from_pipe call count is 1
def test_img2img_pipeline_is_kept_in_base_attribute()
    # after one img2img job: worker._img2img_pipe is the from_pipe result
def test_img2img_default_strength_is_075_when_absent()
    # req has no denoise_strength attribute, and req.denoise_strength=None -> strength 0.75
def test_img2img_zero_strength_is_refused_not_defaulted()
    # req.denoise_strength=0.0 -> ValueError, match "0 < denoise_strength"; pipe not called
@pytest.mark.parametrize("strength", [-0.1, 1.5])
def test_img2img_out_of_range_strength_is_refused(strength)
def test_img2img_zero_effective_steps_is_refused()
    # steps 9, strength 1e-17 -> 9 - int(9 - 9e-17) == 0 -> ValueError that names 9, 1e-17, and 0; pipe not called
    # steps 9, strength 0.1 -> effective 1 -> the pipe runs (test_img2img_low_strength_runs)
def test_img2img_smallest_valid_strength_runs()
    # steps 9: strength 0.6 -> 6, 0.12 -> 2, 0.1 -> 1, 1.0 -> 9 (test_effective_steps_match_diffusers_get_timesteps)
def test_scheduler_stays_in_step_over_mixed_jobs()
    # txt2img, img2img, txt2img, img2img in one worker, each with scheduler_id flow_match_euler ->
    # after each job worker._img2img_pipe.scheduler is worker.pipe.scheduler (once the img2img pipe exists)
def test_cancel_then_next_job_runs()
    # should_cancel returns True in the step callback -> concurrent.futures.CancelledError;
    # release_cache called; a following txt2img job on the same worker returns PNG bytes
def test_failed_img2img_then_txt2img_runs()
    # img2img pipe raises RuntimeError -> error propagates; a following txt2img returns PNG bytes
    # and worker._img2img_pipe is still the same object
def test_job_logs_peak_allocated(caplog)
    # _read_peak_memory patched to return (19 * 2**30, 20 * 2**30) ->
    # one log line with "peak_allocated_gib=19.00" and "peak_reserved_gib=20.00" and "op=txt2img"
def test_peak_is_reset_before_and_read_after_the_pipe_call()
    # record call order: _reset_peak_memory, pipe call, _read_peak_memory
def test_peak_is_logged_when_the_pipe_raises()
def test_job_log_has_exact_bytes(caplog)
    # _read_peak_memory returns (19 * 2**30 + 1, 20 * 2**30) -> the line has
    # "peak_allocated_bytes=20401094657" and "peak_allocated_gib=19.00"
def test_memory_read_failure_does_not_fail_the_job(caplog)
    # _read_peak_memory raises RuntimeError -> run_job returns PNG bytes and the seed;
    # one warning; the job line has "peak_allocated_bytes=unavailable"; release_cache called once
def test_memory_reset_failure_does_not_fail_the_job()
    # _reset_peak_memory raises -> the pipe is still called and run_job returns PNG bytes
def test_memory_read_failure_does_not_mask_a_generation_error()
    # pipe raises ValueError("boom") and _read_peak_memory raises RuntimeError ->
    # run_job raises ValueError("boom"); release_cache called once
def test_invalid_init_image_bytes_raise()                 # Review Focus 3
    # init_image=b"not an image" -> RuntimeError, match "invalid init image bytes"
def test_controlnet_binding_is_refused()                  # Review Focus 5
    # job.controlnet_bindings=[object()] -> RuntimeError, match "does not support ControlNet"; pipe not called
def test_invalid_size_raises()
    # req.size="big" -> RuntimeError, match "Invalid size"
def test_materialized_conditioning_is_refused()
def test_png_metadata_records_request()
    # decode returned bytes; the "lcm" chunk has prompt, seed, size, steps, cfg, scheduler_id
def test_release_cache_runs_after_failure()
    # pipe raises -> execution_policy.release_cache called once
```

The fake pipe must return a real small `PIL.Image` so that PNG encode works.

- [x] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_zimage_worker.py -q -k "txt2img or img2img or negative or controlnet or size or conditioning or png or release or scheduler or cancel or peak"`
Expected: FAIL. The base-class `run_job` is absent or raises `NotImplementedError`.

- [x] **Step 3: Implement `run_job`**

- [x] **Step 4: Run and confirm pass**

Run: `python -m pytest tests/test_zimage_worker.py -q`
Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add backends/zimage_cuda_worker.py tests/test_zimage_worker.py
git commit -m "feat(zimage): add txt2img and img2img execution (STABL-wpnyattk) — next: Task 7 step 1 modes"
```

---

### Task 7: Modes, docs, and full verification (STABL-tzctkzmz)

**Files:**
- Modify: `conf/modes.yml`
- Modify: `docs/WORKER_SELECTION.md`
- Modify: `tests/test_mode_config.py` or the test that loads the repo `conf/modes.yml`

Do the Task 0 review result for `conf/modes.yml` before the edit.

- [x] **Step 1: Write the failing test**

Find the test that loads the real `conf/modes.yml`. Add:

```python
def test_zimage_modes_are_defined():
    turbo, base = modes["ZImage-Turbo"], modes["ZImage-Base"]
    assert (turbo.default_steps, turbo.default_guidance) == (9, 0.0)
    assert (base.default_steps, base.default_guidance) == (30, 4.0)
    for m in (turbo, base):
        assert m.allowed_scheduler_ids == ["flow_match_euler"]
        assert m.default_scheduler_id == "flow_match_euler"
        assert m.default_size == "1024x1024"
        assert m.text_encoder_quantize == "int8"
    assert turbo.allow_custom_negative_prompt is False
    assert turbo.negative_prompt_templates == {}
    assert base.allow_custom_negative_prompt is True
    assert base.negative_prompt_templates == {"default": "blurry, low quality, watermark"}
    assert base.default_negative_prompt_template == "default"
```

Add the regression tests for absent model folders. Each uses a temporary config: one mode that points at an existing folder, plus the two Z-Image modes that point at absent folders.

```python
def test_absent_zimage_folders_do_not_break_mode_config_load(tmp_path, caplog)
    # ModeConfigManager loads; get_mode("Existing") works; get_mode("ZImage-Turbo") works;
    # a warning names each absent Z-Image path
def test_absent_zimage_folders_do_not_fail_strict_controlnet_startup_validation(tmp_path)
    # the "Existing" mode has an enabled controlnet_policy with a valid registry fixture;
    # validate_controlnet_mode_references(mode_config=..., registry=...) returns without an error
    # and detect_model is not called for a Z-Image path
def test_default_mode_is_unchanged_by_zimage_modes()
    # repo conf/modes.yml: default_mode equals its value on main
```

- [x] **Step 2: Run and confirm failure**

Expected: `test_zimage_modes_are_defined` FAILS with `KeyError: 'ZImage-Turbo'`. The absent-folder tests must pass after step 3. A failure there is the stop condition above.

- [x] **Step 3: Add the two modes at the end of the `modes:` block**

```yaml
  ZImage-Turbo:
    model: repos/Z-Image-Turbo-BF16
    scheduler_profile: native
    resolution_set: sdxl
    default_size: 1024x1024
    recommended_size: 1024x1024
    default_steps: 9
    default_guidance: 0.0
    allowed_scheduler_ids:
      - flow_match_euler
    default_scheduler_id: flow_match_euler
    text_encoder_quantize: int8
  ZImage-Base:
    model: repos/Z-Image-Base
    scheduler_profile: native
    resolution_set: sdxl
    default_size: 1024x1024
    recommended_size: 1024x1024
    default_steps: 30
    default_guidance: 4.0
    allowed_scheduler_ids:
      - flow_match_euler
    default_scheduler_id: flow_match_euler
    negative_prompt_templates:
      default: "blurry, low quality, watermark"
    default_negative_prompt_template: default
    allow_custom_negative_prompt: true
    text_encoder_quantize: int8
```

Absent Z-Image folders must not break a configuration that works today. Read on 2026-11-12:

- `ModeConfigManager._validate_paths` logs a warning for an absent model path. It does not raise.
- `validate_controlnet_mode_references` in `server/controlnet_registry.py` skips a mode with no enabled `controlnet_policy`. The two Z-Image modes have none.

Step 1 pins both facts with the regression tests below. If one of them fails, stop. Do not add the modes to `conf/modes.yml`. Move them to an opt-in file, `conf/modes.zimage.example.yml`, state that in the FP comment, and ask the user.

Slice 2 adds a `controlnet_policy` to these modes. That change makes strict startup resolve the model path. Slice 2 must handle it.

- [x] **Step 4: Update `docs/WORKER_SELECTION.md`**

Add one line to the worker list near line 29: `DiffusersZImageCudaWorker` for family `zimage` (Z-Image-Turbo, Z-Image base), module `backends.zimage_cuda_worker`. State that MPS has no `zimage` binding. Then relink only this doc: `drift link docs/WORKER_SELECTION.md`. Read the whole doc first. Do not relink if another statement in it is wrong.

- [ ] **Step 5: Run the full verification**

```bash
python -m pytest tests/test_zimage_family.py tests/test_zimage_text_encoder.py tests/test_zimage_worker.py \
  tests/test_scheduler_registry.py tests/test_mode_config.py tests/test_model_resolution.py \
  tests/test_worker_factory.py tests/test_hunyuandit_worker.py tests/test_family_profiles.py -q
docker compose -f docker-compose.test.yml build test
docker compose -f docker-compose.test.yml run --rm test > /tmp/zimage-test-full.log 2>&1; echo "exit=$?"; tail -5 /tmp/zimage-test-full.log
export LC_ALL=C   # comm needs the same byte order as the committed pair files
drift check > /tmp/zimage-drift-after.log 2>&1; echo "exit=$?"
drift_pairs /tmp/zimage-drift-after.log > /tmp/zimage-drift-after.pairs     # function from Task 0 step 3
D=docs/superpowers/plans/2026-11-12-zimage-cuda-drift
comm -13 $D/baseline.pairs /tmp/zimage-drift-after.pairs | comm -23 - $D/historical-exceptions.pairs   # new pairs outside the exceptions
comm -13 $D/baseline.pairs /tmp/zimage-drift-after.pairs | comm -12 - $D/historical-exceptions.pairs   # exceptions present
comm -23 $D/baseline.pairs /tmp/zimage-drift-after.pairs                                             # resolved pairs
```

Expected: pytest passes. The container suite exits 0. Do not read a full log into the session. Filter it.

Drift pass rule:

- The `drift check` exit code is 0 or 1. Another code is a command failure.
- The after log has at least one document line and no line that starts with `error`.
- The first `comm` prints nothing: no new pair outside the historical exceptions. A `docs/WORKER_SELECTION.md` pair here means the relink did not happen.
- The second `comm` prints only pairs from the exception list. Report them in the review request.
- The third `comm` prints only pairs for `docs/WORKER_SELECTION.md`, or nothing. Another resolved pair means an unplanned relink. Find the cause.

- [ ] **Step 6: Commit and request review**

```bash
git add conf/modes.yml docs/WORKER_SELECTION.md drift.lock tests/
git commit -m "feat(zimage): add ZImage-Turbo and ZImage-Base modes, update worker selection doc (STABL-tzctkzmz) — next: Task 8 certification on enigma, waits for user go"
```

Use `superpowers:requesting-code-review` with no sub-agent, or hand the branch to the user's reviewer. Open the pull request only when the user asks.

---

### Task 8: Certification on enigma (STABL-vnakbitb)

This task is a later gate. Do not start it without a go from the user. It writes no product code.

Host facts: `ssh enigma.lan`, repo `/home/hdd/workspace/Stability-Toys`, GPU RTX 3090. The live container `stability-toys-dev` shares the GPU. Mounts need `--security-opt label=disable` or `:z`. Agree with the user how the branch reaches the server before the first run.

- [ ] **Step 1: Prepare the certification config**

Make a copy of the server config outside the repo. Add two reference modes that are equal to the two Z-Image modes but have `text_encoder_quantize: none`: `ZImage-Turbo-Ref` and `ZImage-Base-Ref`. Do not commit them.

- [ ] **Step 2: Record the frozen inputs before a run**

`fp comment STABL-vnakbitb` with the commit SHA, the spec SHA-256, and the statement that prompts P1 to P5, seeds 101 to 105, and the scoring rules come from spec section 7.1 without change.

- [ ] **Step 3: Functional runs, both modes**

For each of `ZImage-Turbo` and `ZImage-Base`:

```bash
st gen "<prompt>" --mode <mode> --size 1024x1024 --seed 42 --outfile <mode>-t2i.png
st gen "<prompt>" --mode <mode> --size 1024x1024 --seed 42 --init-image <mode>-t2i.png --denoise-strength 0.6 --outfile <mode>-i2i.png
```

Record two separate memory figures for each run. Do not mix them.

- **Peak allocated**: the `peak_allocated_bytes` value in the server log line of the job (Task 6). The server process measures it with `torch.cuda.max_memory_allocated()`, with a reset and a synchronize around the generation. The limit of 20401094656 bytes applies to this figure only.
- **Device used**: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`, sampled each second during the run. It includes the allocator reserve and other processes. Record the highest sample. It has no limit in slice 1.

Before each run, record `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`. Another process on the GPU makes the device figure invalid for comparison. State that in the record.

Record also: pass or fail, seconds, and the job id that links the image to its log line. Use `--scheduler` only if the mode default is not applied.

Pass:
- Each image is coherent. img2img keeps the source composition.
- Turbo int8 txt2img has `peak_allocated_bytes` at or below `19 * 2**30` = 20401094656 in its server log line. Compare the byte value. Do not compare the rounded GiB value.
- A line with `peak_allocated_bytes=unavailable` is a failed measurement. Repeat the run after the cause is fixed.
- A base run that fails for memory at 1024x1024 blocks base certification. Record the first base `peak_allocated_bytes` as the base figure.

- [ ] **Step 4: Store behavior**

Restart the server. Load each mode again. The log must show `store hit` and the same key as the first load. Record the load seconds for miss and hit.

- [ ] **Step 5: Written-text gate, both modes**

Run the 20 images of spec section 7.1: P1 to P5, each with the mode and with its `-Ref` mode, at the frozen seed. No re-run with a different seed.

A person scores each string before a comparison of arms. Record a table for each mode: prompt id, string, int8 score, bf16 score.

Pass for each mode: int8 has at least 5 of 6 correct, and int8 has no more incorrect strings than bf16.

If a `-Ref` arm cannot run for memory, the relative rule has no result. Record that. The mode is then not certified, and the user decides the next step.

- [ ] **Step 6: Record evidence and stop**

`fp comment STABL-vnakbitb` with: commands, pass or fail for each check, peak figures, score tables, image paths, commit SHA. State each failed check plainly. A mode that fails is not certified with int8. Do not change a default to make a check pass.

---

## Self-review record

- Spec coverage: 5.1 and 5.2 in Task 1. 5.3 in Task 2. 5.4 in Tasks 5 and 6. 5.5 store in Task 4, admission in Task 5, mode field in Task 3. 5.6 in Task 7. 5.7 in Tasks 5 and 6. Section 6 tests are spread over Tasks 1 to 7. Section 7 in Task 8.
- Left open on purpose: `/status` scope report for the family. Task 7 step 3 checks startup behavior. If `/status` needs a family row, file a child issue. Do not add it without a failing test.
- The spec lists a latents path. The HunyuanDiT worker has no `run_job_with_latents`, so this plan adds none.
