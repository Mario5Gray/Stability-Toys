# Karras Scheduler Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Do not use subagent-driven development in this repository. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add canonical `dpmpp_2m_karras` and `dpmpp_sde_karras` scheduler IDs whose registry-owned construction kwargs enable Karras sigmas without changing existing scheduler behavior.

**Architecture:** Replace the string-only scheduler registry with immutable `SchedulerSpec` entries containing a Diffusers class path and per-ID kwargs. Keep `get_scheduler_class()` as the class-only resolver, apply kwargs only in `build_scheduler()`, and expose the new IDs through the existing SDXL allowlist and opaque worker metadata flow.

**Tech Stack:** Python 3, Diffusers scheduler APIs, pytest, PyYAML configuration, drift provenance checks.

**Design:** `docs/superpowers/specs/2026-07-10-karras-scheduler-variants-design.md`

---

### Task 1: Add structured scheduler specs and Karras variants

**Files:**
- Modify: `tests/test_scheduler_registry.py`
- Modify: `backends/scheduler_registry.py`

- [x] **Step 1: Check drift bindings before editing**

Run:

```bash
drift refs backends/scheduler_registry.py
drift refs tests/test_scheduler_registry.py
```

Expected: any bound docs are listed for review before code changes; no files are modified.

- [x] **Step 2: Replace the string-registry fixture and add failing behavior tests**

Update `tests/test_scheduler_registry.py` so the fake scheduler accepts kwargs, the old `SCHEDULER_IMPORTS.clear()/update()` fixture is replaced with `monkeypatch` against structured `SCHEDULER_SPECS`, and the new IDs and isolation behavior are asserted:

```python
"""Tests for scheduler registry resolution and policy semantics."""

import sys
from types import SimpleNamespace

import pytest


class _FakeScheduler:
    @classmethod
    def from_config(cls, config, **kwargs):
        return {
            "scheduler": cls.__name__,
            "config": config,
            "kwargs": kwargs,
        }


def test_build_scheduler_known_id_without_extra_kwargs(monkeypatch):
    from backends import scheduler_registry

    module_name = "diffusers.schedulers.scheduling_euler_discrete"
    fake_module = SimpleNamespace(EulerDiscreteScheduler=_FakeScheduler)
    monkeypatch.setattr(
        scheduler_registry,
        "SCHEDULER_SPECS",
        {
            "euler": scheduler_registry.SchedulerSpec(
                f"{module_name}.EulerDiscreteScheduler"
            )
        },
    )
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    built = scheduler_registry.build_scheduler("euler", {"beta": "value"})

    assert built == {
        "scheduler": "_FakeScheduler",
        "config": {"beta": "value"},
        "kwargs": {},
    }


def test_build_scheduler_forwards_spec_kwargs_and_deepcopies_config(monkeypatch):
    from backends import scheduler_registry

    module_name = "diffusers.schedulers.scheduling_dpmsolver_singlestep"
    fake_module = SimpleNamespace(DPMSolverSinglestepScheduler=_FakeScheduler)
    monkeypatch.setattr(
        scheduler_registry,
        "SCHEDULER_SPECS",
        {
            "dpmpp_sde_karras": scheduler_registry.SchedulerSpec(
                f"{module_name}.DPMSolverSinglestepScheduler",
                {"use_karras_sigmas": True},
            )
        },
    )
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    config = {"nested": {"beta": "value"}}

    built = scheduler_registry.build_scheduler("DPMpp_SDE_Karras", config)

    assert built["kwargs"] == {"use_karras_sigmas": True}
    assert built["config"] == config
    assert built["config"] is not config
    assert built["config"]["nested"] is not config["nested"]


def test_list_scheduler_ids_includes_karras_variants():
    from backends.scheduler_registry import list_scheduler_ids

    assert {"dpmpp_2m_karras", "dpmpp_sde_karras"} <= set(list_scheduler_ids())


def test_build_scheduler_unknown_id_raises():
    from backends import scheduler_registry

    with pytest.raises(ValueError, match="Unknown scheduler_id"):
        scheduler_registry.build_scheduler("not-a-scheduler", {})
```

- [x] **Step 3: Run the registry tests to verify RED**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base
python -m pytest tests/test_scheduler_registry.py -q
```

Expected: FAIL because `SchedulerSpec` and `SCHEDULER_SPECS` do not exist and the Karras IDs are not listed.

- [x] **Step 4: Implement the structured registry and kwargs construction seam**

Replace the registry representation and resolver/build functions in `backends/scheduler_registry.py` with:

```python
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SchedulerSpec:
    class_path: str
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)


SCHEDULER_SPECS = {
    "ddim": SchedulerSpec("diffusers.schedulers.scheduling_ddim.DDIMScheduler"),
    "euler": SchedulerSpec(
        "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler"
    ),
    "euler_a": SchedulerSpec(
        "diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler"
    ),
    "dpmpp_2m": SchedulerSpec(
        "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler"
    ),
    "dpmpp_2m_karras": SchedulerSpec(
        "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
        {"use_karras_sigmas": True},
    ),
    "dpmpp_sde": SchedulerSpec(
        "diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler"
    ),
    "dpmpp_sde_karras": SchedulerSpec(
        "diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler",
        {"use_karras_sigmas": True},
    ),
    "lcm": SchedulerSpec("diffusers.schedulers.scheduling_lcm.LCMScheduler"),
}


def normalize_scheduler_id(scheduler_id: str) -> str:
    return str(scheduler_id).strip().lower()


def list_scheduler_ids() -> list[str]:
    return sorted(SCHEDULER_SPECS.keys())


def _get_scheduler_spec(scheduler_id: str) -> SchedulerSpec:
    normalized = normalize_scheduler_id(scheduler_id)
    spec = SCHEDULER_SPECS.get(normalized)
    if spec is None:
        raise ValueError(
            f"Unknown scheduler_id '{scheduler_id}'. Available: {', '.join(list_scheduler_ids())}"
        )
    return spec


def get_scheduler_class(scheduler_id: str) -> Any:
    spec = _get_scheduler_spec(scheduler_id)
    module_name, class_name = spec.class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_scheduler(scheduler_id: str, config: Any) -> Any:
    spec = _get_scheduler_spec(scheduler_id)
    scheduler_cls = get_scheduler_class(scheduler_id)
    return scheduler_cls.from_config(deepcopy(config), **spec.extra_kwargs)
```

Retain the module docstring, `from __future__ import annotations`, `import importlib`, and `from copy import deepcopy` already present. The `dpmpp_sde_karras` entry intentionally uses the existing singlestep class mapping.

- [x] **Step 5: Run the registry tests to verify GREEN**

Run:

```bash
python -m pytest tests/test_scheduler_registry.py -q
```

Expected: `4 passed`.

- [x] **Step 6: Commit the registry checkpoint**

```bash
git add backends/scheduler_registry.py tests/test_scheduler_registry.py
git commit -m "feat(schedulers): add Karras registry variants (STABL-jltuulda)"
```

### Task 2: Prove worker allowlist and metadata propagation

**Files:**
- Modify: `tests/test_cuda_worker_capabilities.py`
- No production change: `backends/cuda_worker.py`

- [ ] **Step 1: Check drift bindings before editing the worker test**

Run:

```bash
drift refs tests/test_cuda_worker_capabilities.py
```

Expected: any bindings are listed; no files are modified.

- [ ] **Step 2: Add the Karras allowlist selection test**

Add this test to `TestSchedulerSelection` in `tests/test_cuda_worker_capabilities.py`:

```python
def test_karras_scheduler_id_is_normalized_and_applied_under_allowlist(self):
    pipe = _make_pipe()
    base = _make_base()
    base.pipe = pipe
    base.model_info = SimpleNamespace(
        default_scheduler_id=None,
        allowed_scheduler_ids=["dpmpp_sde_karras"],
    )
    base._baseline_scheduler_class = MagicMock()
    base._baseline_scheduler_config = {"name": "base"}
    req = SimpleNamespace(scheduler_id=" DPMpp_SDE_Karras ")
    built_scheduler = object()

    with patch(
        "backends.cuda_worker.build_scheduler",
        return_value=built_scheduler,
    ) as mock_build:
        selected = base._apply_request_scheduler(req)

    assert selected == "dpmpp_sde_karras"
    mock_build.assert_called_once_with(
        "dpmpp_sde_karras",
        {"name": "base"},
    )
    assert pipe.scheduler is built_scheduler
```

- [ ] **Step 3: Extend the existing SDXL render test to inspect metadata**

Add `import json` near the top of `tests/test_cuda_worker_capabilities.py`. In `test_sdxl_run_job_forwards_negative_prompt`, return the canonical Karras ID from the existing scheduler mock:

```python
worker._apply_request_scheduler = Mock(return_value="dpmpp_sde_karras")
```

Replace the final `pnginfo.add_text.assert_called_once()` with:

```python
pnginfo.add_text.assert_called_once()
metadata_key, metadata_json = pnginfo.add_text.call_args.args
assert metadata_key == "lcm"
assert json.loads(metadata_json)["scheduler_id"] == "dpmpp_sde_karras"
```

This is characterization coverage of the existing opaque metadata path, so it is expected to pass without a `cuda_worker.py` change.

- [ ] **Step 4: Run the focused worker tests**

Run:

```bash
python -m pytest \
  tests/test_cuda_worker_capabilities.py::TestSchedulerSelection \
  tests/test_cuda_worker_capabilities.py::TestNegativePromptForwarding::test_sdxl_run_job_forwards_negative_prompt \
  -q
```

If the enclosing SDXL class name differs after concurrent changes, use:

```bash
python -m pytest tests/test_cuda_worker_capabilities.py -k 'SchedulerSelection or sdxl_run_job_forwards_negative_prompt' -q
```

Expected: all selected tests PASS, proving allowlist selection and exact PNG metadata propagation with no production worker edit.

- [ ] **Step 5: Commit the worker proof checkpoint**

```bash
git add tests/test_cuda_worker_capabilities.py
git commit -m "test(schedulers): prove Karras worker propagation (STABL-jltuulda)"
```

### Task 3: Expose Karras variants in SDXL mode policy

**Files:**
- Modify: `conf/modes.yml`

- [ ] **Step 1: Check drift bindings before editing shared mode policy**

Run:

```bash
drift refs conf/modes.yml
```

Expected: any bound docs are listed for review before the policy change.

- [ ] **Step 2: Add the two canonical IDs to the SDXL allowlist**

Change only the `modes.SDXL.allowed_scheduler_ids` sequence in `conf/modes.yml`:

```yaml
allowed_scheduler_ids:
  - ddim
  - dpmpp_2m
  - dpmpp_2m_karras
  - dpmpp_sde
  - dpmpp_sde_karras
  - euler
  - lcm
default_scheduler_id: dpmpp_2m
```

Do not change SD1.5 mode allowlists or the SDXL default.

- [ ] **Step 3: Verify the shared policy directly**

Run:

```bash
python -c 'import yaml; data=yaml.safe_load(open("conf/modes.yml")); mode=data["modes"]["SDXL"]; assert {"dpmpp_2m_karras", "dpmpp_sde_karras"} <= set(mode["allowed_scheduler_ids"]); assert mode["default_scheduler_id"] == "dpmpp_2m"; print("SDXL scheduler policy ok")'
```

Expected: `SDXL scheduler policy ok`.

Do not modify `tests/test_mode_config.py`, `tests/test_model_routes.py`, or `tests/test_worker_pool.py`; their scheduler allowlists are synthetic fixtures and do not cover shared `conf/modes.yml`.

- [ ] **Step 4: Run the complete focused regression slice**

Run:

```bash
python -m pytest tests/test_scheduler_registry.py tests/test_cuda_worker_capabilities.py -q
```

Expected: all tests PASS with no failures.

- [ ] **Step 5: Verify diff scope and drift state**

Run:

```bash
git diff --check
git status --short
drift check --changed backends/scheduler_registry.py
drift check --changed tests/test_scheduler_registry.py
drift check --changed tests/test_cuda_worker_capabilities.py
drift check --changed conf/modes.yml
```

Expected: no whitespace errors; only the planned files are changed; every scoped drift check reports `ok` or identifies prose that must be reviewed and updated before proceeding.

- [ ] **Step 6: Commit policy and final verification checkpoint**

```bash
git add conf/modes.yml
git commit -m "config(schedulers): allow Karras variants on SDXL (STABL-jltuulda)"
```

- [ ] **Step 7: Attach revisions and stop for review**

Run:

```bash
fp issue assign STABL-jltuulda --rev "$(git rev-parse HEAD)"
fp comment STABL-jltuulda "Implementation complete on feature/STABL-jltuulda. Registry specs add per-ID kwargs and Karras variants; focused worker tests prove allowlist and PNG metadata propagation; SDXL policy opts in while retaining dpmpp_2m as default. STOP: ready for review. NEXT: run the review cycle; do not start another issue."
```

Expected: FP records the implementation revision and one `STOP/NEXT` review handoff. Do not mark the issue done or advance waveplan state before human review.
