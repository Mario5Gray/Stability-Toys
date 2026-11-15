# MPS Z-Image Turbo and Base Implementation Plan

> For agentic workers: use superpowers:executing-plans inline.
> Repository rules forbid subagent-driven development.
> Track tasks and steps in FP. Do not add Markdown task checkboxes.

**Goal:** Deliver native MPS Turbo and Base generation with current CUDA mode/request LoRA behavior.

**Architecture:** Reuse the existing Z-Image constructor and generation loop through a small MPS subclass.
Add a family-specific bf16 policy and MPS memory logging.
Keep the shared encoder store, LoRA fuse, governor, routes and CLI contracts.

**Tech stack:** Python, Torch MPS, Diffusers Z-Image, Transformers Qwen3, Quanto qint8, PEFT, existing Go CLI.

**Design:** ../specs/2026-11-15-mps-zimage-design.md.
**Review record:** ../reports/2026-11-15-mps-zimage-plan-review.md.
**Parent:** STABL-zubknmov.
**Planning issue:** STABL-eudrdkpt.
**Baseline:** 6c3466050de8432a9ab174bcdacb46bdfe1fb93b.
**Status:** Revised after review at 78c7e48. No implementation task started.

## Authorization and execution

User authorized both variants and this plan.
Do not treat this request as authorization to implement.
After plan review and implementation authorization, execute tasks inline with TDD.
Full-model runs require separate operator go at Tasks 4 and 5.
Push, merge, installed-service restart and deployment remain separate actions.

The planning worktree is .worktrees/mps-zimage-plan.
Branch: docs/stabl-zubknmov-mps-zimage-plan.
Preserve unrelated source-checkout files.
Use a fresh implementation worktree from the reviewed baseline.
Do not install dependencies or run provider calls during plan preparation.

Before implementation:
1. Read AGENTS.md, FP_AGENTS.md and the stopping-point policy.
2. Recheck agent-policy.rdf at the relevant root.
3. Run fp issue list --status in-progress and fp context STABL-zubknmov.
4. Verify the exact approved plan commit.
5. Run drift refs on each bound file before editing.
6. Record current native runtime and exact test baseline.
7. Leave all live tasks pending until operator go.

Sigma owns this plan and its revisions.
Theta is the designated implementation author.
Before Task 1, confirm the active runtime identity is theta.
Prefix implementation FP commands with FP_AGENT_NAME='theta' only after that check.
If the implementation author differs, record the actual identity and revise command prefixes before claiming a task.
Do not use another agent's name for attribution.
This session performs planning only.
Use the dedicated stability-toys environment:
```bash
conda activate stability-toys
python -c 'import sys; print(sys.executable)'
```
Expected interpreter: /Users/darkbit1001/miniforge3/envs/stability-toys/bin/python.

## Task ledger and dependencies

| Task | FP issue | Depends on | State |
|---|---|---|---|
| 1. Policy and worker hooks | STABL-lcfvhjog | Plan approval | todo |
| 2. Encoder-store parity | STABL-mbhlhinl | Plan approval | todo |
| 3. Binding and LoRA contracts | STABL-vklkdewe | Tasks 1, 2 | todo |
| 4. Turbo acceptance | STABL-nxonrqlm | Task 3, operator go | todo |
| 5. Base acceptance | STABL-hrlcjzvh | Task 3, operator go | todo |
| 6. Regression and accepted-scope docs | STABL-ryapqlom | Tasks 4, 5 | todo |

Run Tasks 4 and 5 sequentially on the shared Mac.
One variant passing cannot close the other.
Parent completion requires all six tasks and reviewed evidence.

## File map

| File | Responsibility |
|---|---|
| backends/platforms/mps_policy.py | Extract protected dtype resolver. Preserve default fp16 policy. |
| backends/platforms/zimage_mps_policy.py, new | bf16 Z-Image policy and pre-weight MPS check. |
| backends/zimage_mps_worker.py, new | Shared-family subclass and honest MPS job-memory logging. |
| backends/platforms/mps_bindings.py | Import-clean zimage binding for both variants. |
| tests/test_zimage_mps_policy.py, new | Real-library policy contracts with mocked device calls. |
| tests/test_mps_policy.py | Existing fp16 behavior regression. |
| tests/test_zimage_worker.py | Existing isolated stub cohort plus MPS constructor/job contracts. |
| tests/test_zimage_lora_worker.py | Tiny real LoRA math for CUDA and MPS worker classes. |
| tests/test_zimage_text_encoder.py | Existing content-key and store regression contracts. |
| tests/test_zimage_mps_encoder.py, new | Tiny local-Qwen filesystem store proof on native MPS. |
| tests/test_mps_registration.py | Both factory paths, lazy imports and family set. |
| docs/MPS_BACKEND.md | Accepted family, dtype, telemetry and native evidence. |
| docs/WORKER_SELECTION.md | Selection and unchanged LoRA behavior. |

Do not copy the complete CUDA worker into a second file.
No production change to zimage_cuda_worker.py is expected.
Change it only if a demonstrated shared defect prevents the planned subclass.
Do not change worker_factory.py, API schemas, Go flags or governor authority logic without a failing integration test.

## Task 1: bf16 policy and worker hooks

FP: STABL-lcfvhjog.
Files: mps_policy.py, new zimage_mps_policy.py, new zimage_mps_worker.py,
new test_zimage_mps_policy.py, existing test_zimage_worker.py and test_mps_policy.py.

### Step 1: Add policy RED tests

Create tests/test_zimage_mps_policy.py.
Use real Torch with mocked availability and a patched bf16 check.
Test this minimum executable case:

```python
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch

from backends.platforms.zimage_mps_policy import ZIMAGE_MPS_TORCH_POLICY

def test_zimage_policy_uses_bf16(monkeypatch):
    for name in (
        "CUDA_DTYPE", "CUDA_OFFLOAD", "CUDA_QUANTIZE",
        "CUDA_ENABLE_XFORMERS", "PYTORCH_ENABLE_MPS_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    with (
        patch("torch.backends.mps.is_built", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("backends.platforms.zimage_mps_policy._check_bf16_available") as check,
    ):
        settings = ZIMAGE_MPS_TORCH_POLICY.resolve_settings(SimpleNamespace())
    check.assert_called_once_with()
    assert settings.device == "mps"
    assert settings.dtype is torch.bfloat16
    assert settings.dtype_str == "bf16"
    assert settings.offload == settings.quantize == "none"
    assert settings.enable_xformers is False
```

Add cases with exact expected results:

| Input | Expected |
|---|---|
| MPS not built or unavailable | RuntimeError before bf16 check |
| bf16 check raises | RuntimeError before encoder/pipeline load |
| CUDA_DTYPE absent, fp16, bf16, fp32 | bf16 family dtype |
| CUDA_DEVICE=cuda:3 | Ignored, device remains mps |
| CUDA_OFFLOAD=model or sequential | ValueError, offload |
| CUDA_QUANTIZE=fp8 | ValueError, quantize |
| CUDA_ENABLE_XFORMERS=1 | ValueError, xformers |
| Safe model override masks invalid environment | Same refusal |
| Invalid runtime_offload/runtime_quantize/runtime_enable_xformers | Same refusal |
| PYTORCH_ENABLE_MPS_FALLBACK=1 | ValueError, fallback |
| Existing MPS_TORCH_POLICY with CUDA_DTYPE=bf16 | Existing fp16 refusal |

Run:
```bash
python -m tests.run tests/test_zimage_mps_policy.py -- -q --no-cov -p no:cacheprovider
```
Expected RED: missing policy module.

### Step 2: Extract one dtype seam

Move existing CUDA_DTYPE validation into MpsTorchPolicy._resolve_dtype.
The complete protected method is:

```python
def _resolve_dtype(self) -> tuple[Any, str]:
    import torch

    dtype_str = os.environ.get("CUDA_DTYPE", "fp16").lower().strip()
    if dtype_str != "fp16":
        raise ValueError("MPS requires CUDA_DTYPE=fp16")
    return torch.float16, "fp16"
```

In resolve_settings, after built/available checks, replace existing dtype validation with:
```python
dtype, dtype_str = self._resolve_dtype()
```
Use dtype=dtype and dtype_str=dtype_str in TorchExecutionSettings.
Keep every other validation and default unchanged.
Keep the singleton MPS_TORCH_POLICY unchanged.

### Step 3: Add the dedicated policy

Complete new backends/platforms/zimage_mps_policy.py:

```python
"""Z-Image bf16 MPS execution policy."""

from __future__ import annotations

import os
from typing import Any
from backends.platforms.mps_policy import MpsTorchPolicy


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def _check_bf16_available() -> None:
    import torch

    try:
        with torch.inference_mode():
            value = torch.ones((2, 2), device="mps", dtype=torch.bfloat16)
            result = value @ value
            torch.mps.synchronize()
            if not bool(torch.isfinite(result).all().item()):
                raise RuntimeError("bf16 MPS probe returned non-finite values")
    except Exception as exc:
        raise RuntimeError("Z-Image requires working bf16 MPS operations") from exc

class ZImageMpsTorchPolicy(MpsTorchPolicy):
    def _resolve_dtype(self) -> tuple[Any, str]:
        import torch

        if _bool_env("PYTORCH_ENABLE_MPS_FALLBACK"):
            raise ValueError("Z-Image MPS does not support CPU fallback")
        _check_bf16_available()
        return torch.bfloat16, "bf16"

ZIMAGE_MPS_TORCH_POLICY = ZImageMpsTorchPolicy()
```

This policy changes only Z-Image dtype resolution.
The shared constructor still handles the ignored CUDA_DTYPE warning.

### Step 4: Add worker RED contracts

Append MPS cases to existing tests/test_zimage_worker.py.
That file already belongs to its own stub interpreter.
Reuse its fake_load, _model_info, fake pipeline and request/job fixtures.
Do not import its Torch stubs into a real-library test file.

Required constructor assertion:

```python
def test_mps_constructor_loads_int8_on_mps(zw, fake_load, monkeypatch):
    from backends.zimage_mps_worker import DiffusersZImageMpsWorker
    from backends.platforms.zimage_mps_policy import ZIMAGE_MPS_TORCH_POLICY

    monkeypatch.setattr(
        "backends.platforms.zimage_mps_policy._check_bf16_available",
        lambda: None,
    )
    worker = DiffusersZImageMpsWorker(
        worker_id=0,
        model_path=fake_load.model_dir,
        model_info=_model_info(),
    )
    assert worker.device == "mps"
    assert worker.dtype == zw.torch.bfloat16
    assert worker.execution_policy is ZIMAGE_MPS_TORCH_POLICY
    fake_load.store.assert_called_once_with(
        fake_load.model_dir, device="mps", dtype=zw.torch.bfloat16,
    )
    assert worker._try_reset_peak_memory() is False
```

Patch CUDA availability, synchronization and peak-counter calls to raise in MPS job tests.
Add one case with CUDA_DTYPE=fp16. Assert the ignored-value constructor warning.
Cover txt2img and img2img, cancellation, error cleanup and explicit encoder none.
Assert the generator receives device mps.
Assert policy.release_cache runs after success and error.
Assert telemetry failure logs unavailable fields without losing generated bytes.
Keep existing CUDA cases unchanged.

Run the new cases before adding the worker.
Expected RED: missing worker module.

### Step 5: Add the worker

Complete new backends/zimage_mps_worker.py:

```python
"""MPS worker for Z-Image Turbo and Base."""

from __future__ import annotations

import logging
from typing import Any
import torch

from backends.platforms.zimage_mps_policy import ZIMAGE_MPS_TORCH_POLICY
from backends.zimage_cuda_worker import DiffusersZImageCudaWorker

logger = logging.getLogger(__name__)

class DiffusersZImageMpsWorker(DiffusersZImageCudaWorker):
    execution_policy = ZIMAGE_MPS_TORCH_POLICY
    log_tag = "zimage-mps"

    def _try_reset_peak_memory(self) -> bool:
        return False

    def _try_read_peak_memory(self) -> None:
        return None

    def _log_job(
        self, job_id: Any, op: str, width: int, height: int, steps: int,
        peak: tuple[int, int] | None,
    ) -> None:
        allocated: int | str = "unavailable"
        driver: int | str = "unavailable"
        try:
            torch.mps.synchronize()
            allocated, driver = self.execution_policy.memory_stats()
        except Exception as exc:
            logger.warning("[%s] memory read failed: %r", self.log_tag, exc)
        logger.info(
            "[%s] job=%s op=%s size=%sx%s steps=%s "
            "current_allocated_bytes=%s driver_allocated_bytes=%s "
            "peak_allocated_bytes=unavailable peak_reserved_bytes=unavailable",
            self.log_tag, job_id, op, width, height, steps, allocated, driver,
        )
```

Do not override generation, configure_loras, img2img construction or cleanup.
Do not register this worker publicly yet.

### Step 6: Verify GREEN and record

```bash
python -m tests.run tests/test_zimage_mps_policy.py tests/test_mps_policy.py tests/test_zimage_worker.py -- -q --no-cov -p no:cacheprovider
git diff --check
```

Expected: exit 0, no new failures or skips.
Record exact counts and RED/GREEN evidence in STABL-lcfvhjog.
Commit only the task files.
Commit message must name STABL-lcfvhjog and the exact next task.

## Task 2: int8 encoder-store parity

FP: STABL-mbhlhinl.
Files: new tests/test_zimage_mps_encoder.py, existing tests/test_zimage_text_encoder.py.
Production store change is conditional on a demonstrated defect.

### Step 1: Write a native tiny-model filesystem test

Use real Transformers, Quanto and the existing public store loader.
Create tiny local Qwen3 weights under tmp_path/model/text_encoder.
Use no remote model identifier.

```python
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModel, Qwen3Config
from backends.zimage_text_encoder import load_int8_text_encoder

def test_mps_encoder_filesystem_hit(tmp_path):
    if os.environ.get("RUN_ZIMAGE_MPS_NATIVE") != "1":
        pytest.skip("explicit native MPS test")
    assert torch.backends.mps.is_available(), "native MPS unavailable"
    torch.manual_seed(0)
    root = tmp_path / "model"
    encoder_dir = root / "text_encoder"
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, dtype="bfloat16",
    )
    encoder = AutoModel.from_config(cfg, dtype=torch.bfloat16)
    encoder.save_pretrained(encoder_dir, safe_serialization=True)
    del encoder
    store = tmp_path / "encoder-store"
    ids = torch.tensor([[1, 2, 3, 4]], device="mps")
    with torch.inference_mode():
        fresh = load_int8_text_encoder(
            str(root), device="mps", dtype=torch.bfloat16, store_root=store,
        )
        expected = fresh(ids).last_hidden_state.detach().cpu()
        del fresh
        torch.mps.empty_cache()
        with patch(
            "backends.zimage_text_encoder._build_entry",
            side_effect=AssertionError("store hit rebuilt encoder"),
        ):
            loaded = load_int8_text_encoder(
                str(root), device="mps", dtype=torch.bfloat16, store_root=store,
            )
        actual = loaded(ids).last_hidden_state.detach().cpu()
    assert actual.dtype is torch.bfloat16
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Run this native cohort explicitly outside the restricted sandbox:
```bash
RUN_ZIMAGE_MPS_NATIVE=1 python -m tests.run tests/test_zimage_mps_encoder.py -- -q --no-cov -p no:cacheprovider
```

Expected: exit 0 and no skips.
Without RUN_ZIMAGE_MPS_NATIVE, its skip is intentional offline behavior.
A skip cannot satisfy native acceptance.
The store may already pass. Do not invent a production change to manufacture RED.

### Step 2: Add restart proof and refusal contracts

Extend the native test with two fresh Python processes using subprocess.run(check=True).
Process A constructs the store and writes expected output with safetensors.
Process B patches _build_entry to raise, loads the same entry and compares output.
Both processes use sys.executable and the same explicit store_root.
Neither process can download files.
Save subprocess stdout, stderr and exit codes.

Add ordinary fake-seam tests proving device mps and dtype bf16 reach build and hit paths.
Retain all existing content-hash, rename, corruption, stale-temp and locking tests.
Assert explicit text_encoder_quantize none bypasses the store in worker tests.

If native behavior fails, isolate serialization, dtype restoration or device placement first.
Write a focused failing test before changing the store.
Keep content identity and lock publication semantics intact.
Do not add fp16 construction under the current key.

### Step 3: Verify and record

```bash
python -m tests.run tests/test_zimage_text_encoder.py tests/test_zimage_worker.py -- -q --no-cov -p no:cacheprovider
```

Record ordinary tests separately from the explicit native cohort.
Commit tests and any proven minimal store fix under STABL-mbhlhinl.
Next: Task 3.

## Task 3: binding and CUDA LoRA contracts

FP: STABL-vklkdewe.
Files: mps_bindings.py, test_mps_registration.py,
test_zimage_lora_worker.py, related request/governor tests only if a contract fails.

### Step 1: Add binding RED assertions

Extend existing family-set assertion to include zimage.
Add `zimage` to WORKER_REF_FAMILIES at tests/test_mps_registration.py:142.
The guard at line 175 requires the list to equal the binding set.

The subprocess case asserts the policy identity at line 164.
Z-Image uses a different policy object.
Replace the assertion with a backend identity check:
```python
"assert cls.execution_policy.backend_id == 'mps', cls"
```
Remove the now-unused MPS_TORCH_POLICY import from the subprocess command.
Keep the subclass assertion.

Add:

```python
def test_mps_zimage_binding():
    from backends.platforms.base import ExecutionCapabilities
    from backends.platforms.mps_bindings import MPS_FAMILY_BINDINGS

    cell = MPS_FAMILY_BINDINGS["zimage"]
    assert cell.worker_ref == "backends.zimage_mps_worker.DiffusersZImageMpsWorker"
    assert cell.execution_capabilities == ExecutionCapabilities(True, False, False)
```

Keep SDXL as the unbound-family negative case.
Test both configured variant modes resolve to the same zimage binding.
Retain lazy-import assertions: binding reads must not import Torch or either Z-Image worker.
Cover inproc factory and module-level create_mps_worker subprocess dispatch.

Run binding tests before adding the row.
Expected RED: missing zimage entry or old family-set mismatch.

### Step 2: Add the binding

Add this entry to MPS_FAMILY_BINDINGS using the table's existing declaration style:

```python
MPS_FAMILY_BINDINGS["zimage"] = FamilyPlatformBinding(
    "backends.zimage_mps_worker.DiffusersZImageMpsWorker",
    ExecutionCapabilities(True, False, False),
)
```

This covers both Turbo and Base.
Before editing the bound row, update operator prose to identify the development binding and pending acceptance.
Do not claim accepted support or refresh stale provenance before reviewing that prose.
ControlNet and combined generation stay false.
No new detection or variant override is required.
The row is development wiring until both live acceptance tasks pass.

### Step 3: Extend tiny real LoRA tests

Parameterize existing _worker helper and each existing LoRA test over both classes.
Keep tiny models on CPU for ordinary numerical tests.
The native live tasks prove actual MPS execution.

Use this helper signature and class fixture:

```python
from backends.zimage_mps_worker import DiffusersZImageMpsWorker

@pytest.fixture(params=[DiffusersZImageCudaWorker, DiffusersZImageMpsWorker])
def worker_type(request):
    return request.param

def _worker(worker_type=DiffusersZImageCudaWorker):
    w = worker_type.__new__(worker_type)
    w.pipe = ZImagePipeline(
        scheduler=None, vae=None, text_encoder=None, tokenizer=None,
        transformer=_tiny_transformer(),
    )
    w.worker_id = 0
    w._img2img_pipe = None
    return w
```

Each test receives worker_type and calls _worker(worker_type).
Keep the existing numerical assertions and error types.
The log test must capture the inherited family logger and expect the worker-specific tag.
Cover additive LoRAs, zero strength, alpha conversion, foreign keys, partial fit,
non-finite strength, ambiguous directory, adapter removal and img2img sharing.
Do not add a new format promise beyond current Diffusers conversion support.

### Step 4: Check request authority and admission

Run existing request, route, governor and mode-LoRA cohorts.
Add MPS-provider integration cases only where backend selection affects the result.
Assert the following:

| Trigger | Required result |
|---|---|
| Same ordered file/strength tuple | No reload |
| New tuple or strength | Clean worker reload |
| Explicit plain mode with no request LoRAs | Plain weights |
| No mode and no request list | Existing terminal authority |
| Mode LoRAs plus request LoRAs | Refused before switch |
| Invalid path, duplicate, escape or non-finite strength | Existing pre-admission refusal |
| Wrong-family or partial-fit LoRA | Failed mode load, next valid request recovers |
| ControlNet request | Refused before preprocessing |
| Direct worker ControlNet job | Existing worker refusal |

Do not modify authority-key semantics or error mappings to pass an MPS test.

### Step 5: Verify GREEN

```bash
python -m tests.run tests/test_mps_registration.py tests/test_zimage_lora_worker.py tests/test_diffusers_lora_fuse.py tests/test_mode_loras.py tests/test_request_loras.py tests/test_request_lora_routes.py tests/test_governor_request_loras.py -- -q --no-cov -p no:cacheprovider
python -m tests.run tests/test_progress_bar_suppression.py -- -q --no-cov -p no:cacheprovider
git diff --check
```

Run progress-bar tests separately because their module imports interact with lazy-import assertions.
Record exact counts and failures.
Commit under STABL-vklkdewe.
Next: implementation review before native full-model runs.

## Tasks 4 and 5: separate native acceptance

Task 4: STABL-nxonrqlm, Turbo.
Task 5: STABL-hrlcjzvh, Base.
Both depend on Task 3 and explicit operator go.
Use this complete runbook separately for each variant.

### Step 1: Approve and inventory native assets

Require these explicit inputs before proceeding:
- ZIMAGE_MODEL_ROOT: native root containing both Diffusers folders.
- ZIMAGE_TURBO_MODEL and ZIMAGE_BASE_MODEL: paths relative to that root.
- ZIMAGE_LORA_ROOT: native LoRA root.
- ZIMAGE_TURBO_LORA and ZIMAGE_TURBO_LORA_2: two distinct compatible files beneath the LoRA root.
- ZIMAGE_BASE_LORA and ZIMAGE_BASE_LORA_2: two distinct compatible files beneath the LoRA root.
- ZIMAGE_WRONG_FAMILY_LORA: one file beneath the LoRA root that fails the fit check for both variants.
- ZIMAGE_TURBO_TRIGGER and ZIMAGE_BASE_TRIGGER: approved proof prompts or trigger text.

These are operator inputs, not inferred filenames.
Check model_index.json, all required component files and LoRA containment.
Resolve each model folder through detect_model.
Require family_id zimage.
A present folder with another family fails the inventory.
Record SHA-256 hashes for model configs, weight shards and LoRA files.
Record exact source revision, dirty state, runtime versions and host.
Do not download or copy remote assets without authorization.

Set up an evidence directory without overwriting another run:
```bash
export ZIMAGE_PROOF_DIR="$(mktemp -d /tmp/stabl-zubknmov.XXXXXX)"
mkdir -p "$ZIMAGE_PROOF_DIR/conf" "$ZIMAGE_PROOF_DIR/store" "$ZIMAGE_PROOF_DIR/encoder-store" "$ZIMAGE_PROOF_DIR/bin"
cp -R conf/. "$ZIMAGE_PROOF_DIR/conf/"
```

Write only the copied modes.yml.
Set copied model_root and lora_root from the approved inputs.
Keep original Turbo/Base defaults and negative-prompt policies.
Add four aliases per variant:
- plain int8, with no loras list.
- mode-LoRA int8, with first LoRA at strength 0.7.
- zero-mode-LoRA int8, with first LoRA at strength 0.
- plain unquantized, with text_encoder_quantize none.

Do not change product configuration precedence.
Explicit live CLI steps/cfg values will pin the intended variant settings.
Use a separate empty-default CLI config for the omitted-override default check.


Generate the copied aliases and CLI config with this code:
```python
import copy
import json
import os
from pathlib import Path
from ruamel.yaml import YAML

from backends.family_profiles import resolve_family
from utils.model_detector import detect_model

proof = Path(os.environ["ZIMAGE_PROOF_DIR"])
model_root = Path(os.environ["ZIMAGE_MODEL_ROOT"]).resolve(strict=True)
lora_root = Path(os.environ["ZIMAGE_LORA_ROOT"]).resolve(strict=True)
wrong_lora = (lora_root / os.environ["ZIMAGE_WRONG_FAMILY_LORA"]).resolve(strict=True)
assert wrong_lora.is_file()
assert lora_root in wrong_lora.parents
yaml = YAML()
path = proof / "conf" / "modes.yml"
data = yaml.load(path.read_text())
original = data["modes"]
modes = {}
for label, prefix in (("Turbo", "ZIMAGE_TURBO"), ("Base", "ZIMAGE_BASE")):
    name = "ZImage-" + label
    model = os.environ[prefix + "_MODEL"]
    assert not Path(model).is_absolute()
    assert (model_root / model / "model_index.json").is_file()
    assert resolve_family(detect_model(str(model_root / model))).family_id == "zimage"
    paths = []
    for suffix in ("_LORA", "_LORA_2"):
        weight = (lora_root / os.environ[prefix + suffix]).resolve(strict=True)
        assert weight.is_file()
        assert lora_root in weight.parents
        paths.append(str(weight.relative_to(lora_root)))
    assert paths[0] != paths[1]
    base = copy.deepcopy(original[name])
    base["model"] = model
    base["text_encoder_quantize"] = "int8"
    base.pop("loras", None)
    modes[name] = base
    for suffix, strength in (("-ModeLoRA", 0.7), ("-ZeroLoRA", 0.0)):
        entry = copy.deepcopy(base)
        entry["loras"] = [{"path": paths[0], "strength": strength}]
        modes[name + suffix] = entry
    entry = copy.deepcopy(base)
    entry["text_encoder_quantize"] = "none"
    modes[name + "-EncoderNone"] = entry
data["model_root"] = str(model_root)
data["lora_root"] = str(lora_root)
data["default_mode"] = "ZImage-Turbo"
data["modes"] = modes
with path.open("w") as output:
    yaml.dump(data, output)
config = {"config": {
    "server_url": "http://127.0.0.1:14200",
    "defaults": {
        "generation": {},
        "output_format": "png",
        "output_directory": str(proof),
        "include_meta": False,
    },
}}
(proof / "st.json").write_text(json.dumps(config, indent=2))
```

The full shard/LoRA hash manifest remains required before runs.
Use explicit per-variant/per-isolation output subdirectories.
Keep the common conf, CLI config and encoder-store paths unchanged.
Set ZIMAGE_CASE_DIR to the current variant/isolation output directory.
Use ZIMAGE_CASE_DIR for PNG filenames and saved status/log artifacts.

### Step 2: Build the exact CLI and start an isolated server

Build from reviewed source:
```bash
(cd cli/go && go build -o "$ZIMAGE_PROOF_DIR/bin/st" ./cmd/st)
```

Check port 14200 is free. Do not stop another process.
Run the server in a foreground terminal after operator go:
```bash
env -u PYTORCH_ENABLE_MPS_FALLBACK -u CUDA_DTYPE -u CUDA_OFFLOAD -u CUDA_QUANTIZE -u CUDA_ENABLE_XFORMERS BACKEND=mps MODE_CONFIG_PATH="$ZIMAGE_PROOF_DIR/conf" WORKER_ISOLATION=inproc CONTROLNET_REGISTRY_VALIDATION=off STORAGE_PROVIDER=FILESYSTEM FS_STORAGE_DIR="$ZIMAGE_PROOF_DIR/store" ZIMAGE_ENCODER_STORE="$ZIMAGE_PROOF_DIR/encoder-store" DEFAULT_TIMEOUT=3600 ADMISSION_TIMEOUT_S=1800 NUM_WORKERS=1 python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 14200
```

CONTROLNET_REGISTRY_VALIDATION=off avoids validating unrelated checkpoint paths in the copied registry.
Keep the copied controlnets.yaml present. ControlNet admission remains refused.
Capture stdout/stderr to the run-specific evidence log.
Use a process handle for this server only.
The server.run module hardcodes port 4200 at the inspected baseline.
Do not use PORT=14200 with that module.
Keep client timeout separate from server execution and admission budgets.

In another terminal:
```bash
export ZIMAGE_ST="$ZIMAGE_PROOF_DIR/bin/st"
export ZIMAGE_URL=http://127.0.0.1:14200
export XDG_STATE_HOME="$ZIMAGE_PROOF_DIR/state"
curl --fail --silent --show-error "$ZIMAGE_URL/health"
curl --fail --silent --show-error "$ZIMAGE_URL/api/models/status"
```

Expected: healthy isolated MPS server.
Verify effective config before the first render.

### Step 3: Execute the frozen matrix

Set variant variables explicitly before its matrix:
```bash
export ZIMAGE_MODE=ZImage-Turbo
export ZIMAGE_STEPS=9
export ZIMAGE_CFG=0.0
export ZIMAGE_PROMPT="a red apple on a wooden table, soft light, $ZIMAGE_TURBO_TRIGGER"
export ZIMAGE_LORA="$ZIMAGE_TURBO_LORA"
export ZIMAGE_LORA_2="$ZIMAGE_TURBO_LORA_2"
export ZIMAGE_CASE_DIR="$ZIMAGE_PROOF_DIR/turbo-inproc"
mkdir -p "$ZIMAGE_CASE_DIR"
```

For Base, set:
```bash
export ZIMAGE_MODE=ZImage-Base
export ZIMAGE_STEPS=30
export ZIMAGE_CFG=4.0
export ZIMAGE_PROMPT="a red apple on a wooden table, soft light, $ZIMAGE_BASE_TRIGGER"
export ZIMAGE_LORA="$ZIMAGE_BASE_LORA"
export ZIMAGE_LORA_2="$ZIMAGE_BASE_LORA_2"
export ZIMAGE_CASE_DIR="$ZIMAGE_PROOF_DIR/base-inproc"
mkdir -p "$ZIMAGE_CASE_DIR"
```

Use turbo-subprocess and base-subprocess directory names after restart.
Use seed 4242 and size 1024x1024.
Use each variant's approved trigger in the same prompt for all LoRA comparisons.
Use fixed negative text "blurry, low quality, watermark" for Base.
Pass --negative explicitly for custom-negative cases.
No generation path applies a mode default template at this baseline.
Omit --negative only to record that the negative field is absent.
Do not claim template substitution.

Common command shape:
```bash
"$ZIMAGE_ST" gen --timeout 3700s --server "$ZIMAGE_URL" --config "$ZIMAGE_PROOF_DIR/st.json" --mode "$ZIMAGE_MODE" --prompt "$ZIMAGE_PROMPT" --size 1024x1024 --steps "$ZIMAGE_STEPS" --cfg "$ZIMAGE_CFG" --seed 4242 --outfile "$ZIMAGE_CASE_DIR/plain-1.png"
```

The copied CLI config must use PNG output and a run-specific output directory.
Do not use conflate/history inheritance during the frozen matrix.
Set an explicit plain mode for every plain restoration.

Execute these cases for each isolation mode:

| Case | Operation | Output in ZIMAGE_CASE_DIR | Required evidence |
|---|---|---|---|
| P1 | Plain txt2img | plain-1.png | Baseline |
| P2 | Repeat P1 | plain-2.png | Pixels equal P1 |
| D | Plain call without step/cfg overrides | defaults.png | Variant defaults in PNG |
| I1 | P1 init, strength 0.35 | init-035.png | Recorded strength |
| I2 | Same init, strength 0.9 | init-090.png | Recorded strength, pixels differ from I1 |
| M | Mode-LoRA alias at 0.7 | mode-lora.png | Targets and active set, pixels differ from P1 |
| Z | Zero-mode-LoRA alias | zero-mode-lora.png | Pixels equal P1 |
| R1 | First request LoRA at 0.7 | request-lora-07.png | Active set and merge targets |
| R2 | Repeat R1 | request-lora-07-repeat.png | No reload, pixels equal R1 |
| R3 | First request LoRA at 1.0 | request-lora-10.png | Reload and changed active strength/image |
| R4 | First LoRA at 0.7, second at 0.5 | request-lora-two.png | Both targets and active entries |
| RI | I1 parameters with first request LoRA at 0.7 | request-lora-img2img.png | Pixels differ from I1 |
| RZ | First request LoRA at zero | request-lora-zero.png | Pixels equal P1 |
| P3 | Explicit plain mode, no request LoRA | plain-restored.png | Empty active set, pixels equal P1 |
| E | ZIMAGE_WRONG_FAMILY_LORA | wrong-lora-error.json | Failed load and explicit error. No PNG expected. |
| P4 | Explicit plain after E | plain-recovered.png | Recovery, pixels equal P1 |
| N1, Base | Custom negative "blurry, low quality, watermark", cfg 4 | negative-a.png | Effective negative metadata |
| N2, Base | Custom negative "red apple, wooden table", cfg 4 | negative-b.png | Effective negative metadata, pixels differ from N1 |
| F | Plain unquantized alias | encoder-none.png | Non-flat output, no store loader call |

Pass --outfile with the table's exact PNG path for each successful CLI case.
For E, preserve the error response and command exit instead of expecting a PNG.
Keep stdout/stderr beside each output using its case ID.

Request-LoRA flags:
```bash
--lora "$ZIMAGE_LORA:0.7"
--lora "$ZIMAGE_LORA:0.7" --lora "$ZIMAGE_LORA_2:0.5"
```

Img2img flags:
```bash
--init-image "$ZIMAGE_CASE_DIR/plain-1.png" --denoise-strength 0.35
```

After each case, save /api/models/status and the exact request parameters.
Verify current_mode, current_loras and operation capability fields.
Status lacks an authority epoch at this baseline.
Use bounded worker-load and merge logs to prove reuse/reload.
Do not infer reload count from elapsed time.

Validate HTTP as well as CLI/WebSocket:
send one plain /generate JSON request and one request-LoRA JSON request.
Use identical explicit variant parameters.
Require HTTP 200, a valid PNG and correct current_loras.
Save an invalid path request and its HTTP 400 response.
Save malformed-entry HTTP 422 and mode/request-mixing HTTP 400 responses.
Verify rejection caused no model switch through bounded logs and unchanged status.

### Step 4: Compare decoded pixels

Use signed or floating arrays for differences.
Do not compare PNG file bytes because metadata can differ.

```python
import os
from pathlib import Path
import numpy as np
from PIL import Image

proof_dir = Path(os.environ["ZIMAGE_CASE_DIR"])

def pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == (1024, 1024)
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    assert np.isfinite(rgb).all()
    assert float(rgb.std()) >= 5.0
    return rgb

import json

equal_pairs = {
    "P1=P2": ("plain-1.png", "plain-2.png"),
    "Z=P1": ("zero-mode-lora.png", "plain-1.png"),
    "R2=R1": ("request-lora-07-repeat.png", "request-lora-07.png"),
    "RZ=P1": ("request-lora-zero.png", "plain-1.png"),
    "P3=P1": ("plain-restored.png", "plain-1.png"),
    "P4=P1": ("plain-recovered.png", "plain-1.png"),
}
equality = {
    case: bool(np.array_equal(pixels(proof_dir / a), pixels(proof_dir / b)))
    for case, (a, b) in equal_pairs.items()
}
plain = pixels(proof_dir / "plain-1.png")
lora = pixels(proof_dir / "request-lora-07.png")
lora_changed = not np.array_equal(plain, lora)
mad = float(np.abs(plain.astype(np.float32) - lora.astype(np.float32)).mean())
report = {"equality": equality, "lora_changed": bool(lora_changed), "lora_mad": mad}
(proof_dir / "pixel-checks.json").write_text(json.dumps(report, indent=2))
assert all(equality.values()), "record mismatch and run three-trial diagnostics"
assert lora_changed, "LoRA output did not change"
```

Define proof_dir from ZIMAGE_CASE_DIR before executing this snippet.
Save MAD values as measurements.
No arbitrary MAD threshold establishes LoRA semantics.
User reviews image quality and intended LoRA effect.
A failed equality, flat output or wrong active set fails its case.
Before declaring a product defect, run this diagnostic sequence:
1. Create equality-repeat-1, equality-repeat-2 and equality-repeat-3 beneath the current variant/isolation directory.
2. Point ZIMAGE_CASE_DIR at one fresh trial directory.
3. Run P1, P2, Z, R1, R2, RZ, P3, E and P4 with the original seed and parameters.
4. Preserve each case's reload boundary, active-set snapshot, terminal response and output filename.
5. Run the comparison snippet for that trial and preserve pixel-checks.json.
6. Repeat Steps 2-5 for each remaining trial directory.
7. Record all outcomes before attributing a product defect or requesting a tolerance change.

Compare each trial only against its own P1 and R1.
Never overwrite the original failed outputs.
If all diagnostic trials pass, retain the original mismatch as an intermittent acceptance finding.
Any unresolved mismatch keeps acceptance incomplete.
Escalate the evidence and proposed diagnosis for review.
A tolerance change still requires operator approval.
Do not replace failure with a skip or quietly widen tolerance.

### Step 5: Restart hit and isolation proof

Stop only the acceptance server using its recorded process handle.
Keep its dedicated encoder-store directory.
Restart the same reviewed code with WORKER_ISOLATION=subprocess.
Re-run the matrix in a separate variant/isolation output subdirectory.
Require an encoder-store hit for each variant after process restart.
Require no encoder rebuild for the corresponding unchanged content key.
Do not assume the two variants have equal encoder keys.
Record process exits and restart timestamps.

Run explicit encoder none for each variant in both isolation modes.
It is a diagnostic/configuration option, not a fallback for failing int8 acceptance.
If int8 fails, the parity task remains incomplete.

### Step 6: Record variant result

Store a JSON report for each variant and isolation mode.
Report fields must include:
schema=mps_zimage_acceptance/v1, issue, variant, isolation, revision, dirty state,
host, package versions, model/LoRA hashes, effective requests, PNG paths,
pixel comparisons, status snapshots, log intervals, memory samples,
cold-load/warm-job/reload seconds, test exits and user visual result.

Mark each case passed, failed or unavailable.
Unavailable never means passed.
The report overall result requires every mandatory case and visual review.
Store all artifacts under ZIMAGE_PROOF_DIR and reference them from the corresponding FP issue.
Do not mark Task 4 or 5 done until its own evidence passes.
Do not claim a published or deployed capability from a local acceptance result.

## Task 6: regressions and accepted-scope docs

FP: STABL-ryapqlom.
Files: MPS_BACKEND.md and WORKER_SELECTION.md.
Depends on accepted Turbo and Base evidence.

### Step 1: Run focused regression cohorts

```bash
python -m tests.run tests/test_zimage_family.py tests/test_zimage_worker.py tests/test_zimage_text_encoder.py tests/test_zimage_lora_worker.py tests/test_diffusers_lora_fuse.py tests/test_zimage_mps_policy.py tests/test_mps_policy.py tests/test_mps_registration.py tests/test_mps_worker.py tests/test_hunyuan_mps_worker.py tests/test_backend_runtimes.py -- -q --no-cov -p no:cacheprovider
python -m tests.run tests/test_request_loras.py tests/test_request_lora_routes.py tests/test_governor.py tests/test_governor_request_loras.py tests/test_worker_pool.py tests/test_worker_handle.py -- -q --no-cov -p no:cacheprovider
python -m tests.run tests/test_progress_bar_suppression.py -- -q --no-cov -p no:cacheprovider
```

Expected: exit 0 for each command.
Record actual counts and skips.
The native encoder test runs separately with its explicit opt-in.

Run the native CPU container suite on exact base and candidate revisions.
Use docs/TESTING_IN_DOCKER.md and the CPU test service.
Do not run Linux CUDA containers on this Mac as MPS evidence.
Compare failing node IDs, skip reasons and process exits.
Do not treat historical 19-failure counts as the current baseline.
Investigate every lost baseline pass.
Full native CUDA execution needs an authorized NVIDIA host.
If unavailable, state that limit explicitly rather than claiming GPU regression proof.

### Step 2: Write evidence-bounded operator docs

Run drift refs before editing each document and its bound sources.
Document both zimage variants, bf16 family policy, int8/default and none modes,
mode/request LoRAs, no ControlNet, memory field meanings and accepted isolation modes.
Preserve existing fp16 behavior for SD1.5 and HunyuanDiT.
Document CUDA_DTYPE precedence and unsupported CPU fallback.
Reference exact reviewed revision and separate Turbo/Base evidence.
Keep STABL-puwnngfo PNG recreate limitation visible.

Review each changed prose claim against code and artifacts.
Only then refresh affected drift links.
Run drift check.
The review revision recorded 48 stale document-target pairs in
docs/superpowers/reports/2026-11-15-mps-zimage-drift-baseline.txt.
Compare exact document-target pairs against that file.
Unrelated stale anchors must remain explicit baseline exceptions.

### Step 3: Final review and handoff

```bash
git diff --check
git status --short
git log -5 --oneline
FP_AGENT_NAME='theta' fp tree STABL-zubknmov
```

Review the implementation against every design contract and acceptance case.
No subagent review is permitted by this repository.
Record unresolved findings and exact next action in FP.
Commit only reviewed task files with issue ID and next step.
Assign commits to their issues with fp issue assign --rev.
Keep push, PR, merge, issue closure and installed deployment as separate evidenced states.

## Plan self-review

Coverage:
- Both variants: Tasks 3, 4 and 5.
- Dtype, strict device use and telemetry: Task 1.
- int8/none and filesystem restart: Tasks 1, 2, 4 and 5.
- Mode/request/multiple LoRAs and clean restoration: Tasks 3, 4 and 5.
- Base CFG and negative prompts: Task 5.
- CUDA and existing MPS regression: Task 6.
- Native paths, exact revisions, isolation and authorization: task runbook and execution section.
- Known PNG recreate limitation: design and Task 6.

No implementation or native full-model acceptance is marked complete.
Next action: review this plan and design at the committed revision.
