# SDXL FP8 Runtime Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SDXL FP8 single-file mode load with an explicit low-VRAM runtime policy, avoid redundant runtime FP8 quantization for native-FP8 checkpoints, and support a practical 4-5GB VRAM operating mode via offload/slicing/xformers configuration.

**Architecture:** Reuse the existing capability pipeline (`ModeConfig` -> `WorkerPool.merge_mode_capabilities()` -> `ModelInfo` -> CUDA worker) and extend it with runtime memory-policy fields. `CudaWorkerBase` becomes the single authority for normalizing env defaults and mode overrides, while the SDXL worker keeps its existing `from_single_file()` loader path but stops double-quantizing when `checkpoint_precision == "fp8"`.

**Tech Stack:** Python 3.12, Diffusers, PyTorch CUDA backend, SafeTensors, pytest.

---

## File Structure

- Modify: `server/mode_config.py`
  Purpose: parse and serialize runtime memory-policy overrides alongside existing capability fields.
- Modify: `server/model_routes.py`
  Purpose: expose the runtime policy fields through the existing `/api/modes` response.
- Modify: `utils/model_detector.py`
  Purpose: carry runtime policy fields on `ModelInfo` so worker creation stays capability-driven.
- Modify: `backends/worker_pool.py`
  Purpose: merge authoritative mode-level runtime overrides onto detected model capabilities.
- Modify: `backends/cuda_worker.py`
  Purpose: normalize runtime policy once, let mode overrides win over env vars, and skip redundant Quanto fp8 quantization for native-FP8 checkpoints.
- Modify: `conf/modes.yml`
  Purpose: encode the SDXL FP8 mode with explicit low-VRAM runtime policy.
- Modify: `tests/test_mode_config.py`
  Purpose: verify parsing and round-tripping of runtime policy fields.
- Modify: `tests/test_model_routes.py`
  Purpose: verify runtime policy fields are returned by `/api/modes`.
- Modify: `tests/test_cuda_worker_base.py`
  Purpose: verify model-info runtime overrides beat env defaults.
- Modify: `tests/test_cuda_worker_capabilities.py`
  Purpose: verify native-FP8 SDXL skips runtime quantization and uses the requested low-VRAM runtime policy.
- Modify: `docs/SDXL_WORKER.md`
  Purpose: document the practical low-VRAM SDXL FP8 configuration and expected tradeoffs.

### Task 1: Add Runtime Policy Fields to Mode and Model Metadata

**Files:**
- Modify: `server/mode_config.py`
- Modify: `server/model_routes.py`
- Modify: `utils/model_detector.py`
- Modify: `backends/worker_pool.py`
- Modify: `tests/test_mode_config.py`
- Modify: `tests/test_model_routes.py`

- [ ] **Step 1: Write the failing mode-config test**

```python
def test_mode_config_parses_runtime_policy_overrides(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-fp8
modes:
  sdxl-fp8:
    model: checkpoints/sdxl/model.safetensors
    loader_format: single_file
    checkpoint_precision: fp8
    scheduler_profile: native
    runtime_quantize: none
    runtime_offload: model
    runtime_attention_slicing: true
    runtime_enable_xformers: true
""".strip()
    )

    from server.mode_config import ModeConfigManager

    mode = ModeConfigManager(str(tmp_path)).get_mode("sdxl-fp8")

    assert mode.runtime_quantize == "none"
    assert mode.runtime_offload == "model"
    assert mode.runtime_attention_slicing is True
    assert mode.runtime_enable_xformers is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mode_config.py::test_mode_config_parses_runtime_policy_overrides -v`
Expected: FAIL because the runtime policy fields do not exist on `ModeConfig`.

- [ ] **Step 3: Extend `ModeConfig` and `ModelInfo`**

```python
@dataclass
class ModeConfig:
    ...
    runtime_quantize: Optional[str] = None
    runtime_offload: Optional[str] = None
    runtime_attention_slicing: Optional[bool] = None
    runtime_enable_xformers: Optional[bool] = None
```

```python
@dataclass
class ModelInfo:
    ...
    runtime_quantize: Optional[str] = None
    runtime_offload: Optional[str] = None
    runtime_attention_slicing: Optional[bool] = None
    runtime_enable_xformers: Optional[bool] = None
```

- [ ] **Step 4: Parse, merge, and expose the new fields**

```python
mode = ModeConfig(
    ...
    runtime_quantize=mode_data.get("runtime_quantize"),
    runtime_offload=mode_data.get("runtime_offload"),
    runtime_attention_slicing=mode_data.get("runtime_attention_slicing"),
    runtime_enable_xformers=mode_data.get("runtime_enable_xformers"),
)
```

```python
for field in (
    "loader_format",
    "checkpoint_precision",
    "checkpoint_variant",
    "scheduler_profile",
    "runtime_quantize",
    "runtime_offload",
    "runtime_attention_slicing",
    "runtime_enable_xformers",
    ...
):
    value = getattr(mode, field, None)
    if value is not None:
        setattr(resolved, field, value)
```

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/test_mode_config.py tests/test_model_routes.py tests/test_worker_pool.py -q`
Expected: PASS with runtime policy fields parsed and returned by the API.

### Task 2: Make `CudaWorkerBase` Respect Mode-Level Runtime Policy

**Files:**
- Modify: `backends/cuda_worker.py`
- Modify: `tests/test_cuda_worker_base.py`

- [ ] **Step 1: Write the failing base-worker override test**

```python
def test_model_info_runtime_policy_overrides_env_defaults():
    pipe = _make_pipe()
    model_info = SimpleNamespace(
        runtime_quantize="none",
        runtime_offload="model",
        runtime_attention_slicing=True,
        runtime_enable_xformers=True,
        checkpoint_precision="fp8",
    )

    base = _make_base(
        {"CUDA_QUANTIZE": "fp8", "CUDA_OFFLOAD": "none"},
        model_info=model_info,
    )

    base._setup_pipe_memory_opts(pipe)

    pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=0)
    pipe.enable_attention_slicing.assert_called_once_with(1)
    pipe.enable_xformers_memory_efficient_attention.assert_called_once()
    pipe.to.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cuda_worker_base.py::TestRuntimePolicyOverrides::test_model_info_runtime_policy_overrides_env_defaults -v`
Expected: FAIL because `CudaWorkerBase` ignores `model_info`.

- [ ] **Step 3: Normalize runtime policy in the base worker**

```python
def __init__(self, worker_id: int, model_info: Any | None = None) -> None:
    self.worker_id = worker_id
    self.model_info = model_info
    ...
    self._parse_env()
```

```python
def _parse_env(self) -> None:
    ...
    self._quantize = getattr(self.model_info, "runtime_quantize", None) or env_quantize
    self._offload = getattr(self.model_info, "runtime_offload", None) or env_offload
    self._attention_slicing = (
        getattr(self.model_info, "runtime_attention_slicing", None)
        if getattr(self.model_info, "runtime_attention_slicing", None) is not None
        else env_attention_slicing
    )
    self._enable_xformers = (
        getattr(self.model_info, "runtime_enable_xformers", None)
        if getattr(self.model_info, "runtime_enable_xformers", None) is not None
        else env_enable_xformers
    )
```

- [ ] **Step 4: Pass `model_info` into both CUDA worker constructors**

```python
class DiffusersSDXLCudaWorker(CudaWorkerBase):
    def __init__(self, worker_id: int, model_path: str, model_info: Optional[Any] = None):
        super().__init__(worker_id, model_info=model_info)
```

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py -q`
Expected: PASS for override behavior with no regression in offload routing.

### Task 3: Skip Redundant Runtime FP8 Quantization for Native-FP8 Checkpoints

**Files:**
- Modify: `backends/cuda_worker.py`
- Modify: `tests/test_cuda_worker_capabilities.py`

- [ ] **Step 1: Flip the existing capability test to the desired behavior**

```python
def test_fp8_single_file_checkpoint_skips_runtime_quanto_when_requested():
    pipe = _make_pipe()
    model_info = SimpleNamespace(
        checkpoint_precision="fp8",
        loader_format="single_file",
        runtime_quantize="fp8",
    )
    base = _make_base({"CUDA_QUANTIZE": "fp8"}, model_info=model_info)
    fake_qfloat8 = object()
    fake_quanto = SimpleNamespace(quantize=Mock(), freeze=Mock(), qfloat8=fake_qfloat8)

    with patch.dict(sys.modules, {"optimum.quanto": fake_quanto}):
        base._setup_pipe_memory_opts(pipe)

    fake_quanto.quantize.assert_not_called()
    fake_quanto.freeze.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cuda_worker_capabilities.py::TestCapabilityAwareMemoryOpts::test_fp8_single_file_checkpoint_skips_runtime_quanto_when_requested -v`
Expected: FAIL because the worker still quantizes even when the checkpoint is already fp8.

- [ ] **Step 3: Gate runtime Quanto application**

```python
should_quantize_runtime = (
    self._quantize == "fp8"
    and getattr(self.model_info, "checkpoint_precision", "unknown") != "fp8"
)
if should_quantize_runtime:
    from optimum.quanto import freeze, quantize, qfloat8
    ...
```

- [ ] **Step 4: Keep the SDXL single-file loader path explicit**

```python
pipe = StableDiffusionXLPipeline.from_single_file(
    ckpt_path,
    torch_dtype=self.dtype,
)
```

This task intentionally does not invent a new custom FP8 component loader. The measurable win comes from avoiding double-quantization and pairing the checkpoint with explicit low-VRAM runtime settings.

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/test_cuda_worker_capabilities.py tests/test_worker_factory.py -q`
Expected: PASS for capability-aware loader behavior and no regression in worker selection.

### Task 4: Configure the SDXL FP8 Mode for Low-VRAM Operation

**Files:**
- Modify: `conf/modes.yml`
- Modify: `docs/SDXL_WORKER.md`

- [ ] **Step 1: Update the SDXL mode in `conf/modes.yml`**

```yaml
  SDXL:
    model: checkpoints/sdxl4GB2GBImprovedFP8_fp8FullCheckpoint.safetensors
    loader_format: single_file
    checkpoint_precision: fp8
    checkpoint_variant: sdxl-base
    scheduler_profile: native
    runtime_quantize: none
    runtime_offload: model
    runtime_attention_slicing: true
    runtime_enable_xformers: true
    default_size: 512x512
    recommended_size: 512x512
    default_steps: 11
    default_guidance: 3.0
```

- [ ] **Step 2: Document the tradeoff explicitly**

```markdown
- Pre-quantized FP8 checkpoints are still loaded through `from_single_file()`.
- Native-FP8 checkpoints should not be re-quantized with Quanto.
- To reach ~4-5GB operational VRAM, use `runtime_offload: model` plus xformers and attention slicing.
- This reduces active VRAM at the cost of generation speed.
```

- [ ] **Step 3: Run the targeted verification suite**

Run: `pytest tests/test_mode_config.py tests/test_model_routes.py tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py tests/test_worker_factory.py -q`
Expected: PASS

- [ ] **Step 4: Optional live validation**

Run: `pytest tests/test_sdxl_worker.py::test_basic_generation -v -s`
Expected: PASS if the local SDXL model and CUDA runtime are available. Observe VRAM during worker init and first generation to confirm the practical runtime window.
