# SDXL FP8 Capability Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable capability-driven support for Diffusers-loadable single-file SDXL FP8 checkpoints, starting with `wangkanai/sdxl-fp8`, so they can be configured through the existing mode system and loaded safely at `512x512` on limited VRAM without forcing the wrong scheduler profile.

**Architecture:** Keep the existing worker split (`SD1.5` vs `SDXL`) but extend model detection and mode config with loader capabilities such as `loader_format`, `checkpoint_precision`, `checkpoint_variant`, and `scheduler_profile`. Mode overrides are authoritative and must be merged in the mode-loading path before worker construction. The worker factory should receive already-resolved capabilities and pass the resulting `ModelInfo` into the chosen worker so loader and scheduler policy live in one place and future single-file variants reuse the same path.

**Tech Stack:** Python 3.12, Diffusers, PyTorch CUDA backend, SafeTensors, FastAPI mode management API, pytest.

---

## File Structure

- Modify: `utils/model_detector.py`
  Purpose: carry reusable loader capability metadata alongside existing architecture detection.
- Modify: `server/mode_config.py`
  Purpose: allow mode-level overrides for loader format, checkpoint precision, checkpoint variant, scheduler profile, and recommended generation size.
- Modify: `server/model_routes.py`
  Purpose: expose and persist the new mode fields through the existing REST API.
- Modify: `backends/worker_factory.py`
  Purpose: accept already-resolved model capabilities, choose worker family, and pass those capabilities into the CUDA workers.
- Modify: `backends/worker_pool.py`
  Purpose: merge authoritative mode overrides with detected model capabilities before worker construction.
- Modify: `backends/cuda_worker.py`
  Purpose: select loader policy from capabilities, keep `from_single_file()` reusable, and avoid redundant runtime fp8 quantization when the checkpoint is already fp8.
- Create: `tests/test_mode_config.py`
  Purpose: cover mode parsing and serialization of capability overrides.
- Modify: `tests/test_worker_factory.py`
  Purpose: cover capability-aware worker creation and override merge behavior.
- Create: `tests/test_cuda_worker_capabilities.py`
  Purpose: cover loader-policy selection and “pre-quantized fp8 checkpoint skips Quanto” behavior with fully stubbed modules.
- Modify: `docs/SDXL_WORKER.md`
  Purpose: document how to configure SDXL fp8 single-file checkpoints in `modes.yml`, including scheduler profile.

### Task 1: Add Capability Fields to Model and Mode Metadata

**Files:**
- Modify: `utils/model_detector.py`
- Modify: `server/mode_config.py`
- Modify: `server/model_routes.py`
- Modify: `backends/worker_pool.py`
- Create: `tests/test_mode_config.py`

- [ ] **Step 1: Write the failing mode-config test**

```python
def test_mode_config_parses_loader_capability_overrides(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-fp8
modes:
  sdxl-fp8:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
    default_steps: 30
    default_guidance: 7.5
    loader_format: single_file
    checkpoint_precision: fp8
    checkpoint_variant: sdxl-base
    scheduler_profile: native
    recommended_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sdxl-fp8")

    assert mode.loader_format == "single_file"
    assert mode.checkpoint_precision == "fp8"
    assert mode.checkpoint_variant == "sdxl-base"
    assert mode.scheduler_profile == "native"
    assert mode.recommended_size == "512x512"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mode_config.py::test_mode_config_parses_loader_capability_overrides -v`
Expected: FAIL because `ModeConfig` does not yet expose the new fields.

- [ ] **Step 3: Extend `ModelInfo` and `ModeConfig` with capability fields**

```python
@dataclass
class ModelInfo:
    path: str
    variant: ModelVariant = ModelVariant.UNKNOWN
    cross_attention_dim: Optional[int] = None
    ...
    loader_format: str = "unknown"
    checkpoint_precision: str = "unknown"
    checkpoint_variant: str = "unknown"
```

```python
@dataclass
class ModeConfig:
    name: str
    model: str
    loras: List[LoRAConfig] = field(default_factory=list)
    default_size: str = "512x512"
    default_steps: int = 4
    default_guidance: float = 1.0
    loader_format: Optional[str] = None
    checkpoint_precision: Optional[str] = None
    checkpoint_variant: Optional[str] = None
    scheduler_profile: Optional[str] = None
    recommended_size: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Parse, serialize, and return the new mode fields**

```python
mode = ModeConfig(
    name=mode_name,
    model=mode_data["model"],
    loras=loras,
    default_size=mode_data.get("default_size", "512x512"),
    default_steps=mode_data.get("default_steps", 4),
    default_guidance=mode_data.get("default_guidance", 1.0),
    loader_format=mode_data.get("loader_format"),
    checkpoint_precision=mode_data.get("checkpoint_precision"),
    checkpoint_variant=mode_data.get("checkpoint_variant"),
    scheduler_profile=mode_data.get("scheduler_profile"),
    recommended_size=mode_data.get("recommended_size"),
    metadata=mode_data.get("metadata", {}),
)
```

```python
return {
    "default_mode": modes_dict["default_mode"],
    "modes": {
        name: {
            "model": mode_data["model"],
            "loras": mode_data["loras"],
            "default_size": mode_data["default_size"],
            "default_steps": mode_data["default_steps"],
            "default_guidance": mode_data["default_guidance"],
            "loader_format": mode_data.get("loader_format"),
            "checkpoint_precision": mode_data.get("checkpoint_precision"),
            "checkpoint_variant": mode_data.get("checkpoint_variant"),
            "scheduler_profile": mode_data.get("scheduler_profile"),
            "recommended_size": mode_data.get("recommended_size"),
        }
        for name, mode_data in modes_dict["modes"].items()
    },
}
```

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/test_mode_config.py tests/test_workflow_routes.py tests/test_worker_pool.py -q`
Expected: PASS for the new mode-config test and no regressions in existing config consumers.

- [ ] **Step 6: Commit**

```bash
git add utils/model_detector.py server/mode_config.py server/model_routes.py backends/worker_pool.py tests/test_mode_config.py
git commit -m "feat: add loader capability fields to model and mode metadata"
```

### Task 2: Detect and Merge Loader Capabilities

**Files:**
- Modify: `utils/model_detector.py`
- Modify: `backends/worker_factory.py`
- Modify: `backends/worker_pool.py`
- Modify: `tests/test_worker_factory.py`

- [ ] **Step 1: Write the failing factory test for capability merge**

```python
@patch("utils.model_detector.detect_model")
@patch("backends.cuda_worker.DiffusersSDXLCudaWorker")
def test_create_sdxl_worker_passes_detected_capabilities(mock_worker_cls, mock_detect):
    mock_info = Mock()
    mock_info.cross_attention_dim = 2048
    mock_info.variant = Mock(value="sdxl-base")
    mock_info.confidence = 0.95
    mock_info.loader_format = "single_file"
    mock_info.checkpoint_precision = "fp8"
    mock_info.checkpoint_variant = "sdxl-base"
    mock_detect.return_value = mock_info

    create_cuda_worker(worker_id=3, model_path="/models/checkpoints/sdxl-base.safetensors")

    kwargs = mock_worker_cls.call_args.kwargs
    assert kwargs["model_info"].loader_format == "single_file"
    assert kwargs["model_info"].checkpoint_precision == "fp8"
    assert kwargs["model_info"].scheduler_profile == "native"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_worker_factory.py::TestCreateCudaWorker::test_create_sdxl_worker_passes_detected_capabilities -v`
Expected: FAIL because the factory does not yet pass `model_info`.

- [ ] **Step 3: Add capability inference helpers in the model detector**

```python
def _infer_loader_format(path: str) -> str:
    if os.path.isdir(path):
        return "diffusers_dir"
    if Path(path).suffix.lower() in {".safetensors", ".ckpt"}:
        return "single_file"
    return "unknown"


def _infer_checkpoint_precision(keys: list[str]) -> str:
    lowered = " ".join(keys).lower()
    if "fp8" in lowered or "float8" in lowered:
        return "fp8"
    return "unknown"
```

```python
info.loader_format = _infer_loader_format(path)
if info.variant == ModelVariant.SDXL_BASE:
    info.checkpoint_variant = "sdxl-base"
elif info.variant == ModelVariant.SDXL_REFINER:
    info.checkpoint_variant = "sdxl-refiner"
info.scheduler_profile = "native"
```

- [ ] **Step 4: Pass resolved `ModelInfo` into the worker constructors**

```python
def inspect_model(model_path: str) -> ModelInfo:
    from utils.model_detector import detect_model
    return detect_model(model_path)


def create_cuda_worker(worker_id: int, model_path: str) -> "PipelineWorker":
    model_info = inspect_model(model_path)
    worker_type = _worker_type_from_info(model_info)

    if worker_type == "sdxl":
        from backends.cuda_worker import DiffusersSDXLCudaWorker
        return DiffusersSDXLCudaWorker(worker_id=worker_id, model_path=model_path, model_info=model_info)
```

- [ ] **Step 5: Add authoritative mode-override merge in the worker-pool load path**

```python
def merge_mode_capabilities(model_info: ModelInfo, mode) -> ModelInfo:
    if mode.loader_format:
        model_info.loader_format = mode.loader_format
    if mode.checkpoint_precision:
        model_info.checkpoint_precision = mode.checkpoint_precision
    if mode.checkpoint_variant:
        model_info.checkpoint_variant = mode.checkpoint_variant
    if mode.scheduler_profile:
        model_info.scheduler_profile = mode.scheduler_profile
    return model_info
```

```python
mode = self._mode_config.get_mode(mode_name)
model_info = inspect_model(mode.model_path)
model_info = merge_mode_capabilities(model_info, mode)
worker = self._worker_factory(worker_id=0, model_path=mode.model_path, model_info=model_info)
```

Note: perform this merge in the worker-pool load path where the selected mode is already available. Do not re-read YAML from inside the worker or factory.

- [ ] **Step 6: Run targeted tests**

Run: `pytest tests/test_worker_factory.py tests/test_worker_pool.py -q`
Expected: PASS with updated constructor assertions and no regression in SD1.5/SDXL family selection.

- [ ] **Step 7: Commit**

```bash
git add utils/model_detector.py backends/worker_factory.py backends/worker_pool.py tests/test_worker_factory.py tests/test_worker_pool.py
git commit -m "feat: detect and pass model loader capabilities to workers"
```

### Task 3: Apply Capability-Driven Loader Policies in CUDA Workers

**Files:**
- Modify: `backends/cuda_worker.py`
- Create: `tests/test_cuda_worker_capabilities.py`

- [ ] **Step 1: Write the failing worker capability test**

```python
def test_sdxl_single_file_fp8_uses_from_single_file_and_skips_runtime_quantize(monkeypatch):
    model_info = SimpleNamespace(
        loader_format="single_file",
        checkpoint_precision="fp8",
        checkpoint_variant="sdxl-base",
        scheduler_profile="native",
    )

    with patch("backends.cuda_worker.StableDiffusionXLPipeline.from_single_file") as single_file, \
         patch("backends.cuda_worker.StableDiffusionXLPipeline.from_pretrained") as pretrained:
        pipe = _make_sdxl_pipe()
        single_file.return_value = pipe

        worker = DiffusersSDXLCudaWorker(worker_id=0, model_path="/models/sdxl-base.safetensors", model_info=model_info)

        single_file.assert_called_once()
        pretrained.assert_not_called()
        assert worker._checkpoint_precision == "fp8"
        assert worker._scheduler_profile == "native"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cuda_worker_capabilities.py::test_sdxl_single_file_fp8_uses_from_single_file_and_skips_runtime_quantize -v`
Expected: FAIL because the worker constructor does not yet accept `model_info`.

- [ ] **Step 3: Accept `model_info` in worker constructors and persist normalized capabilities**

```python
class CudaWorkerBase:
    def __init__(self, worker_id: int, model_info: Any | None = None) -> None:
        self.worker_id = worker_id
        self.model_info = model_info
        self._loader_format = getattr(model_info, "loader_format", "unknown")
        self._checkpoint_precision = getattr(model_info, "checkpoint_precision", "unknown")
        self._checkpoint_variant = getattr(model_info, "checkpoint_variant", "unknown")
        self._scheduler_profile = getattr(model_info, "scheduler_profile", "native")
        ...
```

```python
class DiffusersSDXLCudaWorker(CudaWorkerBase):
    def __init__(self, worker_id: int, model_path: str, model_info: Any | None = None):
        super().__init__(worker_id, model_info=model_info)
```

- [ ] **Step 4: Split SDXL load policy into explicit branches**

```python
def _load_sdxl_pipeline(self, ckpt_path: str):
    if self._loader_format == "diffusers_dir":
        return StableDiffusionXLPipeline.from_pretrained(
            ckpt_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None,
        ), "diffusers"

    return StableDiffusionXLPipeline.from_single_file(
        ckpt_path,
        torch_dtype=self.dtype,
        use_safetensors=True,
    ), "single-file"
```

- [ ] **Step 5: Apply scheduler policy and skip redundant Quanto fp8 quantization**

```python
if self._scheduler_profile == "lcm":
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
```

```python
def _setup_pipe_memory_opts(self, pipe):
    should_quantize_runtime = self._quantize == "fp8" and self._checkpoint_precision != "fp8"
    if should_quantize_runtime:
        from optimum.quanto import freeze, quantize, qfloat8
        quantize(pipe.unet, weights=qfloat8)
        freeze(pipe.unet)
        if hasattr(pipe, "text_encoder_2"):
            quantize(pipe.text_encoder_2, weights=qfloat8)
            freeze(pipe.text_encoder_2)
```

Note: standard SDXL checkpoints like `wangkanai/sdxl-fp8` should stay on their native scheduler unless the mode explicitly opts into `lcm`. This change is intentionally about avoiding double-quantization and forcing the wrong scheduler. It should not auto-enable fp8 env vars for unrelated checkpoints.

- [ ] **Step 6: Run worker capability tests**

Run: `pytest tests/test_cuda_worker_base.py tests/test_cuda_worker_capabilities.py -q`
Expected: PASS for both the existing memory-option tests and the new capability-driven loader tests.

- [ ] **Step 7: Commit**

```bash
git add backends/cuda_worker.py tests/test_cuda_worker_capabilities.py tests/test_cuda_worker_base.py
git commit -m "feat: apply capability-driven SDXL single-file loader policy"
```

### Task 4: Document Shared FP8 SDXL Mode Usage and Wire Acceptance

**Files:**
- Modify: `docs/SDXL_WORKER.md`
- Modify: `conf/modes.yml` (example only if the local deployment should expose the model by default)
- Modify: `docs/DYNAMIC_MODEL_LOADING.md`

- [ ] **Step 1: Add documentation example for `wangkanai/sdxl-fp8`**

```yaml
modes:
  sdxl-fp8:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
    default_steps: 30
    default_guidance: 7.5
    loader_format: single_file
    checkpoint_precision: fp8
    checkpoint_variant: sdxl-base
    scheduler_profile: native
    recommended_size: 512x512
```

- [ ] **Step 2: Document supported scope explicitly**

```md
- Supported: Diffusers-loadable SDXL single-file checkpoints, including pre-quantized fp8 checkpoints that load through `from_single_file()`.
- Not guaranteed: arbitrary vendor-specific checkpoint layouts that require custom conversion code.
- Scheduler policy is explicit: standard SDXL checkpoints stay `native`; only LCM-tuned checkpoints should use `lcm`.
- Recommended for 8 GB GPUs: start at `512x512`, use working `xformers`, and enable `CUDA_OFFLOAD=model` when needed.
```

- [ ] **Step 3: Run documentation-adjacent verification**

Run: `pytest tests/test_worker_factory.py tests/test_cuda_worker_capabilities.py tests/test_mode_config.py -q`
Expected: PASS

- [ ] **Step 4: Manual acceptance check**

Run in a CUDA container with working xformers:

```bash
curl -X POST http://localhost:4200/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "sdxl-fp8",
    "prompt": "a serene mountain landscape at sunset, photorealistic",
    "size": "512x512",
    "num_inference_steps": 30,
    "guidance_scale": 7.5
  }'
```

Expected: `200` response and successful SDXL image generation without touching SD1.5 code paths.

- [ ] **Step 5: Commit**

```bash
git add docs/SDXL_WORKER.md docs/DYNAMIC_MODEL_LOADING.md conf/modes.yml
git commit -m "docs: add reusable SDXL fp8 single-file mode configuration"
```

## Self-Review

- Spec coverage:
  - Reusable capability fields: covered in Task 1 and Task 2.
  - Shared loader-policy architecture and scheduler policy: covered in Task 3.
  - Support for the provided SDXL fp8 single-file checkpoint at `512x512`: covered in Task 4.
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to previous task” shortcuts remain.
- Type consistency:
  - Uses `loader_format`, `checkpoint_precision`, `checkpoint_variant`, `scheduler_profile`, and `recommended_size` consistently across detector, mode config, API, worker pool, and worker layers.

## Execution Handoff

Plan complete and saved to `docs/PLAN_sdxl_fp8_capability_loader.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
