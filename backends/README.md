# Backends Package

This package contains all model workers and factory logic for the LCM server.

## Structure

```
backends/
├── base.py              # Base classes and protocols
├── cuda_worker.py       # CUDA workers (SD1.5 and SDXL)
├── rknn_worker.py       # RKNN NPU worker
├── worker_factory.py    # Factory with automatic model detection
└── styles.py            # LoRA style definitions
```

## Worker Factory

The `worker_factory.py` module builds a CUDA worker from an already-resolved model:

```python
from backends.worker_factory import create_cuda_worker
from backends.model_resolution import resolve_model

resolved, binding = resolve_model(model_path, mode)
worker = create_cuda_worker(worker_id=0, resolved=resolved, binding=binding)
```

### How It Works

1. **Resolution**: `resolve_model()` detects the model once, resolves its neutral
   family (`backends.family_profiles`), overlays mode capabilities, and emits a
   portable `ResolvedModel` plus a node-local `LocalModelBinding`.
2. **Dispatch**: `create_cuda_worker()` looks up the canonical CUDA cell from
   `resolved.profile.family_id` in `CUDA_FAMILY_BINDINGS`
   (`backends/platforms/cuda_bindings.py`).
3. **Lazy import**: it resolves the cell's dotted `worker_ref` with `importlib`
   only inside the factory, so status reads and rejected requests never import
   Torch/Diffusers worker code.
4. **Creation**: instantiates the worker from `binding.model_path` and the thawed
   `resolved.info`. A known family with no CUDA cell raises `UnsupportedFamilyError`.

`WorkerPool._load_mode` performs this resolution once per mode load.

## Workers

### DiffusersCudaWorker (SD1.5)

- **Architecture**: SD 1.5, SD 2.0, SD 2.1
- **Cross-attention dim**: 768 (SD1.5), 1024 (SD2.x)
- **Text encoders**: Single CLIP encoder
- **Default resolution**: 512x512
- **Latent space**: 64x64

### DiffusersSDXLCudaWorker (SDXL)

- **Architecture**: SDXL Base, SDXL Refiner
- **Cross-attention dim**: 2048
- **Text encoders**: Dual (CLIP-L + OpenCLIP-G)
- **Default resolution**: 1024x1024
- **Latent space**: 128x128

### RKNNPipelineWorker (NPU)

- **Architecture**: RK3588 NPU-optimized
- **Format**: RKNN compiled models
- **Context**: Multi-core support (NPU_CORE_0/1/2)

## LoRA Support

Both CUDA workers support LoRAs with automatic filtering:

- **SD1.5 Worker**: Only loads LoRAs with `required_cross_attention_dim=768`
- **SDXL Worker**: Only loads LoRAs with `required_cross_attention_dim=2048`

LoRAs are defined in `styles.py` using the `StyleDef` dataclass.

## Usage in Server

CUDA generation is served by `WorkerPool` (via `CudaGenerationRuntime`), which
calls `resolve_model()` once per mode load and passes the resulting
`ResolvedModel` + `LocalModelBinding` to `create_cuda_worker()`. The RKNN path
uses `backends.rknn_runtime.PipelineService` directly. All family resolution and
worker selection is encapsulated in the backends package.
