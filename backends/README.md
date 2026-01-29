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

The `worker_factory.py` module provides automatic worker selection:

```python
from backends.worker_factory import create_cuda_worker

# Automatically detects model type and returns correct worker
worker = create_cuda_worker(worker_id=0)
```

### How It Works

1. **Detection**: `detect_worker_type()` reads `MODEL_ROOT` and `MODEL` env vars
2. **Inspection**: Uses `utils.model_detector` to inspect the model file
3. **Selection**: Returns `"sd15"` or `"sdxl"` based on `cross_attention_dim`
4. **Creation**: `create_cuda_worker()` instantiates the appropriate worker class

### Environment Variables

Both SD1.5 and SDXL use the same variables:

```bash
MODEL_ROOT=/path/to/models
MODEL=model.safetensors
```

The factory automatically detects which worker to use.

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

The server uses the factory for automatic worker creation:

```python
# server/lcm_sr_server.py
from backends.worker_factory import create_cuda_worker

if use_cuda:
    w = create_cuda_worker(worker_id=i)
else:
    w = RKNNPipelineWorker(...)
```

All model detection and selection logic is encapsulated in the backends package.
