# SDXL Worker Documentation

## Overview

The `DiffusersSDXLCudaWorker` is a CUDA-based worker for running Stable Diffusion XL (SDXL) models with LCM scheduling. It follows the same interface as the SD1.5 worker (`DiffusersCudaWorker`) but is optimized for SDXL's unique architecture.

## Key Differences from SD1.5

| Feature | SD1.5 | SDXL |
|---------|-------|------|
| **Text Encoders** | Single (CLIP-L) | Dual (CLIP-L + OpenCLIP-G) |
| **Default Resolution** | 512x512 | 1024x1024 |
| **Latent Space** | 64x64 | 128x128 |
| **Cross-Attention Dim** | 768 | 2048 |
| **VRAM Requirements** | ~4GB | ~12GB |
| **LoRA Compatibility** | SD1.5 LoRAs | SDXL LoRAs |

## Environment Variables

### Required

- **`SDXL_MODEL_ROOT`** or **`MODEL_ROOT`**: Base path to model directory
  ```bash
  export SDXL_MODEL_ROOT=/path/to/sdxl/models
  ```

- **`SDXL_MODEL`** or **`MODEL`**: Model checkpoint name
  ```bash
  export SDXL_MODEL=sdxl-1.0-base.safetensors
  # or for diffusers format:
  export SDXL_MODEL=stable-diffusion-xl-base-1.0
  ```

### Optional

- **`CUDA_DTYPE`**: Model precision (default: `fp16`)
  - `fp16` - Half precision (recommended for SDXL)
  - `bf16` - Brain float 16 (if supported by GPU)
  - `fp32` - Full precision (not recommended - very slow)

- **`CUDA_DEVICE`**: CUDA device to use (default: `cuda:0`)
  ```bash
  export CUDA_DEVICE=cuda:0
  ```

- **`CUDA_ENABLE_XFORMERS`**: Enable xformers memory-efficient attention (default: `0`)
  ```bash
  export CUDA_ENABLE_XFORMERS=1  # Recommended for SDXL
  ```

- **`CUDA_ATTENTION_SLICING`**: Enable attention slicing for memory efficiency (default: `0`)
  ```bash
  export CUDA_ATTENTION_SLICING=1
  ```

## Data Structures

### No New Structures Required

The SDXL worker uses the **same interfaces** as the SD1.5 worker:

- **`Job`**: Wrapper for generation request
  ```python
  @dataclass
  class Job:
      req: GenerateRequest
      fut: Future
      submitted_at: float
  ```

- **`GenerateRequest`**: Request parameters (from FastAPI)
  ```python
  class GenerateRequest(BaseModel):
      prompt: str
      size: str = "1024x1024"  # Default changed for SDXL
      num_inference_steps: int = 4
      guidance_scale: float = 1.0
      seed: Optional[int] = None
      style_lora: StyleLoraRequest = StyleLoraRequest()
  ```

- **`StyleLoraRequest`**: LoRA style configuration
  ```python
  class StyleLoraRequest(BaseModel):
      style: Optional[str] = None  # e.g., "papercut"
      level: int = 0  # 0=off, 1..N=strength preset
  ```

### Return Types

- **`run_job(job) -> tuple[bytes, int]`**
  - Returns: `(png_bytes, seed_used)`

- **`run_job_with_latents(job) -> tuple[bytes, int, bytes]`**
  - Returns: `(png_bytes, seed_used, latents_bytes)`
  - `latents_bytes`: Raw NCHW float16 tensor `[1,4,8,8]` = 512 bytes

## Usage Examples

### 1. Basic Setup

```bash
# Set environment variables
export SDXL_MODEL_ROOT=/models/sdxl
export SDXL_MODEL=sdxl-1.0-base.safetensors
export CUDA_ENABLE_XFORMERS=1
export CUDA_DTYPE=fp16
```

### 2. Using in Server Code

```python
from backends.cuda_worker import DiffusersSDXLCudaWorker

# Create worker
worker = DiffusersSDXLCudaWorker(worker_id=0)

# Create job
from server.lcm_sr_server import GenerateRequest
req = GenerateRequest(
    prompt="a majestic mountain landscape at sunset",
    size="1024x1024",
    num_inference_steps=4,
    guidance_scale=1.0,
    seed=42
)

job = Job(req=req, fut=Future(), submitted_at=time.time())

# Generate image
png_bytes, seed = worker.run_job(job)

# Save result
with open("output.png", "wb") as f:
    f.write(png_bytes)
```

### 3. Integrating into PipelineService

To use SDXL instead of SD1.5, modify `server/lcm_sr_server.py`:

```python
# In PipelineService.__init__():

if use_cuda:
    # Choose worker based on environment variable
    use_sdxl = os.environ.get("USE_SDXL", "0") in ("1", "true", "yes")

    if use_sdxl:
        from backends.cuda_worker import DiffusersSDXLCudaWorker
        w = DiffusersSDXLCudaWorker(worker_id=i)
    else:
        from backends.cuda_worker import DiffusersCudaWorker
        w = DiffusersCudaWorker(worker_id=i)
```

Add environment variable:
```bash
export USE_SDXL=1
```

## Model Format Support

### 1. Single-File Checkpoints (.safetensors, .ckpt)

```bash
export SDXL_MODEL_ROOT=/models/sdxl
export SDXL_MODEL=sdxl-turbo.safetensors
```

File structure:
```
/models/sdxl/
  sdxl-turbo.safetensors
  sdxl-1.0-base.safetensors
```

For pre-quantized FP8 SDXL checkpoints, keep the same single-file loader path
but make the runtime policy explicit:

- Native FP8 checkpoints should not be re-quantized with Quanto.
- `runtime_offload: model` is the main lever for reducing active VRAM into the
  4-5GB range at 512x512.
- `runtime_attention_slicing: true` and `runtime_enable_xformers: true` trade
  speed for lower working-set pressure.

Example mode:

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
```

This profile reduces operational VRAM, but it is slower than keeping the whole
pipeline resident on the GPU.

### 2. Diffusers Format (directory with model_index.json)

```bash
export SDXL_MODEL_ROOT=/models/sdxl
export SDXL_MODEL=stable-diffusion-xl-base-1.0
```

File structure:
```
/models/sdxl/stable-diffusion-xl-base-1.0/
  model_index.json
  text_encoder/
  text_encoder_2/
  unet/
  vae/
  scheduler/
  tokenizer/
  tokenizer_2/
```

## LoRA Support

The SDXL worker automatically filters LoRAs based on cross-attention dimension:

- **SD1.5 LoRAs** (cross_attention_dim=768) → Skipped
- **SDXL LoRAs** (cross_attention_dim=2048) → Loaded

### Adding SDXL LoRAs

Edit `backends/styles.py`:

```python
STYLE_REGISTRY = {
    "cyberpunk_sdxl": StyleDef(
        adapter_name="cyberpunk_sdxl",
        lora_path="/path/to/cyberpunk_sdxl.safetensors",
        levels=[0.3, 0.5, 0.7, 1.0],
        required_cross_attention_dim=2048,  # SDXL
    ),
}
```

## Performance Tuning

### Memory Optimization

1. **Enable xformers** (most effective):
   ```bash
   export CUDA_ENABLE_XFORMERS=1
   ```

2. **Use fp16** (default, recommended):
   ```bash
   export CUDA_DTYPE=fp16
   ```

3. **Enable attention slicing** (if still OOM):
   ```bash
   export CUDA_ATTENTION_SLICING=1
   ```

4. **Reduce resolution**:
   - Use 512x512 or 768x768 instead of 1024x1024

### Speed Optimization

1. **Use LCM models** or apply LCM LoRAs
2. **Reduce inference steps**: 4-8 steps is typical for LCM
3. **Use compiled UNet** (PyTorch 2.0+):
   ```python
   pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
   ```

## Testing

### Run SDXL Worker Tests

```bash
# Set model path
export SDXL_MODEL_ROOT=/path/to/models
export SDXL_MODEL=sdxl-model.safetensors

# Run all tests
pytest tests/test_sdxl_worker.py -v

# Run specific test
pytest tests/test_sdxl_worker.py::test_basic_generation -v -s

# Run with GPU info
pytest tests/test_sdxl_worker.py -v -s --log-cli-level=INFO
```

### Test Coverage

The test suite covers:
- ✅ Worker initialization and model loading
- ✅ Basic image generation
- ✅ Deterministic generation (seed consistency)
- ✅ Generation with latents
- ✅ Multiple resolutions
- ✅ Error handling (invalid size format)
- ✅ Random seed generation

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Enable xformers: `export CUDA_ENABLE_XFORMERS=1`
2. Use fp16: `export CUDA_DTYPE=fp16`
3. Reduce resolution: Use 512x512 or 768x768
4. Enable attention slicing: `export CUDA_ATTENTION_SLICING=1`
5. Close other GPU processes

### Slow Generation

**Symptom**: Generation takes >10 seconds per image

**Solutions**:
1. Ensure you're using an LCM model or LCM scheduler
2. Reduce inference steps to 4-6
3. Enable xformers for memory-efficient attention
4. Check GPU utilization: `nvidia-smi -l 1`

### Wrong Pipeline Type

**Symptom**: Error about SDXL UNet with SD1.5 pipeline

**Solutions**:
- Ensure you're using `DiffusersSDXLCudaWorker` (not `DiffusersCudaWorker`)
- Check that the model is actually SDXL (not SD1.5)

### LoRAs Not Loading

**Symptom**: LoRAs are skipped with cross_attention_dim mismatch

**Solutions**:
- Ensure LoRAs are SDXL-compatible (cross_attention_dim=2048)
- Check LoRA file path is correct
- Verify LoRA format is supported (.safetensors recommended)

## API Compatibility

The SDXL worker is **100% compatible** with the existing FastAPI endpoints:

- ✅ `POST /generate` - Works without changes
- ✅ `POST /superres` - Works without changes
- ✅ Style LoRAs - Works (filtered by cross_attention_dim)
- ✅ Storage integration - Works without changes

**No API changes required** - just swap the worker class.

## Migration from SD1.5 to SDXL

1. **Set environment variables**:
   ```bash
   export USE_SDXL=1
   export SDXL_MODEL_ROOT=/path/to/sdxl
   export SDXL_MODEL=sdxl-model.safetensors
   ```

2. **Update default resolution** (optional):
   ```python
   # In server config
   DEFAULT_SIZE = "1024x1024"  # Was "512x512"
   ```

3. **Update LoRAs** (if using styles):
   - Replace SD1.5 LoRAs with SDXL versions
   - Update `backends/styles.py` with new paths

4. **Increase VRAM allocation**:
   - SD1.5: ~4GB
   - SDXL: ~12GB (with xformers and fp16)

5. **Test thoroughly**:
   ```bash
   pytest tests/test_sdxl_worker.py -v
   ```

## Summary

The SDXL worker provides **drop-in SDXL support** with:
- ✅ Same interface as SD1.5 worker
- ✅ No new data structures
- ✅ Automatic LoRA filtering
- ✅ Comprehensive tests
- ✅ Production-ready error handling
- ✅ Memory-efficient optimizations

**Contract preserved** - the only changes are:
- New worker class: `DiffusersSDXLCudaWorker`
- New environment variables: `SDXL_MODEL_ROOT`, `SDXL_MODEL`
- Recommended resolution: 1024x1024 (instead of 512x512)
