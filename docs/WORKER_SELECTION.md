# Worker Selection Guide

The server **automatically detects** the model type and selects the correct worker (SD1.5 or SDXL).

## How It Works

When the server starts with CUDA backend, it:

1. Calls `backends.worker_factory.create_cuda_worker()`
2. Factory reads `MODEL_ROOT` and `MODEL` environment variables
3. Factory inspects the model file using the detection system
4. Factory automatically creates and returns:
   - `DiffusersCudaWorker` if cross_attention_dim = 768 (SD1.5)
   - `DiffusersSDXLCudaWorker` if cross_attention_dim = 2048 (SDXL)

**No manual configuration needed!**

The detection logic lives in `backends/worker_factory.py`, keeping all model-related logic in the backends package.

## Environment Variables

Same variables for both SD1.5 and SDXL:

```bash
export BACKEND=cuda
export MODEL_ROOT=/path/to/models
export MODEL=model.safetensors
```

The server will automatically detect which worker to use.

## Example Configurations

### Docker with Any Model (SD1.5 or SDXL)

```bash
docker run --rm --gpus all --privileged \
  -v /models:/models:ro \
  -e BACKEND=cuda \
  -e MODEL_ROOT=/models \
  -e MODEL=your-model.safetensors \
  -p 4200:4200 \
  lcm-sd:latest
```

The server logs will show which worker was selected:
```
[ModelDetection] Detecting model type for: /models/your-model.safetensors
[ModelDetection] Detected variant: sdxl-base
[ModelDetection] Cross-attention dim: 2048
[ModelDetection] Using SDXL worker
[PipelineService] Initialized DiffusersSDXLCudaWorker (worker 0)
```

### Environment File (.env)

```bash
BACKEND=cuda
MODEL_ROOT=/models
MODEL=my-model.safetensors    # Auto-detected as SD1.5 or SDXL
CUDA_DEVICE=cuda:0
CUDA_DTYPE=fp16
DEFAULT_SIZE=1024x1024
DEFAULT_STEPS=4
DEFAULT_GUIDANCE=7.5
```

## Startup Logs

Check the logs to see detection results:

**SD1.5 Model:**
```
[ModelDetection] Detecting model type for: /models/dreamshaper-sd15.safetensors
[ModelDetection] Detected variant: sd15
[ModelDetection] Cross-attention dim: 768
[ModelDetection] Using SD1.5 worker
[PipelineService] Initialized DiffusersCudaWorker (worker 0)
```

**SDXL Model:**
```
[ModelDetection] Detecting model type for: /models/sdxl-base.safetensors
[ModelDetection] Detected variant: sdxl-base
[ModelDetection] Cross-attention dim: 2048
[ModelDetection] Using SDXL worker
[PipelineService] Initialized DiffusersSDXLCudaWorker (worker 0)
```

## Troubleshooting

### Error: "RuntimeError: Model not found at: /models/..."

**Problem**: MODEL_ROOT or MODEL is incorrect.

**Solution**: Verify the paths are correct:

```bash
export MODEL_ROOT=/correct/path
export MODEL=correct-filename.safetensors
```

### Error: "RuntimeError: Model detection failed: ..."

**Problem**: The model file is corrupted or unsupported format.

**Solution**:
1. Check the file exists: `ls -la $MODEL_ROOT/$MODEL`
2. Test detection manually: `python model_detector.py $MODEL_ROOT/$MODEL`
3. Verify it's a supported format (.safetensors, .ckpt, or diffusers directory)

### Manual Model Detection

You can test detection before starting the server:

```bash
python -m utils.model_detector /models/your-model.safetensors
```

Output:
```
Variant: sdxl-base
Cross-Attention Dim: 2048
Format: safetensors
Is LoRA: False
Confidence: 0.95
Compatible Worker: backends.cuda_worker.DiffusersSDXLCudaWorker
```

See [MODEL_DETECTOR_EXTENSIBLE.md](MODEL_DETECTOR_EXTENSIBLE.md) for more details on the detection system.
