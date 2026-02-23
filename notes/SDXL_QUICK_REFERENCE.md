# SDXL Worker Quick Reference Card

## 🚀 Quick Start (3 Steps)

```bash
# 1. Set environment
export SDXL_MODEL_ROOT=/path/to/models
export SDXL_MODEL=sdxl-model.safetensors

# 2. Run tests in Docker (never on host!)
./test-sdxl.sh

# 3. Start server
export USE_SDXL=1
./runner.sh
```

## 📦 What Was Added

| File | Purpose |
|------|---------|
| `backends/cuda_worker.py` | +`DiffusersSDXLCudaWorker` class |
| `tests/test_sdxl_worker.py` | Test suite (8 tests) |
| `Dockerfile.test` | **Docker test container** |
| `test-sdxl.sh` | **Docker test runner (use this!)** |
| `run_sdxl_tests.sh` | Legacy test runner (host) |
| `docs/SDXL_WORKER.md` | Full documentation |
| `docs/TESTING_IN_DOCKER.md` | **Docker testing guide** |
| `docs/SDXL_INTEGRATION_EXAMPLE.md` | Integration guide |
| `SDXL_SUMMARY.md` | Implementation summary |

## 🔧 Environment Variables

```bash
# Required
SDXL_MODEL_ROOT=/models/sdxl    # Model directory
SDXL_MODEL=model.safetensors    # Model file

# Recommended
USE_SDXL=1                      # Enable SDXL mode
CUDA_ENABLE_XFORMERS=1          # Memory optimization
CUDA_DTYPE=fp16                 # Half precision
DEFAULT_SIZE=1024x1024          # SDXL native resolution
```

## 💻 Integration Pattern

```python
# In server/lcm_sr_server.py, PipelineService.__init__():

if use_cuda:
    use_sdxl = os.environ.get("USE_SDXL", "0") in ("1", "true", "yes")

    if use_sdxl:
        from backends.cuda_worker import DiffusersSDXLCudaWorker
        w = DiffusersSDXLCudaWorker(worker_id=i)
    else:
        from backends.cuda_worker import DiffusersCudaWorker
        w = DiffusersCudaWorker(worker_id=i)
```

## 🧪 Test Commands (Docker Only!)

**⚠️ Important: Always run tests in Docker, never on host**

```bash
# Recommended: Use test script
./test-sdxl.sh

# With explicit paths
./test-sdxl.sh /path/to/models sdxl-model.safetensors

# Manual Docker command
docker build -f Dockerfile.test -t lcm-sd-test .
docker run --rm --gpus all \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  lcm-sd-test:latest

# Run specific test in Docker
docker run --rm --gpus all \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  lcm-sd-test:latest \
  pytest tests/test_sdxl_worker.py::test_basic_generation -v -s -p no:cov
```

## 🎯 Test Coverage

- ✅ Worker initialization (model loading)
- ✅ Basic generation (critical path)
- ✅ Deterministic generation
- ✅ Generation with latents
- ✅ Multiple resolutions
- ✅ Error handling
- ✅ Random seed generation

## 🌐 API Usage

```bash
# Same endpoint, no changes
curl -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a majestic mountain landscape",
    "size": "1024x1024",
    "num_inference_steps": 4,
    "seed": 42
  }' \
  --output image.png
```

## 📊 SD1.5 vs SDXL

| Feature | SD1.5 | SDXL |
|---------|-------|------|
| Resolution | 512x512 | 1024x1024 |
| VRAM | ~4GB | ~12GB |
| Speed (4 steps) | ~0.3s | ~1.5s |
| Text Encoders | 1 | 2 |
| Cross-Attn Dim | 768 | 2048 |

## 🔍 Troubleshooting

### OOM Error
```bash
export CUDA_ENABLE_XFORMERS=1
export CUDA_DTYPE=fp16
export CUDA_ATTENTION_SLICING=1
```

### Slow Generation
```bash
# Check GPU usage
nvidia-smi

# Enable xformers
export CUDA_ENABLE_XFORMERS=1

# Use LCM model/scheduler
# Reduce steps to 4-6
```

### Model Not Found
```bash
# Check paths
echo $SDXL_MODEL_ROOT/$SDXL_MODEL
ls -lh $SDXL_MODEL_ROOT/$SDXL_MODEL

# Verify environment
env | grep SDXL
```

### Wrong Pipeline
```bash
# Ensure SDXL mode enabled
export USE_SDXL=1

# Check logs
docker logs <container> 2>&1 | grep sdxl
```

## 📝 Contract Compliance

**No changes required to:**
- ✅ `Job` dataclass
- ✅ `GenerateRequest` schema
- ✅ `PipelineService` architecture
- ✅ FastAPI endpoints
- ✅ Client code
- ✅ API contracts

**Same interface:**
```python
def run_job(self, job) -> tuple[bytes, int]:
    """Return (png_bytes, seed_used)"""

def run_job_with_latents(self, job) -> tuple[bytes, int, bytes]:
    """Return (png_bytes, seed_used, latents_bytes)"""
```

## 🎨 LoRA Support

```python
# backends/styles.py
STYLE_REGISTRY = {
    "style_name": StyleDef(
        adapter_name="style_name",
        lora_path="/path/to/lora.safetensors",
        levels=[0.3, 0.5, 0.7, 1.0],
        required_cross_attention_dim=2048,  # SDXL
    ),
}
```

LoRAs filtered automatically:
- SD1.5 LoRAs (768) → Skipped
- SDXL LoRAs (2048) → Loaded

## 💾 Memory Requirements

| Resolution | VRAM (fp16 + xformers) |
|------------|------------------------|
| 512x512 | ~6GB |
| 768x768 | ~9GB |
| 1024x1024 | ~12GB |
| 1536x1536 | ~20GB |

## ⚡ Performance Tips

1. **Enable xformers**: `CUDA_ENABLE_XFORMERS=1` (most important)
2. **Use fp16**: `CUDA_DTYPE=fp16` (default)
3. **Enable VAE tiling**: Automatic
4. **Reduce steps**: 4-6 for LCM
5. **Optimize resolution**: 768x768 if speed critical

## 📚 Documentation Files

- `docs/SDXL_WORKER.md` - Complete worker docs
- `docs/SDXL_INTEGRATION_EXAMPLE.md` - Integration patterns
- `SDXL_SUMMARY.md` - Implementation summary
- `SDXL_QUICK_REFERENCE.md` - This file

## 🎬 Example Session

```bash
# 1. Setup
export SDXL_MODEL_ROOT=/models/sdxl
export SDXL_MODEL=sdxl-1.0-base.safetensors

# 2. Test in Docker
./test-sdxl.sh
# ✅ All 8 tests pass

# 3. Run server
export USE_SDXL=1
export CUDA_ENABLE_XFORMERS=1
./localbuild
./runner.sh

# 4. Generate image
curl -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "size": "1024x1024"}' \
  -o test.png

# 5. Check result
file test.png
# test.png: PNG image data, 1024 x 1024
```

## ✅ Success Checklist

- [ ] Environment variables set
- [ ] Model file exists
- [ ] Docker installed with GPU support
- [ ] Tests pass in Docker (`./test-sdxl.sh`)
- [ ] Server starts without errors
- [ ] Generation endpoint works
- [ ] GPU memory under control
- [ ] Generation speed acceptable

## 📞 Help

1. **Read docs**: `docs/SDXL_WORKER.md`
2. **Docker testing**: `docs/TESTING_IN_DOCKER.md`
3. **Check examples**: `docs/SDXL_INTEGRATION_EXAMPLE.md`
4. **Run tests**: `./test-sdxl.sh` (Docker only!)
5. **Check logs**: `docker logs <container> | tail -100`
6. **Monitor GPU**: `nvidia-smi -l 1`

---

**Ready to use!** Just set environment variables and run.
