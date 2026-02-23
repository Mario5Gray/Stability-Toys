# Testing Guidelines

## ⚠️ Important: Docker-Only Testing Policy

**All tests MUST be run in Docker containers. DO NOT run tests on the host machine.**

### Why Docker-Only?

1. **Isolation** - Prevents host environment contamination
2. **Reproducibility** - Consistent test environment
3. **Clean State** - Fresh environment for each test run
4. **Portability** - Same results across different machines
5. **Safety** - GPU operations isolated from host

## Quick Start

### SDXL Worker Tests

```bash
# Set your model location
export SDXL_MODEL_ROOT=/path/to/models
export SDXL_MODEL=sdxl-model.safetensors

# Run tests in Docker (this is the ONLY way)
./test-sdxl.sh
```

That's it! The script handles:
- ✅ Building the test container
- ✅ Mounting your model directory (read-only)
- ✅ Setting up GPU access
- ✅ Running all 8 SDXL tests
- ✅ Reporting results

### Expected Output

```
========================================
SDXL Worker Test (Docker)
========================================

✓ GPU detected:
NVIDIA GeForce RTX 3090, 24576 MiB

✓ Model configuration:
  SDXL_MODEL_ROOT: /models/sdxl
  SDXL_MODEL:      sdxl-1.0-base.safetensors
  Full path:       /models/sdxl/sdxl-1.0-base.safetensors

========================================
Building test image...
========================================
[Docker build output...]

✓ Test image built successfully

========================================
Running SDXL tests in container...
========================================

tests/test_sdxl_worker.py::test_worker_initialization PASSED
tests/test_sdxl_worker.py::test_basic_generation PASSED
tests/test_sdxl_worker.py::test_deterministic_generation PASSED
tests/test_sdxl_worker.py::test_generation_with_latents PASSED
tests/test_sdxl_worker.py::test_different_resolutions PASSED
tests/test_sdxl_worker.py::test_invalid_size_format PASSED
tests/test_sdxl_worker.py::test_random_seed_generation PASSED

======================== 8 passed in 45.23s ========================

========================================
✓ All tests passed!
========================================
```

## Test Files

| File | Description | When to Use |
|------|-------------|-------------|
| `Dockerfile.test` | Test container definition | Automatic (via test-sdxl.sh) |
| `test-sdxl.sh` | **Main test runner** | **Always use this** |
| `tests/test_sdxl_worker.py` | SDXL test suite | Executed inside container |
| `run_sdxl_tests.sh` | Legacy runner (host) | **DO NOT USE** |

## Testing Workflow

### 1. Development Cycle

```bash
# Make changes to code
vim backends/cuda_worker.py

# Run tests in Docker
./test-sdxl.sh

# If tests fail, check output and fix
# Repeat until tests pass
```

### 2. Run Specific Test

```bash
# Build test image first
docker build -f Dockerfile.test -t lcm-sd-test:latest .

# Run specific test
docker run --rm --gpus all --privileged \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  -e CUDA_ENABLE_XFORMERS=1 \
  lcm-sd-test:latest \
  pytest tests/test_sdxl_worker.py::test_basic_generation -v -s -p no:cov
```

### 3. Debug Mode

```bash
# Run with verbose logging
docker run --rm --gpus all --privileged \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  -e LOG_LEVEL=DEBUG \
  lcm-sd-test:latest
```

### 4. Interactive Debug

```bash
# Start container with bash
docker run --rm -it --gpus all --privileged \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  lcm-sd-test:latest \
  bash

# Inside container, run tests manually
pytest tests/test_sdxl_worker.py -v -s -p no:cov
```

## Prerequisites

### Required
- Docker installed
- NVIDIA GPU with CUDA support
- nvidia-docker runtime
- SDXL model checkpoint

### Check Prerequisites

```bash
# Check Docker
docker --version

# Check GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Check model file
ls -lh $SDXL_MODEL_ROOT/$SDXL_MODEL
```

### Install nvidia-docker (if needed)

```bash
# Ubuntu/Debian
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Troubleshooting

### Test Script Fails to Find Model

```bash
# Check environment variables
echo $SDXL_MODEL_ROOT
echo $SDXL_MODEL

# Check file exists
ls -la $SDXL_MODEL_ROOT/$SDXL_MODEL

# Pass paths explicitly
./test-sdxl.sh /full/path/to/models sdxl-model.safetensors
```

### Docker Build Fails

```bash
# Clear cache and rebuild
docker builder prune -a
docker build -f Dockerfile.test -t lcm-sd-test:latest .
```

### GPU Not Accessible

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# If fails, restart Docker
sudo systemctl restart docker

# Check nvidia-container-runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Tests Fail with OOM

```bash
# Enable memory optimizations
docker run --rm --gpus all \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  -e CUDA_ENABLE_XFORMERS=1 \
  -e CUDA_ATTENTION_SLICING=1 \
  -e CUDA_DTYPE=fp16 \
  lcm-sd-test:latest

# Or use smaller model
export SDXL_MODEL=sdxl-turbo.safetensors
./test-sdxl.sh
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Test SDXL Worker

on: [push, pull_request]

jobs:
  test:
    runs-on: [self-hosted, gpu]

    steps:
    - uses: actions/checkout@v3

    - name: Run SDXL tests
      env:
        SDXL_MODEL_ROOT: /models/sdxl
        SDXL_MODEL: sdxl-1.0-base.safetensors
      run: ./test-sdxl.sh
```

### GitLab CI

```yaml
sdxl-tests:
  image: docker:latest
  services:
    - docker:dind
  tags:
    - gpu
  script:
    - ./test-sdxl.sh /models/sdxl sdxl-1.0-base.safetensors
```

## What Gets Tested

The test suite covers all critical paths:

1. **Worker Initialization** (`test_worker_initialization`)
   - Model loads correctly
   - Pipeline initialized
   - GPU accessible

2. **Basic Generation** (`test_basic_generation`) ⭐ **Critical Path**
   - Generate image from prompt
   - Verify PNG output
   - Check seed consistency

3. **Determinism** (`test_deterministic_generation`)
   - Same seed produces same output
   - Reproducibility verified

4. **Latents Extraction** (`test_generation_with_latents`)
   - Generate image + latents
   - Verify latent format (512 bytes)

5. **Multiple Resolutions** (`test_different_resolutions`)
   - 512x512, 768x768, 1024x1024
   - All resolutions work

6. **Error Handling** (`test_invalid_size_format`)
   - Invalid inputs rejected
   - Error messages correct

7. **Random Seeds** (`test_random_seed_generation`)
   - Auto-generated seeds work
   - Different seeds produce different outputs

## Performance Expectations

| GPU | VRAM | Test Suite Time |
|-----|------|-----------------|
| RTX 3060 | 12GB | ~120s |
| RTX 3080 | 10GB | ~70s |
| RTX 3090 | 24GB | ~45s |
| RTX 4090 | 24GB | ~30s |
| A100 | 40GB | ~25s |

*Including Docker build time (~20s on subsequent runs)*

## Best Practices

1. ✅ **Always use Docker** - Never run tests on host
2. ✅ **Run before commit** - Ensure tests pass before committing
3. ✅ **Check GPU memory** - Monitor with `nvidia-smi`
4. ✅ **Use read-only mounts** - Models mounted with `:ro` flag
5. ✅ **Clean up images** - Remove old test images regularly
6. ✅ **Version test images** - Tag for tracking changes

## Cleaning Up

```bash
# Remove test image
docker rmi lcm-sd-test:latest

# Clean up old images
docker image prune

# Full cleanup (careful!)
docker system prune -a
```

## Summary

**Main Command**:
```bash
./test-sdxl.sh
```

**That's all you need!** The script handles everything:
- Builds Docker image
- Mounts models
- Configures GPU
- Runs tests
- Reports results

**Never run tests on host machine.** Always use Docker.

## Documentation

For more details, see:
- `docs/TESTING_IN_DOCKER.md` - Complete Docker testing guide
- `docs/SDXL_WORKER.md` - SDXL worker documentation
- `SDXL_QUICK_REFERENCE.md` - Quick reference card
