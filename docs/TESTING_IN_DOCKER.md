# Testing SDXL Worker in Docker

**Important**: All tests should be run in Docker containers, not on the host machine.

## Quick Start

```bash
# Set your model path
export SDXL_MODEL_ROOT=/path/to/models
export SDXL_MODEL=sdxl-model.safetensors

# Run tests in Docker
./test-sdxl.sh
```

## Files Overview

| File | Purpose |
|------|---------|
| `Dockerfile.test` | Test container definition |
| `test-sdxl.sh` | Build and run test container |
| `tests/test_sdxl_worker.py` | SDXL test suite |

## Running Tests

### Method 1: Using test-sdxl.sh (Recommended)

```bash
# Option A: Pass paths as arguments
./test-sdxl.sh /path/to/models sdxl-model.safetensors

# Option B: Use environment variables
export SDXL_MODEL_ROOT=/path/to/models
export SDXL_MODEL=sdxl-model.safetensors
./test-sdxl.sh
```

### Method 2: Manual Docker Commands

```bash
# 1. Build test image
docker build -f Dockerfile.test -t lcm-sd-test:latest .

# 2. Run tests
docker run --rm --gpus all --privileged \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  -e CUDA_ENABLE_XFORMERS=1 \
  -e CUDA_DTYPE=fp16 \
  lcm-sd-test:latest
```

### Method 3: Run Specific Tests

```bash
# Build image first
docker build -f Dockerfile.test -t lcm-sd-test:latest .

# Run specific test
docker run --rm --gpus all --privileged \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  lcm-sd-test:latest \
  pytest tests/test_sdxl_worker.py::test_basic_generation -v -s -p no:cov

# Run with more verbose output
docker run --rm --gpus all --privileged \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  -e LOG_LEVEL=DEBUG \
  lcm-sd-test:latest
```

## Test Container Details

### Base Image
- `nvidia/cuda:12.8.0-runtime-ubuntu22.04`
- Includes CUDA runtime for GPU support
- Python 3.12

### Installed Dependencies
- Core: `requirements.txt` (torch, diffusers, transformers, etc.)
- Testing: `pytest`, `pytest-timeout`, `pytest-asyncio`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SDXL_MODEL_ROOT` | Required | Path to model directory (inside container) |
| `SDXL_MODEL` | Required | Model filename |
| `CUDA_ENABLE_XFORMERS` | `1` | Enable memory-efficient attention |
| `CUDA_DTYPE` | `fp16` | Model precision |
| `CUDA_DEVICE` | `cuda:0` | CUDA device |
| `LOG_LEVEL` | `INFO` | Logging level |

### Volume Mounts
- Model directory: `-v /host/path:/models:ro` (read-only)

## Expected Test Output

```
========================================
SDXL Worker Test (Docker)
========================================

✓ GPU detected:
NVIDIA GeForce RTX 3090, 24576 MiB

✓ Model configuration:
  SDXL_MODEL_ROOT: /path/to/models
  SDXL_MODEL:      sdxl-1.0-base.safetensors
  Full path:       /path/to/models/sdxl-1.0-base.safetensors

========================================
Building test image...
========================================

[+] Building 45.2s (18/18) FINISHED
...

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

## Troubleshooting

### GPU Not Detected

```bash
# Check if nvidia-docker-runtime is installed
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit:
# Ubuntu/Debian:
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build Fails

```bash
# Check Docker is running
docker info

# Clear Docker cache and rebuild
docker builder prune -a
docker build -f Dockerfile.test -t lcm-sd-test:latest .
```

### Tests Fail with OOM

```bash
# Check GPU memory
nvidia-smi

# Run with smaller resolution tests only
docker run --rm --gpus all \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  -e CUDA_ENABLE_XFORMERS=1 \
  -e CUDA_ATTENTION_SLICING=1 \
  lcm-sd-test:latest \
  pytest tests/test_sdxl_worker.py::test_basic_generation -v -s -p no:cov
```

### Model Not Found

```bash
# Check volume mount
docker run --rm \
  -v /path/to/models:/models:ro \
  ubuntu:22.04 \
  ls -la /models

# Ensure full path is correct
ls -la /path/to/models/sdxl-model.safetensors
```

### Permission Issues

```bash
# Run with user permissions
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  lcm-sd-test:latest
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: SDXL Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Build test image
      run: docker build -f Dockerfile.test -t lcm-sd-test:latest .

    - name: Run tests
      run: |
        docker run --rm --gpus all \
          -v /models:/models:ro \
          -e SDXL_MODEL_ROOT=/models \
          -e SDXL_MODEL=${{ secrets.SDXL_MODEL }} \
          -e CUDA_ENABLE_XFORMERS=1 \
          lcm-sd-test:latest
```

### GitLab CI Example

```yaml
sdxl-tests:
  image: docker:latest
  services:
    - docker:dind
  tags:
    - gpu
  script:
    - docker build -f Dockerfile.test -t lcm-sd-test:latest .
    - docker run --rm --gpus all
        -v /models:/models:ro
        -e SDXL_MODEL_ROOT=/models
        -e SDXL_MODEL=${SDXL_MODEL}
        lcm-sd-test:latest
```

## Development Workflow

### 1. Make Changes to Worker

```bash
# Edit backends/cuda_worker.py
vim backends/cuda_worker.py
```

### 2. Run Tests in Container

```bash
# Quick test
./test-sdxl.sh

# Or rebuild and test specific
docker build -f Dockerfile.test -t lcm-sd-test:latest .
docker run --rm --gpus all \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  lcm-sd-test:latest \
  pytest tests/test_sdxl_worker.py::test_basic_generation -v -s -p no:cov
```

### 3. Iterate

Repeat steps 1-2 until tests pass.

### 4. Run Full Test Suite

```bash
./test-sdxl.sh
```

## Test Container Optimizations

### Layer Caching

The Dockerfile is optimized for layer caching:
1. System dependencies (rarely change)
2. Python requirements (change occasionally)
3. Application code (change frequently)

### Faster Rebuilds

```bash
# Only rebuild if dependencies changed
docker build -f Dockerfile.test --target dependencies -t lcm-sd-test:deps .

# Full rebuild
docker build -f Dockerfile.test -t lcm-sd-test:latest .
```

### Multi-Stage Build (Optional)

For production, you can create a multi-stage build that runs tests and then creates a smaller runtime image.

## Cleaning Up

```bash
# Remove test image
docker rmi lcm-sd-test:latest

# Remove dangling images
docker image prune

# Full cleanup (careful!)
docker system prune -a
```

## Best Practices

1. ✅ **Always run tests in containers** - Never on host
2. ✅ **Mount models read-only** - Prevent accidental modifications
3. ✅ **Use specific model versions** - Ensure reproducibility
4. ✅ **Enable xformers** - Reduce memory usage
5. ✅ **Check GPU memory** - Monitor with `nvidia-smi`
6. ✅ **Tag test images** - Use versions for tracking
7. ✅ **Clean up regularly** - Remove old images

## Summary

The Docker-based testing approach provides:
- ✅ **Isolation** - No host contamination
- ✅ **Reproducibility** - Consistent environment
- ✅ **Portability** - Run anywhere with Docker
- ✅ **CI/CD Ready** - Easy integration
- ✅ **GPU Support** - Full CUDA access
- ✅ **Clean State** - Fresh environment each run

**Main Command**: `./test-sdxl.sh /path/to/models sdxl-model.safetensors`

That's it! All tests run safely in containers.
