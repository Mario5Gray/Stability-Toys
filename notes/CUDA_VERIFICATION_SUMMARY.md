# CUDA Verification Summary

## What Was Added

To ensure the test container has proper CUDA access, the following components were added:

### 1. **Enhanced Dockerfile.test**

**Changes:**
- ✅ Uses `python:3.12-slim` base image (matches main Dockerfile)
- ✅ Installs CUDA libraries from nvidia.com repos (matches main Dockerfile)
- ✅ Sets CUDA environment variables:
  ```dockerfile
  ENV CUDA_HOME=/usr/local/cuda-12.8
  ENV PATH=${CUDA_HOME}/bin:${PATH}
  ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
  ```

**CUDA Packages Installed:**
- `cuda-cudart-12-8`
- `libcublas-12-8`
- `libcufft-12-8`
- `libcurand-12-8`
- `libcusolver-12-8`
- `libcusparse-12-8`
- `cuda-nvcc-12-8`
- `cuda-toolkit-12-8-config-common`

### 2. **verify_cuda.py Script**

Comprehensive CUDA verification script that checks:
- ✅ PyTorch installation
- ✅ CUDA availability
- ✅ Device count and details
- ✅ CUDA version
- ✅ GPU memory
- ✅ Simple CUDA operation

**Usage:**
```bash
docker run --rm --gpus all --privileged lcm-sd-test:latest python verify_cuda.py
```

### 3. **Enhanced test-sdxl.sh**

Added automatic GPU verification before running tests:
```bash
# Verifies GPU access in container using PyTorch
docker run --rm --gpus all lcm-sd-test:latest python3 -c "import torch; ..."
```

If GPU is not accessible, the script exits with troubleshooting tips.

### 4. **Enhanced Test Suite**

Added `test_cuda_available()` as the first test:
```python
def test_cuda_available():
    """Test that CUDA is available in the container."""
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() > 0
```

Also enhanced `test_worker_initialization()` to verify worker is on CUDA device.

### 5. **Documentation**

- 📄 `docs/CUDA_VERIFICATION.md` - Complete CUDA verification guide
- 📄 `CUDA_VERIFICATION_SUMMARY.md` - This file

## How It Works

### Test Execution Flow

```
1. Host GPU Check
   └─> nvidia-smi on host

2. Build Test Image
   └─> Dockerfile.test with CUDA libs

3. Container GPU Verification
   └─> PyTorch CUDA check in container

4. Run Test Suite
   ├─> test_cuda_available() - Verify CUDA
   ├─> test_worker_initialization() - Verify worker on GPU
   └─> ... other tests

5. Report Results
```

### Verification Layers

| Layer | Check | Tool |
|-------|-------|------|
| **Host** | GPU available | `nvidia-smi` |
| **Docker Runtime** | GPU passthrough | `--gpus all` flag |
| **Container** | CUDA libraries | CUDA packages installed |
| **PyTorch** | CUDA available | `torch.cuda.is_available()` |
| **Worker** | Device placement | `worker.device` |

## Quick Start

### Verify Everything

```bash
# Single command verifies entire stack
./test-sdxl.sh /path/to/models sdxl-model.safetensors
```

This checks:
1. ✓ Host GPU with nvidia-smi
2. ✓ Model file exists
3. ✓ Docker image builds
4. ✓ Container can access GPU
5. ✓ PyTorch can use CUDA
6. ✓ Worker initializes on GPU
7. ✓ All tests pass

### Verify CUDA Only

```bash
# Build image
docker build -f Dockerfile.test -t lcm-sd-test:latest .

# Run verification
docker run --rm --gpus all --privileged lcm-sd-test:latest python verify_cuda.py
```

## Expected Output

### verify_cuda.py Success

```
============================================================
CUDA Verification
============================================================

✓ PyTorch installed: 2.5.1
✓ CUDA available: True
✓ CUDA device count: 1

CUDA Device Details:
------------------------------------------------------------
Device 0: NVIDIA GeForce RTX 3090
  Compute Capability: 8.6
  Total Memory: 24.00 GB
  Multi Processors: 82

✓ PyTorch CUDA version: 12.8
✓ CUDA operation successful: torch.Size([3, 3])
✓ CUDA memory: 2.00 MB allocated, 2.00 MB reserved

============================================================
✓ All CUDA checks passed!
============================================================
```

### test-sdxl.sh Success

```
========================================
SDXL Worker Test (Docker)
========================================

✓ GPU detected:
NVIDIA GeForce RTX 3090, 24576 MiB

✓ Model configuration:
  SDXL_MODEL_ROOT: /models/sdxl
  SDXL_MODEL:      sdxl-1.0-base.safetensors

========================================
Building test image...
========================================
[Docker build output...]

✓ Test image built successfully

========================================
Verifying GPU access in container...
========================================

CUDA available: True
CUDA devices: 1
Device name: NVIDIA GeForce RTX 3090

✓ GPU access verified in container

========================================
Running SDXL tests in container...
========================================

tests/test_sdxl_worker.py::test_cuda_available PASSED
tests/test_sdxl_worker.py::test_worker_initialization PASSED
tests/test_sdxl_worker.py::test_basic_generation PASSED
[... 6 more tests ...]

======================== 9 passed in 45.23s ========================

========================================
✓ All tests passed!
========================================
```

## Troubleshooting

### Issue: CUDA Not Available

**Check 1: Host GPU**
```bash
nvidia-smi
```

**Check 2: Docker GPU Access**
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

**Check 3: Container GPU Access**
```bash
docker run --rm --gpus all --privileged lcm-sd-test:latest python verify_cuda.py
```

### Issue: Tests Skip with "CUDA not available"

This means:
1. CUDA is not accessible in the container
2. Missing `--gpus all` flag
3. nvidia-docker runtime not installed

**Fix:**
```bash
# Install nvidia-container-toolkit
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Files Added/Modified

### New Files
- `verify_cuda.py` - CUDA verification script
- `docs/CUDA_VERIFICATION.md` - CUDA verification guide
- `CUDA_VERIFICATION_SUMMARY.md` - This file

### Modified Files
- `Dockerfile.test` - Added CUDA environment variables and nvcc
- `test-sdxl.sh` - Added container GPU verification step
- `tests/test_sdxl_worker.py` - Added `test_cuda_available()` and device checks

## Benefits

1. **Early Detection** - Catches CUDA issues before model loading
2. **Clear Errors** - Provides specific troubleshooting steps
3. **Automated Verification** - No manual checks needed
4. **CI/CD Ready** - Easy to integrate into pipelines
5. **Comprehensive** - Checks entire stack from host to worker

## Summary

The test container now has **5 layers of CUDA verification**:

1. ✅ Host GPU check (`nvidia-smi`)
2. ✅ Docker runtime check (`--gpus all`)
3. ✅ Container PyTorch check (`torch.cuda.is_available()`)
4. ✅ Test suite CUDA check (`test_cuda_available()`)
5. ✅ Worker device check (`worker.device == cuda:0`)

**Main Command:**
```bash
./test-sdxl.sh /path/to/models sdxl-model.safetensors
```

This single command verifies CUDA at all 5 layers and runs the complete test suite.
