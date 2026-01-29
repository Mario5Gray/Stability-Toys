# CUDA Verification in Docker

This guide explains how to verify CUDA access in test containers.

## Quick Verification

### 1. Using verify_cuda.py Script (Recommended)

```bash
# Build test image
docker build -f Dockerfile.test -t lcm-sd-test:latest .

# Run verification
docker run --rm --gpus all --privileged lcm-sd-test:latest python verify_cuda.py
```

**Expected Output:**
```
============================================================
CUDA Verification
============================================================

✓ PyTorch installed: 2.x.x
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

### 2. Quick PyTorch Check

```bash
docker run --rm --gpus all --privileged lcm-sd-test:latest \
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. CUDA Version Check

```bash
docker run --rm --gpus all --privileged lcm-sd-test:latest \
  python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

## Built-in Test Verification

The test suite automatically verifies CUDA:

1. **Module-level skip marker**: Skips all tests if CUDA unavailable
2. **test_cuda_available()**: First test verifies CUDA is working
3. **test_worker_initialization()**: Verifies worker is on CUDA device

## Troubleshooting

### Problem: "CUDA not available"

**Symptoms:**
```
✗ CUDA available: False
```

**Solutions:**

1. **Check GPU on host:**
   ```bash
   nvidia-smi
   ```

2. **Check nvidia-docker runtime:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Restart Docker:**
   ```bash
   sudo systemctl restart docker
   ```

4. **Install nvidia-container-toolkit:**
   ```bash
   # Ubuntu/Debian
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Problem: "No CUDA devices found"

**Symptoms:**
```
✓ CUDA available: True
✓ CUDA device count: 0
```

**Solutions:**

1. **Check --gpus flag:**
   ```bash
   # Wrong (missing --gpus)
   docker run --rm lcm-sd-test:latest python verify_cuda.py

   # Correct
   docker run --rm --gpus all lcm-sd-test:latest python verify_cuda.py
   ```

2. **Check Docker daemon config:**
   ```bash
   cat /etc/docker/daemon.json
   ```
   Should contain:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

### Problem: "CUDA operation failed"

**Symptoms:**
```
✓ CUDA available: True
✓ CUDA device count: 1
✗ CUDA operation failed: CUDA error: out of memory
```

**Solutions:**

1. **Check GPU memory:**
   ```bash
   nvidia-smi
   ```

2. **Close other GPU processes:**
   ```bash
   # List GPU processes
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   ```

3. **Use smaller test model or reduce batch size**

## Test Script Verification

The `test-sdxl.sh` script automatically verifies CUDA before running tests:

```bash
./test-sdxl.sh
```

**Verification steps:**
1. ✓ Checks nvidia-smi on host
2. ✓ Checks GPU is detected
3. ✓ Builds test image
4. ✓ Verifies GPU access in container (using PyTorch)
5. ✓ Runs tests

If verification fails, the script exits with helpful error messages.

## Manual Verification Steps

### Step 1: Check Host GPU

```bash
nvidia-smi
```

Expected: GPU listed with driver version

### Step 2: Check Docker GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Expected: Same GPU info as step 1

### Step 3: Check Test Container GPU Access

```bash
docker build -f Dockerfile.test -t lcm-sd-test:latest .
docker run --rm --gpus all lcm-sd-test:latest python verify_cuda.py
```

Expected: All checks pass

### Step 4: Run Tests

```bash
./test-sdxl.sh
```

Expected: All 9 tests pass (including test_cuda_available)

## Environment Variables

Set these for CUDA configuration:

```bash
# CUDA device selection
export CUDA_DEVICE=cuda:0  # Use first GPU
export CUDA_DEVICE=cuda:1  # Use second GPU

# CUDA optimizations
export CUDA_ENABLE_XFORMERS=1  # Enable xformers
export CUDA_DTYPE=fp16         # Use half precision
```

## Docker Runtime Options

### Using Specific GPU

```bash
# Use GPU 0 only
docker run --rm --gpus '"device=0"' lcm-sd-test:latest python verify_cuda.py

# Use GPU 1 only
docker run --rm --gpus '"device=1"' lcm-sd-test:latest python verify_cuda.py

# Use multiple GPUs
docker run --rm --gpus '"device=0,1"' lcm-sd-test:latest python verify_cuda.py
```

### Memory Limits

```bash
# Limit GPU memory
docker run --rm --gpus all --memory=16g lcm-sd-test:latest python verify_cuda.py
```

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Verify CUDA in Container
  run: |
    docker build -f Dockerfile.test -t lcm-sd-test:latest .
    docker run --rm --gpus all lcm-sd-test:latest python verify_cuda.py
```

### GitLab CI

```yaml
verify-cuda:
  tags:
    - gpu
  script:
    - docker build -f Dockerfile.test -t lcm-sd-test:latest .
    - docker run --rm --gpus all lcm-sd-test:latest python verify_cuda.py
```

## CUDA Environment in Container

The test container has these CUDA environment variables set:

```dockerfile
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

These ensure CUDA tools and libraries are accessible.

## Summary

**Quick Check:**
```bash
docker run --rm --gpus all lcm-sd-test:latest python verify_cuda.py
```

**Full Test:**
```bash
./test-sdxl.sh
```

Both should show CUDA working correctly before running SDXL tests.
