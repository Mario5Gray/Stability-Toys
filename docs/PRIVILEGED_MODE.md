# Privileged Mode for Docker Tests

## Why --privileged is Required

The test containers run with `--privileged` flag to match the production runtime configuration in `runner.sh`.

### From runner.sh

```bash
docker run --rm -it \
  --network dream-lab_appnet \
  -p 4200:4200 \
  --privileged \      # <-- Required for GPU access
  ...
```

### Reasons for --privileged

1. **GPU Device Access** - Full access to `/dev` devices for CUDA/NPU
2. **RKNN Support** - RK3588 NPU requires privileged access
3. **CUDA Runtime** - Some CUDA operations need extended permissions
4. **Consistency** - Test environment matches production

## Usage

All test Docker commands now include `--privileged`:

### Test Script (Automated)

```bash
./test-sdxl.sh /path/to/models sdxl-model.safetensors
```

The script automatically uses `--privileged`.

### Manual Commands

```bash
# Build
docker build -f Dockerfile.test -t lcm-sd-test:latest .

# Run tests
docker run --rm --gpus all --privileged \
  -v /path/to/models:/models:ro \
  -e SDXL_MODEL_ROOT=/models \
  -e SDXL_MODEL=sdxl-model.safetensors \
  lcm-sd-test:latest

# Verify CUDA
docker run --rm --gpus all --privileged \
  lcm-sd-test:latest python verify_cuda.py

# Interactive debug
docker run --rm -it --gpus all --privileged \
  -v /path/to/models:/models:ro \
  lcm-sd-test:latest bash
```

## What Changed

All Docker commands in the following files now include `--privileged`:

### Scripts
- ✅ `test-sdxl.sh` - Main test runner
- ✅ `verify_cuda.py` - CUDA verification comments

### Documentation
- ✅ `docs/TESTING_IN_DOCKER.md`
- ✅ `docs/CUDA_VERIFICATION.md`
- ✅ `TESTING_README.md`
- ✅ `CUDA_VERIFICATION_SUMMARY.md`
- ✅ `Dockerfile.test` - Usage comments

## Security Note

`--privileged` gives the container nearly all capabilities of the host. This is acceptable for:

✅ **Development/Testing** - Controlled environment, trusted code
✅ **Local Execution** - Single-user machines
✅ **GPU Access** - Required for hardware access

❌ **Not recommended for:**
- Production workloads (unless required)
- Multi-tenant environments
- Untrusted code execution

## Alternative: Specific Capabilities

If you want to avoid `--privileged`, you can use specific capabilities:

```bash
docker run --rm \
  --gpus all \
  --cap-add=SYS_ADMIN \
  --device=/dev/dri \
  --device=/dev/npu \
  -v /path/to/models:/models:ro \
  lcm-sd-test:latest
```

However, `--privileged` is simpler and matches your production setup.

## Compatibility

### RKNN Backend (RK3588 NPU)
```bash
# Requires --privileged for NPU device access
docker run --rm --privileged \
  -e BACKEND=rknn \
  lcm-sd-test:latest
```

### CUDA Backend (NVIDIA GPU)
```bash
# Requires --privileged for full CUDA access
docker run --rm --gpus all --privileged \
  -e BACKEND=cuda \
  lcm-sd-test:latest
```

## Summary

**All test commands now use `--privileged`** to match the production configuration in `runner.sh`.

**Main test command:**
```bash
./test-sdxl.sh /path/to/models sdxl-model.safetensors
```

This ensures test environment perfectly matches production environment.
