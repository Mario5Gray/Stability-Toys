#!/bin/bash
# test-sdxl.sh - Build and run SDXL tests in Docker container
#
# Usage:
#   ./test-sdxl.sh [model_root] [model_name]
#
# Examples:
#   ./test-sdxl.sh /models/sdxl sdxl-1.0-base.safetensors
#   ./test-sdxl.sh  # Uses environment variables

set -e

echo "========================================="
echo "SDXL Worker Test (Docker)"
echo "========================================="
echo ""

# Check if running in WSL or Linux
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. CUDA GPU required."
    exit 1
fi

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ Error: CUDA GPU not detected."
    exit 1
fi

echo "✓ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo ""

# Get model path from args or environment
MODEL_ROOT="${1:-${MODEL_ROOT}}"
MODEL_NAME="${2:-${MODEL}}"

if [ -z "$MODEL_ROOT" ]; then
    echo "❌ Error: Model root not specified."
    echo ""
    echo "Usage:"
    echo "  ./test-sdxl.sh /path/to/models sdxl-model.safetensors"
    echo ""
    echo "Or set environment variables:"
    echo "  export MODEL_ROOT=/path/to/models"
    echo "  export MODEL=sdxl-model.safetensors"
    echo "  ./test-sdxl.sh"
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    echo "❌ Error: Model name not specified."
    echo ""
    echo "Usage:"
    echo "  ./test-sdxl.sh /path/to/models sdxl-model.safetensors"
    exit 1
fi

# Check if model exists
MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"
if [ ! -e "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "✓ Model configuration:"
echo "  MODEL_ROOT: $MODEL_ROOT"
echo "  MODEL:      $MODEL_NAME"
echo "  Full path:  $MODEL_PATH"
echo ""

# Build test image
echo "========================================="
echo "Building test image..."
echo "========================================="
echo ""

docker build -f Dockerfile.test -t lcm-sd-test:latest .

if [ $? -ne 0 ]; then
    echo "❌ Error: Docker build failed"
    exit 1
fi

echo ""
echo "✓ Test image built successfully"
echo ""

# Verify GPU access in container
echo "========================================="
echo "Verifying GPU access in container..."
echo "========================================="
echo ""

docker run --rm --gpus all --privileged lcm-sd-test:latest python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if [ $? -ne 0 ]; then
    echo "❌ Error: GPU not accessible in container"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check nvidia-docker runtime is installed"
    echo "  2. Test with: docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi"
    echo "  3. Restart Docker: sudo systemctl restart docker"
    exit 1
fi

echo ""
echo "✓ GPU access verified in container"
echo ""

# Run tests in container
echo "========================================="
echo "Running SDXL tests in container..."
echo "========================================="
echo ""

# Get absolute path for model root
MODEL_ROOT_ABS=$(realpath "$MODEL_ROOT")

docker run --rm \
    --gpus all \
    --privileged \
    -v "${MODEL_ROOT_ABS}:/models:ro" \
    -e MODEL_ROOT=/models \
    -e MODEL="$MODEL_NAME" \
    -e CUDA_ENABLE_XFORMERS=1 \
    -e CUDA_DTYPE=fp16 \
    -e LOG_LEVEL=INFO \
    lcm-sd-test:latest

TEST_EXIT_CODE=$?

echo ""
echo "========================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "❌ Tests failed with exit code $TEST_EXIT_CODE"
fi
echo "========================================="

exit $TEST_EXIT_CODE
