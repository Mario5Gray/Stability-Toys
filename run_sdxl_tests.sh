#!/bin/bash
# Quick test runner for SDXL worker
# Usage: ./run_sdxl_tests.sh

set -e

echo "========================================="
echo "SDXL Worker Test Runner"
echo "========================================="
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. CUDA GPU required for SDXL tests."
    exit 1
fi

# Check CUDA availability
if ! nvidia-smi &> /dev/null; then
    echo "❌ Error: CUDA GPU not detected."
    exit 1
fi

echo "✓ CUDA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo ""

# Check environment variables
if [ -z "$MODEL_ROOT" ]; then
    echo "❌ Error: MODEL_ROOT not set."
    echo ""
    echo "Set environment variables:"
    echo "  export MODEL_ROOT=/path/to/models"
    echo "  export MODEL=sdxl-model.safetensors"
    exit 1
fi

if [ -z "$MODEL" ]; then
    echo "❌ Error: MODEL not set."
    echo ""
    echo "Set environment variables:"
    echo "  export MODEL=sdxl-model.safetensors"
    exit 1
fi

MODEL_PATH="$MODEL_ROOT/$MODEL"

echo "✓ Environment variables:"
echo "  MODEL_ROOT: $MODEL_ROOT"
echo "  MODEL:      $MODEL"
echo "  Full path:  $MODEL_PATH"
echo ""

# Check if model exists
if [ ! -e "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "✓ Model found at $MODEL_PATH"
echo ""

# Set recommended settings for testing
export CUDA_ENABLE_XFORMERS=1
export CUDA_DTYPE=fp16

echo "✓ Test settings:"
echo "  CUDA_ENABLE_XFORMERS: $CUDA_ENABLE_XFORMERS"
echo "  CUDA_DTYPE:           $CUDA_DTYPE"
echo ""

# Run tests
echo "========================================="
echo "Running SDXL Worker Tests"
echo "========================================="
echo ""

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ Error: pytest not found. Install with:"
    echo "  pip install pytest"
    exit 1
fi

# Run with verbose output, overriding pytest.ini coverage options
pytest tests/test_sdxl_worker.py -v -s --tb=short \
    --override-ini="addopts=" \
    --no-cov 2>/dev/null || \
pytest tests/test_sdxl_worker.py -v -s --tb=short \
    --override-ini="addopts="

echo ""
echo "========================================="
echo "✓ All tests completed!"
echo "========================================="
