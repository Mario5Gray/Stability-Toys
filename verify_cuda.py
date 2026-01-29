#!/usr/bin/env python3
"""
Simple CUDA verification script for Docker containers.

Usage:
  python verify_cuda.py

Or in Docker:
  docker run --rm --gpus all --privileged lcm-sd-test:latest python verify_cuda.py
"""

import sys

def verify_cuda():
    """Verify CUDA is available and working."""
    print("=" * 60)
    print("CUDA Verification")
    print("=" * 60)
    print()

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA available: True")
    else:
        print(f"✗ CUDA available: False")
        print()
        print("Troubleshooting:")
        print("  1. Ensure you're running with --gpus all")
        print("  2. Check nvidia-docker runtime is installed")
        print("  3. Test: docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi")
        return False

    # Check CUDA devices
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA device count: {device_count}")

    if device_count == 0:
        print("✗ No CUDA devices found")
        return False

    # Print device details
    print()
    print("CUDA Device Details:")
    print("-" * 60)
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_props = torch.cuda.get_device_properties(i)

        print(f"Device {i}: {device_name}")
        print(f"  Compute Capability: {device_props.major}.{device_props.minor}")
        print(f"  Total Memory: {device_props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi Processors: {device_props.multi_processor_count}")

    # Check CUDA version
    print()
    print(f"✓ PyTorch CUDA version: {torch.version.cuda}")

    # Try a simple CUDA operation
    print()
    print("Testing CUDA operation...")
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x @ y
        print(f"✓ CUDA operation successful: {z.shape}")

        # Check memory
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"✓ CUDA memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")

    except Exception as e:
        print(f"✗ CUDA operation failed: {e}")
        return False

    print()
    print("=" * 60)
    print("✓ All CUDA checks passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = verify_cuda()
    sys.exit(0 if success else 1)
