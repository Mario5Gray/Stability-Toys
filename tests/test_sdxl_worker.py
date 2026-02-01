"""
Test suite for DiffusersSDXLCudaWorker.

Critical path tests:
1. Worker initialization (model loading)
2. Basic generation (run_job)
3. Generation with latents (run_job_with_latents)

Requirements:
- CUDA GPU with sufficient VRAM (12GB+ recommended for SDXL)
- SDXL model checkpoint
- Set environment variables before running:
  - MODEL_ROOT
  - MODEL (checkpoint name)

Usage:
  export MODEL_ROOT=/path/to/models
  export MODEL=sdxl-1.0.safetensors
  pytest tests/test_sdxl_worker.py -v
"""

import os
import pytest
import torch
from dataclasses import dataclass
from concurrent.futures import Future
from typing import Optional

# Skip all tests if CUDA is not functional
def _cuda_functional():
    """Check if CUDA is actually usable (not just detected)."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.tensor([0.0], device="cuda")
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _cuda_functional(),
    reason="CUDA not functional - SDXL worker requires working GPU"
)


@dataclass
class MockStyleLora:
    """Mock StyleLora for testing."""
    style: Optional[str] = None
    level: int = 0


@dataclass
class MockGenerateRequest:
    """Mock GenerateRequest for testing."""
    prompt: str
    size: str = "1024x1024"
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    seed: Optional[int] = None
    style_lora: MockStyleLora = None

    def __post_init__(self):
        if self.style_lora is None:
            self.style_lora = MockStyleLora()


@dataclass
class MockJob:
    """Mock Job for testing."""
    req: MockGenerateRequest
    fut: Future = None
    submitted_at: float = 0.0

    def __post_init__(self):
        if self.fut is None:
            self.fut = Future()


@pytest.fixture(scope="module")
def sdxl_worker():
    """
    Create an SDXL worker instance.

    This fixture is module-scoped to avoid reloading the model for every test.
    Model loading can take 10-30 seconds depending on hardware.
    """
    # Check required env vars
    model_root = os.environ.get("SDXL_MODEL_ROOT") or os.environ.get("MODEL_ROOT")
    model_name = os.environ.get("SDXL_MODEL") or os.environ.get("MODEL")

    if not model_root:
        pytest.skip("SDXL_MODEL_ROOT (or MODEL_ROOT) not set")
    if not model_name:
        pytest.skip("SDXL_MODEL (or MODEL) not set")

    ckpt_path = os.path.join(model_root, model_name)
    if not os.path.exists(ckpt_path):
        pytest.skip(f"SDXL model not found at {ckpt_path}")

    # Import here to avoid import errors when tests are collected
    from backends.cuda_worker import DiffusersSDXLCudaWorker

    print(f"\n[test] Loading SDXL model from {ckpt_path}")
    worker = DiffusersSDXLCudaWorker(worker_id=0)
    print(f"[test] SDXL worker initialized successfully")

    yield worker

    # Cleanup
    if hasattr(worker, 'pipe'):
        del worker.pipe
    torch.cuda.empty_cache()
    print(f"[test] SDXL worker cleaned up")


def test_cuda_available():
    """Test that CUDA is available in the container."""
    assert torch.cuda.is_available(), "CUDA must be available for SDXL worker"
    assert torch.cuda.device_count() > 0, "At least one CUDA device must be available"

    device_name = torch.cuda.get_device_name(0)
    print(f"[test] CUDA device: {device_name}")
    print(f"[test] CUDA devices count: {torch.cuda.device_count()}")
    print(f"[test] PyTorch CUDA version: {torch.version.cuda}")


def test_worker_initialization(sdxl_worker):
    """Test that the worker initializes correctly."""
    assert sdxl_worker is not None
    assert sdxl_worker.worker_id == 0
    assert hasattr(sdxl_worker, 'pipe')
    assert hasattr(sdxl_worker, 'device')
    assert hasattr(sdxl_worker, 'dtype')

    # Verify SDXL-specific attributes
    assert hasattr(sdxl_worker.pipe, 'text_encoder')
    assert hasattr(sdxl_worker.pipe, 'text_encoder_2')
    assert hasattr(sdxl_worker.pipe, 'unet')
    assert hasattr(sdxl_worker.pipe, 'vae')

    # Verify worker is on CUDA device
    assert 'cuda' in str(sdxl_worker.device).lower(), "Worker must be on CUDA device"

    print(f"[test] Worker device: {sdxl_worker.device}")
    print(f"[test] Worker dtype: {sdxl_worker.dtype}")


def test_basic_generation(sdxl_worker):
    """
    Test basic image generation.

    This is the critical path - if this works, the worker is functional.
    """
    req = MockGenerateRequest(
        prompt="a cat sitting on a mat",
        size="1024x1024",
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42
    )

    job = MockJob(req=req)

    print(f"[test] Running generation: prompt='{req.prompt}', size={req.size}, steps={req.num_inference_steps}")

    png_bytes, seed = sdxl_worker.run_job(job)

    # Verify output
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000  # PNG should be reasonably sized
    assert isinstance(seed, int)
    assert seed == 42

    # Verify PNG header
    assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n', "Output is not a valid PNG"

    print(f"[test] Generation successful: {len(png_bytes)} bytes, seed={seed}")


def test_deterministic_generation(sdxl_worker):
    """Test that the same seed produces the same output."""
    req1 = MockGenerateRequest(
        prompt="a dog in a park",
        size="512x512",  # Smaller for faster test
        num_inference_steps=4,
        seed=12345
    )

    req2 = MockGenerateRequest(
        prompt="a dog in a park",
        size="512x512",
        num_inference_steps=4,
        seed=12345
    )

    job1 = MockJob(req=req1)
    job2 = MockJob(req=req2)

    print(f"[test] Testing determinism with seed=12345")

    png_bytes1, seed1 = sdxl_worker.run_job(job1)
    png_bytes2, seed2 = sdxl_worker.run_job(job2)

    assert seed1 == seed2 == 12345
    assert png_bytes1 == png_bytes2, "Same seed should produce identical output"

    print(f"[test] Determinism verified: outputs are identical")


def test_generation_with_latents(sdxl_worker):
    """Test generation with latents extraction."""
    req = MockGenerateRequest(
        prompt="a beautiful landscape",
        size="1024x1024",
        num_inference_steps=4,
        seed=9999
    )

    job = MockJob(req=req)

    print(f"[test] Running generation with latents")

    png_bytes, seed, latents_bytes = sdxl_worker.run_job_with_latents(job)

    # Verify output
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000
    assert isinstance(seed, int)
    assert seed == 9999

    # Verify latents
    assert isinstance(latents_bytes, bytes)
    # NCHW float16 [1,4,8,8] = 1*4*8*8*2 bytes = 512 bytes
    assert len(latents_bytes) == 512, f"Expected 512 bytes for latents, got {len(latents_bytes)}"

    print(f"[test] Generation with latents successful")


def test_different_resolutions(sdxl_worker):
    """Test generation at different resolutions."""
    resolutions = [
        "512x512",    # Small
        "768x768",    # Medium
        "1024x1024",  # SDXL native
    ]

    for size in resolutions:
        req = MockGenerateRequest(
            prompt="test image",
            size=size,
            num_inference_steps=4,
            seed=777
        )

        job = MockJob(req=req)

        print(f"[test] Testing resolution: {size}")

        png_bytes, seed = sdxl_worker.run_job(job)

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 1000
        assert seed == 777

        print(f"[test] Resolution {size} OK: {len(png_bytes)} bytes")


def test_invalid_size_format(sdxl_worker):
    """Test that invalid size format raises an error."""
    req = MockGenerateRequest(
        prompt="test",
        size="invalid_size",  # Invalid format
        num_inference_steps=4,
    )

    job = MockJob(req=req)

    print(f"[test] Testing invalid size format")

    with pytest.raises(RuntimeError, match="Invalid size"):
        sdxl_worker.run_job(job)

    print(f"[test] Invalid size format correctly rejected")


def test_random_seed_generation(sdxl_worker):
    """Test that worker generates random seeds when none provided."""
    req = MockGenerateRequest(
        prompt="random seed test",
        size="512x512",
        num_inference_steps=4,
        seed=None  # No seed provided
    )

    job = MockJob(req=req)

    print(f"[test] Testing random seed generation")

    png_bytes1, seed1 = sdxl_worker.run_job(job)
    png_bytes2, seed2 = sdxl_worker.run_job(job)

    # Seeds should be different (highly likely)
    assert seed1 != seed2, "Random seeds should be different"
    # But outputs should also be different
    assert png_bytes1 != png_bytes2, "Different seeds should produce different outputs"

    print(f"[test] Random seed generation OK: seed1={seed1}, seed2={seed2}")


if __name__ == "__main__":
    """
    Run tests directly:

    export SDXL_MODEL_ROOT=/path/to/models
    export SDXL_MODEL=sdxl-model.safetensors
    python tests/test_sdxl_worker.py
    """
    pytest.main([__file__, "-v", "-s"])
