"""
Shared pytest fixtures and configuration for Dream Lab tests.
"""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
from PIL import Image
import io
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def test_image_64():
    """Create a 64x64 test image."""
    return Image.new('RGB', (64, 64), color=(100, 150, 200))


@pytest.fixture
def test_image_224():
    """Create a 224x224 test image."""
    return Image.new('RGB', (224, 224), color=(100, 150, 200))


@pytest.fixture
def test_image_512():
    """Create a 512x512 test image."""
    return Image.new('RGB', (512, 512), color=(100, 150, 200))


@pytest.fixture
def test_image_bytes():
    """Create test image as PNG bytes."""
    img = Image.new('RGB', (64, 64), color='red')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def mock_pipeline_worker():
    """Create a mock pipeline worker."""
    from dataclasses import dataclass

    @dataclass
    class GenSpec:
        prompt: str
        size: str
        steps: int
        cfg: float
        seed: int = None

    worker = Mock()
    worker.worker_id = 1

    def run_job(spec):
        seed = spec.seed or 12345
        np.random.seed(seed)
        width, height = map(int, spec.size.split('x'))
        pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue(), seed

    worker.run_job = run_job
    return worker


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "a beautiful landscape",
        "a cat sitting on a windowsill",
        "abstract geometric patterns",
        "a serene mountain lake at sunset",
        "futuristic cityscape at night",
    ]


@pytest.fixture
def sample_seeds():
    """Sample seeds for testing."""
    return [12345, 67890, 11111, 22222, 33333]


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "requires_redis: mark test as requiring Redis")
    config.addinivalue_line("markers", "ws: mark test as WebSocket test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)

        if "slow" in item.name.lower() or "stress" in item.name.lower():
            item.add_marker(pytest.mark.slow)

        mod = getattr(item, "module", None)
        modfile = getattr(mod, "__file__", "") or ""
        basename = os.path.basename(modfile)
        if basename.startswith("test_ws_") or basename == "test_jobs_callback.py":
            item.add_marker(pytest.mark.ws)


def assert_valid_score(score):
    """Assert that a score is valid (between 0 and 1)."""
    assert isinstance(score, float), f"Score must be float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score must be in [0, 1], got {score}"


def assert_valid_image(image):
    """Assert that an image is valid PIL Image."""
    assert isinstance(image, Image.Image), f"Expected PIL Image, got {type(image)}"
    assert image.size[0] > 0 and image.size[1] > 0, "Image has invalid size"


pytest.assert_valid_score = assert_valid_score
pytest.assert_valid_image = assert_valid_image
