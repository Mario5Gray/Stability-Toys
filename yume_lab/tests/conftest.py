"""
Shared pytest fixtures for Yume dream system tests.
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

# Add parent's parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


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
    """Create a 224x224 test image (CLIP size)."""
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
def random_latent_tensor():
    """Create a random latent tensor."""
    import torch
    return torch.randn(1, 4, 8, 8)


@pytest.fixture
def random_latent_numpy():
    """Create a random latent as numpy array."""
    return np.random.randn(1, 4, 8, 8).astype(np.float32)


@pytest_asyncio.fixture
async def mock_redis_client():
    """Create a mock Redis async client."""
    redis = AsyncMock()

    redis.hset = AsyncMock(return_value=True)
    redis.hget = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    redis.zadd = AsyncMock(return_value=True)
    redis.zrevrange = AsyncMock(return_value=[])
    redis.zrange = AsyncMock(return_value=[])
    redis.delete = AsyncMock(return_value=True)
    redis.exists = AsyncMock(return_value=False)
    redis.keys = AsyncMock(return_value=[])

    redis._store = {}
    redis._sorted_sets = {}

    def mock_hset_impl(key, mapping=None, **kwargs):
        if key not in redis._store:
            redis._store[key] = {}
        if mapping:
            redis._store[key].update(mapping)
        redis._store[key].update(kwargs)
        return len(mapping) if mapping else len(kwargs)

    def mock_hgetall_impl(key):
        return redis._store.get(key, {})

    def mock_zadd_impl(name, mapping):
        if name not in redis._sorted_sets:
            redis._sorted_sets[name] = []
        for key, score in mapping.items():
            redis._sorted_sets[name].append((key, score))
        redis._sorted_sets[name].sort(key=lambda x: x[1], reverse=True)
        return len(mapping)

    def mock_zrevrange_impl(name, start, end, withscores=False):
        if name not in redis._sorted_sets:
            return []
        items = redis._sorted_sets[name][start:end+1]
        if withscores:
            return items
        return [item[0] for item in items]

    redis.hset.side_effect = mock_hset_impl
    redis.hgetall.side_effect = mock_hgetall_impl
    redis.zadd.side_effect = mock_zadd_impl
    redis.zrevrange.side_effect = mock_zrevrange_impl

    return redis


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
def mock_clip_model_hf():
    """Create a mock Hugging Face CLIP model."""
    import torch

    model = Mock()
    model.__class__.__name__ = "CLIPModel"

    def get_text_features(**kwargs):
        features = torch.randn(1, 512)
        return features / features.norm(dim=-1, keepdim=True)

    def get_image_features(**kwargs):
        features = torch.randn(1, 512)
        return features / features.norm(dim=-1, keepdim=True)

    model.get_text_features = Mock(side_effect=get_text_features)
    model.get_image_features = Mock(side_effect=get_image_features)
    model.to = Mock(return_value=model)
    model.eval = Mock()

    return model


@pytest.fixture
def mock_clip_processor():
    """Create a mock CLIP processor."""
    import torch

    processor = Mock()

    def process_text(text, **kwargs):
        return {'input_ids': torch.randint(0, 1000, (1, 77))}

    def process_images(images, **kwargs):
        return {'pixel_values': torch.randn(1, 3, 224, 224)}

    def process_call(text=None, images=None, **kwargs):
        if text is not None:
            return process_text(text, **kwargs)
        elif images is not None:
            return process_images(images, **kwargs)
        else:
            raise ValueError("Must provide text or images")

    processor.side_effect = process_call
    return processor


@pytest.fixture
def mock_clip_scorer(mock_clip_model_hf, mock_clip_processor):
    """Create a mock CLIP scorer with realistic scoring."""
    scorer = Mock()

    def score_fn(image, text):
        img_array = np.array(image)
        score = (img_array.mean() / 255.0) * 0.5 + len(text) / 200.0
        return min(1.0, max(0.0, score))

    scorer.score = Mock(side_effect=score_fn)
    return scorer


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


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "requires_redis: mark test as requiring Redis")


def assert_valid_score(score):
    """Assert that a score is valid (between 0 and 1)."""
    assert isinstance(score, float), f"Score must be float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score must be in [0, 1], got {score}"


def assert_valid_image(image):
    """Assert that an image is valid PIL Image."""
    assert isinstance(image, Image.Image), f"Expected PIL Image, got {type(image)}"
    assert image.size[0] > 0 and image.size[1] > 0, "Image has invalid size"


def assert_valid_candidate(candidate):
    """Assert that a DreamCandidate is valid."""
    assert candidate.seed is not None, "Candidate must have seed"
    assert candidate.prompt, "Candidate must have prompt"
    assert candidate.timestamp > 0, "Candidate must have valid timestamp"
    assert candidate.latent_hash, "Candidate must have latent hash"
    assert_valid_score(candidate.score)


pytest.assert_valid_score = assert_valid_score
pytest.assert_valid_image = assert_valid_image
pytest.assert_valid_candidate = assert_valid_candidate
