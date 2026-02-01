"""
Unit tests for DreamWorker class.
Tests dream session management, candidate generation, and scoring.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from PIL import Image
import io

# Mock the DreamWorker and related classes
# In real usage, import from: from yume.dream_worker import DreamWorker, DreamCandidate
# For testing, we'll create minimal versions


@pytest.fixture
def mock_model():
    """Mock LCM pipeline worker."""
    model = Mock()
    model.worker_id = 1
    
    # Mock run_job to return fake PNG bytes
    def mock_run_job(spec):
        # Create a small test image
        img = Image.new('RGB', (64, 64), color='red')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        return png_bytes, spec.seed or 12345
    
    model.run_job = mock_run_job
    return model


@pytest_asyncio.fixture
async def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.hset = AsyncMock(return_value=True)
    redis.zadd = AsyncMock(return_value=True)
    redis.zrevrange = AsyncMock(return_value=[])
    redis.hgetall = AsyncMock(return_value={})
    return redis


@pytest.fixture
def mock_clip_scorer():
    """Mock CLIP scorer."""
    scorer = Mock()
    scorer.score = Mock(return_value=0.75)
    return scorer


@pytest.fixture
def dream_worker(mock_model, mock_redis, mock_clip_scorer):
    """Create DreamWorker instance for testing."""
    # Import or create DreamWorker
    # For this example, assuming we have access to it
    from yume.dream_worker import DreamWorker
    
    config = {
        'top_k': 100,
    }
    
    worker = DreamWorker(
        model=mock_model,
        redis_client=mock_redis,
        clip_scorer=mock_clip_scorer,
        config=config
    )
    
    return worker


class TestDreamWorkerInitialization:
    """Test DreamWorker initialization."""
    
    def test_init_default_config(self, mock_model, mock_redis, dream_worker):
        """Test initialization with default config."""
        
        assert dream_worker.model == mock_model
        assert dream_worker.redis == mock_redis
        assert dream_worker.is_dreaming is False
        assert dream_worker.dream_count == 0
        assert dream_worker.start_time is None
        assert dream_worker.top_k == 100  # default
    
    def test_init_custom_config(self, mock_model, mock_redis):
        """Test initialization with custom config."""
        from yume.dream_worker import DreamWorker
        
        config = {'top_k': 50}
        worker = DreamWorker(mock_model, mock_redis, config=config)
        
        assert worker.top_k == 50
    
    def test_init_with_clip_scorer(self, dream_worker, mock_clip_scorer):
        """Test initialization with CLIP scorer."""
        
        assert dream_worker.clip_scorer == mock_clip_scorer


class TestDreamSession:
    """Test dream session management."""
    
    @pytest.mark.asyncio
    async def test_start_dreaming(self, dream_worker):
        """Test starting a dream session."""
        result = await dream_worker.start_dreaming(
            base_prompt="a beautiful landscape",
            duration_hours=0.001,  # Very short for testing
        )
        
        assert result['status'] == 'started'
        assert result['base_prompt'] == "a beautiful landscape"
        assert dream_worker.is_dreaming is True
        assert dream_worker.start_time is not None
        assert len(dream_worker.prompt_variations) > 0
    
    @pytest.mark.asyncio
    async def test_start_dreaming_while_already_dreaming(self, dream_worker):
        """Test that starting while already dreaming returns error."""
        await dream_worker.start_dreaming("test", duration_hours=1)
        
        result = await dream_worker.start_dreaming("test2", duration_hours=1)
        
        assert 'error' in result
        assert result['error'] == "Already dreaming"
    
    def test_stop_dreaming(self, dream_worker):
        """Test stopping a dream session."""
        dream_worker.is_dreaming = True
        dream_worker.dream_count = 100
        
        result = dream_worker.stop_dreaming()
        
        assert result['status'] == 'stopped'
        assert result['total_dreams'] == 100
        assert dream_worker.is_dreaming is False
    
    def test_get_status(self, dream_worker):
        """Test getting dream status."""
        dream_worker.is_dreaming = True
        dream_worker.dream_count = 50
        dream_worker.dreams_per_second = 10
        dream_worker.start_time = time.time()
        
        status = dream_worker.get_status()
        
        assert status['is_dreaming'] is True
        assert status['dream_count'] == 50
        assert status['dreams_per_second'] == 10
        assert status['elapsed_seconds'] >= 0


class TestCandidateGeneration:
    """Test candidate generation and scoring."""
    
    @pytest.mark.asyncio
    async def test_generate_candidate(self, dream_worker):
        """Test generating a single candidate."""
        dream_worker.prompt_variations = ["test prompt"]
        dream_worker.exploration_strategy = "random"
        
        candidate = await dream_worker._generate_candidate(temperature=0.5)
        
        assert candidate.seed is not None
        assert candidate.prompt == "test prompt"
        assert candidate.score == 0.0  # Not scored yet
        assert candidate.timestamp > 0
        assert candidate.latent_hash is not None
        assert candidate.rendered is False
    
    @pytest.mark.asyncio
    async def test_score_candidate_with_clip(self, dream_worker, mock_clip_scorer):
        """Test scoring candidate with CLIP."""
        from yume.dream_worker import DreamCandidate
        
        candidate = DreamCandidate(
            seed=12345,
            prompt="test",
            score=0.0,
            timestamp=time.time(),
            latent_hash="abc123",
            metadata={'preview_bytes': None}
        )
        
        # Mock that candidate has preview
        img = Image.new('RGB', (64, 64), color='blue')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        candidate.metadata['preview_bytes'] = buf.getvalue()
        
        score = await dream_worker._score_candidate(candidate)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_score_candidate_without_clip(self, dream_worker):
        """Test scoring candidate without CLIP (aesthetic only)."""
        dream_worker.clip_scorer = None
        
        from yume.dream_worker import DreamCandidate
        
        candidate = DreamCandidate(
            seed=12345,
            prompt="test",
            score=0.0,
            timestamp=time.time(),
            latent_hash="abc123",
            metadata={'preview_bytes': None}
        )
        
        # Add preview
        img = Image.new('RGB', (64, 64), color='green')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        candidate.metadata['preview_bytes'] = buf.getvalue()
        
        score = await dream_worker._score_candidate(candidate)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestPromptVariations:
    """Test prompt variation generation."""
    
    def test_generate_variations_low_temperature(self, dream_worker):
        """Test prompt variations with low temperature."""
        variations = dream_worker._generate_prompt_variations(
            "a cat", 
            temperature=0.1
        )
        
        assert len(variations) >= 1
        assert "a cat" in variations
    
    def test_generate_variations_high_temperature(self, dream_worker):
        """Test prompt variations with high temperature."""
        variations = dream_worker._generate_prompt_variations(
            "a dog",
            temperature=0.9
        )
        
        assert len(variations) > 1
        assert "a dog" in variations
        # Should have variations with modifiers
        assert any("," in v for v in variations)


class TestExplorationStrategies:
    """Test seed exploration strategies."""
    
    def test_random_strategy(self, dream_worker):
        """Test random seed generation."""
        dream_worker.exploration_strategy = "random"
        
        seed1 = dream_worker._next_exploration_seed()
        seed2 = dream_worker._next_exploration_seed()
        
        assert seed1 != seed2  # Should be different (usually)
        assert 0 <= seed1 < 2**31
        assert 0 <= seed2 < 2**31
    
    def test_linear_walk_strategy(self, dream_worker):
        """Test linear walk seed generation."""
        dream_worker.exploration_strategy = "linear_walk"
        dream_worker.dream_count = 0
        
        seed1 = dream_worker._next_exploration_seed()
        dream_worker.dream_count = 1
        seed2 = dream_worker._next_exploration_seed()
        
        assert seed2 == seed1 + 1000
    
    def test_grid_strategy(self, dream_worker):
        """Test grid-based seed generation."""
        dream_worker.exploration_strategy = "grid"
        dream_worker.top_k = 100
        dream_worker.dream_count = 0
        
        seed1 = dream_worker._next_exploration_seed()
        dream_worker.dream_count = 1
        seed2 = dream_worker._next_exploration_seed()
        
        assert seed1 != seed2


class TestLatentHashing:
    """Test latent hashing for deduplication."""
    
    def test_hash_numpy_array(self, dream_worker):
        """Test hashing numpy array."""
        latent = np.random.rand(4, 8, 8).astype(np.float32)
        
        hash1 = dream_worker._hash_latent(latent)
        hash2 = dream_worker._hash_latent(latent)
        
        assert hash1 == hash2  # Same data = same hash
        assert len(hash1) == 32  # MD5 hex digest
    
    def test_hash_torch_tensor(self, dream_worker):
        """Test hashing torch tensor."""
        import torch

        latent = torch.randn(1, 4, 8, 8)

        hash1 = dream_worker._hash_latent(latent)
        hash2 = dream_worker._hash_latent(latent)
        
        assert hash1 == hash2
        assert len(hash1) == 32
    
    def test_hash_different_latents(self, dream_worker):
        """Test that different latents have different hashes."""
        latent1 = np.random.rand(4, 8, 8).astype(np.float32)
        latent2 = np.random.rand(4, 8, 8).astype(np.float32)
        
        hash1 = dream_worker._hash_latent(latent1)
        hash2 = dream_worker._hash_latent(latent2)
        
        assert hash1 != hash2


class TestRedisStorage:
    """Test Redis storage operations."""
    
    @pytest.mark.asyncio
    async def test_store_candidate(self, dream_worker, mock_redis):
        """Test storing candidate in Redis."""
        from yume.dream_worker import DreamCandidate
        
        dream_worker.start_time = time.time()
        
        candidate = DreamCandidate(
            seed=12345,
            prompt="test prompt",
            score=0.85,
            timestamp=time.time(),
            latent_hash="abc123",
            rendered=True,
            image_data="base64imagedata",
            metadata={'size': '512x512'}
        )
        
        await dream_worker._store_candidate(candidate)
        
        # Verify Redis calls
        assert mock_redis.hset.called
        assert mock_redis.zadd.called
    
    @pytest.mark.asyncio
    async def test_get_top_dreams_no_session(self, dream_worker):
        """Test getting top dreams with no active session."""
        dream_worker.start_time = None
        
        result = await dream_worker.get_top_dreams()
        
        assert [] == result
    
    @pytest.mark.asyncio
    async def test_get_top_dreams_with_results(self, dream_worker, mock_redis):
        """Test getting top dreams with results."""
        dream_worker.start_time = time.time()
        
        # Mock Redis responses
        mock_redis.zrevrange = AsyncMock(return_value=[
            (b"dream:123:12345", 0.9),
            (b"dream:123:67890", 0.8),
        ])
        
        mock_redis.hgetall = AsyncMock(return_value={
            b'seed': b'12345',
            b'prompt': b'test',
            b'score': b'0.9',
            b'timestamp': b'1234567890.0',
            b'rendered': b'1',
            b'image_data': b'imagedata',
        })
        
        results = await dream_worker.get_top_dreams(limit=10)
        
        assert isinstance(results, list)


class TestFPSTracking:
    """Test FPS tracking."""
    
    def test_update_fps(self, dream_worker):
        """Test FPS counter update."""
        dream_worker.last_fps_check = time.time() - 2.0  # 2 seconds ago
        dream_worker.fps_counter = 20
        
        dream_worker._update_fps()
        
        # After 2 seconds with 20 frames, should be ~10 fps
        assert dream_worker.dreams_per_second > 0
        assert dream_worker.fps_counter == 0  # Reset after update


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    @pytest.mark.asyncio
    async def test_generate_candidate_with_model_error(self, dream_worker, mock_model):
        """Test candidate generation when model fails."""
        mock_model.run_job = Mock(side_effect=Exception("Model error"))
        
        dream_worker.prompt_variations = ["test"]
        candidate = await dream_worker._generate_candidate(0.5)
        
        # Should return a candidate with dummy data, not crash
        assert candidate is not None
        assert candidate.seed is not None
    
    @pytest.mark.asyncio
    async def test_score_candidate_with_scorer_error(self, dream_worker, mock_clip_scorer):
        """Test scoring when CLIP scorer fails."""
        mock_clip_scorer.score = Mock(side_effect=Exception("CLIP error"))
        
        from yume.dream_worker import DreamCandidate
        
        candidate = DreamCandidate(
            seed=12345,
            prompt="test",
            score=0.0,
            timestamp=time.time(),
            latent_hash="abc123",
            metadata={'preview_bytes': b'fake'}
        )
        
        # Should fall back to aesthetic scoring
        score = await dream_worker._score_candidate(candidate)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
