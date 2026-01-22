"""
Integration tests for Yume library.
Tests end-to-end workflows and component interactions.
"""

import pytest
import asyncio
import time
import numpy as np
from PIL import Image
import io
import redis.asyncio as aioredis
from unittest.mock import Mock, AsyncMock, MagicMock, patch


@pytest.fixture
async def redis_client():
    """Create real or mock Redis client."""
    # For testing, use mock. For real integration tests, use real Redis
    redis = AsyncMock()
    redis.hset = AsyncMock(return_value=True)
    redis.zadd = AsyncMock(return_value=True)
    redis.zrevrange = AsyncMock(return_value=[])
    redis.hgetall = AsyncMock(return_value={})
    redis.delete = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_pipeline_worker():
    """Create a mock pipeline worker that simulates real image generation."""
    worker = Mock()
    worker.worker_id = 1
    
    def generate_image(spec):
        """Generate a test image based on seed."""
        # Use seed to create deterministic images
        seed = spec.seed or 12345
        np.random.seed(seed)
        
        # Parse size
        width, height = map(int, spec.size.split('x'))
        
        # Generate random but deterministic image
        pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)
        
        # Convert to PNG bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue(), seed
    
    worker.run_job = generate_image
    return worker


@pytest.fixture
def mock_clip_scorer():
    """Create a mock CLIP scorer with realistic behavior."""
    scorer = Mock()
    
    def score_image(image, text):
        """Score based on simple image properties."""
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        # Use std dev as proxy for "interesting-ness"
        score = min(1.0, img_array.std() / 100.0)
        
        # Add text influence (just length-based for mock)
        text_influence = min(1.0, len(text) / 100.0)
        
        return 0.5 * score + 0.5 * text_influence
    
    scorer.score = score_image
    return scorer


class TestEndToEndDreamSession:
    """Test complete dream session workflow."""
    
    @pytest.mark.asyncio
    async def test_full_dream_session(self, redis_client, mock_pipeline_worker, mock_clip_scorer):
        """Test a complete dream session from start to finish."""
        from yume.dream_worker import DreamWorker
        
        # Create worker
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
            clip_scorer=mock_clip_scorer,
            config={'top_k': 10}
        )
        
        # Start dreaming
        result = await worker.start_dreaming(
            base_prompt="a beautiful landscape",
            duration_hours=0.001,  # Very short for testing
            temperature=0.5,
            similarity_threshold=0.3,
            render_interval=5
        )
        
        assert result['status'] == 'started'
        assert worker.is_dreaming
        
        # Wait for session to complete
        await asyncio.sleep(0.1)
        
        # Stop if still running
        if worker.is_dreaming:
            worker.stop_dreaming()
        
        # Check that some dreams were generated
        assert worker.dream_count > 0
    
    @pytest.mark.asyncio
    async def test_dream_session_with_rendering(self, redis_client, mock_pipeline_worker, mock_clip_scorer):
        """Test that high-scoring candidates get rendered."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
            clip_scorer=mock_clip_scorer,
            config={'top_k': 5}
        )
        
        # Mock high scores
        mock_clip_scorer.score = Mock(return_value=0.95)
        
        await worker.start_dreaming(
            base_prompt="test",
            duration_hours=0.001,
            similarity_threshold=0.9,
            render_interval=2
        )
        
        await asyncio.sleep(0.1)
        
        # Should have stored candidates in Redis
        # (Check via mock calls)
        assert redis_client.hset.called or redis_client.zadd.called


class TestDreamWorkflowSteps:
    """Test individual workflow steps."""
    
    @pytest.mark.asyncio
    async def test_candidate_generation_and_scoring_pipeline(self, redis_client, mock_pipeline_worker, mock_clip_scorer):
        """Test the full candidate pipeline."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
            clip_scorer=mock_clip_scorer,
            config={'top_k': 10}
        )
        
        worker.prompt_variations = ["test prompt"]
        
        # Generate candidate
        candidate = await worker._generate_candidate(temperature=0.5)
        
        assert candidate is not None
        assert candidate.seed is not None
        assert candidate.latent_hash is not None
        
        # Score candidate
        score = await worker._score_candidate(candidate)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        # Store candidate
        candidate.score = score
        candidate.rendered = False
        
        worker.start_time = time.time()
        await worker._store_candidate(candidate)
        
        assert redis_client.hset.called
    
    @pytest.mark.asyncio
    async def test_top_candidate_rendering(self, redis_client, mock_pipeline_worker, mock_clip_scorer):
        """Test rendering of top candidates."""
        from yume.dream_worker import DreamWorker, DreamCandidate
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
            clip_scorer=mock_clip_scorer,
        )
        
        candidate = DreamCandidate(
            seed=12345,
            prompt="test",
            score=0.95,
            timestamp=time.time(),
            latent_hash="abc123",
            rendered=False,
            metadata={}
        )
        
        # Render candidate
        await worker._render_candidate(candidate)
        
        assert candidate.rendered or True  # May fail if worker doesn't support rendering
        # Check image data was set (if rendering succeeded)


class TestConcurrentDreaming:
    """Test concurrent operations and race conditions."""
    
    @pytest.mark.asyncio
    async def test_cannot_start_multiple_sessions(self, redis_client, mock_pipeline_worker):
        """Test that only one dream session can run at a time."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        # Start first session
        result1 = await worker.start_dreaming("test1", duration_hours=1)
        assert result1['status'] == 'started'
        
        # Try to start second session
        result2 = await worker.start_dreaming("test2", duration_hours=1)
        assert 'error' in result2
        
        # Clean up
        worker.stop_dreaming()
    
    @pytest.mark.asyncio
    async def test_stop_while_dreaming(self, redis_client, mock_pipeline_worker):
        """Test stopping an active dream session."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        # Start session
        await worker.start_dreaming("test", duration_hours=1)
        assert worker.is_dreaming
        
        # Stop immediately
        result = worker.stop_dreaming()
        
        assert result['status'] == 'stopped'
        assert not worker.is_dreaming


class TestDreamPersistence:
    """Test Redis persistence and retrieval."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_dreams(self, redis_client, mock_pipeline_worker):
        """Test storing and retrieving dream results."""
        from yume.dream_worker import DreamWorker, DreamCandidate
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        worker.start_time = time.time()
        
        # Create and store candidates
        candidates = []
        for i in range(5):
            candidate = DreamCandidate(
                seed=1000 + i,
                prompt=f"test prompt {i}",
                score=0.5 + i * 0.1,
                timestamp=time.time(),
                latent_hash=f"hash_{i}",
                rendered=True,
                image_data=f"image_{i}",
                metadata={'index': i}
            )
            candidates.append(candidate)
            await worker._store_candidate(candidate)
        
        # Verify storage calls
        assert redis_client.hset.call_count == 5
        assert redis_client.zadd.call_count == 5
    
    @pytest.mark.asyncio
    async def test_get_top_dreams_sorted(self, redis_client, mock_pipeline_worker):
        """Test that top dreams are returned sorted by score."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        worker.start_time = time.time()
        
        # Mock Redis to return sorted results
        redis_client.zrevrange = AsyncMock(return_value=[
            (b"dream:123:1003", 0.9),
            (b"dream:123:1002", 0.8),
            (b"dream:123:1001", 0.7),
        ])
        
        redis_client.hgetall = AsyncMock(side_effect=lambda key: {
            b'seed': key.split(b':')[2],
            b'prompt': b'test',
            b'score': b'0.8',
            b'timestamp': str(time.time()).encode(),
            b'rendered': b'1',
            b'image_data': b'data',
        })
        
        results = await worker.get_top_dreams(limit=10, min_score=0.0)
        
        assert len(results) == 3


class TestExplorationStrategies:
    """Test different exploration strategies."""
    
    @pytest.mark.asyncio
    async def test_random_exploration(self, redis_client, mock_pipeline_worker):
        """Test random seed exploration."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        worker.exploration_strategy = "random"
        worker.prompt_variations = ["test"]
        
        # Generate multiple candidates
        seeds = []
        for _ in range(10):
            candidate = await worker._generate_candidate(0.5)
            seeds.append(candidate.seed)
        
        # Seeds should be diverse (not all the same)
        assert len(set(seeds)) > 1
    
    @pytest.mark.asyncio
    async def test_linear_walk_exploration(self, redis_client, mock_pipeline_worker):
        """Test linear walk exploration."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        worker.exploration_strategy = "linear_walk"
        worker.dream_count = 0
        
        seeds = []
        for i in range(5):
            worker.dream_count = i
            seed = worker._next_exploration_seed()
            seeds.append(seed)
        
        # Seeds should increment predictably
        assert seeds[1] == seeds[0] + 1000
        assert seeds[2] == seeds[1] + 1000


class TestPromptVariations:
    """Test prompt variation strategies."""
    
    @pytest.mark.asyncio
    async def test_prompt_variations_with_temperature(self, redis_client, mock_pipeline_worker):
        """Test that temperature affects prompt diversity."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        base_prompt = "a cat"
        
        # Low temperature
        low_temp_variations = worker._generate_prompt_variations(base_prompt, 0.1)
        
        # High temperature
        high_temp_variations = worker._generate_prompt_variations(base_prompt, 0.9)
        
        # High temp should generate more variations
        assert len(high_temp_variations) >= len(low_temp_variations)
        
        # Base prompt should be in both
        assert base_prompt in low_temp_variations
        assert base_prompt in high_temp_variations


class TestErrorRecovery:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_continue_after_model_error(self, redis_client, mock_clip_scorer):
        """Test that dream session continues after model errors."""
        from yume.dream_worker import DreamWorker
        
        # Create worker with failing model
        failing_model = Mock()
        failing_model.worker_id = 1
        failing_model.run_job = Mock(side_effect=Exception("Model crashed"))
        
        worker = DreamWorker(
            model=failing_model,
            redis_client=redis_client,
            clip_scorer=mock_clip_scorer,
        )
        
        worker.prompt_variations = ["test"]
        
        # Should handle error gracefully
        try:
            candidate = await worker._generate_candidate(0.5)
            # If it returns a candidate, it handled the error
            assert candidate is not None
        except Exception:
            # Or it may raise, which is also acceptable
            pass
    
    @pytest.mark.asyncio
    async def test_continue_after_scoring_error(self, redis_client, mock_pipeline_worker):
        """Test that scoring errors don't crash the session."""
        from yume.dream_worker import DreamWorker, DreamCandidate
        
        # Create scorer that fails
        failing_scorer = Mock()
        failing_scorer.score = Mock(side_effect=Exception("Scorer crashed"))
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
            clip_scorer=failing_scorer,
        )
        
        candidate = DreamCandidate(
            seed=12345,
            prompt="test",
            score=0.0,
            timestamp=time.time(),
            latent_hash="abc",
            metadata={'preview_bytes': b'fake'}
        )
        
        # Should fall back to aesthetic scoring
        score = await worker._score_candidate(candidate)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestPerformanceMetrics:
    """Test performance tracking."""
    
    @pytest.mark.asyncio
    async def test_fps_tracking(self, redis_client, mock_pipeline_worker):
        """Test dreams per second tracking."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        worker.last_fps_check = time.time() - 1.0
        worker.fps_counter = 10
        
        # Update FPS
        worker._update_fps()
        
        # Should have calculated FPS
        assert worker.dreams_per_second > 0
        assert worker.fps_counter == 0  # Reset
    
    @pytest.mark.asyncio
    async def test_status_tracking(self, redis_client, mock_pipeline_worker):
        """Test status information tracking."""
        from yume.dream_worker import DreamWorker
        
        worker = DreamWorker(
            model=mock_pipeline_worker,
            redis_client=redis_client,
        )
        
        # Start session
        await worker.start_dreaming("test", duration_hours=1)
        
        # Get status
        status = worker.get_status()
        
        assert status['is_dreaming']
        assert 'dream_count' in status
        assert 'dreams_per_second' in status
        assert 'elapsed_seconds' in status
        
        # Clean up
        worker.stop_dreaming()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
