"""
Model load/unload lifecycle tests.

Covers the full lifecycle matrix from TODO.md:
1. Load Dreamworker
"""

import gc
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import Future
import sys

# Mock dependencies just long enough to import, then restore immediately.
_MOCKED_MODULES = ['torch', 'torch.cuda', 'diffusers']
_saved_modules = {k: sys.modules.get(k) for k in _MOCKED_MODULES}

for _mod in _MOCKED_MODULES:
    sys.modules[_mod] = MagicMock()

from backends.worker_pool import (
    WorkerPool,
    Job,
    JobType,
    GenerationJob,
    ModeSwitchJob,
    CustomJob,
    reset_worker_pool,
)

# Restore immediately
for _mod, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _orig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mode_config():
    """Two modes: mode-a (SDXL) and mode-b (SD1.5)."""
    config = Mock()
    config.config = Mock()
    config.config.model_root = "/models"

    mode_a = Mock()
    mode_a.name = "mode-a"
    mode_a.model = "sdxl.safetensors"
    mode_a.model_path = "/models/sdxl.safetensors"
    mode_a.loras = []
    mode_a.default_size = "1024x1024"
    mode_a.default_steps = 30
    mode_a.default_guidance = 7.5

    mode_b = Mock()
    mode_b.name = "mode-b"
    mode_b.model = "sd15.safetensors"
    mode_b.model_path = "/models/sd15.safetensors"
    mode_b.loras = []
    mode_b.default_size = "512x512"
    mode_b.default_steps = 4
    mode_b.default_guidance = 1.0

    config.get_mode.side_effect = lambda name: {
        "mode-a": mode_a,
        "mode-b": mode_b,
    }[name]

    config.get_default_mode.return_value = "mode-a"

    return config


@pytest.fixture
def mock_registry():
    """Mock model registry tracking register/unregister calls."""
    registry = Mock()
    registry.get_used_vram.return_value = 0
    registry.can_fit.return_value = True
    registry.register_model = Mock()
    registry.unregister_model = Mock()
    registry.clear = Mock()
    return registry


@pytest.fixture
def mock_worker_factory():
    """Returns Mock workers with run_job returning fake PNG bytes."""
    fake_png = b"\x89PNG_fake_image_data"

    def factory(worker_id: int):
        worker = Mock()
        worker.run_job = Mock(return_value=fake_png)
        return worker

    return Mock(side_effect=factory)


@pytest.fixture
def pool(mock_mode_config, mock_registry, mock_worker_factory):
    """Standard WorkerPool with mode-a loaded on init."""
    reset_worker_pool()
    p = WorkerPool(
        queue_max=10,
        worker_factory=mock_worker_factory,
        mode_config=mock_mode_config,
        registry=mock_registry,
    )
    yield p
    p.shutdown()
    reset_worker_pool()


@pytest.fixture
def empty_pool(mock_mode_config, mock_registry, mock_worker_factory):
    """WorkerPool with no initial worker loaded."""
    reset_worker_pool()

    # Patch _load_mode during __init__ so nothing is loaded
    original_load = WorkerPool._load_mode
    with patch.object(WorkerPool, '_load_mode'):
        p = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )
    # Restore _load_mode for later use
    # (patch.object context manager already restores it)

    # Manually start the worker thread since _load_mode was skipped
    p._start_worker_thread()

    yield p
    p.shutdown()
    reset_worker_pool()

class TestDreamInitWorkerPool:
    """Test dream init with WorkerPool (service=None) code path."""

    @pytest.mark.asyncio
    async def test_dream_init_with_worker_pool(self, tmp_path):
        """service=None + worker_pool with a worker → should succeed past worker lookup."""
        from yume.dream_init import initialize_dream_system

        mock_worker = Mock()
        mock_pool = Mock()
        mock_pool._worker = mock_worker

        app_state = Mock()
        # Patch redis so we don't need a real connection
        with patch('yume.dream_init.redis') as mock_redis:
            mock_redis.from_url.return_value = Mock()
            mock_redis.from_url.return_value.ping = Mock(side_effect=Exception("no redis"))

            result = await initialize_dream_system(
                app_state=app_state,
                service=None,
                backend="cpu",
                worker_pool=mock_pool,
            )
        # Redis fails so init returns False, but we didn't crash on service.workers
        assert result is False

    @pytest.mark.asyncio
    async def test_dream_init_no_service_no_pool(self, tmp_path):
        """Both service and worker_pool are None → returns False."""
        from yume.dream_init import initialize_dream_system

        app_state = Mock()
        with patch('yume.dream_init.redis') as mock_redis:
            mock_conn = Mock()
            mock_conn.ping = Mock(return_value=True)
            mock_redis.from_url.return_value = mock_conn

            result = await initialize_dream_system(
                app_state=app_state,
                service=None,
                backend="cpu",
                worker_pool=None,
            )
        assert result is False
