"""
Model load/unload lifecycle tests.

Covers the full lifecycle matrix from TODO.md:
1. Load model → generate (success)
2. Unload model → generate (fail: "No worker available")
3. Load new mode → generate (success)
4. Unload mode → generate (fail)
5. No mode → generate (fail)
6. Unload when nothing loaded → safe no-op
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


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestBasicLifecycle:
    """Load → generate, unload → fail, double-unload safe."""

    def test_load_and_generate(self, pool):
        """mode-a loaded on init, GenerationJob succeeds."""
        assert pool._current_mode == "mode-a"
        assert pool._worker is not None

        job = GenerationJob(req=Mock())
        result = pool.submit_job(job).result(timeout=5.0)

        assert result == b"\x89PNG_fake_image_data"

    def test_generate_after_unload_fails(self, pool):
        """Unload → GenerationJob raises RuntimeError."""
        pool._unload_current_worker()
        assert pool._worker is None

        job = GenerationJob(req=Mock())
        future = pool.submit_job(job)

        with pytest.raises(RuntimeError, match="No worker available"):
            future.result(timeout=5.0)

    def test_unload_when_already_unloaded_is_safe(self, pool):
        """Double unload produces no error."""
        pool._unload_current_worker()
        pool._unload_current_worker()  # should be a no-op
        assert pool._worker is None


class TestModeSwitchingLifecycle:
    """Mode switching: unload→switch, auto-unload, switch→unload."""

    def test_load_new_mode_after_unload(self, pool, mock_worker_factory):
        """Unload mode-a → switch to mode-b → generate succeeds."""
        pool._unload_current_worker()
        assert pool._worker is None

        future = pool.switch_mode("mode-b")
        future.result(timeout=5.0)

        assert pool._current_mode == "mode-b"
        assert pool._worker is not None

        job = GenerationJob(req=Mock())
        result = pool.submit_job(job).result(timeout=5.0)
        assert result == b"\x89PNG_fake_image_data"

    def test_switch_auto_unloads_old_mode(self, pool, mock_registry):
        """Switch mode-a→mode-b without explicit unload; old mode unregistered."""
        future = pool.switch_mode("mode-b")
        future.result(timeout=5.0)

        assert pool._current_mode == "mode-b"
        # Should have unregistered mode-a during the switch
        mock_registry.unregister_model.assert_any_call("mode-a")

    def test_generate_after_switch_then_unload(self, pool):
        """Switch → generate → unload → generate fails."""
        pool.switch_mode("mode-b").result(timeout=5.0)

        job1 = GenerationJob(req=Mock())
        result = pool.submit_job(job1).result(timeout=5.0)
        assert result == b"\x89PNG_fake_image_data"

        pool._unload_current_worker()

        job2 = GenerationJob(req=Mock())
        with pytest.raises(RuntimeError, match="No worker available"):
            pool.submit_job(job2).result(timeout=5.0)


class TestNoModeScenarios:
    """Tests starting with no mode loaded."""

    def test_generate_without_any_mode(self, empty_pool):
        """Empty pool → generate fails."""
        assert empty_pool._worker is None

        job = GenerationJob(req=Mock())
        future = empty_pool.submit_job(job)

        with pytest.raises(RuntimeError, match="No worker available"):
            future.result(timeout=5.0)

    def test_load_after_starting_empty(self, empty_pool):
        """Empty pool → switch_mode → generate succeeds."""
        empty_pool.switch_mode("mode-b").result(timeout=5.0)

        assert empty_pool._current_mode == "mode-b"
        assert empty_pool._worker is not None

        job = GenerationJob(req=Mock())
        result = empty_pool.submit_job(job).result(timeout=5.0)
        assert result == b"\x89PNG_fake_image_data"

    def test_unload_empty_pool_noop(self, empty_pool):
        """Empty pool unload is safe no-op."""
        empty_pool._unload_current_worker()  # no error
        assert empty_pool._worker is None


class TestRegistryIntegration:
    """Verify registry register/unregister calls during lifecycle."""

    def test_register_on_load(self, pool, mock_registry):
        """Register called with mode name on init."""
        mock_registry.register_model.assert_called_once()
        args = mock_registry.register_model.call_args
        assert args.kwargs['name'] == "mode-a"
        assert args.kwargs['model_path'] == "/models/sdxl.safetensors"

    def test_unregister_on_unload(self, pool, mock_registry):
        """Unregister called with mode name on unload."""
        pool._unload_current_worker()
        mock_registry.unregister_model.assert_called_once_with("mode-a")

    def test_registry_ops_on_switch(self, pool, mock_registry):
        """Unregister old + register new on mode switch."""
        mock_registry.reset_mock()

        pool.switch_mode("mode-b").result(timeout=5.0)

        mock_registry.unregister_model.assert_called_with("mode-a")
        # register_model called for mode-b
        reg_call = mock_registry.register_model.call_args
        assert reg_call.kwargs['name'] == "mode-b"

    def test_vram_tracked(self, pool, mock_registry):
        """get_used_vram called during load."""
        # Called twice per _load_mode (before and after worker creation)
        assert mock_registry.get_used_vram.call_count >= 2


class TestEdgeCases:
    """Edge cases and defensive behavior."""

    def test_triple_unload_safe(self, pool):
        """Three consecutive unloads produce no error."""
        pool._unload_current_worker()
        pool._unload_current_worker()
        pool._unload_current_worker()
        assert pool._worker is None

    def test_switch_to_same_mode_skips(self, pool, mock_worker_factory):
        """Switching to current mode does not recreate worker."""
        calls_before = mock_worker_factory.call_count

        pool.switch_mode("mode-a").result(timeout=5.0)

        assert mock_worker_factory.call_count == calls_before

    def test_generation_job_execute_none(self):
        """GenerationJob.execute(None) raises RuntimeError."""
        job = GenerationJob(req=Mock())
        with pytest.raises(RuntimeError, match="No worker available"):
            job.execute(None)

    @patch('backends.worker_pool.torch.cuda.empty_cache')
    def test_unload_triggers_gc_and_cache_clear(self, mock_empty_cache, pool):
        """Unload calls gc.collect and torch.cuda.empty_cache."""
        mock_gc = Mock()
        with patch.dict('sys.modules', {'gc': mock_gc}):
            # gc is imported locally in _unload_current_worker, so we
            # patch it via the gc module itself
            import gc as real_gc
            with patch.object(real_gc, 'collect') as mock_collect:
                pool._unload_current_worker()
                mock_collect.assert_called()
        mock_empty_cache.assert_called()


class TestFullLifecycleMatrix:
    """Single test walking through all 6 TODO.md lifecycle steps."""

    def test_full_lifecycle_matrix(self, mock_mode_config, mock_registry, mock_worker_factory):
        """
        1. Load model → generate (success)
        2. Unload model → generate (fail)
        3. Load new mode → generate (success)
        4. Unload mode → generate (fail)
        5. No mode → generate (fail)  [already in unloaded state]
        6. Unload when nothing loaded → safe no-op
        """
        reset_worker_pool()
        pool = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        try:
            # Step 1: Load model → generate (success)
            assert pool._current_mode == "mode-a"
            job1 = GenerationJob(req=Mock())
            result1 = pool.submit_job(job1).result(timeout=5.0)
            assert result1 == b"\x89PNG_fake_image_data"

            # Step 2: Unload model → generate (fail)
            pool._unload_current_worker()
            job2 = GenerationJob(req=Mock())
            with pytest.raises(RuntimeError, match="No worker available"):
                pool.submit_job(job2).result(timeout=5.0)

            # Step 3: Load new mode → generate (success)
            pool.switch_mode("mode-b").result(timeout=5.0)
            assert pool._current_mode == "mode-b"
            job3 = GenerationJob(req=Mock())
            result3 = pool.submit_job(job3).result(timeout=5.0)
            assert result3 == b"\x89PNG_fake_image_data"

            # Step 4: Unload mode → generate (fail)
            pool._unload_current_worker()
            job4 = GenerationJob(req=Mock())
            with pytest.raises(RuntimeError, match="No worker available"):
                pool.submit_job(job4).result(timeout=5.0)

            # Step 5: No mode → generate (fail) [still unloaded]
            job5 = GenerationJob(req=Mock())
            with pytest.raises(RuntimeError, match="No worker available"):
                pool.submit_job(job5).result(timeout=5.0)

            # Step 6: Unload when nothing loaded → safe no-op
            pool._unload_current_worker()  # no error

        finally:
            pool.shutdown()
            reset_worker_pool()
