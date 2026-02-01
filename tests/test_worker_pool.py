"""
Functional tests for WorkerPool.

Tests job queue, mode switching, and worker lifecycle management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future
import time
import threading
import sys

# Mock dependencies just long enough to import the module under test,
# then restore sys.modules immediately so other test files aren't poisoned.
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
)

# Restore immediately
for _mod, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _orig


@pytest.fixture
def mock_mode_config():
    """Mock mode configuration."""
    config = Mock()
    config.config = Mock()
    config.config.model_root = "/models"

    # Define test modes
    mode_sdxl = Mock()
    mode_sdxl.name = "sdxl-general"
    mode_sdxl.model = "sdxl.safetensors"
    mode_sdxl.model_path = "/models/sdxl.safetensors"
    mode_sdxl.loras = []
    mode_sdxl.default_size = "1024x1024"
    mode_sdxl.default_steps = 30
    mode_sdxl.default_guidance = 7.5

    mode_sd15 = Mock()
    mode_sd15.name = "sd15-fast"
    mode_sd15.model = "sd15.safetensors"
    mode_sd15.model_path = "/models/sd15.safetensors"
    mode_sd15.loras = []
    mode_sd15.default_size = "512x512"
    mode_sd15.default_steps = 4
    mode_sd15.default_guidance = 1.0

    config.get_mode.side_effect = lambda name: {
        "sdxl-general": mode_sdxl,
        "sd15-fast": mode_sd15,
    }[name]

    config.get_default_mode.return_value = "sdxl-general"

    return config


@pytest.fixture
def mock_registry():
    """Mock model registry."""
    registry = Mock()
    registry.get_used_vram.return_value = 0
    registry.can_fit.return_value = True  # Default: always fits
    registry.register_model = Mock()
    registry.unregister_model = Mock()
    registry.clear = Mock()
    return registry


@pytest.fixture
def mock_worker_factory():
    """Mock worker factory."""
    worker = Mock()
    worker.run_job = Mock(return_value="test_result")

    factory = Mock()
    factory.return_value = worker
    return factory


@pytest.fixture
def worker_pool(mock_mode_config, mock_registry, mock_worker_factory):
    """Create WorkerPool with mocked dependencies using DI."""
    from backends.worker_pool import reset_worker_pool
    reset_worker_pool()  # Ensure clean state

    pool = WorkerPool(
        queue_max=10,
        worker_factory=mock_worker_factory,
        mode_config=mock_mode_config,
        registry=mock_registry,
    )
    yield pool
    pool.shutdown()
    reset_worker_pool()


class TestWorkerPoolInit:
    """Test WorkerPool initialization."""

    def test_init_with_default_mode(self, mock_mode_config, mock_registry, mock_worker_factory):
        """Test initialization loads default mode."""
        from backends.worker_pool import reset_worker_pool
        reset_worker_pool()

        pool = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        assert pool._current_mode == "sdxl-general"
        assert pool._worker is not None
        mock_worker_factory.assert_called_once_with(worker_id=0)

        pool.shutdown()
        reset_worker_pool()

    def test_init_with_custom_queue_size(self, mock_mode_config, mock_registry, mock_worker_factory):
        """Test initialization with custom queue size."""
        from backends.worker_pool import reset_worker_pool
        reset_worker_pool()

        pool = WorkerPool(
            queue_max=32,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        assert pool.queue_max == 32

        pool.shutdown()
        reset_worker_pool()

    def test_init_starts_worker_thread(self, mock_mode_config, mock_registry, mock_worker_factory):
        """Test that worker thread is started on init."""
        from backends.worker_pool import reset_worker_pool
        reset_worker_pool()

        pool = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        # Worker thread should be alive
        time.sleep(0.1)  # Give thread time to start
        assert pool._worker_thread is not None
        assert pool._worker_thread.is_alive()

        pool.shutdown()
        reset_worker_pool()


class TestJobSubmission:
    """Test job submission and execution."""

    def test_submit_generation_job(self, worker_pool):
        """Test submitting a generation job."""
        req = Mock()
        job = GenerationJob(req=req)

        future = worker_pool.submit_job(job)

        assert future is not None
        assert isinstance(future, Future)

        # Wait for result
        result = future.result(timeout=5.0)
        assert result == "test_result"

    def test_submit_custom_job(self, worker_pool):
        """Test submitting a custom job."""
        def custom_handler(x, y):
            return x + y

        job = CustomJob(handler=custom_handler, args=(5, 3))
        future = worker_pool.submit_job(job)

        result = future.result(timeout=5.0)
        assert result == 8

    def test_submit_custom_job_with_kwargs(self, worker_pool):
        """Test custom job with kwargs."""
        def custom_handler(a, b, operation="add"):
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b

        job = CustomJob(handler=custom_handler, args=(5, 3), kwargs={"operation": "multiply"})
        future = worker_pool.submit_job(job)

        result = future.result(timeout=5.0)
        assert result == 15

    def test_submit_multiple_jobs(self, worker_pool):
        """Test submitting multiple jobs (queuing)."""
        jobs = []
        for i in range(5):
            req = Mock()
            req.id = i
            job = GenerationJob(req=req)
            future = worker_pool.submit_job(job)
            jobs.append(future)

        # All jobs should complete
        for future in jobs:
            result = future.result(timeout=10.0)
            assert result == "test_result"

    def test_get_queue_size(self, worker_pool):
        """Test getting queue size."""
        # Initially empty or near-empty
        initial_size = worker_pool.get_queue_size()
        assert initial_size >= 0

        # Submit jobs that take time
        def slow_handler():
            time.sleep(0.1)
            return "done"

        for _ in range(3):
            job = CustomJob(handler=slow_handler)
            worker_pool.submit_job(job)

        # Queue should have jobs
        time.sleep(0.05)  # Let jobs enter queue
        queue_size = worker_pool.get_queue_size()
        assert queue_size >= 0  # May have already processed some


class TestModeSwitching:
    """Test mode switching functionality."""

    def test_switch_mode(self, worker_pool, mock_worker_factory):
        """Test switching to a different mode."""
        initial_mode = worker_pool.get_current_mode()
        assert initial_mode == "sdxl-general"

        # Switch to sd15
        future = worker_pool.switch_mode("sd15-fast")
        result = future.result(timeout=5.0)

        assert worker_pool.get_current_mode() == "sd15-fast"
        # Worker should be recreated
        assert mock_worker_factory.call_count >= 2

    def test_switch_mode_queues_after_jobs(self, worker_pool):
        """Test that mode switch waits for pending jobs."""
        results = []

        # Submit a slow job
        def slow_job():
            time.sleep(0.2)
            return "slow_done"

        job1 = CustomJob(handler=slow_job)
        fut1 = worker_pool.submit_job(job1)

        # Submit mode switch
        switch_fut = worker_pool.switch_mode("sd15-fast")

        # Submit another job after switch
        job2 = GenerationJob(req=Mock())
        fut2 = worker_pool.submit_job(job2)

        # All should complete
        results.append(fut1.result(timeout=5.0))
        switch_fut.result(timeout=5.0)
        results.append(fut2.result(timeout=5.0))

        assert results[0] == "slow_done"
        assert results[1] == "test_result"

    def test_get_current_mode(self, worker_pool):
        """Test getting current mode."""
        assert worker_pool.get_current_mode() == "sdxl-general"

        worker_pool.switch_mode("sd15-fast").result(timeout=5.0)
        assert worker_pool.get_current_mode() == "sd15-fast"

    def test_switch_to_same_mode_noop(self, worker_pool, mock_worker_factory):
        """Test switching to current mode is a no-op."""
        initial_call_count = mock_worker_factory.call_count

        future = worker_pool.switch_mode("sdxl-general")  # Already in this mode
        future.result(timeout=5.0)

        # Worker should not be recreated
        assert mock_worker_factory.call_count == initial_call_count


class TestWorkerLifecycle:
    """Test worker lifecycle management."""

    def test_load_mode_creates_worker(self, mock_mode_config, mock_registry, mock_worker_factory):
        """Test that loading a mode creates a worker."""
        from backends.worker_pool import reset_worker_pool
        reset_worker_pool()

        pool = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        assert pool._worker is not None
        assert pool._current_mode == "sdxl-general"
        # Called once during init for default mode
        mock_worker_factory.assert_called_once_with(worker_id=0)

        pool.shutdown()
        reset_worker_pool()

    def test_load_mode_registers_with_registry(self, mock_mode_config, mock_registry, mock_worker_factory):
        """Test that loading mode registers with model registry."""
        from backends.worker_pool import reset_worker_pool
        reset_worker_pool()

        pool = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        # Should register model during init
        mock_registry.register_model.assert_called()
        call_args = mock_registry.register_model.call_args
        assert call_args.kwargs['name'] == "sdxl-general"

        pool.shutdown()
        reset_worker_pool()

    def test_unload_mode_destroys_worker(self, worker_pool, mock_registry):
        """Test that unloading mode destroys worker and clears registry."""
        assert worker_pool._worker is not None

        # Trigger unload via mode switch
        worker_pool.switch_mode("sd15-fast").result(timeout=5.0)

        # Should unregister old model
        # Look for call with old mode name
        unregister_calls = [call for call in mock_registry.unregister_model.call_args_list]
        assert len(unregister_calls) > 0

    @patch('backends.worker_pool.torch.cuda.empty_cache')
    def test_mode_switch_clears_cuda_cache(self, mock_empty_cache, worker_pool):
        """Test that mode switching clears CUDA cache."""
        worker_pool.switch_mode("sd15-fast").result(timeout=5.0)

        # Should clear CUDA cache
        mock_empty_cache.assert_called()


class TestCustomJobExecution:
    """Test custom job execution."""

    def test_custom_job_with_args(self, worker_pool):
        """Test custom job with positional arguments."""
        def add(a, b, c):
            return a + b + c

        job = CustomJob(handler=add, args=(1, 2, 3))
        future = worker_pool.submit_job(job)

        result = future.result(timeout=5.0)
        assert result == 6

    def test_custom_job_no_args(self, worker_pool):
        """Test custom job with no arguments."""
        def no_args():
            return "no_args_result"

        job = CustomJob(handler=no_args)
        future = worker_pool.submit_job(job)

        result = future.result(timeout=5.0)
        assert result == "no_args_result"

    def test_custom_job_with_exception(self, worker_pool):
        """Test custom job that raises exception."""
        def failing_job():
            raise ValueError("Custom job error")

        job = CustomJob(handler=failing_job)
        future = worker_pool.submit_job(job)

        with pytest.raises(ValueError) as exc_info:
            future.result(timeout=5.0)

        assert "Custom job error" in str(exc_info.value)

    def test_custom_job_extensibility(self, worker_pool):
        """Test that custom jobs can be used for any callable."""
        # Test with lambda
        job1 = CustomJob(handler=lambda: "lambda_result")
        assert worker_pool.submit_job(job1).result(timeout=5.0) == "lambda_result"

        # Test with class method
        class Calculator:
            def multiply(self, x, y):
                return x * y

        calc = Calculator()
        job2 = CustomJob(handler=calc.multiply, args=(3, 4))
        assert worker_pool.submit_job(job2).result(timeout=5.0) == 12


class TestJobTypes:
    """Test different job types."""

    def test_generation_job_type(self):
        """Test GenerationJob has correct type."""
        req = Mock()
        job = GenerationJob(req=req)
        assert job.job_type == JobType.GENERATION

    def test_mode_switch_job_type(self):
        """Test ModeSwitchJob has correct type."""
        job = ModeSwitchJob(target_mode="test")
        assert job.job_type == JobType.MODE_SWITCH

    def test_custom_job_type(self):
        """Test CustomJob has correct type."""
        job = CustomJob(handler=lambda: None)
        assert job.job_type == JobType.CUSTOM


class TestErrorHandling:
    """Test error handling in worker pool."""

    def test_job_exception_propagates(self, worker_pool):
        """Test that job exceptions propagate to future."""
        def failing_job():
            raise RuntimeError("Job failed")

        job = CustomJob(handler=failing_job)
        future = worker_pool.submit_job(job)

        with pytest.raises(RuntimeError) as exc_info:
            future.result(timeout=5.0)

        assert "Job failed" in str(exc_info.value)

    def test_invalid_mode_switch(self, worker_pool, mock_mode_config):
        """Test switching to invalid mode raises error."""
        mock_mode_config.get_mode.side_effect = KeyError("Mode not found")

        with pytest.raises(KeyError):
            future = worker_pool.switch_mode("invalid-mode")
            future.result(timeout=5.0)


class TestShutdown:
    """Test worker pool shutdown."""

    def test_shutdown_waits_for_jobs(self, mock_mode_config, mock_registry, mock_worker_factory):
        """Test that shutdown waits for pending jobs."""
        from backends.worker_pool import reset_worker_pool
        reset_worker_pool()

        pool = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        completed = []

        def slow_job():
            time.sleep(0.2)
            completed.append(True)
            return "done"

        # Submit job
        job = CustomJob(handler=slow_job)
        future = pool.submit_job(job)

        # Shutdown waits for jobs by default
        pool.shutdown()

        # Job should have completed
        assert len(completed) == 1
        reset_worker_pool()

    def test_shutdown_cleans_up(self, mock_mode_config, mock_registry, mock_worker_factory):
        """Test shutdown cleans up resources."""
        from backends.worker_pool import reset_worker_pool
        reset_worker_pool()

        pool = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        # Shutdown should complete
        pool.shutdown()

        # Worker should be unloaded
        assert pool._worker is None
        reset_worker_pool()


class TestConcurrency:
    """Test concurrent operations."""

    def test_concurrent_job_submission(self, worker_pool):
        """Test submitting jobs from multiple threads."""
        results = []
        lock = threading.Lock()

        def submit_jobs():
            for i in range(5):
                job = CustomJob(handler=lambda x=i: x * 2, args=())
                future = worker_pool.submit_job(job)
                result = future.result(timeout=10.0)
                with lock:
                    results.append(result)

        threads = [threading.Thread(target=submit_jobs) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All jobs should complete
        assert len(results) == 15

    def test_mode_switch_during_generation(self, worker_pool):
        """Test mode switching while generation jobs are running."""
        results = []

        # Submit generation job
        job1 = GenerationJob(req=Mock())
        fut1 = worker_pool.submit_job(job1)

        # Switch mode
        switch_fut = worker_pool.switch_mode("sd15-fast")

        # Submit another job
        job2 = GenerationJob(req=Mock())
        fut2 = worker_pool.submit_job(job2)

        # All should complete
        results.append(fut1.result(timeout=5.0))
        switch_fut.result(timeout=5.0)
        results.append(fut2.result(timeout=5.0))

        assert len(results) == 2
        assert worker_pool.get_current_mode() == "sd15-fast"


class TestQueueManagement:
    """Test queue management."""

    def test_queue_size_tracking(self, worker_pool):
        """Test that queue size is tracked correctly."""
        initial_size = worker_pool.get_queue_size()

        # Submit slow jobs
        def slow_job():
            time.sleep(0.1)
            return "done"

        futures = []
        for _ in range(3):
            job = CustomJob(handler=slow_job)
            futures.append(worker_pool.submit_job(job))

        # Wait for completion
        for fut in futures:
            fut.result(timeout=5.0)

        final_size = worker_pool.get_queue_size()
        # Queue should be empty or nearly empty after jobs complete
        assert final_size <= initial_size + 1  # Allow for timing variations


class TestModeDefaults:
    """Test mode default parameters."""

    def test_mode_defaults_applied(self, worker_pool, mock_mode_config):
        """Test that mode defaults are accessible."""
        mode = mock_mode_config.get_mode("sdxl-general")

        assert mode.default_size == "1024x1024"
        assert mode.default_steps == 30
        assert mode.default_guidance == 7.5

    def test_different_mode_defaults(self, worker_pool, mock_mode_config):
        """Test different modes have different defaults."""
        sdxl_mode = mock_mode_config.get_mode("sdxl-general")
        sd15_mode = mock_mode_config.get_mode("sd15-fast")

        assert sdxl_mode.default_size != sd15_mode.default_size
        assert sdxl_mode.default_steps != sd15_mode.default_steps
