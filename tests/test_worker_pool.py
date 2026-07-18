"""
Functional tests for WorkerPool.

Tests job queue, mode switching, and worker lifecycle management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import concurrent.futures
from concurrent.futures import Future
import queue
import time
import threading
import sys
from types import SimpleNamespace

from backends.conditioning.contracts import ConditioningConfig

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
    mode_sdxl.loader_format = "single_file"
    mode_sdxl.checkpoint_precision = "fp8"
    mode_sdxl.checkpoint_variant = "sdxl-base"
    mode_sdxl.scheduler_profile = "native"
    mode_sdxl.recommended_size = "512x512"
    mode_sdxl.runtime_quantize = "none"
    mode_sdxl.runtime_offload = "model"
    mode_sdxl.runtime_attention_slicing = True
    mode_sdxl.runtime_enable_xformers = True
    mode_sdxl.negative_prompt_templates = {"safe_photo": "blurry, watermark"}
    mode_sdxl.default_negative_prompt_template = "safe_photo"
    mode_sdxl.allow_custom_negative_prompt = True
    mode_sdxl.allowed_scheduler_ids = ["euler", "dpmpp_2m"]
    mode_sdxl.default_scheduler_id = "euler"
    mode_sdxl.metadata = {"single_file_config": "configs/sdxl-base"}
    mode_sdxl.conditioning = ConditioningConfig()

    mode_sd15 = Mock()
    mode_sd15.name = "sd15-fast"
    mode_sd15.model = "sd15.safetensors"
    mode_sd15.model_path = "/models/sd15.safetensors"
    mode_sd15.loras = []
    mode_sd15.default_size = "512x512"
    mode_sd15.default_steps = 4
    mode_sd15.default_guidance = 1.0
    mode_sd15.loader_format = None
    mode_sd15.checkpoint_precision = None
    mode_sd15.checkpoint_variant = None
    mode_sd15.scheduler_profile = None
    mode_sd15.recommended_size = None
    mode_sd15.runtime_quantize = None
    mode_sd15.runtime_offload = None
    mode_sd15.runtime_attention_slicing = None
    mode_sd15.runtime_enable_xformers = None
    mode_sd15.negative_prompt_templates = {}
    mode_sd15.default_negative_prompt_template = None
    mode_sd15.allow_custom_negative_prompt = False
    mode_sd15.allowed_scheduler_ids = None
    mode_sd15.default_scheduler_id = None
    mode_sd15.metadata = {}
    mode_sd15.conditioning = ConditioningConfig()

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
    registry.get_allocated_vram.return_value = 0
    registry.get_total_vram.return_value = 8 * 1024**3
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
    worker.configure_conditioning = None

    factory = Mock()
    factory.return_value = worker
    return factory


@pytest.fixture(autouse=True)
def mock_cuda_runtime():
    """Provide numeric CUDA stats for worker-pool logging paths."""
    fake_oom = type("FakeOutOfMemoryError", (RuntimeError,), {})
    with patch("backends.worker_pool.torch.cuda.is_available", return_value=True), \
         patch("backends.worker_pool.torch.cuda.memory_allocated", return_value=0), \
         patch("backends.worker_pool.torch.cuda.memory_reserved", return_value=0), \
         patch("backends.worker_pool.torch.cuda.OutOfMemoryError", new=fake_oom), \
        patch("backends.worker_pool.torch.cuda.empty_cache"):
        yield


def _detected_info(path: str):
    """Lightweight detector output for worker-pool tests (base_arch set so the
    neutral family resolver matches sd15/sdxl)."""
    from utils.model_detector import ModelInfo, ModelVariant

    if "sdxl" in path:
        return ModelInfo(
            path=path,
            variant=ModelVariant.SDXL_BASE,
            cross_attention_dim=2048,
            base_arch="unet",
            confidence=0.95,
            loader_format="single_file",
            checkpoint_precision="unknown",
            checkpoint_variant="sdxl-base",
            scheduler_profile="native",
        )
    return ModelInfo(
        path=path,
        variant=ModelVariant.SD15,
        cross_attention_dim=768,
        base_arch="unet",
        confidence=0.95,
        loader_format="single_file",
        checkpoint_precision="unknown",
        checkpoint_variant="sd15",
        scheduler_profile="lcm",
    )


@pytest.fixture(autouse=True)
def mock_model_detection():
    """Patch the pool's single resolve_model seam.

    The pool now resolves once via resolve_model; patching that keeps tests off
    the filesystem (no real artifact-ref fingerprinting) while still exercising
    real family resolution and the mode-capability overlay.
    """
    from backends.family_profiles import resolve_family
    from backends.model_resolution import (
        LocalModelBinding,
        build_resolved,
        hub_ref,
        merge_mode_capabilities,
    )

    def _resolve(model_path: str, mode):
        raw = _detected_info(model_path)
        enriched = merge_mode_capabilities(raw, mode)
        resolved = build_resolved(
            model_ref=hub_ref("test/repo", None),
            raw_info=raw,
            profile=resolve_family(raw),
            info=enriched,
        )
        return resolved, LocalModelBinding(model_path)

    with patch("backends.worker_pool.resolve_model", side_effect=_resolve):
        yield


@pytest.fixture
def worker_pool(mock_mode_config, mock_registry, mock_worker_factory):
    """Create WorkerPool with mocked dependencies using DI."""
    from backends.worker_pool import reset_worker_pool
    reset_worker_pool()  # Ensure clean state

    with patch("backends.worker_pool.torch.cuda.is_available", return_value=True), \
         patch("backends.worker_pool.torch.cuda.memory_allocated", return_value=0), \
         patch("backends.worker_pool.torch.cuda.memory_reserved", return_value=0), \
         patch("backends.worker_pool.torch.cuda.empty_cache"):
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
        kwargs = mock_worker_factory.call_args.kwargs
        assert kwargs["worker_id"] == 0
        assert kwargs["binding"].model_path == "/models/sdxl.safetensors"
        resolved = kwargs["resolved"]
        assert resolved.profile.family_id == "sdxl"
        # info is the mode-overlaid snapshot (mode wins).
        assert resolved.info.loader_format == "single_file"
        assert resolved.info.checkpoint_precision == "fp8"
        assert resolved.info.scheduler_profile == "native"

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

    def test_overlay_ignores_non_mapping_detected_metadata(self, mock_mode_config):
        """Non-mapping detected metadata must not break the mode overlay; the
        mode's metadata wins. (The pool resolves via merge_mode_capabilities.)"""
        from backends.model_resolution import merge_mode_capabilities
        from utils.model_detector import ModelInfo, ModelVariant

        detected = ModelInfo(
            path="/models/sdxl.safetensors",
            variant=ModelVariant.SDXL_BASE,
            cross_attention_dim=2048,
            base_arch="unet",
        )
        detected.metadata = Mock()  # non-mapping

        resolved = merge_mode_capabilities(detected, mock_mode_config.get_mode("sdxl-general"))

        assert resolved.metadata == {"single_file_config": "configs/sdxl-base"}


class TestJobSubmission:
    """Test job submission and execution."""

    def test_submit_generation_job(self, worker_pool):
        """Test submitting a generation job."""
        req = Mock()
        job = _gen_job(worker_pool, req=req)

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
            job = _gen_job(worker_pool, req=req)
            future = worker_pool.submit_job(job)
            jobs.append(future)

        # All jobs should complete
        for future in jobs:
            result = future.result(timeout=10.0)
            assert result == "test_result"

    def test_submit_generation_job_clears_record_after_success(self, worker_pool):
        req = Mock()
        job = _gen_job(worker_pool, req=req, job_id="job-success")
        fut = worker_pool.submit_job(job)

        assert fut.result(timeout=5.0) == "test_result"
        assert worker_pool._get_job_record("job-success") is None

    def test_submit_generation_job_clears_record_after_failure(
        self,
        worker_pool,
        mock_worker_factory,
    ):
        worker = mock_worker_factory.return_value
        worker.run_job.side_effect = ValueError("boom")

        req = Mock()
        job = _gen_job(worker_pool, req=req, job_id="job-fail")
        fut = worker_pool.submit_job(job)

        with pytest.raises(ValueError):
            fut.result(timeout=5.0)

        assert worker_pool._get_job_record("job-fail") is None

    def test_submit_generation_job_clears_record_on_queue_full(self, worker_pool):
        req = Mock()
        job = _gen_job(worker_pool, req=req, job_id="job-full")

        with patch.object(worker_pool.q, "put", side_effect=queue.Full):
            with pytest.raises(queue.Full):
                worker_pool.submit_job(job)

        assert worker_pool._get_job_record("job-full") is None

    def test_submit_generation_job_uses_blocking_queue_put_when_timeout_requested(self, worker_pool):
        req = Mock()
        job = _gen_job(worker_pool, req=req, job_id="job-timeout")

        with patch.object(worker_pool.q, "put") as put, \
             patch.object(worker_pool.q, "put_nowait") as put_nowait:
            worker_pool.submit_job(job, timeout_s=0.25)

        put.assert_called_once_with(job, timeout=0.25)
        put_nowait.assert_not_called()

    def test_submit_generation_job_uses_pool_default_timeout_when_override_omitted(
        self,
        mock_mode_config,
        mock_registry,
        mock_worker_factory,
    ):
        from backends.worker_pool import reset_worker_pool

        reset_worker_pool()
        pool = WorkerPool(
            queue_max=10,
            queue_timeout_s=0.75,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        try:
            job = _gen_job(pool, req=Mock(), job_id="job-default-timeout")
            with patch.object(pool.q, "put") as put, \
                 patch.object(pool.q, "put_nowait") as put_nowait:
                pool.submit_job(job)

            put.assert_called_once_with(job, timeout=0.75)
            put_nowait.assert_not_called()
        finally:
            pool.shutdown()
            reset_worker_pool()

    def test_submit_generation_job_uses_put_nowait_when_timeout_override_zero(self, worker_pool):
        job = _gen_job(worker_pool, req=Mock(), job_id="job-nowait")

        with patch.object(worker_pool.q, "put") as put, \
             patch.object(worker_pool.q, "put_nowait") as put_nowait:
            worker_pool.submit_job(job, timeout_s=0)

        put.assert_not_called()
        put_nowait.assert_called_once_with(job)

    def test_oom_cancels_pending_generation_jobs_and_unloads_worker(
        self,
        worker_pool,
        mock_worker_factory,
    ):
        import backends.worker_pool as worker_pool_module

        fake_oom = worker_pool_module.torch.cuda.OutOfMemoryError
        worker = mock_worker_factory.return_value
        worker.run_job.side_effect = fake_oom("CUDA out of memory")

        first_future = worker_pool.submit_job(_gen_job(worker_pool, req=Mock(), job_id="job-1"))
        queued_future = worker_pool.submit_job(_gen_job(worker_pool, req=Mock(), job_id="job-2"))

        with pytest.raises(fake_oom):
            first_future.result(timeout=1.0)

        assert queued_future.cancelled()
        assert worker_pool.is_model_loaded() is False

    def test_free_vram_cancels_running_and_queued_generation_jobs(
        self,
        worker_pool,
        mock_worker_factory,
    ):
        started = threading.Event()
        release = threading.Event()

        def blocking_run_job(job):
            started.set()
            release.wait(timeout=5.0)
            return "blocked"

        worker = mock_worker_factory.return_value
        worker.run_job.side_effect = blocking_run_job

        running_future = worker_pool.submit_job(_gen_job(worker_pool, req=Mock(), job_id="job-running"))
        assert started.wait(timeout=1.0)

        queued_future = worker_pool.submit_job(_gen_job(worker_pool, req=Mock(), job_id="job-queued"))
        status = worker_pool.free_vram("manual_free_vram")

        assert status["status"] == "ok"
        assert set(status["cancelled_jobs"]) == {"job-running", "job-queued"}
        assert queued_future.cancelled()
        assert worker_pool.is_model_loaded() is False

        release.set()
        with pytest.raises(concurrent.futures.CancelledError):
            running_future.result(timeout=5.0)

        assert worker_pool._get_job_record("job-running") is None
        assert worker_pool._get_job_record("job-queued") is None

    def test_cancel_queued_generation_job_marks_future_cancelled(
        self,
        worker_pool,
        mock_worker_factory,
    ):
        blocker_started = threading.Event()
        release = threading.Event()

        def blocking_handler():
            blocker_started.set()
            release.wait(timeout=5.0)
            return "blocked"

        blocker = CustomJob(handler=blocking_handler)
        blocker_future = worker_pool.submit_job(blocker)
        assert blocker_started.wait(timeout=1.0)

        worker = mock_worker_factory.return_value
        worker.run_job.reset_mock()

        req = Mock()
        job = _gen_job(worker_pool, req=req, job_id="job-1")
        fut = worker_pool.submit_job(job)
        assert worker_pool.cancel_job("job-1") is True
        assert fut.cancelled()
        assert worker_pool._get_job_record("job-1") is not None
        release.set()
        assert blocker_future.result(timeout=1.0) == "blocked"
        worker_pool.q.join()
        assert worker.run_job.call_count == 0
        assert worker_pool._get_job_record("job-1") is None

    def test_cancel_running_generation_job_discards_late_result(
        self,
        worker_pool,
        mock_worker_factory,
    ):
        started = threading.Event()
        release = threading.Event()
        worker = mock_worker_factory.return_value
        def run_job(job):
            started.set()
            release.wait(timeout=5.0)
            return ("png", 123)

        worker.run_job.side_effect = run_job
        req = Mock()
        fut = worker_pool.submit_job(_gen_job(worker_pool, req=req, job_id="job-2"))
        assert started.wait(timeout=1.0)
        assert worker_pool.cancel_job("job-2") is True
        release.set()
        with pytest.raises(concurrent.futures.CancelledError):
            fut.result(timeout=1.0)
        assert worker_pool._get_job_record("job-2") is None

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

        # Submit mode switch (queues behind the slow job)
        switch_fut = worker_pool.switch_mode("sd15-fast")

        # Job1 then the switch drain in order; the switch installs a new epoch.
        results.append(fut1.result(timeout=5.0))
        switch_fut.result(timeout=5.0)

        # A generation job submitted AFTER the switch is stamped with the new
        # epoch and runs on the new mode. (A job stamped before the switch would
        # be correctly rejected as stale — see TestActiveModelSnapshot.)
        job2 = _gen_job(worker_pool, req=Mock())
        fut2 = worker_pool.submit_job(job2)
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

    def test_reload_current_mode_recreates_worker(self, worker_pool, mock_worker_factory):
        """Test explicit reload of the current mode."""
        initial_call_count = mock_worker_factory.call_count

        result = worker_pool.reload_current_mode()

        assert result == {"status": "reloaded", "mode": "sdxl-general"}
        assert worker_pool.get_current_mode() == "sdxl-general"
        assert worker_pool.is_model_loaded() is True
        assert mock_worker_factory.call_count == initial_call_count + 1


class TestWorkerLifecycle:
    """Test worker lifecycle management."""

    def test_load_mode_configures_conditioning_before_registration_and_thread(
        self,
        mock_mode_config,
        mock_registry,
    ):
        events = []

        class ConfigurableWorker:
            worker_id = 0

            def configure_conditioning(self, config):
                events.append(("configure", config.service))

            def run_job(self, job):
                del job
                return b"png", 1

        mode = mock_mode_config.get_mode("sdxl-general")
        mode.conditioning = ConditioningConfig(service="compel")
        mock_registry.register_model.side_effect = lambda **kwargs: events.append(
            ("register", kwargs["name"])
        )

        with patch.object(
            WorkerPool,
            "_start_worker_thread",
            autospec=True,
            side_effect=lambda pool: events.append(("thread", pool._current_mode)),
        ), patch.object(WorkerPool, "_start_watchdog_thread", return_value=None):
            pool = WorkerPool(
                worker_factory=Mock(return_value=ConfigurableWorker()),
                mode_config=mock_mode_config,
                registry=mock_registry,
            )

        assert events == [
            ("configure", "compel"),
            ("register", "sdxl-general"),
            ("thread", "sdxl-general"),
        ]
        pool.shutdown()

    def test_non_native_config_rejects_worker_without_conditioning_capability(
        self,
        mock_mode_config,
        mock_registry,
        caplog,
    ):
        class IncompatibleWorker:
            worker_id = 0

            def run_job(self, job):
                del job
                return b"png", 1

        mode = mock_mode_config.get_mode("sdxl-general")
        mode.conditioning = ConditioningConfig(service="compel")
        factory = Mock(return_value=IncompatibleWorker())

        pool = WorkerPool(
            worker_factory=factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        assert pool._worker is None
        assert pool._current_mode is None
        assert "does not support conditioning" in caplog.text
        with pytest.raises(RuntimeError, match="does not support conditioning"):
            pool._load_mode("sdxl-general")
        pool.shutdown()

    def test_explicit_native_config_accepts_worker_without_conditioning_capability(
        self,
        mock_mode_config,
        mock_registry,
    ):
        class IncompatibleWorker:
            worker_id = 0

            def run_job(self, job):
                del job
                return b"png", 1

        mode = mock_mode_config.get_mode("sdxl-general")
        mode.conditioning = ConditioningConfig(service="native")
        pool = WorkerPool(
            worker_factory=Mock(return_value=IncompatibleWorker()),
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        assert pool._worker is not None
        assert pool._current_mode == "sdxl-general"
        pool.shutdown()

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
        kwargs = mock_worker_factory.call_args.kwargs
        assert kwargs["worker_id"] == 0
        assert kwargs["binding"].model_path == "/models/sdxl.safetensors"
        assert kwargs["resolved"].info.checkpoint_variant == "sdxl-base"

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

    def test_load_mode_registers_allocator_delta_not_reserved_delta(
        self,
        mock_mode_config,
        mock_registry,
        mock_worker_factory,
    ):
        """Model registration should use allocator growth even when reserved bytes stay flat."""
        from backends.worker_pool import reset_worker_pool

        reset_worker_pool()
        mock_registry.get_used_vram.side_effect = [5 * 1024**3, 5 * 1024**3]
        mock_registry.get_allocated_vram.side_effect = [1 * 1024**3, 3 * 1024**3]

        pool = WorkerPool(
            queue_max=10,
            worker_factory=mock_worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        call_args = mock_registry.register_model.call_args
        assert call_args.kwargs["vram_bytes"] == 2 * 1024**3

        pool.shutdown()
        reset_worker_pool()

    def test_load_mode_merges_capabilities_before_worker_creation(
        self,
        mock_mode_config,
        mock_registry,
        mock_worker_factory,
    ):
        """Mode overrides should become authoritative ModelInfo before factory dispatch."""
        from backends.worker_pool import reset_worker_pool
        from utils.model_detector import ModelInfo, ModelVariant

        from backends.family_profiles import resolve_family
        from backends.model_resolution import (
            LocalModelBinding,
            build_resolved,
            hub_ref,
            merge_mode_capabilities,
        )

        reset_worker_pool()
        # Detected values deliberately differ from the mode so the overlay
        # precedence (mode wins) is observable in resolved.info.
        detected = ModelInfo(
            path="/models/sdxl.safetensors",
            variant=ModelVariant.SDXL_BASE,
            cross_attention_dim=2048,
            base_arch="unet",
            confidence=0.95,
            loader_format="unknown",
            checkpoint_precision="unknown",
            checkpoint_variant="unknown",
            scheduler_profile="lcm",
        )

        def _resolve(model_path, mode):
            enriched = merge_mode_capabilities(detected, mode)
            return (
                build_resolved(
                    model_ref=hub_ref("test/repo", None),
                    raw_info=detected,
                    profile=resolve_family(detected),
                    info=enriched,
                ),
                LocalModelBinding(model_path),
            )

        with patch("backends.worker_pool.resolve_model", side_effect=_resolve):
            pool = WorkerPool(
                queue_max=10,
                worker_factory=mock_worker_factory,
                mode_config=mock_mode_config,
                registry=mock_registry,
            )

        info = mock_worker_factory.call_args.kwargs["resolved"].info
        assert info.loader_format == "single_file"
        assert info.checkpoint_precision == "fp8"
        assert info.checkpoint_variant == "sdxl-base"
        assert info.scheduler_profile == "native"
        assert info.recommended_size == "512x512"
        assert info.negative_prompt_templates == {"safe_photo": "blurry, watermark"}
        assert info.default_negative_prompt_template == "safe_photo"
        assert info.allow_custom_negative_prompt is True
        assert list(info.allowed_scheduler_ids) == ["euler", "dpmpp_2m"]
        assert info.default_scheduler_id == "euler"
        assert info.metadata["single_file_config"] == "configs/sdxl-base"

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

    def test_unload_current_model_does_not_cancel_queued_jobs(self, worker_pool):
        """Test that explicit unload only drops the worker."""
        result = worker_pool.unload_current_model()

        assert result["status"] == "unloaded"
        assert worker_pool.is_model_loaded() is False
        assert worker_pool.get_current_mode() == "sdxl-general"


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
        job = GenerationJob(req=req, resolution_epoch=0)
        assert job.job_type == JobType.GENERATION

    def test_generation_job_controlnet_bindings_default_empty_list(self):
        """Generation jobs should always expose a controlnet binding list."""
        req = Mock()
        job = GenerationJob(req=req, resolution_epoch=0)
        assert job.controlnet_bindings == []

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


class TestControlNetRuntime:
    """Test the CUDA runtime seam for ControlNet binding resolution."""

    def test_cuda_runtime_attaches_controlnet_bindings_before_submit(self, mock_mode_config):
        """CUDA runtime should resolve ordered bindings before queueing work."""
        from backends.platforms.cuda import CudaGenerationRuntime

        pool = Mock()
        pool.get_current_mode.return_value = "sdxl-general"
        pool.submit_job.return_value = Future()
        req = SimpleNamespace(controlnets=[SimpleNamespace(attachment_id="cn_1")])
        mode = mock_mode_config.get_mode("sdxl-general")
        store = Mock()
        detected = SimpleNamespace(variant=SimpleNamespace(value="sdxl-base"))

        with patch("server.mode_config.get_mode_config", return_value=mock_mode_config), \
             patch("utils.model_detector.detect_model", return_value=detected), \
             patch("server.asset_store.get_store", return_value=store), \
             patch("server.controlnet_execution.active_model_family_from_variant", return_value="sdxl"), \
             patch("server.controlnet_execution.resolve_controlnet_bindings", return_value=["binding"]) as resolve:
            runtime = CudaGenerationRuntime(pool=pool)
            runtime.submit_generate(req)

        queued_job = pool.submit_job.call_args[0][0]
        assert queued_job.controlnet_bindings == ["binding"]
        resolve.assert_called_once()
        args, kwargs = resolve.call_args
        assert args == (req,)
        assert kwargs == {"mode": mode, "store": store, "active_family": "sdxl"}

    def test_cuda_runtime_forwards_queue_timeout_to_worker_pool(self):
        """CUDA runtime should preserve timeout-aware queueing semantics."""
        from backends.platforms.cuda import CudaGenerationRuntime

        pool = Mock()
        pool.submit_job.return_value = Future()
        req = SimpleNamespace(controlnets=[])

        runtime = CudaGenerationRuntime(pool=pool)
        runtime.submit_generate(req, timeout_s=0.5)

        queued_job = pool.submit_job.call_args[0][0]
        assert queued_job.req is req
        assert pool.submit_job.call_args.kwargs == {"timeout_s": 0.5}

    def test_cuda_runtime_uses_pool_default_queue_timeout_when_override_omitted(self):
        """CUDA runtime should defer to the pool default when no timeout override is given."""
        from backends.platforms.cuda import CudaGenerationRuntime

        pool = Mock()
        pool.submit_job.return_value = Future()
        req = SimpleNamespace(controlnets=[])

        runtime = CudaGenerationRuntime(pool=pool)
        runtime.submit_generate(req)

        queued_job = pool.submit_job.call_args[0][0]
        assert queued_job.req is req
        assert pool.submit_job.call_args.kwargs == {"timeout_s": None}

    def test_cuda_runtime_requires_active_mode_for_controlnet_requests(self):
        """ControlNet requests should fail clearly if no mode is loaded."""
        from backends.platforms.cuda import CudaGenerationRuntime

        pool = Mock()
        pool.get_current_mode.return_value = None
        req = SimpleNamespace(controlnets=[SimpleNamespace(attachment_id="cn_1")])

        runtime = CudaGenerationRuntime(pool=pool)

        with pytest.raises(RuntimeError, match="before any mode was loaded"):
            runtime.submit_generate(req)

        pool.submit_job.assert_not_called()

    def test_backend_capabilities_only_cuda_supports_controlnet(self):
        """Only the CUDA backend binds a family whose execution cell supports
        ControlNet. Execution claims now live on the family-platform binding,
        not on platform-wide BackendCapabilities."""
        from backends.platforms.cpu import CPUProvider
        from backends.platforms.cuda import CUDAProvider
        from backends.platforms.rknn import RKNNProvider

        cuda_binding = CUDAProvider().family_binding("sd15")
        assert cuda_binding is not None
        assert cuda_binding.execution_capabilities.supports_controlnet is True
        assert CPUProvider().family_binding("sd15") is None
        assert RKNNProvider().family_binding("sd15") is None


class TestDefaultFactory:
    """Test the built-in worker factory used in production."""

    def test_default_worker_factory_forwards_resolved_and_binding(self):
        """Default factory forwards the resolved model and its local binding."""
        from backends.worker_pool import WorkerPool

        resolved = Mock()
        binding = Mock()

        with patch("backends.worker_factory.create_cuda_worker") as mock_create:
            WorkerPool._default_worker_factory(0, resolved, binding)

        mock_create.assert_called_once_with(0, resolved, binding)


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
        job1 = _gen_job(worker_pool, req=Mock())
        fut1 = worker_pool.submit_job(job1)

        # Switch mode


class TestOomRecovery:
    def test_oom_unloads_worker_and_next_request_demand_reloads(
        self,
        mock_mode_config,
        mock_registry,
    ):
        """OOM should tear down the current worker so the next job reloads cleanly."""
        from backends.worker_pool import reset_worker_pool

        reset_worker_pool()

        fake_oom = getattr(sys.modules["backends.worker_pool"].torch.cuda, "OutOfMemoryError", RuntimeError)
        first_worker = Mock()
        first_worker.run_job.side_effect = fake_oom("CUDA out of memory")
        second_worker = Mock()
        second_worker.run_job.return_value = "recovered"
        worker_factory = Mock(side_effect=[first_worker, second_worker])

        pool = WorkerPool(
            queue_max=10,
            worker_factory=worker_factory,
            mode_config=mock_mode_config,
            registry=mock_registry,
        )

        try:
            fut1 = pool.submit_job(_gen_job(pool, req=Mock()))
            with pytest.raises(fake_oom):
                fut1.result(timeout=10.0)

            assert pool._worker is None
            assert pool._current_mode == "sdxl-general"

            fut2 = pool.submit_job(_gen_job(pool, req=Mock()))
            assert fut2.result(timeout=10.0) == "recovered"
            assert worker_factory.call_count == 2
            assert pool._worker is second_worker
        finally:
            pool.shutdown()
            reset_worker_pool()


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


def _gen_job(pool, req=None, **kw):
    """Build a GenerationJob stamped with the pool's current resolution epoch."""
    kw.setdefault("resolution_epoch", pool.current_resolution_epoch())
    return GenerationJob(req=req if req is not None else Mock(), **kw)


class TestActiveModelSnapshot:
    """Atomic snapshot publication, epoch discipline, and the stale-job barrier."""

    def test_successful_load_publishes_snapshot_atomically(self, worker_pool):
        snap = worker_pool.get_active_model_snapshot()
        assert snap is not None
        assert snap.mode_name == "sdxl-general"
        assert snap.resolved.profile.family_id == "sdxl"
        assert snap.binding.model_path == "/models/sdxl.safetensors"
        assert snap.resolution_epoch >= 1
        assert worker_pool.current_resolution_epoch() == snap.resolution_epoch
        assert worker_pool._worker is not None

    def test_snapshot_mode_is_isolated_from_source(self, worker_pool, mock_mode_config):
        snap = worker_pool.get_active_model_snapshot()
        # The published mode is a deep copy: mutating the source cannot reach it.
        assert snap.mode is not mock_mode_config.get_mode("sdxl-general")

    def test_failed_load_leaves_no_snapshot_and_no_worker(
        self, mock_mode_config, mock_registry
    ):
        from backends.worker_pool import reset_worker_pool

        reset_worker_pool()
        failing_factory = Mock(side_effect=RuntimeError("worker boom"))
        with patch("backends.worker_pool.torch.cuda.is_available", return_value=True), \
             patch("backends.worker_pool.torch.cuda.memory_allocated", return_value=0), \
             patch("backends.worker_pool.torch.cuda.memory_reserved", return_value=0), \
             patch("backends.worker_pool.torch.cuda.empty_cache"):
            pool = WorkerPool(
                queue_max=10,
                worker_factory=failing_factory,
                mode_config=mock_mode_config,
                registry=mock_registry,
            )
        assert pool.get_active_model_snapshot() is None
        assert pool._worker is None
        pool.shutdown()
        reset_worker_pool()

    def test_idle_eviction_retains_snapshot_and_epoch(self, worker_pool):
        before = worker_pool.get_active_model_snapshot()
        worker_pool._last_activity = time.monotonic() - 10_000  # force idle
        worker_pool._evict_if_idle()
        assert worker_pool._worker is None
        after = worker_pool.get_active_model_snapshot()
        assert after is before
        assert after.resolution_epoch == before.resolution_epoch

    def test_demand_reload_reuses_resolved_without_redetection(self, worker_pool):
        epoch = worker_pool.current_resolution_epoch()
        worker_pool._last_activity = time.monotonic() - 10_000
        worker_pool._evict_if_idle()
        assert worker_pool._worker is None

        # If demand reload re-detected, this patched resolve_model would explode.
        with patch("backends.worker_pool.resolve_model",
                   side_effect=AssertionError("demand reload re-detected")):
            fut = worker_pool.submit_job(_gen_job(worker_pool, resolution_epoch=epoch))
            assert fut.result(timeout=5) == "test_result"

        assert worker_pool._worker is not None
        assert worker_pool.current_resolution_epoch() == epoch  # retained, not bumped

    def test_reresolve_installs_new_epoch(self, worker_pool):
        e1 = worker_pool.current_resolution_epoch()
        worker_pool._load_mode("sd15-fast")
        e2 = worker_pool.current_resolution_epoch()
        assert e2 == e1 + 1
        assert worker_pool.get_active_model_snapshot().mode_name == "sd15-fast"

    def test_stale_generation_job_raises_before_run_job(self, worker_pool):
        from backends.worker_pool import StaleResolutionError

        old_epoch = worker_pool.current_resolution_epoch()
        worker = worker_pool._worker
        worker.run_job.reset_mock()

        worker_pool._load_mode("sd15-fast")  # epoch advances

        stale = _gen_job(worker_pool, resolution_epoch=old_epoch)
        fut = worker_pool.submit_job(stale)
        with pytest.raises(StaleResolutionError):
            fut.result(timeout=5)
        worker.run_job.assert_not_called()
