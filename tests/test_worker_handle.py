"""InProcessWorkerHandle isolation tests.

Proves the handle can start a worker, run a job through the backplane, and
tear down — without the Governor. The handle drives the JobSink; the test
attaches a _FutureBridge to the returned Publisher.
"""
import sys
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import Future

import pytest

# Mock torch just long enough to import (same pattern as test_worker_pool.py)
_MOCKED_MODULES = ['torch', 'torch.cuda', 'diffusers']
_saved_modules = {k: sys.modules.get(k) for k in _MOCKED_MODULES}
for _mod in _MOCKED_MODULES:
    sys.modules[_mod] = MagicMock()
# health() reads driver-truth VRAM via torch.cuda.mem_get_info() -> (free, total);
# model the real 2-tuple API on the stub so unpacking doesn't yield 0 values.
sys.modules['torch'].cuda.is_available.return_value = True
sys.modules['torch'].cuda.mem_get_info.return_value = (0, 0)

from backends.worker_handle import InProcessWorkerHandle, WorkerHealth
from backends.governor import GenerationJob, _FutureBridge
from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob
from backends.conditioning.contracts import ConditioningConfig

for _mod, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _orig


def _make_mock_worker(result="test_result"):
    """Build a mock PipelineWorker whose run_job returns a fixed result."""
    worker = Mock()
    worker.run_job = Mock(return_value=result)
    worker.configure_conditioning = None
    return worker


def _make_mode():
    """A mode with an empty (default) ConditioningConfig so start()'s conditioning
    guard is satisfied without the worker needing configure_conditioning — matches
    _load_mode with a non-configurable mode."""
    mode = Mock()
    mode.conditioning = ConditioningConfig()
    return mode


def _make_handle(worker=None):
    """Build an InProcessWorkerHandle with a mock factory."""
    if worker is None:
        worker = _make_mock_worker()
    factory = Mock(return_value=worker)
    handle = InProcessWorkerHandle(worker_factory=factory)
    return handle, worker


def test_handle_starts_worker():
    """start() provisions the worker via the factory."""
    handle, worker = _make_handle()
    resolved = Mock()
    binding = Mock()
    mode = _make_mode()
    handle.start(resolved, binding, mode)
    assert handle._worker is worker
    assert handle.health().state == "ready"


def test_handle_submit_drives_backplane_and_returns_publisher():
    """submit(job) opens a JobSink, runs job.execute(worker), emits result+
    complete, and returns a Publisher the caller subscribes to."""
    handle, worker = _make_handle(_make_mock_worker(result="png_bytes"))
    handle.start(Mock(), Mock(), _make_mode())

    job = GenerationJob(req=Mock(), resolution_epoch=0)

    publisher = handle.submit(job)
    fut = Future()
    publisher.subscribe(_FutureBridge(fut))
    result = fut.result(timeout=2.0)
    assert result == "png_bytes"


def test_handle_submit_emits_error_on_job_failure():
    """If job.execute raises, the handle emits sink.error and the Future
    gets the exception."""
    worker = _make_mock_worker()
    boom = RuntimeError("backend exploded")
    worker.run_job = Mock(side_effect=boom)
    handle, _ = _make_handle(worker=worker)
    handle.start(Mock(), Mock(), _make_mode())

    job = GenerationJob(req=Mock(), resolution_epoch=0)
    publisher = handle.submit(job)
    fut = Future()
    publisher.subscribe(_FutureBridge(fut))
    with pytest.raises(RuntimeError) as ei:
        fut.result(timeout=2.0)
    assert ei.value is boom  # live instance preserved


def test_handle_unload_frees_worker():
    """unload() drops the worker reference and calls empty_cache."""
    handle, worker = _make_handle()
    handle.start(Mock(), Mock(), _make_mode())
    assert handle._worker is not None
    handle.unload()
    assert handle._worker is None
    assert handle.health().state == "dead"


def test_handle_stop_same_as_unload_in_proc():
    """In v1 (in-proc), stop() is the same as unload()."""
    handle, _ = _make_handle()
    handle.start(Mock(), Mock(), _make_mode())
    handle.stop()
    assert handle._worker is None


def test_handle_health_reports_busy_during_job():
    """health() reports 'busy' while a job is running."""
    # This test uses a slow worker to observe the busy state
    worker = _make_mock_worker()
    done_event = threading.Event()
    def slow_run_job(job):
        done_event.wait(timeout=2.0)
        return "done"
    worker.run_job = slow_run_job
    handle, _ = _make_handle(worker=worker)
    handle.start(Mock(), Mock(), _make_mode())

    job = GenerationJob(req=Mock(), resolution_epoch=0)
    publisher = handle.submit(job)
    fut = Future()
    publisher.subscribe(_FutureBridge(fut))

    # While the job runs, health should be busy
    time.sleep(0.1)
    assert handle.health().state == "busy"

    done_event.set()
    assert fut.result(timeout=2.0) == "done"
    assert handle.health().state == "ready"


def test_health_reports_driver_truth_vram_fields():
    """WorkerHealth exposes driver-truth free/total VRAM (mem_get_info), not the
    torch-allocator vram_bytes (spec §8.1, aligning with STABL-sqqlkmdl)."""
    handle, _ = _make_handle()
    handle.start(Mock(), Mock(), _make_mode())
    h = handle.health()
    assert hasattr(h, "vram_free_bytes")
    assert hasattr(h, "vram_total_bytes")
    assert not hasattr(h, "vram_bytes")
    assert isinstance(h.vram_free_bytes, int)
    assert isinstance(h.vram_total_bytes, int)


# --- Task 7: InProcessWorkerHandle injects DeviceMemory + owns Registration ---
from backends.device_memory import DeviceMemorySnapshot, MemoryTopology


def _stub_dm():
    dm = Mock()
    dm.available_for_load.return_value = 20 * 1024**3
    dm.cached_snapshot.return_value = DeviceMemorySnapshot(
        device_uuid="GPU-test", topology=MemoryTopology.DISCRETE,
        total_bytes=24 * 1024**3, free_bytes=20 * 1024**3, consumers=(),
    )
    return dm


def test_health_reads_device_memory_not_torch():
    dm = _stub_dm()
    h = InProcessWorkerHandle(lambda **kw: object(), device_memory=dm)
    health = h.health()
    assert health.vram_free_bytes == 20 * 1024**3
    assert health.vram_total_bytes == 24 * 1024**3


def test_start_registers_consumer_unload_closes():
    dm = _stub_dm()
    registration = Mock()
    dm.register.return_value = registration
    h = InProcessWorkerHandle(lambda **kw: object(), device_memory=dm)

    class _Mode:
        conditioning = Mock(requires_configurable_worker=lambda: False)

    h.start(resolved_mode=Mock(), binding=Mock(), mode=_Mode())
    dm.register.assert_called_once()
    consumer = dm.register.call_args.args[0]
    assert consumer.label == "worker"

    h.unload()
    registration.close.assert_called_once()


def test_double_unload_closes_once():
    dm = _stub_dm()
    registration = Mock()
    dm.register.return_value = registration
    h = InProcessWorkerHandle(lambda **kw: object(), device_memory=dm)

    class _Mode:
        conditioning = Mock(requires_configurable_worker=lambda: False)

    h.start(resolved_mode=Mock(), binding=Mock(), mode=_Mode())
    h.unload()
    h.unload()
    registration.close.assert_called_once()  # handle guards re-close
