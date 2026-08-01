"""Production wiring for WORKER_ISOLATION=subprocess (M-A)."""
from unittest.mock import Mock

from backends.worker_handle_subprocess import SubprocessWorkerHandle
from backends.worker_pool import get_worker_pool, reset_worker_pool


def _minimal_deps():
    registry = Mock()
    registry.get_total_vram.return_value = 0
    registry.get_used_vram.return_value = 0
    registry.get_allocated_vram.return_value = 0
    registry.register_model = Mock()
    registry.unregister_model = Mock()

    mode_config = Mock()
    mode_config.get_default_mode.return_value = "none"
    mode_config.get_mode.side_effect = KeyError("no mode")

    return {"registry": registry, "mode_config": mode_config}


def test_worker_isolation_subprocess_env_creates_subprocess_handle(monkeypatch):
    """WORKER_ISOLATION=subprocess makes get_worker_pool inject a
    SubprocessWorkerHandle backed by the real CudaWorker factory."""
    monkeypatch.setenv("WORKER_ISOLATION", "subprocess")
    monkeypatch.setenv("BACKEND", "cuda")
    reset_worker_pool()
    try:
        pool = get_worker_pool(**_minimal_deps())
        assert isinstance(pool._governor._handle, SubprocessWorkerHandle)
        assert pool._governor._handle._factory_ref == "backends.worker_factory.create_cuda_worker"
    finally:
        reset_worker_pool()
        monkeypatch.delenv("WORKER_ISOLATION", raising=False)


def test_worker_isolation_default_uses_inproc(monkeypatch):
    """Without WORKER_ISOLATION=subprocess the pool keeps the in-proc handle."""
    monkeypatch.delenv("WORKER_ISOLATION", raising=False)
    monkeypatch.setenv("BACKEND", "cuda")
    reset_worker_pool()
    try:
        pool = get_worker_pool(**_minimal_deps())
        from backends.worker_handle import InProcessWorkerHandle
        assert isinstance(pool._governor._handle, InProcessWorkerHandle)
    finally:
        reset_worker_pool()


# --- per-process VRAM attribution wiring (STABL-xtkhoidu) --------------------


def test_subprocess_handle_registers_the_child_as_a_consumer_and_deregisters_on_stop():
    """Symmetry with InProcessWorkerHandle, which has always registered in start()
    and closed in unload(). Without this the subprocess path registers NOTHING and
    100% of worker VRAM lands in unattributed_bytes."""
    from unittest.mock import MagicMock
    from backends.worker_handle_subprocess import SubprocessWorkerHandle

    dm = MagicMock()
    registration = MagicMock()
    dm.register.return_value = registration

    h = SubprocessWorkerHandle(
        "tests._fault_worker.make_fault_worker", device_memory=dm
    )
    h.start(None, None, None)
    try:
        assert dm.register.called, "the child was never registered with DeviceMemory"
        consumer = dm.register.call_args[0][0]
        assert consumer.label == "worker"
    finally:
        h.stop()
    assert registration.close.called, "stop() left a dead child registered"


def test_subprocess_mode_registers_the_parent_process_separately(monkeypatch):
    """In subprocess mode the parent still holds GPU memory of its own (superres),
    and the child holds the worker. Two processes, two consumers, no overlap.

    In INPROC mode the parent must NOT get a second consumer: it is the same
    process as the worker, and two consumers reporting the same process-global
    torch counters double-count.
    """
    from backends.device_memory import ProcessMemoryConsumer

    monkeypatch.setenv("WORKER_ISOLATION", "subprocess")
    monkeypatch.setenv("BACKEND", "cuda")
    reset_worker_pool()
    try:
        pool = get_worker_pool(**_minimal_deps())
        labels = [
            c.label for c in getattr(pool, "_parent_consumers", [])
            if isinstance(c, ProcessMemoryConsumer)
        ]
        assert "server" in labels, (
            "subprocess mode did not register a parent-process consumer, so superres "
            "VRAM stays in unattributed_bytes"
        )
    finally:
        reset_worker_pool()


def test_inproc_mode_does_not_add_a_second_parent_consumer(monkeypatch):
    """The double-count guard, at the wiring level."""
    monkeypatch.delenv("WORKER_ISOLATION", raising=False)
    monkeypatch.setenv("BACKEND", "cuda")
    reset_worker_pool()
    try:
        pool = get_worker_pool(**_minimal_deps())
        assert not getattr(pool, "_parent_consumers", []), (
            "inproc mode registered a parent consumer alongside the in-proc worker "
            "consumer — the same process counted twice"
        )
    finally:
        reset_worker_pool()
