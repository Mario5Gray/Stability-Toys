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
