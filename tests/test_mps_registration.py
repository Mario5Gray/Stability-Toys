import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException


@pytest.fixture
def mps_backend(monkeypatch):
    from backends.platform_registry import reset_backend_provider
    from backends.worker_pool import reset_worker_pool

    monkeypatch.setenv("BACKEND", "mps")
    reset_worker_pool()
    reset_backend_provider()
    yield
    reset_worker_pool()
    reset_backend_provider()


def _minimal_deps():
    mode_config = Mock()
    mode_config.get_default_mode.return_value = "missing"
    mode_config.get_mode.side_effect = KeyError("missing")
    registry = Mock()
    return {"mode_config": mode_config, "registry": registry, "device_memory": Mock()}


def _resolved_sd15():
    from backends.family_profiles import SD15_PROFILE
    from tests.test_worker_factory import _resolved

    return _resolved(SD15_PROFILE)


def test_mps_provider_has_empty_bindings_and_shared_runtime(mps_backend):
    from backends.platform_registry import get_backend_provider
    from backends.platforms.mps_bindings import MPS_FAMILY_BINDINGS
    from backends.platforms.pool_runtime import PoolGenerationRuntime
    from backends.platforms.base import UnsupportedFamilyError

    provider = get_backend_provider()
    assert provider.backend_id == "mps"
    assert MPS_FAMILY_BINDINGS == {}
    assert provider.family_binding("sd15") is None
    assert isinstance(provider.create_generation_runtime(pool=Mock()), PoolGenerationRuntime)
    assert provider.capabilities().supports_generation is False
    assert provider.capabilities().supports_superres is False
    assert provider.create_model_registry().get_vram_stats()["backend"] == "mps"
    with pytest.raises(UnsupportedFamilyError, match="family 'sd15' has no MPS platform binding"):
        provider.create_worker_factory()(
            0,
            SimpleNamespace(profile=SimpleNamespace(family_id="sd15")),
            None,
        )


def test_mps_provider_import_does_not_import_cuda_worker():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from backends.platforms.mps import MPSProvider; "
            "from backends.platforms.mps_bindings import MPS_FAMILY_BINDINGS; "
            "import sys; assert 'backends.cuda_worker' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_mps_binding_import_is_torch_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from backends.platforms.mps_bindings import MPS_FAMILY_BINDINGS; "
            "import sys; assert not {'torch', 'diffusers', 'backends.cuda_worker'} "
            "& sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_mps_provider_reads_current_binding_table(mps_backend, monkeypatch):
    from backends.platforms.base import ExecutionCapabilities, FamilyPlatformBinding
    from backends.platforms.mps import MPSProvider
    import backends.platforms.mps_bindings as mps_bindings

    cell = FamilyPlatformBinding(
        "example.worker",
        ExecutionCapabilities(False, False, False),
    )
    replacement = {"future": cell}
    monkeypatch.setattr(mps_bindings, "MPS_FAMILY_BINDINGS", replacement)
    provider = MPSProvider()
    assert provider.family_binding("future") is cell
    assert provider.create_worker_factory().keywords["family_bindings"] is replacement


def test_mps_runtime_passes_policy_on_first_pool_call(mps_backend, monkeypatch):
    from backends.platform_registry import get_backend_provider
    from backends.platforms.mps_policy import MPS_TORCH_POLICY
    import backends.worker_pool as worker_pool

    calls = []
    sentinel = Mock()
    monkeypatch.setattr(
        worker_pool, "get_worker_pool", lambda **kwargs: calls.append(kwargs) or sentinel
    )
    runtime = get_backend_provider().create_generation_runtime()
    assert runtime._pool is sentinel
    assert calls == [{"parent_execution_policy": MPS_TORCH_POLICY}]


@pytest.mark.parametrize("isolation", ["inproc", "subprocess"])
def test_mps_pool_selects_mps_factory_without_cuda(mps_backend, monkeypatch, isolation):
    from backends.platforms.base import UnsupportedFamilyError
    from backends.worker_handle import InProcessWorkerHandle
    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from backends.worker_pool import get_worker_pool

    monkeypatch.setenv("WORKER_ISOLATION", isolation)
    pool = get_worker_pool(**_minimal_deps())
    handle = pool._governor._handle
    resolved = SimpleNamespace(profile=SimpleNamespace(family_id="sdxl"))
    if isolation == "inproc":
        assert isinstance(handle, InProcessWorkerHandle)
        with pytest.raises(
            UnsupportedFamilyError, match="family 'sdxl' has no MPS platform binding"
        ):
            handle._worker_factory(0, resolved, None)
    else:
        assert isinstance(handle, SubprocessWorkerHandle)
        assert handle._factory_ref == "backends.worker_factory.create_mps_worker"
        assert pool._parent_consumers[0]._execution_policy.backend_id == "mps"


def test_mps_subprocess_entry_rejects_every_family(mps_backend):
    from backends.platforms.base import UnsupportedFamilyError
    from backends.worker_factory import create_mps_worker

    for family_id in ("sd15", "sdxl", "hunyuandit"):
        with pytest.raises(
            UnsupportedFamilyError,
            match=f"family '{family_id}' has no MPS platform binding",
        ):
            resolved = SimpleNamespace(profile=SimpleNamespace(family_id=family_id))
            create_mps_worker(0, resolved, None)


@pytest.mark.parametrize("isolation", ["inproc", "subprocess"])
def test_mps_pool_rejects_known_family_before_worker_start(mps_backend, monkeypatch, isolation):
    from backends.platforms.base import UnsupportedFamilyError
    from backends.worker_pool import get_worker_pool

    monkeypatch.setenv("WORKER_ISOLATION", isolation)
    mode = SimpleNamespace(model_path="/models/sd15", loras=[])
    dependencies = _minimal_deps()
    dependencies["device_memory"].snapshot.return_value = SimpleNamespace(consumers=())
    dependencies["mode_config"].get_default_mode.return_value = "sd15"
    dependencies["mode_config"].get_mode.side_effect = None
    dependencies["mode_config"].get_mode.return_value = mode
    monkeypatch.setattr(
        "backends.governor.resolve_model",
        lambda path, selected_mode: (_resolved_sd15(), None),
    )

    pool = get_worker_pool(**dependencies)
    assert pool.is_model_loaded() is False
    assert pool.get_current_mode() is None
    with pytest.raises(
        UnsupportedFamilyError,
        match="family 'sd15' has no MPS platform binding",
    ):
        pool.switch_mode("sd15")
    if isolation == "subprocess":
        assert pool._governor._handle._proc is None


def test_mps_subprocess_child_uses_mps_binding_table(mps_backend):
    from backends.worker_handle_subprocess import SubprocessWorkerHandle, WorkerStartError
    from backends.model_resolution import LocalModelBinding

    handle = SubprocessWorkerHandle("backends.worker_factory.create_mps_worker")
    try:
        with pytest.raises(
            WorkerStartError,
            match="family 'sd15' has no MPS platform binding",
        ):
            handle.start(_resolved_sd15(), LocalModelBinding("/models/sd15"), None)
    finally:
        handle.stop()


def test_superres_settings_accept_mps_without_cuda():
    from server.superres_http import load_superres_runtime_settings

    settings = load_superres_runtime_settings({"BACKEND": "mps"}, cuda_available=True)
    assert settings.backend == "mps"
    assert settings.use_cuda is False


def test_legacy_pipeline_service_rejects_mps(mps_backend, monkeypatch):
    from server import lcm_sr_server as server

    monkeypatch.setattr(server, "BACKEND", "mps")
    with pytest.raises(RuntimeError, match="BACKEND=mps uses the WorkerPool runtime"):
        server.PipelineService(paths=Mock(), num_workers=1, queue_max=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("isolation", ["inproc", "subprocess"])
async def test_server_lifespan_boots_with_mps_and_no_model(mps_backend, monkeypatch, isolation):
    from backends import governor
    from server import lcm_sr_server as server

    monkeypatch.setenv("WORKER_ISOLATION", isolation)
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "off")
    monkeypatch.setenv("METRICS_ENABLED", "0")
    monkeypatch.setattr(server, "BACKEND", "mps")
    mode = SimpleNamespace(model_path="/models/sd15", loras=[])
    mode_config = Mock()
    mode_config.get_default_mode.return_value = "sd15"
    mode_config.get_mode.return_value = mode
    mode_config.list_modes.return_value = ["sd15"]
    monkeypatch.setattr(governor, "get_mode_config", lambda: mode_config)
    monkeypatch.setattr(server, "get_mode_config", lambda: mode_config)
    monkeypatch.setattr(
        governor,
        "resolve_model",
        lambda path, selected_mode: (_resolved_sd15(), None),
    )
    monkeypatch.setattr(server, "start_config_watcher", lambda *args: None)
    monkeypatch.setattr(server, "stop_config_watcher", lambda: None)
    monkeypatch.setattr(server, "register_job_hook", lambda: None)
    monkeypatch.setattr(server, "_close_providers", lambda app: None)
    monkeypatch.setattr(server.StorageProvider, "make_storage_provider_from_env", lambda: Mock())
    monkeypatch.setattr(server.MetricsSampler, "start", lambda self: None)
    monkeypatch.setattr(server.MetricsSampler, "stop", lambda self: None)

    async with server.lifespan(server.app):
        assert server.app.state.backend_provider.backend_id == "mps"
        assert server.app.state.use_mode_system is True
        assert server.app.state.worker_pool.is_model_loaded() is False
        if isolation == "subprocess":
            consumer = server.app.state.worker_pool._parent_consumers[0]
            assert consumer._execution_policy.backend_id == "mps"
        assert "backends.cuda_worker" not in sys.modules


@pytest.mark.asyncio
async def test_mps_mode_switch_reports_missing_binding(mps_backend, monkeypatch):
    from server import model_routes
    from backends.platforms.base import UnsupportedFamilyError

    mode = SimpleNamespace(model_path="/models/example")
    config = Mock()
    config.get_mode.return_value = mode
    pool = Mock()
    pool.get_current_mode.return_value = None
    monkeypatch.setattr(model_routes, "get_mode_config", lambda: config)
    monkeypatch.setattr(model_routes, "get_worker_pool", lambda: pool)
    pool.switch_mode.side_effect = UnsupportedFamilyError(
        "family 'sd15' has no MPS platform binding"
    )

    with pytest.raises(HTTPException) as error:
        await model_routes.switch_mode(model_routes.ModeSwitchRequest(mode="example"))
    assert error.value.status_code == 422
    assert error.value.detail == "family 'sd15' has no MPS platform binding"
    pool.switch_mode.assert_called_once_with("example")
