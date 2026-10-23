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


def _resolved_sdxl():
    """SDXL is the still-unbound family on MPS, so it carries the negative tests."""
    from backends.family_profiles import SDXL_PROFILE
    from tests.test_worker_factory import _resolved
    from utils.model_detector import ModelVariant

    return _resolved(
        SDXL_PROFILE,
        variant=ModelVariant.SDXL_BASE,
        cad=2048,
        checkpoint_variant="sdxl",
    )


def test_mps_provider_binds_sd15_and_shares_runtime(mps_backend):
    from backends.platform_registry import get_backend_provider
    from backends.platforms.base import ExecutionCapabilities, UnsupportedFamilyError
    from backends.platforms.mps_bindings import MPS_FAMILY_BINDINGS
    from backends.platforms.pool_runtime import PoolGenerationRuntime

    provider = get_backend_provider()
    assert provider.backend_id == "mps"
    assert set(MPS_FAMILY_BINDINGS) == {"sd15"}
    cell = provider.family_binding("sd15")
    assert cell.worker_ref == "backends.mps_worker.DiffusersMpsWorker"
    # Text-to-image and img2img. ControlNet and the combined operation stay
    # false. The cell is the real admission gate for all three.
    assert cell.execution_capabilities == ExecutionCapabilities(True, False, False)
    assert isinstance(provider.create_generation_runtime(pool=Mock()), PoolGenerationRuntime)
    assert provider.capabilities().supports_superres is False
    assert provider.create_model_registry().get_vram_stats()["backend"] == "mps"
    for family_id in ("sdxl", "hunyuandit"):
        assert provider.family_binding(family_id) is None
        with pytest.raises(
            UnsupportedFamilyError,
            match=f"family '{family_id}' has no MPS platform binding",
        ):
            provider.create_worker_factory()(
                0,
                SimpleNamespace(profile=SimpleNamespace(family_id=family_id)),
                None,
            )


class _FakeMpsWorker:
    """Stand-in for the real worker, so factory selection needs no Torch import.

    Importing `backends.mps_worker` in-process would pull in `backends.cuda_worker`
    and break the import-cleanliness assertions elsewhere in this file.
    """

    built: dict = {}

    def __init__(self, worker_id, model_path, model_info=None, family_profile=None):
        _FakeMpsWorker.built = {
            "worker_id": worker_id,
            "model_path": model_path,
            "family_id": family_profile.family_id,
        }


def test_mps_factory_builds_the_bound_sd15_worker(mps_backend, monkeypatch):
    from backends.model_resolution import LocalModelBinding
    from backends.platforms.base import ExecutionCapabilities, FamilyPlatformBinding
    import backends.platforms.mps_bindings as mps_bindings
    from backends.worker_factory import create_mps_worker

    _FakeMpsWorker.built = {}
    monkeypatch.setitem(
        mps_bindings.MPS_FAMILY_BINDINGS,
        "sd15",
        FamilyPlatformBinding(
            "tests.test_mps_registration._FakeMpsWorker",
            ExecutionCapabilities(False, False, False),
        ),
    )
    worker = create_mps_worker(3, _resolved_sd15(), LocalModelBinding("/models/sd15"))

    assert isinstance(worker, _FakeMpsWorker)
    assert _FakeMpsWorker.built == {
        "worker_id": 3,
        "model_path": "/models/sd15",
        "family_id": "sd15",
    }


def test_mps_sd15_worker_ref_resolves_in_a_fresh_interpreter():
    """The bound dotted ref must import and subclass the shared worker base.

    This runs in a subprocess because importing the worker pulls in Torch, which
    the in-process import-cleanliness tests above forbid.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from backends.platforms.mps_bindings import MPS_FAMILY_BINDINGS; "
            "from backends.worker_factory import _resolve_worker_class; "
            "from backends.cuda_worker import CudaWorkerBase; "
            "from backends.platforms.mps_policy import MPS_TORCH_POLICY; "
            "cls = _resolve_worker_class(MPS_FAMILY_BINDINGS['sd15'].worker_ref); "
            "assert issubclass(cls, CudaWorkerBase), cls; "
            "assert cls.execution_policy is MPS_TORCH_POLICY",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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


def test_mps_subprocess_entry_rejects_unbound_families(mps_backend):
    from backends.platforms.base import UnsupportedFamilyError
    from backends.worker_factory import create_mps_worker

    for family_id in ("sdxl", "hunyuandit"):
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
    mode = SimpleNamespace(model_path="/models/sdxl", loras=[])
    dependencies = _minimal_deps()
    dependencies["device_memory"].snapshot.return_value = SimpleNamespace(consumers=())
    dependencies["mode_config"].get_default_mode.return_value = "sdxl"
    dependencies["mode_config"].get_mode.side_effect = None
    dependencies["mode_config"].get_mode.return_value = mode
    monkeypatch.setattr(
        "backends.governor.resolve_model",
        lambda path, selected_mode: (_resolved_sdxl(), None),
    )

    pool = get_worker_pool(**dependencies)
    assert pool.is_model_loaded() is False
    assert pool.get_current_mode() is None
    with pytest.raises(
        UnsupportedFamilyError,
        match="family 'sdxl' has no MPS platform binding",
    ):
        pool.switch_mode("sdxl")
    if isolation == "subprocess":
        assert pool._governor._handle._proc is None


def test_mps_subprocess_child_uses_mps_binding_table(mps_backend):
    from backends.worker_handle_subprocess import SubprocessWorkerHandle, WorkerStartError
    from backends.model_resolution import LocalModelBinding

    handle = SubprocessWorkerHandle("backends.worker_factory.create_mps_worker")
    try:
        with pytest.raises(
            WorkerStartError,
            match="family 'sdxl' has no MPS platform binding",
        ):
            handle.start(_resolved_sdxl(), LocalModelBinding("/models/sdxl"), None)
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
    mode = SimpleNamespace(model_path="/models/sdxl", loras=[])
    mode_config = Mock()
    mode_config.get_default_mode.return_value = "sdxl"
    mode_config.get_mode.return_value = mode
    mode_config.list_modes.return_value = ["sdxl"]
    monkeypatch.setattr(governor, "get_mode_config", lambda: mode_config)
    monkeypatch.setattr(server, "get_mode_config", lambda: mode_config)
    monkeypatch.setattr(
        governor,
        "resolve_model",
        lambda path, selected_mode: (_resolved_sdxl(), None),
    )
    monkeypatch.setattr(server, "start_config_watcher", lambda *args: None)
    monkeypatch.setattr(server, "stop_config_watcher", lambda: None)
    monkeypatch.setattr(server, "register_job_hook", lambda: None)
    monkeypatch.setattr(server, "_close_providers", lambda app: None)
    monkeypatch.setattr(server.StorageProvider, "make_storage_provider_from_env", lambda: Mock())
    monkeypatch.setattr(server.MetricsSampler, "start", lambda self: None)
    monkeypatch.setattr(server.MetricsSampler, "stop", lambda self: None)

    # Measure what this test means: the MPS boot must not import the CUDA
    # worker. A bare `not in sys.modules` check also fails when an EARLIER test
    # file in the same pytest process imported it (tests/test_mps_worker.py
    # does, through backends.mps_worker), which says nothing about the lifespan.
    # The absolute guarantee is covered by the subprocess tests above.
    cuda_worker_preloaded = "backends.cuda_worker" in sys.modules

    async with server.lifespan(server.app):
        assert server.app.state.backend_provider.backend_id == "mps"
        assert server.app.state.use_mode_system is True
        assert server.app.state.worker_pool.is_model_loaded() is False
        if isolation == "subprocess":
            consumer = server.app.state.worker_pool._parent_consumers[0]
            assert consumer._execution_policy.backend_id == "mps"
        if not cuda_worker_preloaded:
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
        "family 'sdxl' has no MPS platform binding"
    )

    with pytest.raises(HTTPException) as error:
        await model_routes.switch_mode(model_routes.ModeSwitchRequest(mode="example"))
    assert error.value.status_code == 422
    assert error.value.detail == "family 'sdxl' has no MPS platform binding"
    pool.switch_mode.assert_called_once_with("example")


def test_mps_registry_reports_loaded_model_but_no_device_total(mps_backend):
    """Slice 1 keeps the placeholder registry. Its zero total means "no device
    measurement", not "MPS uses no memory". The loaded name and count are real.
    """
    from backends.platform_registry import get_backend_provider

    provider = get_backend_provider()
    registry = provider.create_model_registry()
    registry.register_model("lcm-general", "/models/diffusers/lcm-dreamshaper-v7")

    stats = registry.get_vram_stats()
    assert stats["backend"] == "mps"
    assert stats["device"] == "MPS placeholder"
    assert stats["models_loaded"] == 1
    assert [model["name"] for model in stats["models"]] == ["lcm-general"]
    assert registry.list_models() == ["lcm-general"]
    assert registry.get_total_vram() == 0
    assert provider.capabilities().supports_model_registry_stats is False


def test_mps_reports_generation_after_live_acceptance(mps_backend):
    """The flag feeds `/api/models/status` only. The real generation gate is the
    `sd15` cell. The flag turned true as the LAST step of STABL-hvvogszy, after
    a live MPS render in both isolation modes passed the validity checks in
    docs/MPS_BACKEND.md. MPS binds one family. Its cell admits text-to-image
    and img2img. ControlNet and the combined operation stay false.
    """
    from backends.platform_registry import get_backend_provider

    capabilities = get_backend_provider().capabilities()
    assert capabilities.supports_generation is True
    assert capabilities.supports_modes is True
    assert capabilities.supports_superres is False
    assert capabilities.supports_model_registry_stats is False
