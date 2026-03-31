from concurrent.futures import Future
import types

import pytest


def test_resolve_superres_backend_prefers_explicit_cuda():
    from server.superres_service import resolve_superres_backend

    assert resolve_superres_backend(backend="cuda", use_cuda=True) == "cuda"


def test_resolve_superres_backend_uses_cuda_for_auto_when_generation_is_cuda():
    from server.superres_service import resolve_superres_backend

    assert resolve_superres_backend(backend="auto", use_cuda=True) == "cuda"


def test_resolve_superres_backend_falls_back_to_rknn_for_auto_without_cuda():
    from server.superres_service import resolve_superres_backend

    assert resolve_superres_backend(backend="auto", use_cuda=False) == "rknn"


def test_create_superres_service_uses_selected_factory():
    from server.superres_service import create_superres_service

    created = []

    class FakeService:
        def submit(self, image_bytes: bytes, *, out_format: str, quality: int, magnitude: int, timeout_s: float = 0.25):
            fut = Future()
            fut.set_result(image_bytes)
            return fut

        def shutdown(self):
            return None

    def rknn_factory(**kwargs):
        created.append(("rknn", kwargs))
        return FakeService()

    def cuda_factory(**kwargs):
        created.append(("cuda", kwargs))
        return FakeService()

    service = create_superres_service(
        backend_kind="rknn",
        rknn_factory=rknn_factory,
        cuda_factory=cuda_factory,
        model_path="/tmp/model.rknn",
        num_workers=1,
        queue_max=32,
        input_size=224,
        output_size=672,
    )

    assert isinstance(service, FakeService)
    assert created == [
        (
            "rknn",
            {
                "model_path": "/tmp/model.rknn",
                "num_workers": 1,
                "queue_max": 32,
                "input_size": 224,
                "output_size": 672,
            },
        )
    ]


def test_create_superres_service_builds_default_cuda_service():
    from server.superres_service import create_superres_service

    service = create_superres_service(
        backend_kind="cuda",
        model_path="/tmp/model.pth",
        num_workers=1,
        queue_max=32,
        input_size=224,
        output_size=672,
    )

    assert service.__class__.__name__ == "CudaSuperResService"


def test_load_cuda_superres_config_parses_env_defaults():
    from server.superres_service import load_cuda_superres_config

    config = load_cuda_superres_config(
        {
            "CUDA_SR_MODEL": "/models/sr/RealESRGAN_x4plus.pth",
        }
    )

    assert config.model_path == "/models/sr/RealESRGAN_x4plus.pth"
    assert config.tile == 0
    assert config.use_fp16 is True
    assert config.device == "cuda:0"


def test_cuda_superres_service_lazy_loads_worker_and_reuses_it():
    from server.superres_service import CudaSuperResConfig, CudaSuperResService

    created = []

    class FakeWorker:
        def __init__(self):
            self.closed = False

        def upscale_bytes(self, image_bytes: bytes, *, magnitude: int, out_format: str, quality: int) -> bytes:
            return image_bytes + f":{magnitude}:{out_format}:{quality}".encode()

        def close(self):
            self.closed = True

    def worker_factory(config):
        created.append(config)
        return FakeWorker()

    service = CudaSuperResService(
        config=CudaSuperResConfig(
            model_path="/models/sr/RealESRGAN_x4plus.pth",
            tile=256,
            use_fp16=True,
            device="cuda:0",
        ),
        queue_max=4,
        worker_factory=worker_factory,
    )

    assert created == []

    fut1 = service.submit(b"img", out_format="png", quality=92, magnitude=2)
    assert fut1.result(timeout=1) == b"img:2:png:92"
    assert len(created) == 1

    fut2 = service.submit(b"img2", out_format="jpeg", quality=80, magnitude=1)
    assert fut2.result(timeout=1) == b"img2:1:jpeg:80"
    assert len(created) == 1

    service.shutdown()


def test_cuda_superres_service_unloads_worker_after_oom_and_recovers():
    from server.superres_service import CudaSuperResConfig, CudaSuperResService

    created = []
    closed = []

    class OomWorker:
        def __init__(self, name: str, fail: bool):
            self.name = name
            self.fail = fail

        def upscale_bytes(self, image_bytes: bytes, *, magnitude: int, out_format: str, quality: int) -> bytes:
            if self.fail:
                raise RuntimeError("CUDA out of memory while upscaling")
            return image_bytes + f":{self.name}".encode()

        def close(self):
            closed.append(self.name)

    def worker_factory(config):
        idx = len(created)
        created.append(config.model_path)
        if idx == 0:
            return OomWorker("first", True)
        return OomWorker("second", False)

    service = CudaSuperResService(
        config=CudaSuperResConfig(
            model_path="/models/sr/RealESRGAN_x4plus.pth",
            tile=0,
            use_fp16=True,
            device="cuda:0",
        ),
        queue_max=4,
        worker_factory=worker_factory,
    )

    fut1 = service.submit(b"img", out_format="png", quality=92, magnitude=1)
    with pytest.raises(RuntimeError, match="out of memory"):
        fut1.result(timeout=1)

    assert closed == ["first"]

    fut2 = service.submit(b"img", out_format="png", quality=92, magnitude=1)
    assert fut2.result(timeout=1) == b"img:second"

    service.shutdown()


def test_torchvision_functional_tensor_shim_installs_alias_when_missing():
    from server.superres_service import ensure_torchvision_functional_tensor_compat

    fake_functional = types.SimpleNamespace(rgb_to_grayscale=lambda x: x)
    sys_modules = {}

    def fake_import(name: str):
        if name == "torchvision.transforms.functional_tensor":
            raise ModuleNotFoundError(name)
        if name == "torchvision.transforms.functional":
            return fake_functional
        raise AssertionError(f"unexpected import: {name}")

    ensure_torchvision_functional_tensor_compat(import_module=fake_import, sys_modules=sys_modules)

    shim = sys_modules["torchvision.transforms.functional_tensor"]
    assert shim.rgb_to_grayscale is fake_functional.rgb_to_grayscale
