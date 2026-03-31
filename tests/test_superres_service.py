from concurrent.futures import Future

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


def test_create_superres_service_requires_matching_factory():
    from server.superres_service import create_superres_service

    with pytest.raises(RuntimeError, match="CUDA super-resolution service is not configured"):
        create_superres_service(
            backend_kind="cuda",
            model_path="/tmp/model.pth",
            num_workers=1,
            queue_max=32,
            input_size=224,
            output_size=672,
        )
