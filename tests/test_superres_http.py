from concurrent.futures import Future

import pytest


def test_load_superres_runtime_settings_requires_explicit_backend():
    from server.superres_http import load_superres_runtime_settings

    with pytest.raises(RuntimeError, match="BACKEND must be set explicitly"):
        load_superres_runtime_settings({}, cuda_available=False)


def test_load_superres_runtime_settings_preserves_explicit_backend():
    from server.superres_http import load_superres_runtime_settings

    settings = load_superres_runtime_settings(
        {
            "BACKEND": "rknn",
            "MODEL_ROOT": "/models",
            "SR_ENABLED": "1",
            "SR_INPUT_SIZE": "224",
            "SR_OUTPUT_SIZE": "672",
            "SR_NUM_WORKERS": "2",
            "SR_QUEUE_MAX": "16",
            "SR_REQUEST_TIMEOUT": "45",
            "SR_MAX_PIXELS": "123456",
        },
        cuda_available=False,
    )

    assert settings.enabled is True
    assert settings.backend == "rknn"
    assert settings.use_cuda is False
    assert settings.sr_model_path == "/models/super-resolution-10.rknn"
    assert settings.sr_input_size == 224
    assert settings.sr_output_size == 672
    assert settings.sr_num_workers == 2
    assert settings.sr_queue_max == 16
    assert settings.sr_request_timeout == 45.0
    assert settings.sr_max_pixels == 123456


def test_initialize_superres_service_selects_rknn_backend():
    from server.superres_http import initialize_superres_service

    created = []

    def rknn_factory(**kwargs):
        created.append(("rknn", kwargs))
        return object()

    service = initialize_superres_service(
        enabled=True,
        backend="rknn",
        use_cuda=False,
        sr_model_path="/models/super-resolution-10.rknn",
        sr_num_workers=1,
        sr_queue_max=32,
        sr_input_size=224,
        sr_output_size=672,
        sr_max_pixels=24_000_000,
        path_exists=lambda path: True,
        rknn_factory=rknn_factory,
    )

    assert service is not None
    assert created == [
        (
            "rknn",
            {
                "model_path": "/models/super-resolution-10.rknn",
                "num_workers": 1,
                "queue_max": 32,
                "input_size": 224,
                "output_size": 672,
                "max_pixels": 24_000_000,
            },
        )
    ]


def test_initialize_superres_service_selects_explicit_cuda_backend():
    from server.superres_http import initialize_superres_service

    created = []

    def cuda_factory(**kwargs):
        created.append(("cuda", kwargs))
        return object()

    service = initialize_superres_service(
        enabled=True,
        backend="cuda",
        use_cuda=True,
        sr_model_path="/models/super-resolution-10.rknn",
        sr_num_workers=1,
        sr_queue_max=32,
        sr_input_size=224,
        sr_output_size=672,
        sr_max_pixels=24_000_000,
        environ={
            "CUDA_SR_MODEL": "/models/sr/RealESRGAN_x4plus.pth",
            "CUDA_SR_TILE": "256",
            "CUDA_SR_FP16": "1",
        },
        path_exists=lambda path: True,
        cuda_factory=cuda_factory,
    )

    assert service is not None
    assert len(created) == 1
    kind, kwargs = created[0]
    assert kind == "cuda"
    assert kwargs["model_path"] == "/models/sr/RealESRGAN_x4plus.pth"
    assert kwargs["num_workers"] == 1
    assert kwargs["queue_max"] == 32
    assert kwargs["input_size"] == 224
    assert kwargs["output_size"] == 672
    assert kwargs["config"].model_path == "/models/sr/RealESRGAN_x4plus.pth"
    assert kwargs["config"].tile == 256


def test_initialize_superres_service_returns_none_when_disabled():
    from server.superres_http import initialize_superres_service

    assert (
        initialize_superres_service(
            enabled=False,
            backend="cuda",
            use_cuda=True,
            sr_model_path="/models/super-resolution-10.rknn",
            sr_num_workers=1,
            sr_queue_max=32,
            sr_input_size=224,
            sr_output_size=672,
            sr_max_pixels=24_000_000,
        )
        is None
    )


def test_submit_superres_uses_shared_service_contract():
    from server.superres_http import submit_superres

    calls = []

    class FakeService:
        def submit(self, image_bytes: bytes, *, out_format: str, quality: int, magnitude: int, timeout_s: float = 0.25):
            calls.append((image_bytes, out_format, quality, magnitude, timeout_s))
            fut = Future()
            fut.set_result(b"upscaled")
            return fut

    out = submit_superres(
        sr_service=FakeService(),
        image_bytes=b"png",
        out_format="jpeg",
        quality=88,
        magnitude=2,
        queue_timeout_s=0.25,
        request_timeout_s=5.0,
    )

    assert out == b"upscaled"
    assert calls == [(b"png", "jpeg", 88, 2, 0.25)]


def test_submit_superres_raises_queue_full_verbatim():
    from server.superres_http import submit_superres

    class FakeService:
        def submit(self, image_bytes: bytes, *, out_format: str, quality: int, magnitude: int, timeout_s: float = 0.25):
            fut = Future()
            fut.set_exception(RuntimeError("Queue full"))
            return fut

    with pytest.raises(RuntimeError, match="Queue full"):
        submit_superres(
            sr_service=FakeService(),
            image_bytes=b"png",
            out_format="png",
            quality=92,
            magnitude=1,
            queue_timeout_s=0.25,
            request_timeout_s=5.0,
        )


def test_build_superres_headers_uses_service_metadata():
    from server.superres_http import build_superres_headers

    class FakeService:
        model_path = "/models/sr/RealESRGAN_x4plus.pth"
        scale_per_pass = 4

    headers = build_superres_headers(FakeService(), magnitude=2, out_format="jpeg")

    assert headers == {
        "X-SR-Model": "RealESRGAN_x4plus.pth",
        "X-SR-Passes": "2",
        "X-SR-Scale-Per-Pass": "4",
        "X-SR-Format": "jpeg",
    }
