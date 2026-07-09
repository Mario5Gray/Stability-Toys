from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_generate_uses_provider_runtime(monkeypatch):
    from server import lcm_sr_server

    fut = Future()
    fut.set_result((b"png-bytes", 1234))

    runtime = SimpleNamespace(
        get_current_mode=lambda: None,
        is_model_loaded=lambda: False,
        get_queue_size=lambda: 0,
        submit_generate=lambda req: fut,
    )
    monkeypatch.setattr(lcm_sr_server.app.state, "generation_runtime", runtime, raising=False)
    monkeypatch.setattr(lcm_sr_server.app.state, "sr_service", None, raising=False)
    monkeypatch.setattr(lcm_sr_server.app.state, "storage", None, raising=False)

    req = lcm_sr_server.GenerateRequest(prompt="owl")
    response = lcm_sr_server.generate(req)

    assert response.body == b"png-bytes"
    assert response.headers["x-seed"] == "1234"


def test_run_generate_from_dict_uses_provider_runtime(monkeypatch):
    from server import lcm_sr_server

    fut = Future()
    fut.set_result((b"png-bytes", 1234))

    runtime = SimpleNamespace(
        submit_generate=lambda req: fut,
    )
    monkeypatch.setattr(lcm_sr_server.app.state, "generation_runtime", runtime, raising=False)
    monkeypatch.setattr(lcm_sr_server.app.state, "sr_service", None, raising=False)

    out_bytes, seed, headers = lcm_sr_server._run_generate_from_dict({"prompt": "owl"})

    assert out_bytes == b"png-bytes"
    assert seed == 1234
    assert headers["X-Seed"] == "1234"


def test_cuda_provider_creates_runtime_without_server_branching():
    from backends.platforms.cuda import CUDAProvider

    provider = CUDAProvider()
    runtime = provider.create_generation_runtime(queue_max=4, pool=SimpleNamespace())

    assert runtime.__class__.__name__ == "CudaGenerationRuntime"


def test_rknn_provider_creates_runtime_without_server_dependency():
    from backends.platforms.rknn import RKNNProvider

    fake_service = SimpleNamespace(
        submit=lambda req, timeout_s=0.25: Future(),
        q=SimpleNamespace(qsize=lambda: 0),
        shutdown=lambda: None,
    )

    with patch("backends.rknn_runtime.PipelineService.get_instance", return_value=fake_service), \
         patch("backends.rknn_runtime.build_rknn_context_cfgs_for_rk3588", return_value=[{"worker_id": 0}]):
        provider = RKNNProvider()
        runtime = provider.create_generation_runtime(
            paths=SimpleNamespace(),
            num_workers=1,
            queue_max=4,
            use_rknn_context_cfgs="0",
        )

    assert runtime.__class__.__name__ == "RknnGenerationRuntime"


def _build_rknn_runtime_with_capturing_service():
    """Helper: build an RknnGenerationRuntime backed by a service that records submit() kwargs."""
    captured: dict[str, object] = {}

    def _capture(req, *, timeout_s):
        captured["req"] = req
        captured["timeout_s"] = timeout_s
        return Future()

    fake_service = SimpleNamespace(
        submit=_capture,
        q=SimpleNamespace(qsize=lambda: 0),
        shutdown=lambda: None,
    )

    with patch("backends.rknn_runtime.PipelineService.get_instance", return_value=fake_service), \
         patch("backends.rknn_runtime.build_rknn_context_cfgs_for_rk3588", return_value=[{"worker_id": 0}]):
        from backends.platforms.rknn import RKNNProvider

        runtime = RKNNProvider().create_generation_runtime(
            paths=SimpleNamespace(),
            num_workers=1,
            queue_max=4,
            use_rknn_context_cfgs="0",
        )
    return runtime, captured


def test_rknn_runtime_uses_shared_default_queue_timeout(monkeypatch):
    """RKNN runtime must honor the same WORKER_QUEUE_TIMEOUT_S the rest of the platform uses."""
    monkeypatch.setattr("backends.worker_pool.DEFAULT_QUEUE_TIMEOUT_S", 0.99)
    runtime, captured = _build_rknn_runtime_with_capturing_service()

    runtime.submit_generate({"prompt": "owl"})

    assert captured["timeout_s"] == 0.99


def test_rknn_runtime_explicit_timeout_overrides_shared_default(monkeypatch):
    """Caller-supplied timeout_s must win over the shared default."""
    monkeypatch.setattr("backends.worker_pool.DEFAULT_QUEUE_TIMEOUT_S", 0.99)
    runtime, captured = _build_rknn_runtime_with_capturing_service()

    runtime.submit_generate({"prompt": "owl"}, timeout_s=0.5)

    assert captured["timeout_s"] == 0.5


def test_cpu_generation_runtime_raises_clear_error():
    from backends.platforms.cpu import CPUProvider

    runtime = CPUProvider().create_generation_runtime(queue_max=1)

    with pytest.raises(NotImplementedError, match="BACKEND=cpu generation is not implemented"):
        runtime.submit_generate({"prompt": "owl"})


def test_mlx_generation_runtime_raises_clear_error():
    from backends.platforms.mlx import MLXProvider

    runtime = MLXProvider().create_generation_runtime(queue_max=1)

    with pytest.raises(NotImplementedError, match="BACKEND=mlx generation is not implemented"):
        runtime.submit_generate({"prompt": "owl"})


def test_cuda_provider_reports_supports_img2img_and_controlnet():
    from backends.platforms.cuda import CUDAProvider

    caps = CUDAProvider().capabilities()
    assert caps.supports_img2img_and_controlnet is True
