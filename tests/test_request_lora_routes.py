"""Request LoRAs on /generate, the WS route, and /api/models/status (STABL-fvwsvjvn).

Validation runs before admission. A rule failure is HTTP 400 or WS job:error.
A malformed entry is HTTP 422 or WS job:error. Neither calls admission.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.lcm_sr_server as srv
from backends.mode_loras import LoraSpec, RequestLoraError
from server import model_routes
from server.ws_routes import ws_router
from tests.snapshot_test_helpers import install_mode_backed


class _StopAfterAdmission(RuntimeError):
    """Ends the request right after admission; the test reads the admission args."""


@pytest.fixture
def root(tmp_path) -> Path:
    r = tmp_path / "root"
    r.mkdir()
    (r / "a.safetensors").write_bytes(b"x")
    return r


@pytest.fixture
def outside(tmp_path) -> Path:
    o = tmp_path / "outside"
    o.mkdir()
    (o / "x.safetensors").write_bytes(b"x")
    return o


def _mode_config(root: Path):
    return SimpleNamespace(config=SimpleNamespace(lora_root=str(root)))


# --------------------------------------------------------------------------- HTTP


@pytest.fixture
def http(root):
    """A TestClient on a small app that serves srv.generate with a mock runtime."""
    runtime = MagicMock(spec=["switch_mode", "get_current_mode", "admit_generation",
                              "get_active_model_snapshot", "submit_generate", "wait_for_result"])
    runtime.admit_generation.side_effect = _StopAfterAdmission("stop")
    test_app = FastAPI()
    test_app.post("/generate")(srv.generate)
    mock_app = MagicMock()
    mock_app.state.generation_runtime = runtime
    with patch.object(srv, "app", mock_app), \
            patch("server.lcm_sr_server.get_mode_config", return_value=_mode_config(root)):
        yield TestClient(test_app, raise_server_exceptions=False), runtime


def test_http_outside_root_is_400_and_not_admitted(http, outside):
    client, runtime = http

    r = client.post("/generate", json={"prompt": "p", "loras": [{"path": str(outside / "x.safetensors")}]})

    assert r.status_code == 400
    assert "lora_root" in r.json()["detail"]
    runtime.admit_generation.assert_not_called()


def test_http_valid_loras_reach_admission(http, root):
    client, runtime = http

    client.post("/generate", json={
        "prompt": "p", "mode": "plain", "loras": [{"path": "a.safetensors", "strength": 0.8}],
    })

    runtime.admit_generation.assert_called_once_with(
        "plain", (LoraSpec(str((root / "a.safetensors").resolve()), 0.8),)
    )


def test_http_no_loras_admits_with_mode_only(http):
    client, runtime = http

    client.post("/generate", json={"prompt": "p", "mode": "plain", "loras": []})

    runtime.admit_generation.assert_called_once_with("plain")


def test_http_admission_request_lora_error_is_400(http):
    client, runtime = http
    runtime.admit_generation.side_effect = RequestLoraError(
        "mode 'styled' has its own loras: list; send request LoRAs to a mode without loras:"
    )

    r = client.post("/generate", json={"prompt": "p", "mode": "styled", "loras": [{"path": "a.safetensors"}]})

    assert r.status_code == 400
    assert "styled" in r.json()["detail"]


@pytest.mark.parametrize(
    "loras",
    [
        [{"path": "a", "strength": "abc"}],
        [{"strength": 0.5}],
        "a.safetensors",
    ],
)
def test_http_malformed_entries_are_422_and_not_admitted(http, loras):
    client, runtime = http

    r = client.post("/generate", json={"prompt": "p", "loras": loras})

    assert r.status_code == 422
    runtime.admit_generation.assert_not_called()


def test_http_sidecar_escape_is_400_and_not_admitted(http, root, outside):
    client, runtime = http
    (root / "d").mkdir()
    (root / "d" / "adapter_model.safetensors").write_bytes(b"x")
    cfg = outside / "adapter_config.json"
    cfg.write_text("{}")
    (root / "d" / "adapter_config.json").symlink_to(cfg)

    r = client.post("/generate", json={"prompt": "p", "loras": [{"path": "d"}]})

    assert r.status_code == 400
    assert "adapter_config.json" in r.json()["detail"]
    runtime.admit_generation.assert_not_called()


# --------------------------------------------------------------------------- WS


@pytest.fixture
def ws(root):
    app = FastAPI()
    app.include_router(ws_router)
    app.state.use_mode_system = True
    app.state.service = None
    app.state.sr_service = None
    app.state.storage = None
    pool = MagicMock()
    app.state.worker_pool = pool
    install_mode_backed(app.state, pool, SimpleNamespace(
        name="plain", default_size="512x512", default_steps=4, default_guidance=1.0,
        resolution_options=[{"size": "512x512", "aspect_ratio": "1:1"}],
    ))
    with patch("server.ws_routes.get_mode_config", return_value=_mode_config(root)), \
            TestClient(app) as client:
        yield client, pool


def _ws_submit(client, params):
    with client.websocket_connect("/v1/ws") as sock:
        sock.receive_json()  # status
        sock.send_json({"type": "job:submit", "id": "t1", "jobType": "generate", "params": params})
        ack = sock.receive_json()
        assert ack["type"] == "job:ack"
        return sock.receive_json()


def test_ws_outside_root_is_job_error_and_not_admitted(ws, outside):
    client, pool = ws

    frame = _ws_submit(client, {"prompt": "p", "loras": [{"path": str(outside / "x.safetensors")}]})

    assert frame["type"] == "job:error"
    assert "lora_root" in frame["error"]
    pool.admit_generation.assert_not_called()


def test_ws_malformed_entry_is_job_error_and_not_admitted(ws):
    client, pool = ws

    frame = _ws_submit(client, {"prompt": "p", "loras": [{"path": "a", "strength": "abc"}]})

    assert frame["type"] == "job:error"
    pool.admit_generation.assert_not_called()


def test_ws_valid_loras_reach_admission(ws, root):
    client, pool = ws
    pool.admit_generation.side_effect = RuntimeError("stop")

    _ws_submit(client, {"prompt": "p", "mode": "plain", "loras": [{"path": "a.safetensors", "strength": 0.8}]})

    pool.admit_generation.assert_called_once_with(
        "plain", (LoraSpec(str((root / "a.safetensors").resolve()), 0.8),)
    )


# --------------------------------------------------------------------------- status


async def test_models_status_has_current_loras():
    runtime = Mock()
    runtime.get_active_model_snapshot.return_value = None
    runtime.get_current_mode.return_value = None
    runtime.is_model_loaded.return_value = True
    runtime.get_queue_size.return_value = 0
    runtime.get_current_loras.return_value = [{"path": "/l/a.safetensors", "strength": 0.8}]
    registry = Mock()
    registry.get_vram_stats.return_value = {}
    provider = Mock()
    provider.backend_id = "cuda"
    provider.capabilities.return_value = SimpleNamespace(
        supports_generation=True, supports_modes=True, supports_superres=True,
        supports_model_registry_stats=True, supports_img2img=True,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(generation_runtime=None)))

    with patch("server.model_routes.get_backend_provider", return_value=provider), \
            patch("server.model_routes.get_generation_runtime", return_value=runtime), \
            patch("server.model_routes.get_model_registry", return_value=registry), \
            patch.dict(os.environ, {"BACKEND_VERSION": "dev"}, clear=False):
        data = await model_routes.get_models_status(request)

    assert data["current_loras"] == [{"path": "/l/a.safetensors", "strength": 0.8}]


# ------------------------------------------------- runtimes without the mode system
# A runtime with no mode system cannot load LoRAs. Request LoRAs must be refused,
# never dropped: a job must not run on a LoRA set other than the one it named.


def test_http_loras_without_mode_system_is_400(root):
    runtime = MagicMock(spec=["submit_generate"])
    test_app = FastAPI()
    test_app.post("/generate")(srv.generate)
    mock_app = MagicMock()
    mock_app.state.generation_runtime = runtime
    with patch.object(srv, "app", mock_app):
        client = TestClient(test_app, raise_server_exceptions=False)
        r = client.post("/generate", json={"prompt": "p", "loras": [{"path": "a.safetensors"}]})

    assert r.status_code == 400
    assert "mode system" in r.json()["detail"]
    runtime.submit_generate.assert_not_called()


def test_ws_loras_without_mode_system_is_job_error():
    app = FastAPI()
    app.include_router(ws_router)
    app.state.use_mode_system = False
    app.state.service = MagicMock()
    app.state.sr_service = None
    app.state.storage = None
    with TestClient(app) as client:
        frame = _ws_submit(client, {"prompt": "p", "loras": [{"path": "a.safetensors"}]})

    assert frame["type"] == "job:error"
    assert "mode system" in frame["error"]
    app.state.service.submit.assert_not_called()
