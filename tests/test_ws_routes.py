"""
Tests for server/ws_routes.py — WebSocket endpoint integration tests.

Uses Starlette TestClient with real WebSocket connections against a
minimal FastAPI app that mounts the WS router.
"""

import asyncio
import concurrent.futures
import json
import queue
import time
import types
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import ws_routes
from server.ws_hub import WSHub, hub
from server.ws_routes import ws_router
from server.upload_routes import upload_router, UPLOADS


# ---------------------------------------------------------------------------
# Minimal app for testing (no heavy backends)
# ---------------------------------------------------------------------------

def _make_test_app():
    app = FastAPI()
    app.include_router(ws_router)
    app.include_router(upload_router)

    # Minimal app.state stubs
    app.state.use_mode_system = False
    app.state.service = None
    app.state.sr_service = None
    app.state.storage = None
    return app


app = _make_test_app()
client = TestClient(app)


# ---------------------------------------------------------------------------
# Connection + system:status on connect
# ---------------------------------------------------------------------------

class TestWSConnection:
    def test_connect_and_receive_status(self):
        with client.websocket_connect("/v1/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "system:status"
            assert "ts" in msg
            assert "mode" in msg

    def test_connect_disconnect_clean(self):
        """Connect and disconnect without errors."""
        with client.websocket_connect("/v1/ws") as ws:
            msg = ws.receive_json()  # consume status
            assert msg["type"] == "system:status"
        # If we get here without exception, disconnect was clean


# ---------------------------------------------------------------------------
# Ping / Pong
# ---------------------------------------------------------------------------

class TestPingPong:
    def test_ping_returns_pong(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"

    def test_ping_with_id(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "ping", "id": "p1"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"


# ---------------------------------------------------------------------------
# Invalid messages
# ---------------------------------------------------------------------------

class TestInvalidMessages:
    def test_invalid_json(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_text("not json{{{")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Invalid JSON" in msg["error"]

    def test_unknown_type(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "nonexistent:action", "id": "x1"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unknown type" in msg["error"]
            assert msg.get("id") == "x1"


# ---------------------------------------------------------------------------
# job:submit — ack path (generate, without real backend)
# ---------------------------------------------------------------------------

class TestJobSubmit:
    def test_finalize_mode_generate_request_replaces_env_defaults_and_accepts_mode_defaults(self):
        from server.generation_constraints import finalize_mode_generate_request

        req = SimpleNamespace(
            size="640x640",
            num_inference_steps=6,
            guidance_scale=1.5,
        )
        mode = SimpleNamespace(
            name="SDXL",
            default_size="1024x1024",
            default_steps=30,
            default_guidance=7.0,
            resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        )

        finalize_mode_generate_request(
            req,
            mode,
            env_default_size="640x640",
            env_default_steps=6,
            env_default_guidance=1.5,
        )

        assert req.size == "1024x1024"
        assert req.num_inference_steps == 30
        assert req.guidance_scale == 7.0

    def test_build_generate_request_uses_env_defaults_for_omitted_generation_controls(self):
        fake_lcm_module = types.ModuleType("server.lcm_sr_server")

        class _FakeGenerateRequest:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        fake_lcm_module.GenerateRequest = _FakeGenerateRequest
        original_lcm_module = sys.modules.get("server.lcm_sr_server")
        sys.modules["server.lcm_sr_server"] = fake_lcm_module

        try:
            with patch.dict(os.environ, {
                "DEFAULT_SIZE": "640x640",
                "DEFAULT_STEPS": "6",
                "DEFAULT_GUIDANCE": "1.5",
            }, clear=False):
                req = ws_routes._build_generate_request({"prompt": "a cat"})
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module

        assert req.size == "640x640"
        assert req.num_inference_steps == 6
        assert req.guidance_scale == 1.5

    def test_generate_ack_then_error_no_backend(self):
        """Submit a generate job — should ack, then error since no backend."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({
                "type": "job:submit",
                "id": "t1",
                "jobType": "generate",
                "params": {
                    "prompt": "a cat",
                    "size": "512x512",
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "seed": 12345678,
                },
            })
            ack = ws.receive_json()
            assert ack["type"] == "job:ack"
            assert ack["id"] == "t1"
            assert "jobId" in ack

            # Without a real backend, expect job:error
            err = ws.receive_json()
            assert err["type"] == "job:error"
            assert err["jobId"] == ack["jobId"]

    def test_generate_mode_system_forwards_negative_prompt_and_scheduler(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "sdxl-general"
        fut = MagicMock()
        fut.result.return_value = (b"png", 123)
        pool.submit_job.return_value = fut
        app.state.worker_pool = pool
        app.state.storage = None

        fake_lcm_module = types.ModuleType("server.lcm_sr_server")

        class _FakeGenerateRequest:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def _fake_store_image_blob(*args, **kwargs):
            return None

        fake_lcm_module.GenerateRequest = _FakeGenerateRequest
        fake_lcm_module._store_image_blob = _fake_store_image_blob
        original_lcm_module = sys.modules.get("server.lcm_sr_server")
        sys.modules["server.lcm_sr_server"] = fake_lcm_module

        fake_worker_pool_module = types.ModuleType("backends.worker_pool")

        class _FakeGenerationJob:
            def __init__(self, req, init_image=None):
                self.req = req
                self.init_image = init_image
                self.job_id = "backend-job-123"

        fake_worker_pool_module.GenerationJob = _FakeGenerationJob
        original_worker_pool_module = sys.modules.get("backends.worker_pool")
        sys.modules["backends.worker_pool"] = fake_worker_pool_module

        try:
            with patch("server.ws_routes.get_mode_config") as get_mode_config:
                get_mode_config.return_value = SimpleNamespace(
                    get_mode=lambda name: SimpleNamespace(
                        name=name,
                        default_size="512x512",
                        default_steps=4,
                        default_guidance=1.0,
                        resolution_options=[{"size": "512x512", "aspect_ratio": "1:1"}],
                    )
                )

                with client.websocket_connect("/v1/ws") as ws:
                    ws.receive_json()  # consume status
                    ws.send_json({
                        "type": "job:submit",
                        "id": "t-neg",
                        "jobType": "generate",
                        "params": {
                            "prompt": "a cat",
                            "negative_prompt": "blurry, watermark",
                            "scheduler_id": "euler",
                            "size": "512x512",
                            "num_inference_steps": 4,
                            "guidance_scale": 1.0,
                            "seed": 12345678,
                        },
                    })
                    ack = ws.receive_json()
                    assert ack["type"] == "job:ack"

                    done = ws.receive_json()
                    assert done["type"] == "job:complete"
                    assert done["jobId"] == ack["jobId"]

            submitted_job = pool.submit_job.call_args.args[0]
            assert ack["jobId"] == submitted_job.job_id
            assert submitted_job.req.negative_prompt == "blurry, watermark"
            assert submitted_job.req.scheduler_id == "euler"
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module
            if original_worker_pool_module is None:
                sys.modules.pop("backends.worker_pool", None)
            else:
                sys.modules["backends.worker_pool"] = original_worker_pool_module
            app.state.use_mode_system = False
            app.state.worker_pool = None

    def test_generate_mode_system_backend_failure_reports_job_error(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "sdxl-general"
        fut = MagicMock()
        fut.result.side_effect = RuntimeError("backend exploded")
        pool.submit_job.return_value = fut
        app.state.worker_pool = pool
        app.state.storage = None

        fake_lcm_module = types.ModuleType("server.lcm_sr_server")

        class _FakeGenerateRequest:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def _fake_store_image_blob(*args, **kwargs):
            return None

        fake_lcm_module.GenerateRequest = _FakeGenerateRequest
        fake_lcm_module._store_image_blob = _fake_store_image_blob
        original_lcm_module = sys.modules.get("server.lcm_sr_server")
        sys.modules["server.lcm_sr_server"] = fake_lcm_module

        fake_worker_pool_module = types.ModuleType("backends.worker_pool")

        class _FakeGenerationJob:
            def __init__(self, req, init_image=None):
                self.req = req
                self.init_image = init_image
                self.job_id = "backend-job-123"

        fake_worker_pool_module.GenerationJob = _FakeGenerationJob
        original_worker_pool_module = sys.modules.get("backends.worker_pool")
        sys.modules["backends.worker_pool"] = fake_worker_pool_module

        try:
            with patch("server.ws_routes.get_mode_config") as get_mode_config:
                get_mode_config.return_value = SimpleNamespace(
                    get_mode=lambda name: SimpleNamespace(
                        name=name,
                        default_size="512x512",
                        default_steps=4,
                        default_guidance=1.0,
                        resolution_options=[{"size": "512x512", "aspect_ratio": "1:1"}],
                    )
                )

                with client.websocket_connect("/v1/ws") as ws:
                    ws.receive_json()  # consume status
                    ws.send_json({
                        "type": "job:submit",
                        "id": "t-fail",
                        "jobType": "generate",
                        "params": {
                            "prompt": "a cat",
                            "size": "512x512",
                            "num_inference_steps": 4,
                            "guidance_scale": 1.0,
                            "seed": 12345678,
                        },
                    })
                    ack = ws.receive_json()
                    assert ack["type"] == "job:ack"

                    err = ws.receive_json()
                    assert err["type"] == "job:error"
                    assert err["jobId"] == ack["jobId"]
                    assert "backend exploded" in err["error"]
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module
            if original_worker_pool_module is None:
                sys.modules.pop("backends.worker_pool", None)
            else:
                sys.modules["backends.worker_pool"] = original_worker_pool_module
            app.state.use_mode_system = False
            app.state.worker_pool = None

    def test_generate_mode_system_rejects_invalid_size_before_submit(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "SDXL"
        app.state.worker_pool = pool
        app.state.storage = None

        fake_lcm_module = types.ModuleType("server.lcm_sr_server")

        class _FakeGenerateRequest:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def _fake_store_image_blob(*args, **kwargs):
            return None

        fake_lcm_module.GenerateRequest = _FakeGenerateRequest
        fake_lcm_module._store_image_blob = _fake_store_image_blob
        original_lcm_module = sys.modules.get("server.lcm_sr_server")
        sys.modules["server.lcm_sr_server"] = fake_lcm_module
        fake_worker_pool_module = types.ModuleType("backends.worker_pool")

        class _FakeGenerationJob:
            def __init__(self, req, init_image=None):
                self.req = req
                self.init_image = init_image
                self.job_id = "backend-job-123"

        fake_worker_pool_module.GenerationJob = _FakeGenerationJob
        original_worker_pool_module = sys.modules.get("backends.worker_pool")
        sys.modules["backends.worker_pool"] = fake_worker_pool_module

        try:
            with patch("server.ws_routes.get_mode_config") as get_mode_config:
                get_mode_config.return_value = SimpleNamespace(
                    get_mode=lambda name: SimpleNamespace(
                        name=name,
                        default_size="1024x1024",
                        default_steps=4,
                        default_guidance=1.0,
                        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
                    )
                )

                with client.websocket_connect("/v1/ws") as ws:
                    ws.receive_json()  # consume status
                    ws.send_json({
                        "type": "job:submit",
                        "id": "t-invalid-size",
                        "jobType": "generate",
                        "params": {
                            "prompt": "a cat",
                            "size": "768x768",
                            "num_inference_steps": 4,
                            "guidance_scale": 1.0,
                            "seed": 12345678,
                        },
                    })

                    ack = ws.receive_json()
                    assert ack["type"] == "job:ack"
                    assert ack["id"] == "t-invalid-size"

                    err = ws.receive_json()
                    assert err["type"] == "job:error"
                    assert err["jobId"] == ack["jobId"]
                    assert "size '768x768' is not allowed for mode 'SDXL'" in err["error"]

                pool.submit_job.assert_not_called()
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module
            if original_worker_pool_module is None:
                sys.modules.pop("backends.worker_pool", None)
            else:
                sys.modules["backends.worker_pool"] = original_worker_pool_module
            app.state.use_mode_system = False
            app.state.worker_pool = None

    def test_generate_mode_system_mode_lookup_failure_reports_ack_then_job_error(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "SDXL"
        app.state.worker_pool = pool
        app.state.storage = None

        fake_lcm_module = types.ModuleType("server.lcm_sr_server")

        class _FakeGenerateRequest:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        fake_lcm_module.GenerateRequest = _FakeGenerateRequest
        fake_lcm_module._store_image_blob = lambda *args, **kwargs: None
        original_lcm_module = sys.modules.get("server.lcm_sr_server")
        sys.modules["server.lcm_sr_server"] = fake_lcm_module

        fake_worker_pool_module = types.ModuleType("backends.worker_pool")

        class _FakeGenerationJob:
            def __init__(self, req, init_image=None):
                self.req = req
                self.init_image = init_image
                self.job_id = "backend-job-123"

        fake_worker_pool_module.GenerationJob = _FakeGenerationJob
        original_worker_pool_module = sys.modules.get("backends.worker_pool")
        sys.modules["backends.worker_pool"] = fake_worker_pool_module

        try:
            with patch("server.ws_routes.get_mode_config", side_effect=RuntimeError("mode config unavailable")):
                with client.websocket_connect("/v1/ws") as ws:
                    ws.receive_json()  # consume status
                    ws.send_json({
                        "type": "job:submit",
                        "id": "t-mode-config",
                        "jobType": "generate",
                        "params": {
                            "prompt": "a cat",
                            "size": "512x512",
                            "num_inference_steps": 4,
                            "guidance_scale": 1.0,
                            "seed": 12345678,
                        },
                    })

                    ack = ws.receive_json()
                    assert ack["type"] == "job:ack"
                    assert ack["id"] == "t-mode-config"

                    err = ws.receive_json()
                    assert err["type"] == "job:error"
                    assert err["jobId"] == ack["jobId"]
                    assert "mode config unavailable" in err["error"]

                pool.submit_job.assert_not_called()
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module
            if original_worker_pool_module is None:
                sys.modules.pop("backends.worker_pool", None)
            else:
                sys.modules["backends.worker_pool"] = original_worker_pool_module
            app.state.use_mode_system = False
            app.state.worker_pool = None

    def test_generate_mode_system_queue_full_reports_ack_then_job_error(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "SDXL"
        pool.submit_job.side_effect = queue.Full()
        app.state.worker_pool = pool
        app.state.storage = None

        fake_lcm_module = types.ModuleType("server.lcm_sr_server")

        class _FakeGenerateRequest:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        fake_lcm_module.GenerateRequest = _FakeGenerateRequest
        fake_lcm_module._store_image_blob = lambda *args, **kwargs: None
        original_lcm_module = sys.modules.get("server.lcm_sr_server")
        sys.modules["server.lcm_sr_server"] = fake_lcm_module

        fake_worker_pool_module = types.ModuleType("backends.worker_pool")

        class _FakeGenerationJob:
            def __init__(self, req, init_image=None):
                self.req = req
                self.init_image = init_image
                self.job_id = "backend-job-123"

        fake_worker_pool_module.GenerationJob = _FakeGenerationJob
        original_worker_pool_module = sys.modules.get("backends.worker_pool")
        sys.modules["backends.worker_pool"] = fake_worker_pool_module

        try:
            with patch("server.ws_routes.get_mode_config") as get_mode_config:
                get_mode_config.return_value = SimpleNamespace(
                    get_mode=lambda name: SimpleNamespace(
                        name=name,
                        default_size="512x512",
                        default_steps=4,
                        default_guidance=1.0,
                        resolution_options=[{"size": "512x512", "aspect_ratio": "1:1"}],
                    )
                )

                with client.websocket_connect("/v1/ws") as ws:
                    ws.receive_json()  # consume status
                    ws.send_json({
                        "type": "job:submit",
                        "id": "t-queue-full",
                        "jobType": "generate",
                        "params": {
                            "prompt": "a cat",
                            "size": "512x512",
                            "num_inference_steps": 4,
                            "guidance_scale": 1.0,
                            "seed": 12345678,
                        },
                    })

                    ack = ws.receive_json()
                    assert ack["type"] == "job:ack"
                    assert ack["id"] == "t-queue-full"

                    err = ws.receive_json()
                    assert err["type"] == "job:error"
                    assert err["jobId"] == ack["jobId"]
                    assert err["error"] == "Queue full"
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module
            if original_worker_pool_module is None:
                sys.modules.pop("backends.worker_pool", None)
            else:
                sys.modules["backends.worker_pool"] = original_worker_pool_module
            app.state.use_mode_system = False
            app.state.worker_pool = None

    def test_generate_mode_system_backend_cancellation_reports_job_error(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "sdxl-general"
        fut = MagicMock()
        fut.result.side_effect = concurrent.futures.CancelledError()
        pool.submit_job.return_value = fut
        app.state.worker_pool = pool
        app.state.storage = None

        fake_lcm_module = types.ModuleType("server.lcm_sr_server")

        class _FakeGenerateRequest:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def _fake_store_image_blob(*args, **kwargs):
            return None

        fake_lcm_module.GenerateRequest = _FakeGenerateRequest
        fake_lcm_module._store_image_blob = _fake_store_image_blob
        original_lcm_module = sys.modules.get("server.lcm_sr_server")
        sys.modules["server.lcm_sr_server"] = fake_lcm_module

        fake_worker_pool_module = types.ModuleType("backends.worker_pool")

        class _FakeGenerationJob:
            def __init__(self, req, init_image=None):
                self.req = req
                self.init_image = init_image
                self.job_id = "backend-job-123"

        fake_worker_pool_module.GenerationJob = _FakeGenerationJob
        original_worker_pool_module = sys.modules.get("backends.worker_pool")
        sys.modules["backends.worker_pool"] = fake_worker_pool_module

        try:
            with patch("server.ws_routes.get_mode_config") as get_mode_config:
                get_mode_config.return_value = SimpleNamespace(
                    get_mode=lambda name: SimpleNamespace(
                        name=name,
                        default_size="512x512",
                        default_steps=4,
                        default_guidance=1.0,
                        resolution_options=[{"size": "512x512", "aspect_ratio": "1:1"}],
                    )
                )

                with client.websocket_connect("/v1/ws") as ws:
                    ws.receive_json()  # consume status
                    ws.send_json({
                        "type": "job:submit",
                        "id": "t-cancel",
                        "jobType": "generate",
                        "params": {
                            "prompt": "a cat",
                            "size": "512x512",
                            "num_inference_steps": 4,
                            "guidance_scale": 1.0,
                            "seed": 12345678,
                        },
                    })
                    ack = ws.receive_json()
                    assert ack["type"] == "job:ack"

                    err = ws.receive_json()
                    assert err["type"] == "job:error"
                    assert err["jobId"] == ack["jobId"]
                    assert err["error"] == "Cancelled by backend"
        finally:
            if original_lcm_module is None:
                sys.modules.pop("server.lcm_sr_server", None)
            else:
                sys.modules["server.lcm_sr_server"] = original_lcm_module
            if original_worker_pool_module is None:
                sys.modules.pop("backends.worker_pool", None)
            else:
                sys.modules["backends.worker_pool"] = original_worker_pool_module
            app.state.use_mode_system = False
            app.state.worker_pool = None

    def test_unknown_job_type(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({
                "type": "job:submit",
                "id": "t2",
                "jobType": "invalid_type",
                "params": {},
            })
            ack = ws.receive_json()
            assert ack["type"] == "job:ack"
            # Should get an error for unknown jobType
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Unknown jobType" in err["error"]


# ---------------------------------------------------------------------------
# job:cancel / job:priority stubs
# ---------------------------------------------------------------------------

class TestJobStubs:
    def test_job_cancel_ack_reports_backend_cancel_result(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "sdxl-general"
        pool.cancel_job.return_value = {"status": "canceled", "job_id": "abc123"}
        app.state.worker_pool = pool

        try:
            with client.websocket_connect("/v1/ws") as ws:
                ws.receive_json()  # consume status
                ws.send_json({"type": "job:cancel", "id": "c1", "jobId": "abc123"})
                msg = ws.receive_json()
                assert msg["type"] == "job:cancel:ack"
                assert msg["detail"] == "canceled"
            pool.cancel_job.assert_called_once_with("abc123")
        finally:
            app.state.use_mode_system = False
            app.state.worker_pool = None

    def test_job_cancel_ack_reports_not_found_when_backend_cannot_cancel(self):
        app.state.use_mode_system = True
        pool = MagicMock()
        pool.get_current_mode.return_value = "sdxl-general"
        pool.cancel_job.return_value = False
        app.state.worker_pool = pool

        try:
            with client.websocket_connect("/v1/ws") as ws:
                ws.receive_json()  # consume status
                ws.send_json({"type": "job:cancel", "id": "c2", "jobId": "missing"})
                msg = ws.receive_json()
                assert msg["type"] == "job:cancel:ack"
                assert msg["detail"] == "not_found"
            pool.cancel_job.assert_called_once_with("missing")
        finally:
            app.state.use_mode_system = False
            app.state.worker_pool = None

    def test_job_priority_ack(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "job:priority", "id": "p1"})
            msg = ws.receive_json()
            assert msg["type"] == "job:priority:ack"


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

class TestUpload:
    def test_upload_returns_file_ref(self):
        """POST /v1/upload should return a fileRef."""
        import io
        data = b"fake image bytes"
        resp = client.post(
            "/v1/upload",
            files={"file": ("test.png", io.BytesIO(data), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "fileRef" in body
        ref = body["fileRef"]
        assert len(ref) == 32  # hex uuid

        # Verify stored
        from server.upload_routes import resolve_file_ref
        assert resolve_file_ref(ref) == data

    def test_upload_empty_file_400(self):
        import io
        resp = client.post(
            "/v1/upload",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert resp.status_code == 400

    def test_resolve_unknown_ref_raises(self):
        from server.upload_routes import resolve_file_ref
        with pytest.raises(KeyError):
            resolve_file_ref("nonexistent")
