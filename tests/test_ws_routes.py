"""
Tests for server/ws_routes.py — WebSocket endpoint integration tests.

Uses Starlette TestClient with real WebSocket connections against a
minimal FastAPI app that mounts the WS router.
"""

import asyncio
import json
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    def test_job_cancel_ack(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "job:cancel", "id": "c1", "jobId": "j1"})
            msg = ws.receive_json()
            assert msg["type"] == "job:cancel:ack"

    def test_job_priority_ack(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "job:priority", "id": "p1"})
            msg = ws.receive_json()
            assert msg["type"] == "job:priority:ack"


# ---------------------------------------------------------------------------
# dream:* — errors when worker not initialized
# ---------------------------------------------------------------------------

class TestDreamHandlers:
    """Dream handlers should return an error when worker is unavailable.

    Outside Docker, the yume module may fail to import (missing /app/logs/torch.log).
    Both "not initialized" and import errors are valid — the key assertion is
    that the client receives a {"type": "error", ...} envelope, not a crash.
    """

    def test_dream_start_no_worker(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({
                "type": "dream:start",
                "id": "d1",
                "params": {"prompt": "sunset"},
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "error" in msg

    def test_dream_stop_no_worker(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "dream:stop", "id": "d2"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_dream_top_no_worker(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "dream:top", "id": "d3"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_dream_guide_no_worker(self):
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()  # consume status
            ws.send_json({"type": "dream:guide", "id": "d4", "params": {"prompt": "new"}})
            msg = ws.receive_json()
            assert msg["type"] == "error"


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
