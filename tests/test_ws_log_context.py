"""STABL-bpsfmoke: job_id correlation on the event loop.

Two layers:
- the contextvar property the WS design LEANS on (create_task copies the context)
- the wiring, driven through a real WebSocket against the same minimal app
  tests/test_ws_routes.py uses
"""
import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import log_context, ws_routes
from server.ws_hub import hub
from server.ws_routes import ws_router


# ---------------------------------------------------------------------------
# The property the design leans on
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_task_created_under_a_bind_INHERITS_the_id():
    """If this ever stops holding, every _run_generate log line loses its job_id
    silently — nothing else in the suite would notice."""
    seen = []

    async def child():
        seen.append(log_context.current_job_id())

    with log_context.bind_job_id("job-42"):
        await asyncio.create_task(child())
    assert seen == ["job-42"]


@pytest.mark.asyncio
async def test_a_task_created_AFTER_the_bind_exits_does_not_see_it():
    seen = []

    async def child():
        seen.append(log_context.current_job_id())

    with log_context.bind_job_id("job-42"):
        pass
    await asyncio.create_task(child())
    assert seen == [None]


# ---------------------------------------------------------------------------
# Wiring, through a real socket
# ---------------------------------------------------------------------------

def _make_test_app():
    app = FastAPI()
    app.include_router(ws_router)
    app.state.use_mode_system = False
    app.state.service = None
    app.state.sr_service = None
    app.state.storage = None
    return app


_app = _make_test_app()
_client_cm = TestClient(_app)
client = _client_cm.__enter__()


@pytest.fixture(scope="module", autouse=True)
def _close_test_client():
    yield
    _client_cm.__exit__(None, None, None)


@pytest.fixture
def _probe_handler():
    """A handler that reports whatever job_id is bound when the loop calls it.

    Registered and removed around the test so HANDLERS is not mutated for the rest
    of the session.
    """
    async def _probe(ws, msg, client_id):
        return {"type": "probe:seen", "jobId": log_context.current_job_id()}

    ws_routes.HANDLERS["probe:read"] = _probe
    yield
    ws_routes.HANDLERS.pop("probe:read", None)


def test_job_submit_binds_the_minted_id_into_the_generate_task(monkeypatch):
    """handle_job_submit sets the var; asyncio.create_task copies the context, so
    the generation task inherits it without a bind of its own."""
    async def _fake_run_generate(ws, client_id, job_id, params):
        await hub.send(client_id, {
            "type": "probe:generate",
            "jobId": job_id,
            "seen": log_context.current_job_id(),
        })

    monkeypatch.setattr(ws_routes, "_run_generate", _fake_run_generate)

    with client.websocket_connect("/v1/ws") as ws:
        ws.receive_json()   # system:status on connect
        ws.send_json({"type": "job:submit", "id": "c1", "jobType": "generate",
                      "params": {"prompt": "x"}})
        ack = ws.receive_json()
        probe = ws.receive_json()

    assert ack["type"] == "job:ack"
    assert probe["type"] == "probe:generate"
    assert probe["seen"] == ack["jobId"]


def test_a_handler_set_id_does_NOT_leak_to_the_next_message_on_the_SAME_socket(
    monkeypatch, _probe_handler
):
    """The failure this task exists to prevent. Two messages, one connection: the
    second must not inherit the first's correlation id."""
    async def _fake_run_generate(ws, client_id, job_id, params):
        return None

    monkeypatch.setattr(ws_routes, "_run_generate", _fake_run_generate)

    with client.websocket_connect("/v1/ws") as ws:
        ws.receive_json()   # system:status
        ws.send_json({"type": "job:submit", "id": "c1", "jobType": "generate",
                      "params": {"prompt": "x"}})
        ack = ws.receive_json()
        ws.send_json({"type": "probe:read", "id": "c2"})
        probe = ws.receive_json()

    assert ack["jobId"]                     # a real id WAS minted and bound
    assert probe["type"] == "probe:seen"
    assert probe["jobId"] is None           # and it did not survive the message


def test_the_loop_starts_each_message_clean_even_with_no_prior_submit(_probe_handler):
    with client.websocket_connect("/v1/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "probe:read", "id": "c1"})
        assert ws.receive_json()["jobId"] is None
