"""The /superres handler must NOT run the blocking upscale on the event loop.

A mag-2 CUDA upscale takes seconds; if it runs inline in the async handler it
freezes the single asyncio loop for its whole duration, stalling every other
request (/status, /health, WS) — observed as status-poll timeouts during SR
even in WORKER_ISOLATION=subprocess mode (superres is a 2nd in-parent GPU
consumer, STABL-qfjfflrx). The fix offloads it via asyncio.to_thread; these
tests lock that behavior by asserting submit_superres runs OFF the loop thread.
"""
import threading
from types import SimpleNamespace

import pytest


class _FakeUpload:
    filename = "in.png"

    async def read(self):
        return b"imgdata"


async def test_superres_offloads_blocking_call_off_event_loop(monkeypatch):
    from server import lcm_sr_server

    # asyncio_mode=auto -> this coroutine runs ON the event-loop thread.
    loop_thread = threading.current_thread()
    ran_on = {}

    def fake_submit(**kwargs):
        ran_on["thread"] = threading.current_thread()
        return b"sr-bytes"

    monkeypatch.setattr(lcm_sr_server, "submit_superres", fake_submit)
    monkeypatch.setattr(lcm_sr_server, "build_superres_headers", lambda *a, **k: {})
    monkeypatch.setattr(lcm_sr_server.app.state, "sr_service", object(), raising=False)
    monkeypatch.setattr(lcm_sr_server.app.state, "sr_settings",
                        SimpleNamespace(sr_request_timeout=30.0), raising=False)
    monkeypatch.setattr(lcm_sr_server.app.state, "storage", None, raising=False)

    resp = await lcm_sr_server.superres(
        file=_FakeUpload(), magnitude=2, out_format="png", quality=92
    )

    assert resp.body == b"sr-bytes"
    # The blocking call must have been offloaded — a worker thread, not the loop.
    assert ran_on["thread"] is not loop_thread


async def test_superres_v1_also_offloads(monkeypatch):
    """/v1/superres delegates to superres(), so it inherits the offload."""
    from server import lcm_sr_server

    loop_thread = threading.current_thread()
    ran_on = {}

    def fake_submit(**kwargs):
        ran_on["thread"] = threading.current_thread()
        return b"x"

    monkeypatch.setattr(lcm_sr_server, "submit_superres", fake_submit)
    monkeypatch.setattr(lcm_sr_server, "build_superres_headers", lambda *a, **k: {})
    monkeypatch.setattr(lcm_sr_server.app.state, "sr_service", object(), raising=False)
    monkeypatch.setattr(lcm_sr_server.app.state, "sr_settings",
                        SimpleNamespace(sr_request_timeout=30.0), raising=False)
    monkeypatch.setattr(lcm_sr_server.app.state, "storage", None, raising=False)

    await lcm_sr_server.superres_v1(
        file=_FakeUpload(), magnitude=2, out_format="png", quality=92
    )
    assert ran_on["thread"] is not loop_thread
