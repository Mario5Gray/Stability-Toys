"""
Tests for server/ws_hub.py — WSHub connection manager.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

# Ensure project root importable
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.ws_hub import WSHub


def _make_ws(*, fail_send=False):
    """Create a mock WebSocket."""
    ws = AsyncMock()
    if fail_send:
        ws.send_json.side_effect = RuntimeError("connection closed")
    return ws


@pytest.mark.asyncio
async def test_connect_disconnect():
    hub = WSHub()
    ws = _make_ws()
    await hub.connect(ws, "c1")
    assert hub.client_count == 1
    await hub.disconnect("c1")
    assert hub.client_count == 0


@pytest.mark.asyncio
async def test_disconnect_unknown_is_noop():
    hub = WSHub()
    await hub.disconnect("nonexistent")
    assert hub.client_count == 0


@pytest.mark.asyncio
async def test_send_to_client():
    hub = WSHub()
    ws = _make_ws()
    await hub.connect(ws, "c1")
    await hub.send("c1", {"type": "pong"})
    ws.send_json.assert_awaited_once_with({"type": "pong"})


@pytest.mark.asyncio
async def test_send_to_unknown_client_is_noop():
    hub = WSHub()
    await hub.send("nonexistent", {"type": "pong"})  # should not raise


@pytest.mark.asyncio
async def test_send_removes_dead_client():
    hub = WSHub()
    ws = _make_ws(fail_send=True)
    await hub.connect(ws, "dead")
    assert hub.client_count == 1
    await hub.send("dead", {"type": "test"})
    assert hub.client_count == 0


@pytest.mark.asyncio
async def test_broadcast():
    hub = WSHub()
    ws1 = _make_ws()
    ws2 = _make_ws()
    await hub.connect(ws1, "c1")
    await hub.connect(ws2, "c2")

    msg = {"type": "system:status", "mode": "test"}
    await hub.broadcast(msg)

    ws1.send_json.assert_awaited_once_with(msg)
    ws2.send_json.assert_awaited_once_with(msg)


@pytest.mark.asyncio
async def test_broadcast_removes_dead_clients():
    hub = WSHub()
    ws_good = _make_ws()
    ws_bad = _make_ws(fail_send=True)
    await hub.connect(ws_good, "good")
    await hub.connect(ws_bad, "bad")
    assert hub.client_count == 2

    await hub.broadcast({"type": "test"})

    assert hub.client_count == 1
    ws_good.send_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiple_connect_same_id_replaces():
    hub = WSHub()
    ws1 = _make_ws()
    ws2 = _make_ws()
    await hub.connect(ws1, "c1")
    await hub.connect(ws2, "c1")
    assert hub.client_count == 1
    await hub.send("c1", {"type": "test"})
    # Should send to ws2, not ws1
    ws2.send_json.assert_awaited_once()
    ws1.send_json.assert_not_awaited()
