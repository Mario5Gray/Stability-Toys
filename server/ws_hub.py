"""
ws_hub.py — WebSocket connection manager + event bus.

Singleton hub for managing WS clients and broadcasting messages.
All messages are JSON envelopes: {"type": "domain:action", ...}
"""

import asyncio
import logging
from typing import Dict

from fastapi import WebSocket

from server.metrics import record

logger = logging.getLogger(__name__)


def _count_out(msg: dict) -> None:
    """Count one outbound message actually written to a socket.

    Instrumented here rather than in websocket_endpoint because
    _status_broadcaster calls broadcast() every 5s entirely outside that loop —
    the most frequent outbound message would otherwise be invisible.

    Outbound `type` is SERVER-generated, so it is bounded by our own vocabulary.
    """
    msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else "unknown"
    record(lambda met: met.ws_messages_total.labels(
        type=msg_type, direction="out").inc())


class WSHub:
    def __init__(self):
        self._clients: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, client_id: str) -> None:
        async with self._lock:
            self._clients[client_id] = ws
            count = len(self._clients)
        logger.info("WS client connected: %s (%d total)", client_id, count)
        record(lambda met: (
            met.ws_sessions_total.inc(),
            met.ws_connections_active.set(count),
        ))

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)
            count = len(self._clients)
        logger.info("WS client disconnected: %s (%d total)", client_id, count)
        # SET, never dec: disconnect is idempotent on the hub (a double
        # disconnect pops nothing) and both send() and broadcast() reap dead
        # clients through this same path, so an inc/dec pair would drift negative.
        record(lambda met: met.ws_connections_active.set(count))

    async def send(self, client_id: str, msg: dict) -> None:
        """Send to one client. Removes dead clients on failure."""
        async with self._lock:
            ws = self._clients.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_json(msg)
        except Exception as e:
            logger.warning("Failed to send to %s: %s: %s, removing", client_id, type(e).__name__, e)
            await self.disconnect(client_id)
        else:
            _count_out(msg)

    async def broadcast(self, msg: dict) -> None:
        """Send to all connected clients. Tolerates failures."""
        async with self._lock:
            snapshot = list(self._clients.items())
        dead = []
        for cid, ws in snapshot:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(cid)
            else:
                # Per RECIPIENT, not per broadcast call: that is the number that
                # reflects actual socket writes.
                _count_out(msg)
        for cid in dead:
            await self.disconnect(cid)

    @property
    def client_count(self) -> int:
        return len(self._clients)


hub = WSHub()
