"""
ws_hub.py — WebSocket connection manager + event bus.

Singleton hub for managing WS clients and broadcasting messages.
All messages are JSON envelopes: {"type": "domain:action", ...}
"""

import asyncio
import logging
from typing import Any, Dict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WSHub:
    def __init__(self):
        self._clients: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, client_id: str) -> None:
        async with self._lock:
            self._clients[client_id] = ws
        logger.info("WS client connected: %s (%d total)", client_id, len(self._clients))

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)
        logger.info("WS client disconnected: %s (%d total)", client_id, len(self._clients))

    async def send(self, client_id: str, msg: dict) -> None:
        """Send to one client. Removes dead clients on failure."""
        async with self._lock:
            ws = self._clients.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_json(msg)
        except Exception:
            logger.warning("Failed to send to %s, removing", client_id)
            await self.disconnect(client_id)

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
        for cid in dead:
            await self.disconnect(cid)

    @property
    def client_count(self) -> int:
        return len(self._clients)


hub = WSHub()
