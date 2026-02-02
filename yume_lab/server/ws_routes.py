

import asyncio
import json
import logging
import os
import time
import uuid
import queue
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.ws_hub import hub
from server.upload_routes import resolve_file_ref
from invokers.jobs import (
    jobs_put, jobs_get, set_on_update,
)

logger = logging.getLogger(__name__)

ws_router = APIRouter()


# ---------------------------------------------------------------------------
# dream:start / stop / top / guide
# ---------------------------------------------------------------------------

@_handler("dream:start")
async def handle_dream_start(ws: WebSocket, msg: dict, client_id: str) -> dict:
    from yume.dream_worker import get_dream_worker
    worker = get_dream_worker()
    if worker is None:
        return _error("Dream worker not initialized", msg.get("id"))

    p = msg.get("params", {})
    result = await worker.start_dreaming(
        base_prompt=p.get("prompt", ""),
        duration_hours=p.get("duration_hours", 1.0),
        temperature=p.get("temperature", 0.5),
        similarity_threshold=p.get("similarity_threshold", 0.7),
        render_interval=p.get("render_interval", 100),
        exploration_strategy=p.get("strategy", "random"),
    )
    return {"type": "dream:started", "id": msg.get("id"), **result}


@_handler("dream:stop")
async def handle_dream_stop(ws: WebSocket, msg: dict, client_id: str) -> dict:
    from yume.dream_worker import get_dream_worker
    worker = get_dream_worker()
    if worker is None:
        return _error("Dream worker not initialized", msg.get("id"))
    result = worker.stop_dreaming()
    return {"type": "dream:stopped", "id": msg.get("id"), **result}


@_handler("dream:top")
async def handle_dream_top(ws: WebSocket, msg: dict, client_id: str) -> dict:
    from yume.dream_worker import get_dream_worker
    worker = get_dream_worker()
    if worker is None:
        return _error("Dream worker not initialized", msg.get("id"))

    limit = msg.get("limit", 50)
    min_score = msg.get("min_score", 0.0)
    results = await worker.get_top_dreams(limit, min_score)
    return {"type": "dream:top:result", "id": msg.get("id"), "dreams": results or []}


@_handler("dream:guide")
async def handle_dream_guide(ws: WebSocket, msg: dict, client_id: str) -> dict:
    from yume.dream_worker import get_dream_worker
    worker = get_dream_worker()
    if worker is None:
        return _error("Dream worker not initialized", msg.get("id"))

    p = msg.get("params", {})
    # Update base params on the worker if it supports it
    updated = {}
    if hasattr(worker, "base_prompt") and "prompt" in p:
        worker.base_prompt = p["prompt"]
        updated["prompt"] = p["prompt"]
    if hasattr(worker, "temperature") and "temperature" in p:
        worker.temperature = p["temperature"]
        updated["temperature"] = p["temperature"]

    return {"type": "dream:guide:ack", "id": msg.get("id"), "updated": updated}
