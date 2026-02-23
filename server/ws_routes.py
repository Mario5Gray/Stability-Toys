"""
ws_routes.py — WebSocket endpoint + message dispatcher.

Single route: /v1/ws
All messages are JSON envelopes: {"type": "domain:action", ...}
"""

import asyncio
import json
import logging
import os
import time
import uuid
import queue
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
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
# Job update callback → WS push
# ---------------------------------------------------------------------------

def _on_job_update(job_id: str, snapshot: dict) -> None:
    """
    Called from invokers/jobs.py on every mutation (from any thread).
    Schedules a broadcast of job:progress via the hub.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    msg = {
        "type": "job:progress",
        "jobId": job_id,
        "status": snapshot.get("status"),
        "progress": snapshot.get("progress"),
    }

    if loop is not None and loop.is_running():
        loop.create_task(hub.broadcast(msg))
    else:
        # From a worker thread — need to schedule onto the event loop
        # We'll store the loop ref at startup (set in register_job_hook)
        _loop = getattr(_on_job_update, "_loop", None)
        if _loop is not None:
            _loop.call_soon_threadsafe(asyncio.ensure_future, hub.broadcast(msg))


def register_job_hook() -> None:
    """Call once at startup to wire jobs.py → WS push."""
    _on_job_update._loop = asyncio.get_running_loop()
    set_on_update(_on_job_update)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(msg: str, corr_id: str = None) -> dict:
    d = {"type": "error", "error": msg}
    if corr_id:
        d["id"] = corr_id
    return d


def _get_app_state(ws: WebSocket):
    return ws.app.state


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

HANDLERS: Dict[str, Any] = {}  # populated below

# Track running async tasks so we can cancel them
_running_tasks: Dict[str, asyncio.Task] = {}


def _track_task(job_id: str, task: asyncio.Task) -> None:
    _running_tasks[job_id] = task
    task.add_done_callback(lambda _: _running_tasks.pop(job_id, None))


def _handler(msg_type: str):
    def decorator(fn):
        HANDLERS[msg_type] = fn
        return fn
    return decorator


def _post_bytes(url: str, body: bytes, content_type: str) -> int:
    req = urlrequest.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=5) as resp:
        return resp.status


# ---------------------------------------------------------------------------
# ping / pong
# ---------------------------------------------------------------------------

@_handler("ping")
async def handle_ping(ws: WebSocket, msg: dict, client_id: str) -> dict:
    return {"type": "pong"}


# ---------------------------------------------------------------------------
# job:submit
# ---------------------------------------------------------------------------

@_handler("job:submit")
async def handle_job_submit(ws: WebSocket, msg: dict, client_id: str) -> None:
    corr_id = msg.get("id")
    job_type = msg.get("jobType", "generate")
    params = msg.get("params", {})
    job_id = uuid.uuid4().hex[:12]

    # Ack immediately
    await hub.send(client_id, {
        "type": "job:ack",
        "id": corr_id,
        "jobId": job_id,
    })

    if job_type == "generate":
        t = asyncio.create_task(_run_generate(ws, client_id, job_id, params))
        _track_task(job_id, t)
    elif job_type == "comfy":
        t = asyncio.create_task(_run_comfy(ws, client_id, job_id, msg))
        _track_task(job_id, t)
    elif job_type == "sr":
        t = asyncio.create_task(_run_sr(ws, client_id, job_id, msg))
        _track_task(job_id, t)
    else:
        await hub.send(client_id, _error(f"Unknown jobType: {job_type}", corr_id))


# ---------------------------------------------------------------------------
# telemetry:otlp
# ---------------------------------------------------------------------------

@_handler("telemetry:otlp")
async def handle_telemetry_otlp(ws: WebSocket, msg: dict, client_id: str) -> dict:
    endpoint = os.environ.get("OTEL_PROXY_ENDPOINT", "").strip()
    if not endpoint:
        return {"type": "telemetry:ack", "id": msg.get("id"), "status": "noop"}

    payload = msg.get("payload")
    if payload is None:
        return _error("Missing payload", msg.get("id"))

    content_type = msg.get("contentType", "application/json")
    try:
        body = json.dumps(payload).encode("utf-8")
        status = await asyncio.to_thread(_post_bytes, endpoint, body, content_type)
    except HTTPError as e:
        logger.warning("[telemetry] collector error %s", e)
        return _error("collector error", msg.get("id"))
    except URLError as e:
        logger.warning("[telemetry] collector unavailable %s", e)
        return _error("collector unavailable", msg.get("id"))

    return {"type": "telemetry:ack", "id": msg.get("id"), "status": status}


# ---------------------------------------------------------------------------
# job:cancel (stub)
# ---------------------------------------------------------------------------

@_handler("job:cancel")
async def handle_job_cancel(ws: WebSocket, msg: dict, client_id: str) -> dict:
    job_id = msg.get("jobId")
    task = _running_tasks.get(job_id)
    if task and not task.done():
        task.cancel()
        return {"type": "job:cancel:ack", "id": msg.get("id"), "jobId": job_id, "detail": "cancelled"}
    return {"type": "job:cancel:ack", "id": msg.get("id"), "jobId": job_id, "detail": "no running task found"}


# ---------------------------------------------------------------------------
# job:priority (stub)
# ---------------------------------------------------------------------------

@_handler("job:priority")
async def handle_job_priority(ws: WebSocket, msg: dict, client_id: str) -> dict:
    return {"type": "job:priority:ack", "id": msg.get("id"), "detail": "priority not yet implemented"}


# ---------------------------------------------------------------------------
# Generate job runner
# ---------------------------------------------------------------------------

async def _run_generate(ws: WebSocket, client_id: str, job_id: str, params: dict) -> None:
    """Run a generate job using the same code path as POST /generate."""
    try:
        state = _get_app_state(ws)
        from server.lcm_sr_server import GenerateRequest, _store_image_blob

        req = GenerateRequest(
            prompt=params.get("prompt", ""),
            size=params.get("size", os.environ.get("DEFAULT_SIZE", "512x512")),
            num_inference_steps=params.get("num_inference_steps", params.get("steps", 4)),
            guidance_scale=params.get("guidance_scale", params.get("cfg", 1.0)),
            seed=params.get("seed"),
            superres=params.get("superres", False),
            superres_magnitude=params.get("superres_magnitude", 2),
            denoise_strength=params.get("denoise_strength", 0.75),
        )

        # Resolve optional init image reference
        init_image_bytes = None
        init_image_ref = params.get("init_image_ref")
        if init_image_ref:
            try:
                init_image_bytes = resolve_file_ref(init_image_ref)
            except KeyError as e:
                await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": str(e)})
                return

        # Submit to appropriate backend
        if getattr(state, "use_mode_system", False):
            from backends.worker_pool import GenerationJob
            pool = state.worker_pool
            job = GenerationJob(req=req, init_image=init_image_bytes)
            try:
                fut = pool.submit_job(job)
            except queue.Full:
                await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Queue full"})
                return
        else:
            service = state.service
            fut = service.submit(req, timeout_s=0.25)

        timeout = float(os.environ.get("DEFAULT_TIMEOUT", "120"))

        # Run blocking future in thread
        loop = asyncio.get_running_loop()
        png_bytes, seed = await loop.run_in_executor(None, lambda: fut.result(timeout=timeout))

        out_bytes = png_bytes
        did_sr = False
        sr_mag = int(req.superres_magnitude or 2)

        # Optional super-resolution
        if req.superres:
            sr_service = getattr(state, "sr_service", None)
            if sr_service is not None:
                sr_timeout = float(os.environ.get("SR_REQUEST_TIMEOUT", "120"))
                sr_fut = sr_service.submit(
                    image_bytes=png_bytes,
                    out_format=req.superres_format,
                    quality=req.superres_quality,
                    magnitude=sr_mag,
                    timeout_s=0.25,
                )
                out_bytes = await loop.run_in_executor(None, lambda: sr_fut.result(timeout=sr_timeout))
                did_sr = True

        # Store in storage
        storage = getattr(state, "storage", None)
        image_key = _store_image_blob(
            storage,
            out_bytes=out_bytes,
            media_type="image/png",
            req=req,
            seed=int(seed),
            did_superres=did_sr,
            sr_mag=sr_mag,
        )

        outputs = []
        if image_key:
            outputs.append({"url": f"/storage/{image_key}", "key": image_key})

        await hub.send(client_id, {
            "type": "job:complete",
            "jobId": job_id,
            "outputs": outputs,
            "meta": {
                "seed": int(seed),
                "backend": os.environ.get("BACKEND", "auto"),
                "sr": did_sr,
            },
        })

    except asyncio.CancelledError:
        logger.info("Generate job %s cancelled by client", job_id)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Cancelled by client"})

    except Exception as e:
        logger.error("Generate job %s failed: %s", job_id, e, exc_info=True)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": str(e)})


# ---------------------------------------------------------------------------
# Comfy job runner
# ---------------------------------------------------------------------------

async def _run_comfy(ws: WebSocket, client_id: str, job_id: str, msg: dict) -> None:
    """Run a comfy job, reusing _run_job from comfy_routes."""
    try:
        from server.comfy_routes import _run_job, inv as comfy_inv

        params = msg.get("params", {})
        workflow_id = msg.get("workflowId")
        file_ref = msg.get("inputImage", "").replace("fileRef:", "")

        if not workflow_id:
            await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Missing workflowId"})
            return

        # Resolve file ref
        try:
            image_bytes = resolve_file_ref(file_ref)
        except KeyError as e:
            await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": str(e)})
            return

        # Upload image to ComfyUI
        up = comfy_inv.upload_image(image_bytes, filename=f"{job_id}.png", image_type="input")
        uploaded = {
            "name": up.get("name") or up.get("filename"),
            "subfolder": up.get("subfolder", ""),
            "type": up.get("type", "input"),
        }

        # Create job record
        jobs_put(job_id, {
            "id": job_id,
            "workflowId": workflow_id,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "outputs": [],
            "heartbeat_at": None,
            "updated_at": time.time(),
            "comfy": {"client_id": None, "prompt_id": None},
            "progress": {
                "nodes_total": 0, "nodes_seen": 0,
                "current_node": None, "node_progression": [],
                "fraction": 0.0,
            },
        })

        # Run in thread (blocking ComfyUI WS)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_job, job_id, workflow_id, params, uploaded)

        # Get final state
        final = jobs_get(job_id)
        if final and final.get("status") == "done":
            await hub.send(client_id, {
                "type": "job:complete",
                "jobId": job_id,
                "outputs": final.get("outputs", []),
            })
        else:
            await hub.send(client_id, {
                "type": "job:error",
                "jobId": job_id,
                "error": (final or {}).get("error", "Unknown error"),
            })

    except Exception as e:
        logger.error("Comfy job %s failed: %s", job_id, e, exc_info=True)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": str(e)})


# ---------------------------------------------------------------------------
# SR job runner
# ---------------------------------------------------------------------------

async def _run_sr(ws: WebSocket, client_id: str, job_id: str, msg: dict) -> None:
    """Run a standalone super-resolution job."""
    try:
        state = _get_app_state(ws)
        sr_service = getattr(state, "sr_service", None)
        if sr_service is None:
            await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "SR service disabled"})
            return

        file_ref = (msg.get("fileRef") or "").replace("fileRef:", "")
        magnitude = msg.get("magnitude", 2)

        try:
            image_bytes = resolve_file_ref(file_ref)
        except KeyError as e:
            await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": str(e)})
            return

        sr_fut = sr_service.submit(
            image_bytes=image_bytes,
            out_format="png",
            quality=92,
            magnitude=int(magnitude),
            timeout_s=0.25,
        )

        loop = asyncio.get_running_loop()
        sr_timeout = float(os.environ.get("SR_REQUEST_TIMEOUT", "120"))
        out_bytes = await loop.run_in_executor(None, lambda: sr_fut.result(timeout=sr_timeout))

        # Store result
        storage = getattr(state, "storage", None)
        image_key = None
        if storage:
            from persistence.storage_provider import StorageProvider
            image_key = StorageProvider._new_key("sr_image")
            storage.put(image_key, out_bytes, content_type="image/png", meta={
                "sr_only": True, "sr_magnitude": magnitude,
            })

        outputs = []
        if image_key:
            outputs.append({"url": f"/storage/{image_key}", "key": image_key})

        await hub.send(client_id, {
            "type": "job:complete",
            "jobId": job_id,
            "outputs": outputs,
        })

    except Exception as e:
        logger.error("SR job %s failed: %s", job_id, e, exc_info=True)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": str(e)})


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@ws_router.websocket("/v1/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = uuid.uuid4().hex[:12]
    await hub.connect(ws, client_id)

    # Send initial system:status
    try:
        state = _get_app_state(ws)
        await hub.send(client_id, _build_status(state))
    except Exception:
        pass

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                await hub.send(client_id, _error("Invalid JSON"))
                continue

            msg_type = msg.get("type")
            handler = HANDLERS.get(msg_type)
            if handler is None:
                await hub.send(client_id, _error(f"Unknown type: {msg_type}", msg.get("id")))
                continue

            try:
                result = await handler(ws, msg, client_id)
                if result is not None:
                    await hub.send(client_id, result)
            except Exception as e:
                logger.error("Handler %s failed: %s", msg_type, e, exc_info=True)
                await hub.send(client_id, _error(str(e), msg.get("id")))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WS connection error for %s: %s", client_id, e)
    finally:
        await hub.disconnect(client_id)


# ---------------------------------------------------------------------------
# Status builder
# ---------------------------------------------------------------------------

def _build_status(state) -> dict:
    """Build a system:status message from app state."""
    status: dict = {"type": "system:status", "ts": time.time()}

    # Mode
    if getattr(state, "use_mode_system", False):
        pool = getattr(state, "worker_pool", None)
        status["mode"] = pool.get_current_mode() if pool else None
    else:
        status["mode"] = "legacy"

    # VRAM (best-effort)
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.mem_get_info()
            status["vram"] = {"free_mb": mem[0] // (1024 * 1024), "total_mb": mem[1] // (1024 * 1024)}
    except Exception:
        pass

    # Storage health
    storage = getattr(state, "storage", None)
    if storage is not None:
        try:
            status["storage"] = storage.health()
        except Exception:
            status["storage"] = {"ok": False}

    status["ws_clients"] = hub.client_count
    return status
