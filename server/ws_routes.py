"""
ws_routes.py — WebSocket endpoint + message dispatcher.

Single route: /v1/ws
All messages are JSON envelopes: {"type": "domain:action", ...}
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import time
import uuid
import queue
from urllib.error import URLError, HTTPError
from typing import Any, Dict, Optional, List

from server.http_utils import post_bytes

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backends.chat_client import ChatCompletionsClient, ChatConfig
from server.generation_constraints import finalize_mode_generate_request
from server.mode_config import get_mode_config
from server.ws_hub import hub
from server import log_context
from server.tracing import get_tracer
from server.upload_routes import resolve_file_ref
from server.metrics import record
from invokers.jobs import (
    jobs_put, jobs_get, set_on_update,
)

logger = logging.getLogger(__name__)

ws_router = APIRouter()


# ---------------------------------------------------------------------------
# Job update callback → WS push
# ---------------------------------------------------------------------------

def make_generation_progress_forwarder(loop, client_id: str, job_id: str):
    """Return an on_progress(step, total, stage) that streams a job:progress frame
    to ONE client for a generation (STABL-zueslhah).

    The callback fires on the worker/drain thread (in-proc dispatch or the
    subprocess drain), so it schedules the async hub.send onto the captured event
    loop thread-safely — the same pattern _on_job_update uses. A dead loop
    (client gone / shutdown) is swallowed: progress must never break generation.
    """
    def _forward(step: int, total: int, stage: str) -> None:
        msg = {
            "type": "job:progress",
            "jobId": job_id,
            "step": step,
            "total": total,
            "stage": stage,
        }
        try:
            loop.call_soon_threadsafe(asyncio.ensure_future, hub.send(client_id, msg))
        except RuntimeError:
            pass  # event loop closed
    return _forward


def _on_job_update(job_id: str, snapshot: dict) -> None:
    """
    Called from invokers/jobs.py on every mutation (from any thread).
    Schedules a broadcast of job:progress via the hub.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    progress = snapshot.get("progress") or {}
    delta = _progress_delta(progress)
    msg = {
        "type": "job:progress",
        "jobId": job_id,
        "status": snapshot.get("status"),
        "progress": snapshot.get("progress"),
    }
    if delta is not None:
        msg["delta"] = delta

    if loop is not None and loop.is_running():
        loop.create_task(hub.broadcast(msg))
    else:
        # From a worker thread — need to schedule onto the event loop
        # We'll store the loop ref at startup (set in register_job_hook)
        _loop = getattr(_on_job_update, "_loop", None)
        if _loop is not None:
            _loop.call_soon_threadsafe(asyncio.ensure_future, hub.broadcast(msg))


def _progress_delta(progress: dict) -> "str | None":
    fraction = progress.get("fraction") or 0.0
    if not fraction:
        return None
    pct = int(fraction * 100)
    nodes_total = progress.get("nodes_total") or 0
    if nodes_total > 0:
        nodes_seen = progress.get("nodes_seen") or 0
        return f"node {nodes_seen}/{nodes_total} ({pct}%)"
    return f"{pct}%"


def register_job_hook() -> None:
    """Call once at startup to wire jobs.py → WS push."""
    _on_job_update._loop = asyncio.get_running_loop()
    set_on_update(_on_job_update)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(msg: str, corr_id: Optional[str] = None) -> dict:
    d = {"type": "error", "error": msg}
    if corr_id:
        d["id"] = corr_id
    return d


def _get_app_state(ws: WebSocket):
    return ws.app.state


def _requests_controlnet(params: dict) -> bool:
    return bool(params.get("controlnets"))


def _supports_controlnet(provider: Any) -> bool:
    if provider is None:
        return False
    try:
        capabilities = provider.capabilities()
    except Exception:
        return False
    return getattr(capabilities, "supports_controlnet", False) is True


def _supports_img2img_and_controlnet(provider: Any) -> bool:
    if provider is None:
        return False
    try:
        capabilities = provider.capabilities()
    except Exception:
        return False
    return getattr(capabilities, "supports_img2img_and_controlnet", False) is True


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


# --- inbound metrics (STABL-xmsrxvto) ---

_INVALID_JSON = object()


def _inbound_type(raw_type) -> str:
    """Map a client-supplied `type` onto a BOUNDED label value.

    The client controls this string entirely, so it must never reach a label
    unchecked — one 4 KB value or one million distinct ones would be equally
    fatal. Malformed JSON gets its own value because it means something different
    from an unrecognised type: a broken client versus a client asking for
    something that does not exist.

    `raw in HANDLERS` HASHES its argument, and an unhashable client value
    (`{"type": {}}`) raises TypeError. NOTE: this guard does not fix
    STABL-gzfzzsdq — `HANDLERS.get(msg_type)` in the message loop has the same
    hazard and still drops the connection. This only stops the metrics code from
    being the first thing to raise.
    """
    if raw_type is _INVALID_JSON:
        return "invalid_json"
    try:
        return raw_type if raw_type in HANDLERS else "unknown"
    except TypeError:            # unhashable type field, e.g. {"type": {"a": 1}}
        return "unknown"


def _count_in(raw_type) -> None:
    label = _inbound_type(raw_type)
    record(lambda met: met.ws_messages_total.labels(
        type=label, direction="in").inc())


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
    log_context.job_id_var.set(job_id)   # reset by the message loop's bind
    state = _get_app_state(ws)
    fut = None
    req = None
    pre_submit_job_error: Optional[str] = None
    pre_submit_artifacts: list = []
    controlnet_bindings: list = []

    if job_type == "generate" and getattr(state, "use_mode_system", False):
        from backends.worker_pool import GenerationJob

        # One snapshot read: mode defaults, family-cell admission, ControlNet
        # compatibility, and job epoch all originate here. No get_current_mode /
        # get_mode_config / detect_model after this capture. Captured inside the
        # try so a snapshot/admission failure returns job:error after ack rather
        # than crashing the WS loop.
        snapshot = None
        try:
            req = _build_generate_request(params)
            # Admit against the mode this request TARGETS, established atomically with
            # the switch (STABL-ltefhpkk / STABL-iuiwzthc). With no target this returns
            # the active snapshot, exactly as get_active_model_snapshot() did.
            snapshot = state.worker_pool.admit_generation(getattr(req, "mode", None))
            has_init_image = bool(params.get("init_image_ref"))
            if snapshot is not None:
                mode = snapshot.mode
                # Family-cell operation matrix, before any ControlNet preprocessing.
                from server.controlnet_execution import admit_generation_operation
                admit_generation_operation(
                    req,
                    snapshot=snapshot,
                    provider=getattr(state, "backend_provider", None),
                    has_init_image=has_init_image,
                )
                finalize_mode_generate_request(
                    req,
                    mode,
                    env_default_size=os.environ.get("DEFAULT_SIZE", "512x512"),
                    env_default_steps=int(os.environ.get("DEFAULT_STEPS", "4")),
                    env_default_guidance=float(os.environ.get("DEFAULT_GUIDANCE", "1.0")),
                )
                from server.controlnet_constraints import enforce_controlnet_policy
                enforce_controlnet_policy(req, mode)
                from server.controlnet_preprocessing import preprocess_controlnet_attachments
                from server.asset_store import get_store
                pre_submit_artifacts = preprocess_controlnet_attachments(req, get_store())
                req._controlnet_artifacts = pre_submit_artifacts
                if getattr(req, "controlnets", None):
                    from server.controlnet_execution import resolve_controlnet_bindings
                    controlnet_bindings = resolve_controlnet_bindings(
                        req,
                        mode=mode,
                        store=get_store(),
                        active_family=snapshot.resolved.profile.family_id,
                    )
            elif getattr(req, "controlnets", None):
                # No active snapshot (no model loaded / non-CUDA family): there is
                # no family binding to admit ControlNet — stub it exactly as the
                # non-mode-system _run_generate path does.
                from server.controlnet_constraints import ensure_controlnet_dispatch_supported
                ensure_controlnet_dispatch_supported(req, supports_controlnet=False)
        except Exception as e:
            pre_submit_job_error = str(e)
        init_image_bytes = None
        init_image_ref = params.get("init_image_ref")
        if init_image_ref:
            try:
                init_image_bytes = resolve_file_ref(init_image_ref)
            except KeyError as e:
                await hub.send(client_id, _error(str(e), corr_id))
                return

        if pre_submit_job_error is None and snapshot is None:
            # admit_generation returns None only when there is genuinely no model and
            # none was targeted. The current_resolution_epoch() fallback is gone: it
            # stamped the PRE-switch epoch, which is the STABL-ltefhpkk race.
            pre_submit_job_error = "No model loaded"

        if pre_submit_job_error is None:
            job = GenerationJob(
                req=req,
                init_image=init_image_bytes,
                controlnet_bindings=controlnet_bindings,
                resolution_epoch=snapshot.resolution_epoch,
            )
            try:
                _prog = make_generation_progress_forwarder(
                    asyncio.get_running_loop(), client_id, job.job_id
                )
                fut = state.worker_pool.submit_job(job, on_progress=_prog)
            except queue.Full:
                pre_submit_job_error = "Queue full"
            else:
                job_id = job.job_id
                # The Governor's id from here on — the one the dispatch-thread logs
                # will carry, so both halves of the job's life correlate on one
                # value. No second token needed: the loop's bind restores whatever
                # was in place before this handler ran.
                log_context.job_id_var.set(job_id)

    # Ack immediately
    await hub.send(client_id, {
        "type": "job:ack",
        "id": corr_id,
        "jobId": job_id,
    })

    if pre_submit_job_error is not None:
        error_frame: dict = {
            "type": "job:error",
            "jobId": job_id,
            "error": pre_submit_job_error,
        }
        if pre_submit_artifacts:
            error_frame["controlnet_artifacts"] = [artifact.model_dump() for artifact in pre_submit_artifacts]
        await hub.send(client_id, error_frame)
        return

    if job_type == "generate":
        if getattr(state, "use_mode_system", False):
            t = asyncio.create_task(_run_generate_from_future(ws, client_id, job_id, req, fut))  # type: ignore[arg-type]
        else:
            t = asyncio.create_task(_run_generate(ws, client_id, job_id, params))
        _track_task(job_id, t)
    elif job_type == "comfy":
        t = asyncio.create_task(_run_comfy(ws, client_id, job_id, msg))
        _track_task(job_id, t)
    elif job_type == "sr":
        t = asyncio.create_task(_run_sr(ws, client_id, job_id, msg))
        _track_task(job_id, t)
    elif job_type == "chat":
        t = asyncio.create_task(_run_chat(ws, client_id, job_id, params))
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
        status = await asyncio.to_thread(post_bytes, endpoint, body, content_type)
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
    state = _get_app_state(ws)
    if getattr(state, "use_mode_system", False):
        pool = getattr(state, "worker_pool", None)
        if pool is None:
            return {"type": "job:cancel:ack", "id": msg.get("id"), "jobId": job_id, "detail": "no worker pool"}

        result = pool.cancel_job(job_id)
        if isinstance(result, dict):
            detail = result.get("status", "canceled")
        elif result:
            detail = "canceled"
        else:
            detail = "not_found"
        return {"type": "job:cancel:ack", "id": msg.get("id"), "jobId": job_id, "detail": detail}

    task = _running_tasks.get(job_id)  # type: ignore[arg-type]
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

class _BackendCancelledError(Exception):
    """Internal marker for backend-future cancellation."""


def _build_generate_request(params: dict):
    from server.lcm_sr_server import GenerateRequest

    return GenerateRequest(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt"),
        # The CLI has always sent this (config/precedence.go sets params["mode"] and
        # the whole GenParams map ships in the submit frame); it was silently dropped
        # here, which is why the Governor could not admit against the target mode.
        mode=params.get("mode"),
        scheduler_id=params.get("scheduler_id"),
        size=params.get("size", os.environ.get("DEFAULT_SIZE", "512x512")),
        num_inference_steps=params.get(
            "num_inference_steps",
            params.get("steps", int(os.environ.get("DEFAULT_STEPS", "4"))),
        ),
        guidance_scale=params.get(
            "guidance_scale",
            params.get("cfg", float(os.environ.get("DEFAULT_GUIDANCE", "1.0"))),
        ),
        seed=params.get("seed"),
        superres=params.get("superres", False),
        superres_magnitude=params.get("superres_magnitude", 2),
        denoise_strength=params.get("denoise_strength", 0.75),
        controlnets=params.get("controlnets"),
    )


def _resolve_chat_config(state, params: dict) -> Optional[tuple[ChatConfig, Optional[int]]]:
    """Resolve chat config and mode maximum length for the active mode."""
    mode_name = params.get("mode")
    mode_config = get_mode_config()

    if not mode_name:
        if getattr(state, "use_mode_system", False):
            pool = getattr(state, "worker_pool", None)
            if pool is not None:
                mode_name = pool.get_current_mode()
        if not mode_name:
            mode_name = mode_config.get_default_mode()

    if not mode_name:
        return None

    mode = mode_config.get_mode(mode_name)
    maximum_len = getattr(mode, "maximum_len", None)
    chat_cfg = mode_config.resolve_chat_config(
        mode_name,
        overrides={
            "model": params.get("model"),
            "max_tokens": params.get("max_tokens"),
            "temperature": params.get("temperature"),
            "system_prompt": params.get("system_prompt"),
        },
    )
    if chat_cfg is None:
        return None

    return (
        ChatConfig(
            endpoint=chat_cfg.endpoint,
            model=chat_cfg.model,
            api_key_env=chat_cfg.api_key_env,
            max_tokens=chat_cfg.max_tokens,
            temperature=chat_cfg.temperature,
            system_prompt=chat_cfg.system_prompt,
        ),
        maximum_len,
    )


def _build_chat_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        for item in system_prompt.split("\n\n"):
            content = item.strip()
            if content:
                messages.append({"role": "system", "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


async def _run_chat(ws: WebSocket, client_id: str, job_id: str, params: dict) -> None:
    """Run a chat completions job via an OpenAI-compatible backend."""
    try:
        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Missing prompt"})
            return

        state = _get_app_state(ws)
        chat_context = _resolve_chat_config(state, params)
        if chat_context is None:
            await hub.send(
                client_id,
                {"type": "job:error", "jobId": job_id, "error": "chat not configured for this mode"},
            )
            return
        chat_cfg, maximum_len = chat_context

        client = ChatCompletionsClient(chat_cfg)
        stream = bool(params.get("stream", True))
        max_tokens = chat_cfg.max_tokens
        if maximum_len is not None:
            max_tokens = min(max_tokens, int(maximum_len))
        temperature = chat_cfg.temperature
        messages = _build_chat_messages(prompt, chat_cfg.system_prompt)

        if stream:
            chunks: List[str] = []
            async for delta in client.stream(messages, max_tokens=max_tokens, temperature=temperature):
                chunks.append(delta)
                await hub.send(
                    client_id,
                    {"type": "job:progress", "jobId": job_id, "delta": delta},
                )
            full_text = "".join(chunks)
        else:
            full_text = await client.complete(messages, max_tokens=max_tokens, temperature=temperature)

        await hub.send(
            client_id,
            {
                "type": "job:complete",
                "jobId": job_id,
                "outputs": [{"text": full_text}],
                "meta": {
                    "model": chat_cfg.model,
                    "endpoint_base": chat_cfg.endpoint.rstrip("/"),
                },
            },
        )
    except Exception as e:
        logger.error("Chat job %s failed: %s", job_id, e, exc_info=True)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": str(e)})


async def _run_generate_from_future(ws: WebSocket, client_id: str, job_id: str, req, fut) -> None:
    try:
        await _finish_generate(ws, client_id, job_id, req, fut)
    except asyncio.CancelledError:
        logger.info("Generate job %s cancelled by client", job_id)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Cancelled by client"})
    except _BackendCancelledError:
        logger.info("Generate job %s cancelled by backend", job_id)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Cancelled by backend"})
    except Exception as e:
        logger.error("Generate job %s failed: %s", job_id, e, exc_info=True)
        # Admission already passed and preprocessing may have emitted control
        # maps; surface them on a post-admission dispatch/runtime failure (matches
        # the HTTP contract). Matrix rejections happen earlier, before preprocess.
        error_frame: dict = {"type": "job:error", "jobId": job_id, "error": str(e)}
        artifacts = getattr(req, "_controlnet_artifacts", None)
        if artifacts:
            error_frame["controlnet_artifacts"] = [a.model_dump() for a in artifacts]
        await hub.send(client_id, error_frame)


def _resolve_backend_future_result(pool, fut):
    """Resolve a generate's future under the Governor's two budgets (STABL-atzqpcte).

    The pool is passed in rather than the old flat timeout: only the Governor knows
    whether the job is still QUEUED (behind a mode switch's model load) or genuinely
    EXECUTING, and that distinction is the whole fix.
    """
    try:
        return pool.wait_for_result(fut)
    except concurrent.futures.CancelledError as e:
        raise _BackendCancelledError() from e


async def _finish_generate(ws: WebSocket, client_id: str, job_id: str, req, fut) -> None:
    state = _get_app_state(ws)
    from server.lcm_sr_server import _store_image_blob

    # STABL-atzqpcte: two budgets, split at true execution start. A flat
    # fut.result(timeout=DEFAULT_TIMEOUT) here charged queue wait AND the mode
    # switch's model load to a budget meant for generation — the field failure was a
    # WebSocket timeout during a HunyuanDiT load on the first inline --mode generate.
    loop = asyncio.get_running_loop()
    png_bytes, seed = await loop.run_in_executor(
        None, _resolve_backend_future_result, state.worker_pool, fut
    )

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
            "backend": os.environ.get("BACKEND", ""),
            "sr": did_sr,
        },
        **(
            {
                "controlnet_artifacts": [
                    artifact.model_dump()
                    for artifact in getattr(req, "_controlnet_artifacts", [])
                ]
            }
            if getattr(req, "_controlnet_artifacts", None)
            else {}
        ),
    })


async def _run_generate(ws: WebSocket, client_id: str, job_id: str, params: dict) -> None:
    """Run a generate job using the same code path as POST /generate."""
    _run_artifacts: list = []
    try:
        state = _get_app_state(ws)
        req = _build_generate_request(params)

        from server.controlnet_preprocessing import preprocess_controlnet_attachments
        from server.asset_store import get_store
        _run_artifacts = preprocess_controlnet_attachments(req, get_store())
        req._controlnet_artifacts = _run_artifacts

        from server.controlnet_constraints import ensure_controlnet_dispatch_supported
        ensure_controlnet_dispatch_supported(req, supports_controlnet=False)

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
            job = GenerationJob(
                req=req,
                init_image=init_image_bytes,
                resolution_epoch=pool.current_resolution_epoch(),
            )
            try:
                fut = pool.submit_job(job)
            except queue.Full:
                await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Queue full"})
                return
            job_id = job.job_id
        else:
            service = state.service
            fut = service.submit(req, timeout_s=0.25)

        await _finish_generate(ws, client_id, job_id, req, fut)

    except asyncio.CancelledError:
        logger.info("Generate job %s cancelled by client", job_id)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Cancelled by client"})

    except Exception as e:
        logger.error("Generate job %s failed: %s", job_id, e, exc_info=True)
        error_frame: dict = {"type": "job:error", "jobId": job_id, "error": str(e)}
        if _run_artifacts:
            error_frame["controlnet_artifacts"] = [artifact.model_dump() for artifact in _run_artifacts]
        await hub.send(client_id, error_frame)


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
            # STABL-qnlaclof: the span opens HERE, before the parse — not on the
            # bind below. Two exits `continue` past that point (malformed JSON,
            # unrecognised type), so a span anchored later would cover only
            # messages that already parsed AND resolved to a handler, making a
            # broken client invisible and the loop look idle.
            #
            # This scope is deliberately WIDER than the job_id bind's. The bind
            # exists to CLEAR the id so a handler can set its own; its narrowness
            # is the point. The span represents serving one inbound message, which
            # starts at the read. Do not "fix" one to match the other.
            with get_tracer(__name__).start_as_current_span("ws.message") as span:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    span.set_attribute("messaging.type", _inbound_type(_INVALID_JSON))
                    _count_in(_INVALID_JSON)
                    await hub.send(client_id, _error("Invalid JSON"))
                    continue

                msg_type = msg.get("type")
                # Bounded before it can name anything: the client controls this
                # string entirely, and an unbounded span attribute is the same
                # cardinality failure as an unbounded label.
                span.set_attribute("messaging.type", _inbound_type(msg_type))
                _count_in(msg_type)
                handler = HANDLERS.get(msg_type)
                if handler is None:
                    await hub.send(client_id, _error(f"Unknown type: {msg_type}", msg.get("id")))
                    continue

                try:
                    # STABL-bpsfmoke: every message starts with a clean correlation id.
                    # Handlers SET it (job:submit does, once it has minted one); binding
                    # None here is what stops it surviving into the next message on the
                    # same connection. One place, so a handler added later cannot forget.
                    # Tasks the handler spawns are unaffected — create_task copies the
                    # context at creation, so they keep the id this bind later restores.
                    with log_context.bind_job_id(None):
                        result = await handler(ws, msg, client_id)
                        # Inside the bind: emitting a handler's reply is still part of
                        # serving that message, so a send failure logs under the same
                        # correlation id the handler established.
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

    # VRAM (best-effort) — through the SEAM, never torch directly (STABL-qfjfflrx).
    # This used to call torch.cuda.mem_get_info() inline: a direct-CUDA bypass of
    # DeviceMemory's accounting that also bound the parent process to CUDA, which is
    # what makes a CUDA-free parent impossible. It duplicated numbers model_routes
    # already serves from the registry, and had accumulated an isinstance() guard
    # against a stubbed torch leaking in — a symptom of reading the wrong source.
    try:
        pool = getattr(state, "worker_pool", None)
        if pool is not None:
            stats = pool.get_vram_stats()
            free_mb = int(stats["available_gb"] * 1024)
            total_mb = int(stats["total_gb"] * 1024)
            status["vram"] = {"free_mb": free_mb, "total_mb": total_mb}
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
