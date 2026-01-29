from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from invokers.comfy_client import ComfyUIInvoker
from invokers.workflow_store import WorkflowSpec, WorkflowStore

# import the helpers you actually call
from invokers.jobs import jobs_put, jobs_get, jobs_append_unique, jobs_update_path

logger = logging.getLogger("comfy.jobs")
router = APIRouter()

# ---- configure these for your environment ----
COMFY_BASE_URL = "https://node2:8189"  # or https://node2:8188 if TLS in front

WORKFLOWS = {
    "LCM_CYBERPONY_XL": WorkflowSpec(
        workflow_id="LCM_CYBERPONY_XL",
        prompt_path="/workflows/LCM_CYBERPONY_XL.json",
        load_image_node="46",
        sampler_node="50",
    ),
}

inv = ComfyUIInvoker(base_url=COMFY_BASE_URL, verify_tls=True)
store = WorkflowStore(WORKFLOWS)


@router.post("/v1/comfy/jobs")
async def start_job(
    workflowId: str = Form(...),
    params: str = Form("{}"),
    image: UploadFile | None = File(None),
):
    if workflowId not in WORKFLOWS:
        raise HTTPException(400, f"Unknown workflowId={workflowId}")

    try:
        params_obj = json.loads(params or "{}")
        if not isinstance(params_obj, dict):
            raise ValueError("params must be a JSON object")
    except Exception as e:
        raise HTTPException(400, f"Bad params JSON: {e}")

    logger.debug(
        "start_job workflowId=%s params=%s image=%s",
        workflowId,
        params_obj,
        (image.filename if image else None),
    )

    if image is None:
        # preserving your behavior (500). Consider 400 semantics later if you want.
        raise HTTPException(500, "Job failed to initialize.")

    # Read bytes once
    content = await image.read()

    logger.debug(
        "uploaded image bytes=%s filename=%s content_type=%s",
        len(content),
        image.filename,
        image.content_type,
    )

    # Upload exactly once
    up = inv.upload_image(content, filename=image.filename, image_type="input")
    uploaded = {
        "name": up.get("name") or up.get("filename"),
        "subfolder": up.get("subfolder", ""),
        "type": up.get("type", "input"),
    }

    job_id = str(uuid.uuid4())

    # Create job record (additive fields won't break old clients)
    jobs_put(
        job_id,
        {
            "id": job_id,
            "workflowId": workflowId,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "outputs": [],
            # additive
            "heartbeat_at": None,
            "updated_at": time.time(),
            "comfy": {"client_id": None, "prompt_id": None},
            "progress": {
                "nodes_total": 0,
                "nodes_seen": 0,
                "current_node": None,
                "node_progression": [],
                "fraction": 0.0,
            },
        },
    )

    t = threading.Thread(
        target=_run_job,
        args=(job_id, workflowId, params_obj, uploaded),
        daemon=True,
        name=f"comfy-job-{job_id[:8]}",
    )
    t.start()

    return {"job_id": job_id, "jobId": job_id, "id": job_id}


@router.get("/v1/comfy/jobs/{job_id}")
def get_job(job_id: str):
    j = jobs_get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


def _run_job(
    job_id: str,
    workflow_id: str,
    params: Dict[str, Any],
    uploaded_image: Optional[Dict[str, Any]],
) -> None:
    """
    Runs one ComfyUI-backed job and updates JOBS via jobs_* helpers.

    External-facing response shape is preserved:
      - status: "running" | "done" | "error"
      - started_at / finished_at timestamps
      - outputs: list of {id, url, filename, type, subfolder}
      - error: string on failure
    """

    ws = None

    # --- local aliases (tiny speedup, avoids global lookups inside callback) ---
    _update = jobs_update_path
    _append_unique = jobs_append_unique
    _time = time.time
    _uuid4 = uuid.uuid4

    try:
        # Mark running
        now0 = _time()
        _update(job_id, "status", "running")
        _update(job_id, "started_at", now0)
        _update(job_id, "heartbeat_at", now0)

        # Resolve uploaded image name
        uploaded_name = None
        if uploaded_image:
            uploaded_name = uploaded_image.get("name") or uploaded_image.get("filename")

        # Build prompt graph
        prompt_graph = store.make_prompt(
            workflow_id,
            uploaded_filename=uploaded_name,
            steps=params.get("steps"),
            cfg=params.get("cfg"),
            denoise=params.get("denoise"),
            seed=params.get("seed"),
        )

        logger.info(
            "make_prompt workflow=%s image=%s steps=%s cfg=%s denoise=%s seed=%s nodes=%d",
            workflow_id,
            uploaded_name,
            params.get("steps"),
            params.get("cfg"),
            params.get("denoise"),
            params.get("seed"),
            len(prompt_graph),
        )

        # Nodes total known immediately
        nodes_total = len(prompt_graph)
        _update(job_id, "progress.nodes_total", nodes_total)

        # Establish Comfy client_id (needed before opening WS)
        client_id = _uuid4().hex
        _update(job_id, "comfy.client_id", client_id)

        # Open websocket (per-job)
        ws = inv.open_ws(client_id)

        # Submit prompt
        prompt_id = inv.submit_prompt(prompt_graph, client_id=client_id)
        _update(job_id, "comfy.prompt_id", prompt_id)

        # Track unique nodes observed for percentage approximation
        seen: set[str] = set()

        # Snapshot max_wait_s once
        max_wait_s = float(params.get("max_wait_s") or 900)

        def on_node(node: Any) -> None:
            """
            Called for each node event for this prompt_id.
            Only writes to JOBS via jobs_* helpers to avoid races.
            """
            now = _time()
            _update(job_id, "heartbeat_at", now)
            _update(job_id, "progress.current_node", node)

            # Terminal event: node == None => done
            if node is None:
                _update(job_id, "progress.fraction", 1.0)
                return

            node_s = str(node)

            # Maintain ordered progression list; dedupe repeats
            _append_unique(job_id, "progress.node_progression", node_s)

            if node_s not in seen:
                seen.add(node_s)
                seen_count = len(seen)
                _update(job_id, "progress.nodes_seen", seen_count)

                denom = nodes_total if nodes_total > 0 else 1
                frac = seen_count / denom
                if frac > 0.95:
                    frac = 0.95
                _update(job_id, "progress.fraction", frac)

        # Wait + stream progress
        inv.wait_with_node_progress(ws, prompt_id, on_node=on_node, max_wait_s=max_wait_s)

        # Fetch outputs from history
        refs = inv.get_history_outputs(prompt_id)

        base = inv.base_url.rstrip("/")
        outputs = []
        for ref in refs:
            url = f"{base}/view?filename={ref.filename}&type={ref.type}&subfolder={ref.subfolder}"
            out_id = f"{job_id}:{ref.type}:{ref.subfolder}:{ref.filename}"
            outputs.append(
                {
                    "id": out_id,
                    "url": url,
                    "filename": ref.filename,
                    "type": ref.type,
                    "subfolder": ref.subfolder,
                }
            )

        # Mark done
        _update(job_id, "status", "done")
        _update(job_id, "outputs", outputs)
        _update(job_id, "finished_at", _time())
        _update(job_id, "progress.fraction", 1.0)

    except Exception:
        error_id = _uuid4().hex[:8]
        logger.exception("Job %s failed (ref=%s)", job_id, error_id)

        _update(job_id, "status", "error")
        _update(job_id, "error", f"Job failed (ref {error_id})")
        _update(job_id, "finished_at", _time())

    finally:
        try:
            if ws is not None:
                ws.close()
        except Exception:
            pass
