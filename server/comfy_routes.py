# server/comfy_routes.py
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

logger = logging.getLogger("comfy.jobs")
router = APIRouter()

# ---- configure these for your environment ----
COMFY_BASE_URL = "https://node2:8189"  # or https://node2:8188 if TLS in front

# Example: point to your API prompt JSON (uploaded file path shown for reference)
# In your real project, store these under /app/workflows_api/*.json
WORKFLOWS = {
    # Adjust node IDs to match THIS prompt graph.
    # You can open the API prompt JSON and find which node is LoadImage + KSampler.
    "TRACKING-LCM-DIFFS": WorkflowSpec(
        workflow_id="TRACKING-LCM-DIFFS",
        prompt_path="/workflows/Tracking-LCM-DIFFS-API.json",  # <-- replace in container
        load_image_node="46",
        sampler_node="50",
        # pos_text_node="6", neg_text_node="7",  # optional if present
    ),
}

inv = ComfyUIInvoker(base_url=COMFY_BASE_URL, verify_tls=True)
store = WorkflowStore(WORKFLOWS)

JOBS: Dict[str, Dict[str, Any]] = {}


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

    logger.info("start_job workflowId=%s params=%s image=%s",
                workflowId, params_obj, (image.filename if image else None))
    print("start_job workflowId=%s params=%s image=%s",
                workflowId, params_obj, (image.filename if image else None))
    
    uploaded = None
    if image is not None:
        content = await image.read()

        logger.info(
            "uploaded image bytes=%s filename=%s content_type=%s",
            len(content),
            image.filename,
            image.content_type,
        )

        uploaded = inv.upload_image(
            content,
            filename=image.filename,
            image_type="input",
        )

        uploaded = inv.upload_image(content, filename=image.filename, image_type="input")
        uploaded = {
            "name": uploaded.get("name") or uploaded.get("filename"),
            "subfolder": uploaded.get("subfolder", ""),
            "type": uploaded.get("type", "input"),
        }

        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "id": job_id,
            "workflowId": workflowId,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "outputs": [],
        }

        t = threading.Thread(
            target=_run_job,
            args=(job_id, workflowId, params_obj, uploaded),
            daemon=True,
        )
        t.start()
    else:
        logger.info("no image provided in request")
        raise HTTPException(500, "Job failed to initialize.")

    return {"job_id": job_id, "jobId": job_id, "id": job_id}


@router.get("/v1/comfy/jobs/{job_id}")
def get_job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


def _run_job(job_id: str, workflow_id: str, params: Dict[str, Any], uploaded_image: Optional[Dict[str, Any]]) -> None:
    error_id = None
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = time.time()

        uploaded_name = None
        if uploaded_image:
            uploaded_name = uploaded_image.get("name") or uploaded_image.get("filename")

        prompt_graph = store.make_prompt(
            workflow_id,
            uploaded_filename=uploaded_name,
            steps=params.get("steps"),
            cfg=params.get("cfg"),
            denoise=params.get("denoise"),
            seed=params.get("seed"),
            # prompt_text=params.get("prompt"), negative_text=params.get("negative_prompt"),
        )

        logger.info(
          "LoadImage node %s image=%r (uploaded_name=%r)",
          store.specs[workflow_id].load_image_node,
          prompt_graph.get(store.specs[workflow_id].load_image_node, {}).get("inputs", {}).get("image"),
          uploaded_name,
        )

        result = inv.invoke(prompt_graph, max_wait_s=float(params.get("max_wait_s") or 900))

        # Return outputs as URLs your frontend can load (through ComfyUI /view)
        base = inv.base_url.rstrip("/")
        outputs = []
        for ref in result.outputs:
            # If comfy is behind your backend, you may want to proxy /view instead.
            url = f"{base}/view?filename={ref.filename}&type={ref.type}&subfolder={ref.subfolder}"
            outputs.append({"url": url, "filename": ref.filename, "type": ref.type, "subfolder": ref.subfolder})

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["outputs"] = outputs
        JOBS[job_id]["finished_at"] = time.time()

    except Exception:
        # log full traceback, user sees safe error
        error_id = uuid.uuid4().hex[:8]
        logger.exception("Job %s failed (ref=%s)", job_id, error_id)

        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"Job failed (ref {error_id})"
        JOBS[job_id]["finished_at"] = time.time()