"""
Workflow management API endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.workflow_config import get_workflow_config, reload_workflow_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["workflows"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class WorkflowCreateRequest(BaseModel):
    display_name: str
    description: str = ""
    default_size: str = "512x512"
    default_steps: int = 20
    default_cfg: float = 7.0
    tags: List[str] = Field(default_factory=list)
    workflow: Dict[str, Any] = Field(default_factory=dict)


class WorkflowsBulkSaveRequest(BaseModel):
    default_workflow: str
    workflows: Dict[str, Any]


@router.get("/workflows")
async def list_workflows():
    """
    List all workflows (summary only; workflow JSON omitted).
    """
    config = get_workflow_config()
    data = config.to_dict(include_workflow=False)
    return {
        "default_workflow": data["default_workflow"],
        "workflows": data["workflows"],
    }


@router.get("/workflows/{name}")
async def get_workflow(name: str):
    """
    Get a single workflow (full detail + JSON).
    """
    config = get_workflow_config()
    try:
        wf = config.get_workflow(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")

    return {
        "name": wf.name,
        "display_name": wf.display_name,
        "description": wf.description,
        "default_size": wf.default_size,
        "default_steps": wf.default_steps,
        "default_cfg": wf.default_cfg,
        "tags": wf.tags,
        "created_at": wf.created_at,
        "updated_at": wf.updated_at,
        "workflow": wf.workflow,
    }


@router.post("/workflows/{name}")
async def create_or_update_workflow(name: str, request: WorkflowCreateRequest):
    """
    Create or update a workflow.
    """
    config = get_workflow_config()
    data = config.to_dict(include_workflow=True)
    workflows = data["workflows"]

    is_new = name not in workflows
    now = _now_iso()
    existing = workflows.get(name, {})

    workflows[name] = {
        "display_name": request.display_name,
        "description": request.description,
        "default_size": request.default_size,
        "default_steps": request.default_steps,
        "default_cfg": request.default_cfg,
        "tags": request.tags,
        "workflow": request.workflow,
        "created_at": existing.get("created_at", now if is_new else ""),
        "updated_at": now,
    }

    if not data.get("default_workflow") or data["default_workflow"] not in workflows:
        data["default_workflow"] = name

    try:
        config.save_config(data)
        return {"status": "saved", "workflows": list(workflows.keys())}
    except Exception as e:
        logger.error(f"[API] Save workflow failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/workflows")
async def save_all_workflows(request: WorkflowsBulkSaveRequest):
    """
    Save full workflows config, write to disk and reload.
    """
    config = get_workflow_config()
    data = request.model_dump()

    if not data.get("workflows"):
        raise HTTPException(status_code=400, detail="At least one workflow must exist")

    if data["default_workflow"] not in data["workflows"]:
        raise HTTPException(
            status_code=400,
            detail=f"default_workflow '{data['default_workflow']}' not found in workflows",
        )

    try:
        config.save_config(data)
        return {"status": "saved", "workflows": list(data["workflows"].keys())}
    except Exception as e:
        logger.error(f"[API] Save workflows failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workflows/{name}")
async def delete_workflow(name: str):
    """
    Delete a workflow.
    """
    config = get_workflow_config()
    data = config.to_dict(include_workflow=True)

    if name not in data["workflows"]:
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")

    if name == data["default_workflow"]:
        raise HTTPException(status_code=400, detail="Cannot delete default workflow")

    del data["workflows"][name]

    try:
        config.save_config(data)
        return {"status": "deleted", "workflow": name}
    except Exception as e:
        logger.error(f"[API] Delete workflow failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/reload")
async def reload_workflows_config():
    """
    Reload workflows.yml configuration from disk.
    """
    try:
        reload_workflow_config()
        config = get_workflow_config()
        workflows = config.list_workflows()

        logger.info(f"[API] Workflow configuration reloaded: {len(workflows)} workflows")

        return {
            "status": "reloaded",
            "workflows_count": len(workflows),
            "workflows": workflows,
            "default_workflow": config.get_default_workflow(),
        }
    except Exception as e:
        logger.error(f"[API] Workflow config reload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload workflow configuration: {e}",
        )
