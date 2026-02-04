"""
Workflow configuration management.

Loads and validates conf/workflows.yml configuration file containing:
- Default workflow name
- Workflow definitions (metadata + ComfyUI workflow JSON)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for a single ComfyUI workflow."""
    name: str
    display_name: str
    description: str = ""
    workflow: Dict[str, Any] = field(default_factory=dict)
    default_size: str = "512x512"
    default_steps: int = 20
    default_cfg: float = 7.0
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class WorkflowsYAML:
    """Root configuration from workflows.yml."""
    default_workflow: str
    workflows: Dict[str, WorkflowConfig]


class WorkflowConfigManager:
    """
    Manages workflow configurations from conf/workflows.yml.

    Responsibilities:
    - Load and validate workflows.yml
    - Provide access to workflow definitions
    """

    def __init__(self, config_path: str = "conf/workflows.yml"):
        self.config_path = Path(config_path)
        self.config: Optional[WorkflowsYAML] = None
        self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"workflows.yml not found at {self.config_path}. "
                f"Create this file to define ComfyUI workflows."
            )

        logger.info(f"[WorkflowConfig] Loading configuration from {self.config_path}")

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("workflows.yml is empty")

        if "default_workflow" not in data:
            raise ValueError("workflows.yml missing required field: default_workflow")
        if "workflows" not in data or not data["workflows"]:
            raise ValueError("workflows.yml missing or empty: workflows")

        default_workflow = data["default_workflow"]
        workflows: Dict[str, WorkflowConfig] = {}

        for wf_name, wf_data in data["workflows"].items():
            display_name = wf_data.get("display_name") or wf_name
            workflow = wf_data.get("workflow") or {}
            if not isinstance(workflow, dict):
                raise ValueError(f"Workflow '{wf_name}' has invalid workflow JSON")

            tags = wf_data.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            if not isinstance(tags, list):
                raise ValueError(f"Workflow '{wf_name}' has invalid tags list")

            workflows[wf_name] = WorkflowConfig(
                name=wf_name,
                display_name=display_name,
                description=wf_data.get("description", ""),
                workflow=workflow,
                default_size=wf_data.get("default_size", "512x512"),
                default_steps=wf_data.get("default_steps", 20),
                default_cfg=wf_data.get("default_cfg", 7.0),
                created_at=wf_data.get("created_at", ""),
                updated_at=wf_data.get("updated_at", ""),
                tags=tags,
            )

        if default_workflow not in workflows:
            raise ValueError(
                f"default_workflow '{default_workflow}' not found in workflows. "
                f"Available workflows: {list(workflows.keys())}"
            )

        self.config = WorkflowsYAML(
            default_workflow=default_workflow,
            workflows=workflows,
        )

        logger.info(f"[WorkflowConfig] Loaded {len(workflows)} workflows")
        logger.info(f"[WorkflowConfig] Default workflow: {default_workflow}")

    def save_config(self, data: Dict[str, Any]):
        """
        Save configuration data to workflows.yml and reload.

        Args:
            data: Dict with default_workflow and workflows
        """
        yaml_data = {
            "default_workflow": data["default_workflow"],
            "workflows": {},
        }

        for wf_name, wf_data in data["workflows"].items():
            yaml_entry = {
                "display_name": wf_data.get("display_name", wf_name),
                "description": wf_data.get("description", ""),
                "default_size": wf_data.get("default_size", "512x512"),
                "default_steps": wf_data.get("default_steps", 20),
                "default_cfg": wf_data.get("default_cfg", 7.0),
                "tags": wf_data.get("tags", []),
                "workflow": wf_data.get("workflow", {}),
                "created_at": wf_data.get("created_at", ""),
                "updated_at": wf_data.get("updated_at", ""),
            }
            yaml_data["workflows"][wf_name] = yaml_entry

        tmp_path = self.config_path.with_suffix(".yml.tmp")
        with open(tmp_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        tmp_path.rename(self.config_path)

        logger.info(f"[WorkflowConfig] Saved configuration to {self.config_path}")
        self._load_config()

    def reload(self):
        logger.info("[WorkflowConfig] Reloading configuration")
        self._load_config()

    def get_workflow(self, name: str) -> WorkflowConfig:
        if name not in self.config.workflows:
            raise KeyError(
                f"Workflow '{name}' not found. Available: {list(self.config.workflows.keys())}"
            )
        return self.config.workflows[name]

    def list_workflows(self) -> List[str]:
        return list(self.config.workflows.keys())

    def get_default_workflow(self) -> str:
        return self.config.default_workflow

    def to_dict(self, include_workflow: bool = True) -> Dict[str, Any]:
        return {
            "default_workflow": self.config.default_workflow,
            "workflows": {
                name: {
                    "display_name": wf.display_name,
                    "description": wf.description,
                    "default_size": wf.default_size,
                    "default_steps": wf.default_steps,
                    "default_cfg": wf.default_cfg,
                    "tags": wf.tags,
                    "created_at": wf.created_at,
                    "updated_at": wf.updated_at,
                    **({"workflow": wf.workflow} if include_workflow else {}),
                }
                for name, wf in self.config.workflows.items()
            },
        }


_config_manager: Optional[WorkflowConfigManager] = None


def get_workflow_config() -> WorkflowConfigManager:
    """Get global workflow configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = WorkflowConfigManager()
    return _config_manager


def reload_workflow_config():
    """Reload global workflow configuration from disk."""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload()
    else:
        _config_manager = WorkflowConfigManager()
