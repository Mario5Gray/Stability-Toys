import os
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import APIRouter

logger = logging.getLogger(__name__)

KEYMAP_CONFIG_PATH = os.environ.get("KEYMAP_CONFIG_PATH", "conf/keymap.yml")
REPO_ROOT = Path(__file__).resolve().parents[1]

router = APIRouter(prefix="/api", tags=["keymap"])


def _resolve_keymap_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@router.get("/keymap/defaults")
def get_keymap_defaults() -> Dict[str, Any]:
    path = _resolve_keymap_config_path(KEYMAP_CONFIG_PATH)
    if not path.exists():
        logger.warning("[keymap] %s missing; returning empty defaults", path)
        return {"keymap": {}}
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        logger.warning("[keymap] failed to parse %s: %s", path, exc)
        return {"keymap": {}}
    keymap = data.get("keymap") or {}
    if not isinstance(keymap, dict):
        logger.warning("[keymap] %s 'keymap' is not a mapping", path)
        return {"keymap": {}}
    return {"keymap": keymap}
