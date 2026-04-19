import os
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import APIRouter

logger = logging.getLogger(__name__)

KEYMAP_CONFIG_PATH = os.environ.get("KEYMAP_CONFIG_PATH", "conf/keymap.yml")

router = APIRouter(prefix="/api", tags=["keymap"])


@router.get("/keymap/defaults")
def get_keymap_defaults() -> Dict[str, Any]:
    path = Path(KEYMAP_CONFIG_PATH)
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
