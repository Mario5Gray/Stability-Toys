"""Runtime log-level overrides (STABL-ataigkdk).

The Spring model: the artifact ships a baked config as DECLARED DEFAULTS, and the
environment overrides loggers at startup. `logging_config.py` is the declaration;
this module is the override.

Precedence, highest first:

    LOG_LEVELS[name]  >  declared per-logger level  >  LOG_LEVEL  >  INFO

Called after `dictConfig` has run — from the FastAPI lifespan (which fires on BOTH
entry paths) and from the spawned worker child's bootstrap.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from server.logging_config import LEVEL_TRACKING_LOGGERS

logger = logging.getLogger(__name__)

_VALID = {"CRITICAL", "FATAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG", "NOTSET"}


def parse_log_levels(raw: Optional[str]) -> Dict[str, str]:
    """Parse ``name=LEVEL,name=LEVEL``. Malformed entries are SKIPPED, not fatal.

    One bad entry in a deployment environment must not cost every other entry and
    must not take down startup. An empty key is the root logger, matching
    LOGGING_CONFIG's own convention.
    """
    out: Dict[str, str] = {}
    for chunk in (raw or "").split(","):
        if "=" not in chunk:
            if chunk.strip():
                logger.warning("[log_levels] ignoring malformed entry %r", chunk.strip())
            continue
        name, _, level = chunk.partition("=")
        name, level = name.strip(), level.strip().upper()
        if level not in _VALID:
            logger.warning("[log_levels] ignoring unknown level %r for logger %r", level, name)
            continue
        out[name] = level
    return out


def apply_runtime_levels() -> Dict[str, str]:
    """Apply the precedence chain to the live logging tree. Returns what it set.

    Non-raising by contract: this runs inside the FastAPI lifespan, where an
    exception fails server startup, and inside the spawned child's bootstrap,
    where one hangs a parent blocked on the _READY handshake. It still LOGS its
    own failure — a config path that fails silently is its own bug.
    """
    # `applied` records what was ACTUALLY set, not what was intended: it is
    # returned into a startup log line, and a line that reports levels it failed
    # to apply is at its most misleading exactly when someone is debugging
    # logging. `desired` is the intent; entries move across one at a time.
    desired: Dict[str, str] = {}
    applied: Dict[str, str] = {}
    try:
        raw = os.getenv("LOG_LEVEL")
        root_level = (raw or "INFO").strip().upper()
        if root_level not in _VALID:
            # INFO, NOT the baked value. For a TRACKING logger the baked level is
            # a snapshot of the build environment, not a declaration — leaving it
            # in place would preserve exactly the frozen-config bug this module
            # exists to fix. Absent already means INFO in logging_config.py;
            # invalid gets the same answer, loudly.
            logger.warning("[log_levels] unknown LOG_LEVEL %r; falling back to INFO", raw)
            root_level = "INFO"

        # Unconditional: every tracking logger resolves to SOMETHING on every run.
        # A declared level (comfy.jobs) is an intent and is not in this set — that
        # distinction is the whole reason LEVEL_TRACKING_LOGGERS exists.
        for name in LEVEL_TRACKING_LOGGERS:
            desired[name] = root_level

        # Per-logger overrides win over everything, including a declared level.
        # Unparseable entries are SKIPPED rather than defaulted: an override that
        # cannot be read simply does not apply, leaving the declared or tracking
        # level intact. That asymmetry with LOG_LEVEL above is deliberate.
        desired.update(parse_log_levels(os.getenv("LOG_LEVELS")))

        for name, level in desired.items():
            # getLogger CREATES an unknown logger, which is deliberate: a level
            # set for a module that has not imported yet is waiting when it does.
            logging.getLogger(name).setLevel(level)
            applied[name] = level
    except Exception:
        logger.warning("[log_levels] failed to apply runtime levels", exc_info=True)
    return applied
