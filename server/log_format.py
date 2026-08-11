"""Structured log formatting (STABL-bpsfmoke).

One formatter class serves both shapes. The choice is made ONCE, in ``__init__``,
because that is `dictConfig` time — see the class docstring.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from server import log_context
from utils.env import Quotes, env_str

TEXT = "text"
JSON = "json"
_DEFAULT = TEXT

# LogRecord attributes that are either rendered under a different name above or
# are noise in a structured payload. Anything NOT in here that a caller attached
# via logging's `extra=` is passed through, which is what makes the field set
# extensible without touching this module.
_RESERVED = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName",
})


def resolve_log_format(value: Optional[str] = None) -> str:
    """Normalise a LOG_FORMAT value. Unrecognised input degrades to text.

    Never raises: a typo in a deployment env must not stop the process from
    logging, and a hard failure here would be invisible (there is nowhere to
    report it yet).
    """
    # env_str at the boundary; the normalisation below stays a pure function of
    # its argument, so an explicit `value` still bypasses the environment entirely
    # (which is what the formatter's `log_format=` kwarg relies on).
    raw = value if value is not None else env_str("LOG_FORMAT", _DEFAULT, quotes=Quotes.ALLOW)
    candidate = (raw or "").strip().lower()
    return candidate if candidate in (TEXT, JSON) else _DEFAULT


def _iso_utc(created: float) -> str:
    return (
        datetime.fromtimestamp(created, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class StabilityFormatter(logging.Formatter):
    """Text or JSON, decided AT CONSTRUCTION — which is `dictConfig` time.

    That timing is the point. `docker/runtime/live-test.Dockerfile` materialises
    LOGGING_CONFIG to /app/logging_config.json at BUILD time, so anything
    `logging_config.py` reads from the environment at import is baked into the
    image (this is why LOG_LEVEL is not runtime-settable on the dev path). A
    formatter OBJECT is built when uvicorn loads that file at container start, so
    LOG_FORMAT is a real runtime switch on both entry paths.
    """

    def __init__(self, fmt=None, datefmt=None, style="%", log_format=None):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.log_format = resolve_log_format(log_format)

    def format(self, record: logging.LogRecord) -> str:
        if self.log_format != JSON:
            return super().format(record)
        try:
            return json.dumps(self.build_payload(record), default=str)
        except Exception:       # noqa: BLE001 — see the class of failure below
            # A raise here does not produce a worse line, it produces NO line:
            # logging catches it in handleError and prints a traceback instead of
            # the record. Falling back to the text renderer keeps the message.
            return super().format(record)

    def build_payload(self, record: logging.LogRecord) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp": _iso_utc(record.created),
            "level": record.levelname,
            "logger": record.name,
            "thread": record.threadName,
            "message": record.getMessage(),
        }
        payload.update(log_context.static_fields())

        job_id = log_context.current_job_id()
        if job_id is not None:
            payload["job_id"] = job_id

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            payload[key] = value
        return payload
