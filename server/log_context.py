"""Correlation fields for structured logs (STABL-bpsfmoke).

Deliberately imports nothing from `backends` and nothing heavy from `server`. The
JSON formatter calls into this module on EVERY log record — including records
emitted while other modules are still importing, and including records from the
spawned worker child before it has imported torch.

Two scopes, because the process has two:

- ``job_id`` is a ContextVar. It is per-request on the event loop and per-iteration
  on the dispatch thread, and contextvars are the only mechanism that gives both
  without threading an argument through every call site.
- ``mode``/``device_uuid``/``hostname``/``pid`` are process-global. They are the
  same for every line the process writes, so a lock-guarded dict costs less than a
  ContextVar lookup and is readable from any thread.
"""
from __future__ import annotations

import os
import socket
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

job_id_var: ContextVar[Optional[str]] = ContextVar("st_log_job_id", default=None)

_lock = threading.Lock()
_static: Dict[str, Any] = {}


def refresh_process_fields() -> None:
    """(Re)compute the fields that identify THIS process.

    Called at import, and again by the spawned worker child. Spawn re-imports, so
    the child computes its own values anyway — the explicit call is what makes that
    guarantee testable rather than incidental, and it is the one line that would
    matter if the start method ever changed.
    """
    with _lock:
        _static["pid"] = os.getpid()
        try:
            _static["hostname"] = socket.gethostname()
        except Exception:       # noqa: BLE001 — a nameless host still logs
            _static.pop("hostname", None)


def set_static_field(name: str, value: Optional[Any]) -> None:
    """Set a process-wide log field, or remove it when ``value`` is None.

    None REMOVES rather than storing a null: a field that cannot be determined must
    be absent from the payload, so that downstream cannot confuse "unknown" with a
    real value (the ABSENT-NEVER-ZERO rule from STABL-cxbwwgly).
    """
    with _lock:
        if value is None:
            _static.pop(name, None)
        else:
            _static[name] = value


def static_fields() -> Dict[str, Any]:
    """A COPY of the process-wide fields — the caller must not be able to mutate
    the registry by editing a payload it is building."""
    with _lock:
        return dict(_static)


def current_job_id() -> Optional[str]:
    return job_id_var.get()


@contextmanager
def bind_job_id(job_id: Optional[str]) -> Iterator[None]:
    """Bind ``job_id`` for the duration of the block, then restore what was there.

    Restore, NOT clear: nested binds are legal and the outer value has to survive.
    """
    token = job_id_var.set(job_id)
    try:
        yield
    finally:
        job_id_var.reset(token)


refresh_process_fields()
