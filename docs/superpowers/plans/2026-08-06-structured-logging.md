# Structured Loki-Ready Server Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do not use sub-agent driven development** — `AGENTS.md` forbids it in this repo.

**Goal:** Emit structured JSON logs carrying `job_id` correlation from every process that writes to the container's stdout — the server, and the spawned worker child where generation actually happens.

**Architecture:** A formatter class (`server/log_format.StabilityFormatter`) referenced by dotted path from `LOGGING_CONFIG`, so it survives the dev image's build-time materialisation of that dict to `/app/logging_config.json`. It resolves `LOG_FORMAT` **at construction**, which is `dictConfig` time — i.e. container start on both entry paths. Correlation fields come from `server/log_context.py`: a `contextvars.ContextVar` for `job_id` plus a small process-global registry for `mode`, `device_uuid`, `hostname`, `pid`.

**Tech Stack:** stdlib `logging`, `logging.config.dictConfig`, `contextvars`, `json`. No new dependency.

**FP:** `STABL-bpsfmoke` (child of umbrella `STABL-oxbwjwvu`)
**Spec:** `docs/superpowers/specs/2026-08-03-server-observability-seams-design.md` §7

## Global Constraints

- **Inert by default.** `LOG_FORMAT` defaults to `text`; the existing text format is byte-for-byte unchanged when unset or set to anything unrecognised. Same rationale as `METRICS_ENABLED` (spec §1).
- **Nothing added to the dispatch path may raise.** `STABL-hdzggeir`: an exception inside the dispatch loop's own handler kills the loop, the queue goes permanently dead, and `shutdown()` blocks forever on `q.join()`. Every new call in `backends/governor.py` is either provably non-raising or wrapped.
- **A formatter that raises loses the line.** `logging` routes a formatter exception to `handleError`, which prints a traceback *in place of* the record. The JSON path degrades to the text path rather than propagating.
- **Absent, never null.** A field that cannot be determined is omitted from the JSON object; it is never emitted as `null` or `""`. Carried directly from `STABL-cxbwwgly` — `null` and "genuinely empty" must stay distinguishable downstream.
- **`server/superres_cli.py` is out of scope.** It is a CLI; `print` *is* its output contract (spec §7.3).
- **No log-shipping infrastructure in this repo.** No promtail/alloy/collector, no Grafana. `../continuous` owns that.
- Python env: `conda activate stability-toys` before any `pytest`; use `python`, not `python3`.

---

## File Structure

| File | Responsibility |
|---|---|
| `server/log_context.py` *(create)* | `job_id` ContextVar + process-global field registry. Imports nothing from `backends`. |
| `server/log_format.py` *(create)* | `StabilityFormatter`, `resolve_log_format()`, the JSON payload shape. |
| `server/logging_config.py` *(modify)* | Formatters become `"()"` factory references to `StabilityFormatter`. |
| `backends/governor.py` *(modify)* | Bind `job_id` per dispatch-loop iteration; publish `mode` on load/unload. |
| `server/ws_routes.py` *(modify)* | Clear `job_id` per inbound message; set it where the id is minted. |
| `backends/worker_handle_subprocess.py` *(modify)* | `_worker_main` applies `LOGGING_CONFIG` before anything heavy is imported. |
| `server/lcm_sr_server.py`, `server/superres_service.py`, `backends/cuda_worker.py`, `backends/rknnlcm.py` *(modify)* | `print()` → `logger`. |
| `docs/observability-contract.md` *(modify)* | The log field set — the cross-repo half of this work. |
| `tests/test_log_context.py`, `tests/test_log_format.py`, `tests/test_logging_wiring.py` *(create)* | Unit + wiring coverage. |

### Correction to the spec's print inventory

Spec §7.3 lists `server/advisor_service.py` among the in-scope files. **It contains no `print` calls** — the two `grep` hits are `build_evidence_fingerprint(`. The real in-scope inventory is **34 calls in four files**:

| File | Calls |
|---|---|
| `backends/cuda_worker.py` | 26 |
| `backends/rknnlcm.py` | 5 |
| `server/lcm_sr_server.py` | 2 |
| `server/superres_service.py` | 1 |

43 total = 34 in scope + 7 in `superres_cli.py` (excluded) + 2 false matches.

---

## Task 1: Log context — `job_id` and the process-global fields

**Files:**
- Create: `server/log_context.py`
- Test: `tests/test_log_context.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `job_id_var: ContextVar[Optional[str]]`, `bind_job_id(job_id) -> contextmanager`, `current_job_id() -> Optional[str]`, `set_static_field(name: str, value: Optional[Any]) -> None`, `static_fields() -> Dict[str, Any]`, `refresh_process_fields() -> None`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_log_context.py
"""STABL-bpsfmoke: the correlation fields a structured log line carries."""
import os
import threading

import pytest

from server import log_context


@pytest.fixture(autouse=True)
def _isolate():
    saved = log_context.static_fields()
    yield
    for key in list(log_context.static_fields()):
        log_context.set_static_field(key, None)
    for key, value in saved.items():
        log_context.set_static_field(key, value)


def test_job_id_is_absent_until_bound():
    assert log_context.current_job_id() is None


def test_bind_job_id_sets_and_restores():
    with log_context.bind_job_id("abc123"):
        assert log_context.current_job_id() == "abc123"
    assert log_context.current_job_id() is None


def test_bind_job_id_restores_even_when_the_body_raises():
    with pytest.raises(ValueError):
        with log_context.bind_job_id("abc123"):
            raise ValueError("boom")
    assert log_context.current_job_id() is None


def test_nested_binds_restore_the_outer_value():
    with log_context.bind_job_id("outer"):
        with log_context.bind_job_id("inner"):
            assert log_context.current_job_id() == "inner"
        assert log_context.current_job_id() == "outer"


def test_a_bind_does_NOT_leak_into_a_thread_started_from_it():
    """The dispatch loop is a LONG-LIVED thread that copies nothing from the
    submitter. This test pins that fact, because it is the reason the loop has to
    set the var itself rather than inherit it (spec 7.4)."""
    seen = []
    with log_context.bind_job_id("submitter-job"):
        t = threading.Thread(target=lambda: seen.append(log_context.current_job_id()))
        t.start()
        t.join()
    assert seen == [None]


def test_process_fields_carry_pid_and_hostname():
    fields = log_context.static_fields()
    assert fields["pid"] == os.getpid()
    assert isinstance(fields["hostname"], str) and fields["hostname"]


def test_static_fields_returns_a_COPY():
    fields = log_context.static_fields()
    fields["mode"] = "tampered"
    assert "mode" not in log_context.static_fields()


def test_set_static_field_with_None_REMOVES_the_field():
    log_context.set_static_field("mode", "sdxl")
    assert log_context.static_fields()["mode"] == "sdxl"
    log_context.set_static_field("mode", None)
    assert "mode" not in log_context.static_fields()
```

- [x] **Step 2: Run to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_log_context.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'server.log_context'`.

- [x] **Step 3: Write the implementation**

```python
# server/log_context.py
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
```

- [x] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_log_context.py -q`
Expected: 8 passed.

- [x] **Step 5: Commit**

```bash
git add server/log_context.py tests/test_log_context.py
git commit -m "feat(logging): job_id contextvar and process field registry (STABL-bpsfmoke) — next: Task 2, the JSON formatter"
```

---

## Task 2: `StabilityFormatter` and the `LOG_FORMAT` switch

**Files:**
- Create: `server/log_format.py`
- Modify: `server/logging_config.py`
- Test: `tests/test_log_format.py`

**Interfaces:**
- Consumes: `server.log_context.static_fields`, `server.log_context.current_job_id`.
- Produces: `StabilityFormatter(fmt=None, datefmt=None, style="%", log_format=None)`, `resolve_log_format(value=None) -> str`, module constants `TEXT = "text"` / `JSON = "json"`, and `LOGGING_CONFIG` whose `default` and `access` formatters are `"()"` references to `StabilityFormatter`.

### Why construction time is the whole design

`docker/runtime/live-test.Dockerfile:34` materialises `LOGGING_CONFIG` to
`/app/logging_config.json` **at image build time**. Anything `logging_config.py`
reads from the environment at module import is therefore frozen into the image —
which is exactly why `LOG_LEVEL` is not runtime-settable on the dev path today.

A formatter *object*, by contrast, is constructed when `dictConfig` runs — which is
container start, on both entry paths. Reading `LOG_FORMAT` in `__init__` makes it a
genuine runtime switch in dev and prod alike, from a dict that was serialised months
earlier.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_log_format.py
"""STABL-bpsfmoke: the JSON formatter and the LOG_FORMAT switch."""
import json
import logging
import logging.config

import pytest

from server import log_context, log_format
from server.log_format import StabilityFormatter

TEXT_FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def _record(msg="hello", name="st.test", level=logging.INFO, **extra):
    rec = logging.LogRecord(name, level, "/x.py", 10, msg, (), None)
    for key, value in extra.items():
        setattr(rec, key, value)
    return rec


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    yield


# --- the switch --------------------------------------------------------------

def test_default_is_text():
    assert log_format.resolve_log_format() == log_format.TEXT


def test_json_is_selected_case_insensitively_and_trimmed(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "  JSON ")
    assert log_format.resolve_log_format() == log_format.JSON


def test_an_UNRECOGNISED_value_falls_back_to_text_rather_than_raising(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "logfmt")
    assert log_format.resolve_log_format() == log_format.TEXT


def test_an_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    assert log_format.resolve_log_format("text") == log_format.TEXT


# --- text mode is byte-for-byte the old behaviour ----------------------------

def test_text_mode_renders_exactly_what_the_plain_formatter_renders():
    rec = _record()
    ours = StabilityFormatter(fmt=TEXT_FMT).format(rec)
    theirs = logging.Formatter(fmt=TEXT_FMT).format(rec)
    assert ours == theirs


# --- json payload ------------------------------------------------------------

def test_json_mode_emits_the_documented_field_set():
    payload = json.loads(StabilityFormatter(fmt=TEXT_FMT, log_format="json").format(_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "st.test"
    assert payload["message"] == "hello"
    assert payload["pid"] == log_context.static_fields()["pid"]
    assert payload["hostname"] == log_context.static_fields()["hostname"]
    assert "thread" in payload and "timestamp" in payload


def test_json_timestamp_is_iso8601_utc():
    payload = json.loads(StabilityFormatter(log_format="json").format(_record()))
    assert payload["timestamp"].endswith("Z")
    assert "T" in payload["timestamp"]


def test_message_is_INTERPOLATED_not_left_as_a_template():
    rec = logging.LogRecord("st.test", logging.INFO, "/x.py", 10, "n=%s", ("7",), None)
    payload = json.loads(StabilityFormatter(log_format="json").format(rec))
    assert payload["message"] == "n=7"


def test_job_id_is_ABSENT_when_nothing_is_bound():
    payload = json.loads(StabilityFormatter(log_format="json").format(_record()))
    assert "job_id" not in payload


def test_job_id_is_present_when_bound():
    with log_context.bind_job_id("deadbeef"):
        payload = json.loads(StabilityFormatter(log_format="json").format(_record()))
    assert payload["job_id"] == "deadbeef"


def test_process_static_fields_are_merged(monkeypatch):
    log_context.set_static_field("mode", "sdxl-base")
    try:
        payload = json.loads(StabilityFormatter(log_format="json").format(_record()))
    finally:
        log_context.set_static_field("mode", None)
    assert payload["mode"] == "sdxl-base"


def test_exception_text_rides_the_payload():
    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        import sys
        rec = logging.LogRecord("st.test", logging.ERROR, "/x.py", 10, "failed", (), sys.exc_info())
    payload = json.loads(StabilityFormatter(log_format="json").format(rec))
    assert "RuntimeError: kaboom" in payload["exception"]


def test_extra_attributes_survive_as_top_level_fields():
    payload = json.loads(StabilityFormatter(log_format="json").format(_record(mode_epoch=4)))
    assert payload["mode_epoch"] == 4


def test_an_unserialisable_extra_is_STRINGIFIED_not_dropped():
    payload = json.loads(StabilityFormatter(log_format="json").format(_record(thing=object())))
    assert payload["thing"].startswith("<object object")


def test_a_formatter_failure_DEGRADES_to_text_instead_of_losing_the_line(monkeypatch):
    """logging routes a formatter exception to handleError, which prints a
    traceback IN PLACE OF the record. Text is worse than JSON; nothing is worse
    than text."""
    monkeypatch.setattr(
        log_format.StabilityFormatter, "build_payload",
        lambda self, record: (_ for _ in ()).throw(RuntimeError("nope")),
    )
    out = StabilityFormatter(fmt=TEXT_FMT, log_format="json").format(_record())
    assert "hello" in out


# --- dictConfig compatibility ------------------------------------------------

def test_dictConfig_accepts_the_formatter_under_the_key_named_format():
    """dictConfig passes a custom formatter's config keys as KWARGS. The key in
    LOGGING_CONFIG is `format`, and the constructor parameter is `fmt` — CPython's
    configure_formatter retries with `fmt` on the resulting TypeError. This test
    pins that retry, because the whole config file depends on it."""
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "probe": {"()": "server.log_format.StabilityFormatter", "format": TEXT_FMT},
        },
        "handlers": {"probe": {"class": "logging.StreamHandler", "formatter": "probe"}},
        "loggers": {"st.probe": {"handlers": ["probe"], "level": "INFO"}},
    })
    handler = logging.getLogger("st.probe").handlers[0]
    assert isinstance(handler.formatter, StabilityFormatter)
    assert handler.formatter.format(_record()).endswith("hello")
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_log_format.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'server.log_format'`.

- [x] **Step 3: Write the formatter**

```python
# server/log_format.py
"""Structured log formatting (STABL-bpsfmoke).

One formatter class serves both shapes. The choice is made ONCE, in ``__init__``,
because that is `dictConfig` time — see the module note below.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from server import log_context

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
    raw = value if value is not None else os.getenv("LOG_FORMAT", _DEFAULT)
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
```

- [x] **Step 4: Point `LOGGING_CONFIG` at it**

In `server/logging_config.py`, replace the two formatter entries:

```python
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # <-- critical
    "formatters": {
        # Dotted-path reference, NOT an imperatively attached instance. The dev
        # image materialises this dict to /app/logging_config.json at build time
        # and passes it to uvicorn as --log-config, so a formatter that is only
        # reachable from Python works in prod and silently does nothing in dev
        # (spec 7.1). The class resolves LOG_FORMAT when dictConfig CONSTRUCTS
        # it, which is container start on both paths.
        "default": {
            "()": "server.log_format.StabilityFormatter",
            "format": "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        },
        "access": {
            "()": "server.log_format.StabilityFormatter",
            "format": "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        },
    },
    ...unchanged...
}
```

- [x] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_log_format.py -q`
Expected: 16 passed.

- [x] **Step 6: Prove the config still survives JSON round-tripping**

The dev image runs `json.dump(LOGGING_CONFIG, ...)`. If any value stopped being
JSON-serialisable, the dev image would fail to build — a failure that appears at
`docker build`, not in the suite. Add to `tests/test_logging_wiring.py`:

```python
# tests/test_logging_wiring.py
"""STABL-bpsfmoke: the two entry paths that must both get the config."""
import json
import logging
import logging.config

from server.log_format import StabilityFormatter
from server.logging_config import LOGGING_CONFIG


def test_the_config_is_json_serialisable():
    """docker/runtime/live-test.Dockerfile does exactly this at BUILD time. A
    non-serialisable value here breaks the dev image, not the suite — so the suite
    has to check it."""
    json.dumps(LOGGING_CONFIG)


def test_a_ROUND_TRIPPED_config_still_builds_the_formatter():
    """The dev path never sees the Python dict. It sees the file. Configure from
    the round trip, which is what uvicorn --log-config actually loads."""
    materialised = json.loads(json.dumps(LOGGING_CONFIG))
    logging.config.dictConfig(materialised)
    handler = logging.getLogger("uvicorn.error").handlers[0]
    assert isinstance(handler.formatter, StabilityFormatter)


def test_LOG_FORMAT_is_read_at_dictConfig_TIME_not_at_import_time(monkeypatch):
    """The reason the formatter is a dotted reference rather than a value. The
    dict was serialised at image build; the env var is read now."""
    materialised = json.loads(json.dumps(LOGGING_CONFIG))
    monkeypatch.setenv("LOG_FORMAT", "json")
    logging.config.dictConfig(materialised)
    assert logging.getLogger("uvicorn.error").handlers[0].formatter.log_format == "json"


def test_the_default_stays_TEXT_when_LOG_FORMAT_is_unset(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
    assert logging.getLogger("uvicorn.error").handlers[0].formatter.log_format == "text"
```

Run: `python -m pytest tests/test_logging_wiring.py -q`
Expected: 4 passed.

- [x] **Step 7: Restore logging after these tests**

`dictConfig` mutates global logging state and these tests run mid-suite. Add to
`tests/test_logging_wiring.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def _restore_logging():
    """dictConfig is GLOBAL. Without this, every test file collected after this
    one inherits whatever handler set the last test here installed."""
    saved_level = logging.getLogger().level
    saved_handlers = list(logging.getLogger().handlers)
    yield
    logging.config.dictConfig(LOGGING_CONFIG)
    root = logging.getLogger()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
```

Run: `python -m pytest tests/test_logging_wiring.py tests/test_log_format.py tests/test_metrics.py -q`
Expected: all pass — the third file is there to catch cross-file logging leakage.

- [x] **Step 8: Commit**

```bash
git add server/log_format.py server/logging_config.py tests/test_log_format.py tests/test_logging_wiring.py
git commit -m "feat(logging): StabilityFormatter with a runtime LOG_FORMAT switch (STABL-bpsfmoke) — next: Task 3, job_id on the dispatch loop"
```

---

## Task 3: `job_id` on the dispatch thread, `mode` on the process

**Files:**
- Modify: `backends/governor.py` (`_dispatch_loop` ~`:1001`–`:1178`, `_load_mode`, `_unload_current_worker`)
- Test: `tests/test_governor_log_context.py` *(create)*

**Interfaces:**
- Consumes: `server.log_context.job_id_var`, `server.log_context.set_static_field`.
- Produces: nothing new on the Governor's public surface.

### The two traps

**Trap 1 — the reset must be in the `finally`, next to `task_done()`.** The dispatch
loop is one thread processing many jobs and it inherits no context from the
submitter. If it sets and never resets, job N's id appears on job N+1's logs — and
on the loop's own idle logs between jobs. That is worse than an absent field,
because it is *plausible*: it survives review and misleads an operator reading a
production incident.

**Trap 2 — do not re-indent the loop body.** Wrapping ~170 lines in a `with` block
makes the diff unreviewable and risks a mis-indented branch inside the OOM recovery
path. Set the token explicitly and reset it in the existing `finally`. Zero
indentation change.

### How the "between jobs" assertion is actually made

The reset cannot be observed from the test thread — the dispatch loop has its own
context and no probe runs on it between iterations. It **can** be observed through a
log handler: `Handler.emit` runs on the *emitting* thread, so reading
`log_context.current_job_id()` inside `emit` reads the dispatch thread's context at
the moment it logged. `_dispatch_loop` emits `"[Governor] Dispatch loop stopped"`
(`:1179`) on that thread after the last job — with a set-and-never-reset
implementation, that line carries the finished job's id.

The Governor setup below is `test_governor_submit_job_resolves_future_through_handle`
(`tests/test_governor.py:265`) with a recording `run_job`. Reuse it verbatim rather
than inventing a second way to stand a Governor up.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_governor_log_context.py
"""STABL-bpsfmoke: job_id correlation across the dispatch thread."""
import logging
from unittest.mock import Mock, patch

from backends.governor import Governor, GenerationJob
from backends.model_resolution import LocalModelBinding
from server import log_context


class _JobIdProbe(logging.Handler):
    """Records the contextvar AS SEEN BY THE EMITTING THREAD.

    emit() runs on whichever thread logged, so this is the only way to observe the
    dispatch loop's context from a test.
    """

    def __init__(self):
        super().__init__()
        self.seen = []

    def emit(self, record):
        self.seen.append((record.getMessage(), log_context.current_job_id()))


def _governor(run_job):
    """Verbatim from tests/test_governor.py:265, with an injectable run_job."""
    from backends.conditioning.contracts import ConditioningConfig
    from backends.worker_handle import InProcessWorkerHandle

    worker = Mock()
    worker.run_job = run_job
    worker.configure_conditioning = None
    handle = InProcessWorkerHandle(worker_factory=Mock(return_value=worker))

    mode_config = Mock()
    mode = Mock()
    mode.model_path = "/models/test.safetensors"
    mode.loras = []
    mode.conditioning = ConditioningConfig()
    mode_config.get_mode.return_value = mode
    mode_config.get_default_mode.return_value = "test-mode"

    registry = Mock()
    registry.get_used_vram.return_value = 0
    registry.get_allocated_vram.return_value = 0
    registry.get_total_vram.return_value = 8 * 1024**3
    registry.register_model = Mock()

    return Governor(
        worker_factory=Mock(return_value=worker),
        handle=handle,
        mode_config=mode_config,
        registry=registry,
    )


def _resolve(model_path, mode):
    return Mock(), LocalModelBinding(model_path)


def test_a_job_sees_its_own_id():
    seen = []
    with patch("backends.governor.resolve_model", side_effect=_resolve):
        gov = _governor(lambda *a, **k: (seen.append(log_context.current_job_id()), "png")[1])
        job = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
        assert gov.submit_job(job).result(timeout=5.0) == "png"
        gov.shutdown()
    assert seen == [job.job_id]


def test_each_job_sees_its_OWN_id_not_the_previous_one():
    seen = []
    with patch("backends.governor.resolve_model", side_effect=_resolve):
        gov = _governor(lambda *a, **k: (seen.append(log_context.current_job_id()), "png")[1])
        first = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
        gov.submit_job(first).result(timeout=5.0)
        second = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
        gov.submit_job(second).result(timeout=5.0)
        gov.shutdown()
    assert seen == [first.job_id, second.job_id]


def test_the_loop_does_NOT_carry_a_finished_jobs_id_into_its_OWN_lines():
    """The failure this test exists for. Set-without-reset leaves the last job's id
    on every line the dispatch thread emits afterwards — which reads as a real
    correlation, survives review, and misleads whoever is reading the incident."""
    probe = _JobIdProbe()
    gov_logger = logging.getLogger("backends.governor")
    gov_logger.addHandler(probe)
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"))
            job = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
            gov.submit_job(job).result(timeout=5.0)
            gov.shutdown()      # "[Governor] Dispatch loop stopped" — same thread
    finally:
        gov_logger.removeHandler(probe)

    stopped = [jid for msg, jid in probe.seen if "Dispatch loop stopped" in msg]
    assert stopped, "the loop never logged its stop line — test cannot conclude"
    assert stopped == [None] * len(stopped)


def test_mode_is_published_as_a_process_field_on_load():
    """`mode` is process-global, not per-job: every line the process writes while a
    mode is resident belongs to that mode, including lines from threads that never
    saw a job."""
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"))
            assert log_context.static_fields()["mode"] == "test-mode"
            gov.shutdown()
    finally:
        log_context.set_static_field("mode", None)


def test_mode_is_REMOVED_on_unload_not_set_to_empty():
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"))
            gov.unload_current_model()
            assert "mode" not in log_context.static_fields()
            gov.shutdown()
    finally:
        log_context.set_static_field("mode", None)
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_governor_log_context.py -q`
Expected: FAIL — `test_a_job_sees_its_own_id` gets `None`.

- [x] **Step 3: Bind in the dispatch loop**

In `backends/governor.py`, add the import at module top alongside the existing
`from server.metrics import get_metrics`:

```python
from server import log_context
```

Then, in `_dispatch_loop`, between the `q.get` and the existing `try:` (currently
`:1005`–`:1006`):

```python
            except queue.Empty:
                continue
            # STABL-bpsfmoke: bind the correlation id for THIS iteration. The token
            # is set here rather than inside a `with` so the ~170-line body keeps
            # its indentation and the diff stays reviewable; the reset lands in the
            # existing finally, next to task_done(), which is the only place that
            # runs on every exit path (including the `continue` at the cancel
            # check). Without the reset, job N's id appears on job N+1's lines and
            # on the loop's own idle lines — plausible, and wrong.
            _log_token = log_context.job_id_var.set(getattr(job, "job_id", None))
            try:
```

and in the `finally` at `:1176`:

```python
            finally:
                log_context.job_id_var.reset(_log_token)
                self._last_activity = time.monotonic()
                self.q.task_done()
```

`ContextVar.set`/`reset` cannot raise for a valid token, and `getattr(..., None)`
cannot raise — so this satisfies the `STABL-hdzggeir` constraint without a wrapper.

- [x] **Step 4: Publish `mode` on load, unload, AND demand reload**

> **Plan defect found in execution.** The original step named only `_load_mode` and
> `_unload_current_worker` and was **wrong**. `_reload_from_snapshot` (`:605`) brings
> the worker back after an idle eviction *without* going through `_load_mode` — it is
> the epoch-neutral path — and the eviction that preceded it already cleared the
> field. Omit the third site and every line after an evict/reload cycle claims no
> mode is resident while one is. The existing `_publish_mode_active` call in that
> method is the tell: anywhere the metric is republished, the log field must be too.
> Test: `test_a_DEMAND_RELOAD_republishes_mode`.

Find the point in `_load_mode` where the load has SUCCEEDED and the new mode is
recorded as current (`self._current_mode = ...`), and add immediately after it:

```python
                # Process-global, not per-job: threads with no job of their own
                # (sampler, watchdog, uvicorn) still belong to the resident mode.
                self._log_field("mode", mode_name)
```

In `_unload_current_worker`, after the worker is torn down:

```python
                self._log_field("mode", None)   # REMOVE the field; see set_static_field
```

And in `_reload_from_snapshot`, immediately before the existing
`self._metric(...)` block that calls `_publish_mode_active`:

```python
        self._log_field("mode", _mode_name)
```

And add the guarded helper next to the existing `_metric`:

```python
    @staticmethod
    def _log_field(name: str, value) -> None:
        """Guarded exactly like _metric, for the same reason: this runs on the
        dispatch thread and inside lifecycle paths, and STABL-hdzggeir says nothing
        added there may be allowed to raise."""
        try:
            log_context.set_static_field(name, value)
        except Exception:       # noqa: BLE001
            logger.debug("[Governor] log field %s failed", name, exc_info=True)
```

- [x] **Step 5: Publish `device_uuid` once, where the provider is selected**

Without this the field is documented and never emitted in production — and Task 7's
contract test would NOT catch it, because that test sets the field itself. A
documented field nothing writes is the exact drift the contract exists to prevent.

In `Governor.__init__`, immediately after `self._dm = device_memory` (`:403`):

```python
        # One publication point, because provider selection is a singleton and the
        # uuid never changes for the life of the process (STABL-bpsfmoke).
        # getattr, not attribute access: NullDeviceMemory reports "unknown" and a
        # provider without the attribute must degrade to an ABSENT field, not raise.
        self._log_field("device_uuid", getattr(self._dm, "device_uuid", None))
```

Add the corresponding test to `tests/test_governor_log_context.py`:

```python
def test_device_uuid_is_published_from_the_selected_provider():
    dm = Mock()
    dm.device_uuid = "GPU-abc123"
    dm.snapshot.return_value = Mock(consumers=())
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"), device_memory=dm)
            assert log_context.static_fields()["device_uuid"] == "GPU-abc123"
            gov.shutdown()
    finally:
        log_context.set_static_field("device_uuid", None)
        log_context.set_static_field("mode", None)


def test_a_provider_without_a_uuid_leaves_the_field_ABSENT():
    dm = Mock(spec=["snapshot", "cached_snapshot", "reclaim"])   # no device_uuid
    dm.snapshot.return_value = Mock(consumers=())
    try:
        with patch("backends.governor.resolve_model", side_effect=_resolve):
            gov = _governor(Mock(return_value="png"), device_memory=dm)
            assert "device_uuid" not in log_context.static_fields()
            gov.shutdown()
    finally:
        log_context.set_static_field("mode", None)
```

This needs `_governor(...)` to take `device_memory=None` and forward it to the
`Governor(...)` constructor — add that parameter to the helper.

**`Mock(spec=[...])` is deliberate.** A bare `Mock()` auto-creates `device_uuid` and
returns a child Mock, so the absent-field test would pass without exercising
anything. That exact failure cost a green-for-the-wrong-reason test in
`STABL-asawxgvp`.

- [x] **Step 6: Run to verify it passes**

Run: `python -m pytest tests/test_governor_log_context.py -q`
Expected: 7 passed.

- [x] **Step 7: Run the whole Governor suite**

Run: `python -m pytest tests/test_governor.py tests/test_worker_pool.py -q`
Expected: no new failures against the baseline on `main`.

- [x] **Step 8: Commit**

```bash
git add backends/governor.py tests/test_governor_log_context.py
git commit -m "feat(logging): bind job_id per dispatch iteration, publish mode (STABL-bpsfmoke) — next: Task 4, the WS and HTTP entry points"
```

---

## Task 4: `job_id` on the event loop — the WebSocket path

**Files:**
- Modify: `server/ws_routes.py` (the message loop's handler invocation at `:902`; `handle_job_submit` `:227` and `:320`)
- Test: `tests/test_ws_log_context.py` *(create)*

### Every HTTP generation path is out of reach, by prior design

Not just the compat endpoints — **both** HTTP entry points have the same shape, and
the limitation is structural rather than an oversight in either one:

| Handler | Submits | Waits | Logs without an id |
|---|---|---|---|
| `POST /generate` | `lcm_sr_server.py:678` / `:682` | `:699` | `:701` `logger.error("Generate endpoint failed: ...")` |
| compat runner `_run_generate_from_dict` | `:900` / `:902` | `:905` | anything it raises through |

`runtime.submit_generate(req)` returns **only a future** in both. That is deliberate
and load-bearing: `STABL-atzqpcte` established that the waiter keys on *future
identity*, not job id, precisely because an id-keyed API fixes WebSocket and silently
leaves HTTP broken. The comment at `lcm_sr_server.py:696-697` already says so in
place.

Binding `job_id` on either path means adding an id to the runtime's return — an API
change with its own review, not a logging change. **Do not do it in this task.** The
dispatch-thread lines for an HTTP-submitted job still carry the id the Governor
minted (Task 3); what is missing is correlation on the HTTP *handlers' own* lines.

This is stated in the contract doc as a known gap covering **both** paths. Scoping it
to "the compat endpoints" would understate it, and an operator seeing no `job_id` on
a `/generate` failure line would read a documented limitation as a regression.

**Interfaces:**
- Consumes: `server.log_context.job_id_var`, `server.log_context.bind_job_id`.
- Produces: nothing new.

### The design, and why it is not symmetric with Task 3

`asyncio.create_task` **copies the current context** at creation. So a `set` in
`handle_job_submit` is inherited, for free and correctly scoped, by every
`_run_generate` / `_run_comfy` / `_run_sr` / `_run_chat` task spawned at
`ws_routes.py:342`–`:354`. The tasks need no binding of their own.

What that leaves is the *handler frame itself* leaking into the next inbound
message on the same connection. Rather than a `try/finally` around a 100-line
handler body, bind `None` around the **handler invocation in the message loop**.
That is one place, it covers every handler including ones added later, and it makes
leakage structurally impossible rather than a rule someone has to remember.

Note `handle_job_submit` reassigns `job_id` at `:320` from `job.job_id` — the
Governor's id, which is the one the dispatch-thread logs will carry. Set the var a
second time there so both halves of a job's life correlate on the same value. No
second token is needed: the loop-level bind restores whatever was there.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_ws_log_context.py
"""STABL-bpsfmoke: job_id correlation on the event loop."""
import asyncio

import pytest

from server import log_context


@pytest.mark.asyncio
async def test_a_task_created_under_a_bind_INHERITS_the_id():
    """The property the WS design leans on. If this ever stops holding, every
    _run_generate log line loses its job_id silently."""
    seen = []

    async def child():
        seen.append(log_context.current_job_id())

    with log_context.bind_job_id("job-42"):
        await asyncio.create_task(child())
    assert seen == ["job-42"]


@pytest.mark.asyncio
async def test_a_task_created_AFTER_the_bind_exits_does_not_see_it():
    seen = []

    async def child():
        seen.append(log_context.current_job_id())

    with log_context.bind_job_id("job-42"):
        pass
    await asyncio.create_task(child())
    assert seen == [None]
```

Then the wiring test, using whatever WebSocket test client the existing
`tests/test_ws_*.py` files use — **read one first and follow it**:

```python
def test_a_handler_that_sets_the_id_does_not_leak_it_to_the_next_message():
    """Two messages on ONE connection. The second must not inherit the first's
    correlation id."""
    # Register a probe handler that records log_context.current_job_id(),
    # send job:submit then the probe message on the same socket,
    # assert the probe saw None.
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_ws_log_context.py -q`
Expected: the leak test FAILS (probe sees the submitted job's id).

- [x] **Step 3: Clear per inbound message**

In `server/ws_routes.py:902`, the message loop invokes the resolved handler as
`result = await handler(ws, msg, client_id)`. Wrap exactly that call:

```python
                # STABL-bpsfmoke: every message starts with a clean correlation id.
                # Handlers SET it (job:submit does, once it has minted one); binding
                # None here is what stops it surviving into the next message on the
                # same connection. One place, so a handler added later cannot forget.
                with log_context.bind_job_id(None):
                    result = await handler(ws, msg, client_id)
```

Match the surrounding indentation — `:902` sits inside the loop's `try`.

- [x] **Step 4: Set it where the id is minted**

In `handle_job_submit`, immediately after `job_id = uuid.uuid4().hex[:12]` (`:227`):

```python
    log_context.job_id_var.set(job_id)   # reset by the message loop's bind
```

and immediately after `job_id = job.job_id` (`:320`):

```python
        log_context.job_id_var.set(job_id)   # the Governor's id — the one the
                                             # dispatch-thread logs will carry
```

- [x] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_ws_log_context.py -q`
Expected: 3 passed.

- [x] **Step 6: Run the WS suites**

Run: `python -m pytest tests/test_ws_routes.py tests/test_ws_metrics.py -q`
Expected: no new failures. (Adjust the file list to whatever `ls tests/test_ws*` shows.)

- [x] **Step 7: Commit**

```bash
git add server/ws_routes.py tests/test_ws_log_context.py
git commit -m "feat(logging): bind job_id on the WebSocket entry path (STABL-bpsfmoke) — next: Task 5, the spawned worker child"
```

---

## Task 5: The spawned worker child configures its own logging

**Files:**
- Modify: `backends/worker_handle_subprocess.py` (`_worker_main`, `:180`)
- Test: `tests/test_worker_child_logging.py` *(create)*

**Interfaces:**
- Consumes: `server.logging_config.LOGGING_CONFIG`, `server.log_context.refresh_process_fields`.
- Produces: nothing.

### Why this is in scope and not a follow-up

Under `WORKER_ISOLATION=subprocess` — the **production** path — generation happens
in the child. The child inherits the parent's stdout and sees neither `run.py`'s
`dictConfig` nor uvicorn's `--log-config`. The container's log stream is therefore
JSON from the parent interleaved with default-formatted lines from the child, and
the interleaved lines are the ones about the actual work. A JSON logging story that
omits the worker is not a logging story.

`backends` already imports from `server` in ten places (`governor.py:25-26`,
`worker_pool.py:21`, and others), so there is no import-direction objection here —
and this is a *process bootstrap*, not a layer dependency.

- [x] **Step 1: Write the failing test**

> **Two plan defects found in execution.**
>
> 1. **The child must report its failures on the queue.** As first written, a
>    child that raised (which is exactly what RED is) put nothing on the queue, so
>    the assertion became a `q.get` timeout: **122 seconds to learn `ImportError`**.
>    Wrap the child body and put `("error", False, traceback)`.
> 2. **Calling `_configure_child_logging()` directly does not test the wiring.**
>    That test passes unchanged if `_worker_main` stops calling it. A second test
>    parses `_worker_main` with `ast` and asserts the bootstrap is its **first**
>    statement after the docstring — pinning both the call and the ordering torch's
>    import-time logging depends on.

```python
# tests/test_worker_child_logging.py
"""STABL-bpsfmoke: the spawned child configures the same logging the parent has."""
import json
import logging
import multiprocessing as mp
import os
import sys


def _child(queue):
    """Runs in a REAL spawned process. Anything the parent configured is absent
    here — that absence is the bug under test."""
    os.environ["LOG_FORMAT"] = "json"
    from backends.worker_handle_subprocess import _configure_child_logging
    _configure_child_logging()
    from server.log_format import StabilityFormatter
    handler = logging.getLogger().handlers[0]
    rec = logging.LogRecord("st.child", logging.INFO, "/x.py", 1, "from the child", (), None)
    queue.put((isinstance(handler.formatter, StabilityFormatter),
               handler.formatter.format(rec)))


def test_the_child_gets_the_json_formatter_and_its_OWN_pid():
    ctx = mp.get_context("spawn")     # spawn, NOT fork — the facet-3 invariant
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q,))
    p.start()
    is_ours, line = q.get(timeout=60)
    p.join(timeout=30)
    assert is_ours
    payload = json.loads(line)
    assert payload["message"] == "from the child"
    assert payload["pid"] == p.pid          # ITS pid, not the parent's
    assert payload["pid"] != os.getpid()
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_worker_child_logging.py -q`
Expected: FAIL — `ImportError: cannot import name '_configure_child_logging'`.

- [x] **Step 3: Add the child bootstrap**

In `backends/worker_handle_subprocess.py`, above `_worker_main`:

```python
def _configure_child_logging() -> None:
    """Apply the server's logging config inside the spawned child (STABL-bpsfmoke).

    The child inherits the parent's STDOUT but none of its logging configuration:
    it never runs run.py's dictConfig and never sees uvicorn's --log-config. Under
    WORKER_ISOLATION=subprocess that means the container's log stream is JSON from
    the parent interleaved with default-formatted lines from the child — and the
    child is where generation happens.

    Imported lazily and wrapped: this is a process bootstrap and it runs before the
    worker exists, so a failure here must degrade to unconfigured logging rather
    than take down a child the parent is waiting on.
    """
    try:
        import logging.config

        from server.log_context import refresh_process_fields
        from server.logging_config import LOGGING_CONFIG

        logging.config.dictConfig(LOGGING_CONFIG)
        refresh_process_fields()        # this process's OWN pid and hostname
    except Exception:                   # noqa: BLE001 — a child that cannot
        pass                            # configure logging must still run
```

and make it the **first statement** of `_worker_main`, before the existing `try:`:

```python
def _worker_main(conn, factory_ref, wire_resolved, binding, mode, control_conn=None):
    """..."""
    _configure_child_logging()      # BEFORE anything heavy imports: torch and
                                    # diffusers both log at import, and those
                                    # lines are worth having in the same shape.
    try:
        from backends.model_resolution import resolved_model_from_json_dict
```

- [x] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_worker_child_logging.py -q`
Expected: 1 passed.

- [x] **Step 5: Run the subprocess handle suite**

Run: `python -m pytest tests/test_worker_handle_subprocess.py -q`
Expected: no new failures.

- [x] **Step 6: Commit**

```bash
git add backends/worker_handle_subprocess.py tests/test_worker_child_logging.py
git commit -m "feat(logging): configure logging in the spawned worker child (STABL-bpsfmoke) — next: Task 6, print() to logger"
```

---

## Task 6: `print()` → `logger`

**Files:**
- Modify: `backends/cuda_worker.py` (26), `backends/rknnlcm.py` (5), `server/lcm_sr_server.py` (2), `server/superres_service.py` (1)
- Test: `tests/test_no_print_in_server_runtime.py` *(create)*

**Interfaces:** none — this is a mechanical conversion with a guard test.

### Rules for the conversion

- **Level by content, not by uniformity.** `[cuda] FAILED to load style LoRA` is
  `logger.error(..., exc_info=True)` where an exception is in hand;
  `[cuda] loading diffusers` is `logger.info`; the three `debug dump ... failed`
  lines in `cuda_worker.py` (`:187`, `:236`, `:245`) are `logger.debug` — they are
  diagnostics behind `HUNYUAN_DEBUG_DUMP`, and promoting them would make an opt-in
  tool noisy for everyone who never enabled it.
- **Keep the message text identical**, including the `[cuda]` / `[sdxl-cuda]` /
  `[hunyuandit-cuda]` prefixes. They are grep targets in existing notes and
  acceptance scripts. The logger *name* additionally carries the module, which is
  the structured version of the same information, but do not remove the prefix in
  this task.
- **Use lazy `%s` interpolation** (`logger.info("x=%s", x)`) rather than f-strings
  for anything on the job path. An f-string is evaluated even when the level is
  disabled.
- **Each file gets one module-level logger**: `logger = logging.getLogger(__name__)`,
  added next to the existing imports if not already present.
- Do **not** touch `server/superres_cli.py`.

- [ ] **Step 1: Write the guard test**

```python
# tests/test_no_print_in_server_runtime.py
"""STABL-bpsfmoke: server-runtime output goes through logging, not stdout.

A print bypasses level, formatter and structure, and lands in the middle of a
stream something downstream is parsing as JSON.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# server/superres_cli.py is EXCLUDED on purpose: it is a CLI and print IS its
# output contract. Converting it would be a regression, not a fix (spec 7.3).
EXCLUDED = {"server/superres_cli.py"}

IN_SCOPE = sorted(
    p for d in ("server", "backends")
    for p in ROOT.joinpath(d).rglob("*.py")
    if str(p.relative_to(ROOT)) not in EXCLUDED
)


@pytest.mark.parametrize("path", IN_SCOPE, ids=lambda p: str(p.name))
def test_no_print_calls(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert not offenders, f"{path.relative_to(ROOT)} prints at lines {offenders}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_no_print_in_server_runtime.py -q`
Expected: 4 failures, naming the four files and their line numbers.

- [ ] **Step 3: Convert `server/lcm_sr_server.py` and `server/superres_service.py`**

Three calls total. `lcm_sr_server.py:228` and `:231` are startup lines → `logger.info`.
`superres_service.py:207` (`[SR] worker N loaded <path>`) is a lifecycle line →
`logger.info`.

- [ ] **Step 4: Convert `backends/rknnlcm.py`**

Five calls, all timing output on the RKNN job path (`:539`, `:589`, `:620`, `:623`,
`:626`). These are per-job measurements → `logger.info` with lazy interpolation.
`:620` uses positional `print("scale:", a, "vae_inf:", b, ...)` — rewrite as a
single format string, do not pass multiple positional args to `logger.info`.

- [ ] **Step 5: Convert `backends/cuda_worker.py`**

Twenty-six calls. Work top to bottom; the multi-line `print(` calls at `:474`,
`:985`, `:1014`, `:1362`, `:1392`, `:1718` need their continuation lines checked.
Levels: `:187`/`:236`/`:245` → `debug`; `:486` (`xformers enable failed`),
`:1004`/`:1382` (`FAILED to load style LoRA`) → `warning` or `error` with
`exc_info=True` where an exception object is in scope; everything else → `info`.

- [ ] **Step 6: Run to verify it passes**

Run: `python -m pytest tests/test_no_print_in_server_runtime.py -q`
Expected: all parametrised cases pass.

- [ ] **Step 7: Run the worker suites**

Run: `python -m pytest tests/test_cuda_worker.py tests/test_sdxl_worker.py tests/test_hunyuandit_worker.py -q`
Expected: no new failures. If a test asserted on captured stdout, it must move to
`caplog` — note it in the commit if so.

- [ ] **Step 8: Commit**

```bash
git add backends/cuda_worker.py backends/rknnlcm.py server/lcm_sr_server.py server/superres_service.py tests/test_no_print_in_server_runtime.py
git commit -m "refactor(logging): route 34 server-runtime prints through logging (STABL-bpsfmoke) — next: Task 7, the log field contract"
```

---

## Task 7: The contract, the doc⇄code test, and closeout

**Files:**
- Modify: `docs/observability-contract.md`
- Test: `tests/test_log_format.py` (extend)

**Interfaces:** none.

### Why the field set belongs in the contract doc

`docs/observability-contract.md` is what `../continuous` consumes. Metric names are
already there. Log field names are the same kind of artifact — a downstream Loki
query breaks the same way a broken metric name does — and the doc is the only place
that is already a shared interface rather than a repo-local note.

- [ ] **Step 1: Write the bidirectional field test**

```python
# append to tests/test_log_format.py
import pathlib
import re

CONTRACT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "observability-contract.md"


def _documented_log_fields():
    """Field names from the contract's log-field table: | `name` | ... |"""
    section = CONTRACT.read_text().split("## Structured logs")[1].split("\n## ")[0]
    return set(re.findall(r"^\|\s*`([a-z_]+)`\s*\|", section, re.M))


def _emitted_log_fields():
    log_context.set_static_field("mode", "m")
    log_context.set_static_field("device_uuid", "GPU-x")
    try:
        with log_context.bind_job_id("j"):
            rec = logging.LogRecord("st.test", logging.ERROR, "/x.py", 1, "m", (), None)
            rec.exc_info = None
            payload = json.loads(StabilityFormatter(log_format="json").format(rec))
    finally:
        log_context.set_static_field("mode", None)
        log_context.set_static_field("device_uuid", None)
    return set(payload)


def test_every_emitted_log_field_is_documented():
    undocumented = _emitted_log_fields() - _documented_log_fields()
    assert not undocumented, f"emitted but not in the contract: {sorted(undocumented)}"


def test_every_documented_log_field_is_actually_emitted():
    """The direction that catches prose drift. It caught invented metric names in
    STABL-xmsrxvto; the same failure mode applies to field names."""
    optional = {"exception", "stack"}   # only present on a failing record
    missing = _documented_log_fields() - _emitted_log_fields() - optional
    assert not missing, f"documented but never emitted: {sorted(missing)}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_log_format.py -k log_field -q`
Expected: FAIL — the contract has no `## Structured logs` section yet (IndexError
on the split, or an empty documented set).

- [ ] **Step 3: Write the contract section**

Append to `docs/observability-contract.md`:

````markdown
## Structured logs

Emitted to **stdout**, one JSON object per line, when `LOG_FORMAT=json`. Default is
`text` — the unchanged human format. There is no log-shipping configuration in this
repo; `../continuous` owns collection.

Both processes that write to the container's stdout emit this shape: the server, and
the spawned worker child under `WORKER_ISOLATION=subprocess`. `pid` distinguishes
them.

| Field | Always? | Meaning |
|---|---|---|
| `timestamp` | yes | ISO 8601, UTC, milliseconds, `Z`-suffixed |
| `level` | yes | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |
| `logger` | yes | Python logger name — the emitting module |
| `thread` | yes | Thread name. The dispatch loop and the event loop are different threads and this is how you tell them apart |
| `message` | yes | Interpolated message text |
| `pid` | yes | Emitting process. Server and worker child both write here |
| `hostname` | yes | Deliberately a log field and NOT a metric label — see the label policy above |
| `mode` | while a mode is resident | Active mode name. Absent when nothing is loaded |
| `device_uuid` | when a device is resolved | Stable GPU identity |
| `job_id` | during a job | Correlation id. Present on the WebSocket handler AND on the dispatch-thread lines for the same job. **Known gap:** no HTTP handler line carries it — see below |
| `exception` | on a failing record | Formatted traceback |
| `stack` | on `stack_info=True` | Formatted stack |

Any field a caller attaches via `logging`'s `extra=` appears alongside these.

**No HTTP handler line carries `job_id`. This is a documented limitation, not a
regression.** It applies to **both** HTTP generation paths — `POST /generate` and the
compat endpoints' runner — and for the same reason: `submit_generate()` returns only a
future, deliberately (`STABL-atzqpcte` — an id-keyed waiter API fixes WebSocket and
silently leaves HTTP broken). So a `/generate` failure line has no `job_id` by design.

The generation's own **dispatch-thread** lines still carry the id, so the work is
correlatable; what is not correlatable is the HTTP request that asked for it. Closing
this needs a runtime API change, not a formatter change.

Correlate a WebSocket job end to end; for an HTTP request, correlate by time and mode
against the dispatch-thread lines.

**`job_id` correlation spans two threads.** The WS handler runs on the event loop and
the generation runs on the Governor's dispatch thread; contextvars do not cross that
boundary on their own, so the dispatch loop sets and **resets** the id per iteration.
A missing `job_id` on a dispatch line is a bug; a *wrong* one would be worse, which is
why the reset sits in the loop's `finally` next to `task_done()`.

Correlate a whole job with:

```logql
{container="stability-toys"} | json | job_id = "<id>"
```
````

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_log_format.py -q`
Expected: all pass, including both contract directions.

- [ ] **Step 5: Check drift bindings**

```bash
drift refs docs/observability-contract.md
drift refs server/logging_config.py
drift check
```

If the contract doc is bound and its prose is now stale, **update the prose first**,
then `drift link`, then `drift check` again. Never relink without reviewing the
prose.

- [ ] **Step 6: Run the full suite**

```bash
python -m pytest -q 2>&1 | tail -5
```

Baseline on `main` is **1323 passed, 9 skipped, 1 xfailed**. Record the number from
**this** tree — run the suite on the tree you are about to summarise, not the one you
had when you started typing.

- [ ] **Step 7: Live verification, JSON on**

```bash
LOG_FORMAT=json python -c "
import logging, logging.config
from server.logging_config import LOGGING_CONFIG
from server import log_context
logging.config.dictConfig(LOGGING_CONFIG)
log_context.set_static_field('mode', 'sdxl-base')
with log_context.bind_job_id('abc123def456'):
    logging.getLogger('st.demo').info('generation started steps=%s', 4)
"
```

Expected: one JSON line carrying `job_id`, `mode`, `pid`, `hostname`. Then run it
again with `LOG_FORMAT` unset and confirm the output is the unchanged text format.

- [ ] **Step 8: Commit and close out**

```bash
git add docs/observability-contract.md docs/superpowers/plans/2026-08-06-structured-logging.md tests/test_log_format.py
git commit -m "docs(logging): document the structured log field set (STABL-bpsfmoke) — next: PR"
fp issue assign STABL-bpsfmoke --rev <sha1>,<sha2>,...   # every commit, in order
fp comment STABL-bpsfmoke "STOP: ... NEXT: ... DECIDED: ..."
```

---

## Deferred, tracked, NOT done here

| Item | Why |
|---|---|
| Runtime `LOG_LEVEL` on the dev path | Same build-time-materialisation problem the formatter solves, but `level` is a plain value in the dict with no factory hook. `dictConfig` has no env interpolation. Needs its own decision — a `()` filter, or an entrypoint that regenerates the file. |
| `server/lcm_sr_server.py:1032`'s `from logging_config import LOGGING_CONFIG` | Missing the `server.` prefix, so `python server/lcm_sr_server.py` would `ImportError`. Dead path (the dev CMD imports the app; prod uses `run.py`), pre-existing, and fixing it is not a logging-format change. File separately if it matters. |
| `job_id` on HTTP handler lines — **both** `POST /generate` and the compat runner | `submit_generate()` returns only a future by design (`STABL-atzqpcte`). Needs a runtime API change with its own review. Documented as a known gap in the contract, scoped to both paths so a missing id on `/generate` does not read as a regression. |
| `server/superres_cli.py` | Out of scope by decision, not by omission. |
| Promtail / alloy / Grafana | `../continuous` owns log shipping. Explicit non-goal. |
| Tracing | `STABL-qnlaclof`. |
