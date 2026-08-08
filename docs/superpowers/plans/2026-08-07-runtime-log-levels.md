# Runtime Per-Logger Log Levels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do not use sub-agent driven development** — `AGENTS.md` forbids it in this repo.

**Goal:** Make `LOG_LEVEL` and per-logger levels overridable at runtime on every entry path, without removing the baked config — the Spring model, where the artifact ships declared defaults and the environment overrides them.

**Architecture:** `logging_config.py` names which loggers *track* `LOG_LEVEL` and builds its dict from that one tuple. A new `server/log_levels.py` applies the precedence chain after `dictConfig` has run, called from the FastAPI `lifespan` (both entry paths) and from `_configure_child_logging` (the spawned worker child).

**Tech Stack:** stdlib `logging`, `os.environ`. No new dependency.

**FP:** `STABL-ataigkdk`

## Global Constraints

- **The baked config stays.** It is the declared default, not the problem. `comfy.jobs: DEBUG` in particular must survive a runtime `LOG_LEVEL` change — it is a deliberate per-logger declaration.
- **Precedence, exactly:** `LOG_LEVELS[name]` > declared per-logger level > `LOG_LEVEL` (root default) > `INFO`.
- **`INFO` is a real floor, not a documentation flourish.** An absent *or invalid*
  `LOG_LEVEL` resolves the tracking loggers to `INFO`. It does **not** leave them
  at the baked value — see "Two kinds of baked value" below.
- **Nothing here may raise.** This runs in `lifespan` (fails server startup) and in the spawned child's bootstrap (hangs a parent waiting on `_READY`). Every parse and every application is guarded.
- **Test against `json.loads(json.dumps(LOGGING_CONFIG))`.** The dev path never sees the Python dict, and the dev path is the only place the frozen-level bug exists. A test against the in-memory object cannot reproduce it.
- **Option 2 — regenerating `/app/logging_config.json` at build or entrypoint — is REJECTED on the record** as a kludge. Do not reintroduce it.
- Python env: `conda activate stability-toys`; use `python`, not `python3`.

---

## Why this is not "just call setLevel"

Three findings from tracing the current code. Each one changes the design.

### 1. In the materialised JSON, declared and derived levels are indistinguishable

```text
(root)           level='INFO'    <- from LOG_LEVEL
comfy.jobs       level='DEBUG'   <- DECLARED, must survive
uvicorn          level='INFO'    <- from LOG_LEVEL
uvicorn.error    level='INFO'    <- from LOG_LEVEL
uvicorn.access   level='INFO'    <- from LOG_LEVEL
```

`LOG_LEVEL` is substituted at import (`logging_config.py:4`), so by the time anything can read the config back, the provenance is gone. An override that re-applies `LOG_LEVEL` to every configured logger would silently stomp `comfy.jobs` — destroying the exact declaration the Spring model exists to protect.

**Therefore `logging_config.py` must export the tracking set, and build its dict from it.** Two lists that must agree is a bug waiting; one tuple that generates both cannot drift.

### 1b. Two kinds of baked value, and only one of them has authority

The distinction above has a consequence the first draft of this plan missed, caught
in review:

| baked level | what it is | on absent/invalid `LOG_LEVEL` |
|---|---|---|
| `comfy.jobs: "DEBUG"` | a **source literal** — someone decided this logger is worth DEBUG | **survives**; it is an intent |
| `uvicorn: "INFO"` | a **snapshot of whatever environment the build ran in** | **replaced** by `INFO`; it is an accident |

A tracking logger's baked level is not a declaration. It is build-environment
leakage frozen into an artifact — which is the entire bug this issue exists to fix.
So "on invalid input, leave the baked value alone" **preserves exactly the frozen
state**, in a costume. The floor in the precedence chain has to be real: absent or
invalid `LOG_LEVEL` resolves the tracking loggers to `INFO`.

This mirrors what `logging_config.py` already says — `os.getenv("LOG_LEVEL", "INFO")`
means absent is `INFO`. Invalid gets the same answer, plus a warning.

**`LOG_LEVELS` is deliberately different.** A malformed or unknown-level entry there
is *skipped*, leaving the declared or tracking level in place. That is not
inconsistent: a per-logger override that cannot be parsed simply does not apply,
whereas a broken `LOG_LEVEL` still has to resolve to something for loggers whose
level is defined as "whatever `LOG_LEVEL` says".

### 2. Setting the root logger is not enough, and it looks like it works

`LOGGING_CONFIG` assigns **explicit** levels to `uvicorn`, `uvicorn.error` and `uvicorn.access`. An explicit child level shadows the root, so `LOG_LEVEL=DEBUG` applied only to root leaves all three at the baked value — while root itself visibly changes, which is what makes the bug convincing.

### 3. The child is already fine; only the dev parent is frozen

`_configure_child_logging` (`worker_handle_subprocess.py:197`) re-imports `server.logging_config` in a **fresh spawned process**, so its module-level `os.getenv("LOG_LEVEL")` reads the live environment. The child already honours a runtime `LOG_LEVEL` today.

Only the **dev parent** is frozen, because it alone consumes the build-time `/app/logging_config.json`. The child still needs `apply_runtime_levels()` for the *per-logger* `LOG_LEVELS` half, but not for `LOG_LEVEL`.

Handlers are all `NOTSET` (verified), so filtering is entirely at logger level. Nothing here touches handlers.

---

## File Structure

| File | Responsibility |
|---|---|
| `server/logging_config.py` *(modify)* | Export `LEVEL_TRACKING_LOGGERS`; build the dict from it |
| `server/log_levels.py` *(create)* | Parse `LOG_LEVELS`, apply the precedence chain |
| `server/lcm_sr_server.py` *(modify)* | Call it from `lifespan` |
| `backends/worker_handle_subprocess.py` *(modify)* | Call it from `_configure_child_logging` |
| `docs/observability-contract.md` *(modify)* | Document the two env vars and the precedence |
| `tests/test_log_levels.py` *(create)* | Unit + precedence + the dev-path reproduction |

---

## Task 1: `logging_config.py` names what tracks `LOG_LEVEL`

**Files:**
- Modify: `server/logging_config.py`
- Test: `tests/test_log_levels.py` *(create)*

**Interfaces:**
- Produces: `LEVEL_TRACKING_LOGGERS: tuple[str, ...]` — logger names whose level is `LOG_LEVEL` rather than a declared literal. `LOGGING_CONFIG` is built from it.

This task is a **refactor with no behaviour change**. The rendered config must be byte-identical.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_log_levels.py
"""STABL-ataigkdk: runtime per-logger log levels."""
import json

from server import logging_config
from server.logging_config import LEVEL_TRACKING_LOGGERS, LOGGING_CONFIG


def test_the_tracking_set_names_root_and_the_uvicorn_loggers():
    assert set(LEVEL_TRACKING_LOGGERS) == {"", "uvicorn", "uvicorn.error", "uvicorn.access"}


def test_comfy_jobs_is_NOT_level_tracking():
    """It is a DECLARED default (Spring sense) and must survive a runtime
    LOG_LEVEL change. If it ever joins the tracking set, that is the bug."""
    assert "comfy.jobs" not in LEVEL_TRACKING_LOGGERS
    assert LOGGING_CONFIG["loggers"]["comfy.jobs"]["level"] == "DEBUG"


def test_every_tracking_logger_actually_carries_LOG_LEVEL():
    """The two must not drift: a name in the tuple that the dict does not
    configure, or configures with a literal, is a silent no-op at override time."""
    for name in LEVEL_TRACKING_LOGGERS:
        assert name in LOGGING_CONFIG["loggers"], f"{name!r} is tracked but not configured"
        assert LOGGING_CONFIG["loggers"][name]["level"] == logging_config.LOG_LEVEL


def test_every_configured_logger_is_either_tracking_or_declared():
    """No third category. A logger that is neither will silently never respond to
    LOG_LEVEL and will not be recognised as a deliberate declaration either."""
    declared = {"comfy.jobs"}
    assert set(LOGGING_CONFIG["loggers"]) == set(LEVEL_TRACKING_LOGGERS) | declared


def test_the_config_is_still_json_serialisable():
    """The dev image does this at BUILD time; breaking it breaks the image, not
    the suite."""
    json.dumps(LOGGING_CONFIG)
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_log_levels.py -q`
Expected: `ImportError: cannot import name 'LEVEL_TRACKING_LOGGERS'`.

- [x] **Step 3: Capture the current rendered config, to prove the refactor is a no-op**

```bash
python -c "
import json
from server.logging_config import LOGGING_CONFIG
print(json.dumps(LOGGING_CONFIG, sort_keys=True, indent=2))
" > /tmp/logging-config-before.json
```

- [x] **Step 4: Refactor `logging_config.py`**

```python
# logging_config.py
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Loggers whose level IS LOG_LEVEL, as opposed to a declared literal like
# comfy.jobs's DEBUG. The runtime override (server/log_levels.py) re-applies
# LOG_LEVEL to exactly these and leaves declared levels alone.
#
# This tuple exists because the distinction is UNRECOVERABLE later: LOG_LEVEL is
# substituted at import, and the dev image materialises the dict to JSON at build
# time, so a reader of that file sees "INFO" and "DEBUG" as equally literal. The
# dict below is built FROM this tuple so the two cannot drift (STABL-ataigkdk).
LEVEL_TRACKING_LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access")

_LEVEL_TRACKING_HANDLERS = {
    "uvicorn.access": ["access"],       # the one that uses the access formatter
}

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # <-- critical
    "formatters": {
        # ...unchanged, keep the existing comment block verbatim...
    },
    "handlers": {
        # ...unchanged...
    },
    "loggers": {
        **{
            name: {
                "handlers": _LEVEL_TRACKING_HANDLERS.get(name, ["default"]),
                "level": LOG_LEVEL,
                # The root logger takes no `propagate` key; every other logger
                # sets it False so uvicorn's records are not double-emitted.
                **({} if name == "" else {"propagate": False}),
            }
            for name in LEVEL_TRACKING_LOGGERS
        },
        # DECLARED default for one logger, in the Spring sense: the baked config
        # states an intent and the environment may override it per logger. Live —
        # server/comfy_routes.py:18 uses it.
        "comfy.jobs": {
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
```

**Keep the existing formatter comment block verbatim.** It explains why `LOG_FORMAT`
is a `"()"` reference and is the reason that half works.

- [x] **Step 5: Prove the rendered config is unchanged**

```bash
python -c "
import json
from server.logging_config import LOGGING_CONFIG
print(json.dumps(LOGGING_CONFIG, sort_keys=True, indent=2))
" > /tmp/logging-config-after.json
diff /tmp/logging-config-before.json /tmp/logging-config-after.json && echo "IDENTICAL"
```

Expected: `IDENTICAL`. **If it differs, the refactor changed behaviour** — most
likely `propagate` on root, or the `access` handler on `uvicorn.access`. Fix
before continuing; do not accept a diff here.

- [x] **Step 6: Run to verify the tests pass**

Run: `python -m pytest tests/test_log_levels.py tests/test_logging_wiring.py -q`
Expected: all pass.

- [x] **Step 7: Commit**

```bash
git add server/logging_config.py tests/test_log_levels.py
git commit -m "refactor(logging): name which loggers track LOG_LEVEL (STABL-ataigkdk) — next: Task 2, apply_runtime_levels"
```

---

## Task 2: `apply_runtime_levels()`

**Files:**
- Create: `server/log_levels.py`
- Test: `tests/test_log_levels.py` (extend)

**Interfaces:**
- Consumes: `server.logging_config.LEVEL_TRACKING_LOGGERS`.
- Produces: `apply_runtime_levels() -> Dict[str, str]` (logger name → applied level, for logging and tests), `parse_log_levels(raw: Optional[str]) -> Dict[str, str]`.

### The env surface

| var | meaning |
|---|---|
| `LOG_LEVEL` | root default, applied to `LEVEL_TRACKING_LOGGERS`. Existing behaviour, unchanged meaning. |
| `LOG_LEVELS` | per-logger overrides: `"comfy.jobs=WARNING,backends.governor=DEBUG"` |

**Why one variable with explicit keys, and not Spring's `LOG_LEVEL_COMFY_JOBS`.**
Spring can mangle `logging.level.org.springframework.web` into
`LOGGING_LEVEL_ORG_SPRINGFRAMEWORK_WEB` because Java packages contain no
underscores. Ours do — `server.ws_routes`, `backends.cuda_worker`,
`server.log_format`. `SERVER_WS_ROUTES` cannot be reversed: `server.ws_routes` or
`server.ws.routes`? The only repair is to forward-transform every *known* logger
and compare, but loggers are created lazily on first `getLogger()`, so the known
set is incomplete at startup. A per-var scheme could therefore only target loggers
that already exist, **and the failure is silent** — set the var, nothing happens,
no error. Explicit keys have none of that.

**`getLogger(name)` on an unknown name CREATES it**, which is a feature here: a
level set for a logger whose module has not imported yet is waiting when it does.

- [x] **Step 1: Write the failing tests**

```python
# append to tests/test_log_levels.py
import logging
import logging.config

import pytest

from server import log_levels
from server.log_levels import apply_runtime_levels, parse_log_levels


@pytest.fixture(autouse=True)
def _restore_logging():
    """dictConfig and setLevel are GLOBAL. Snapshot the levels this file touches
    and put them back, or every later test file inherits them."""
    names = ["", "comfy.jobs", "uvicorn", "uvicorn.error", "uvicorn.access",
             "st.probe", "st.probe.child"]
    saved = {n: logging.getLogger(n).level for n in names}
    yield
    logging.config.dictConfig(LOGGING_CONFIG)
    for n, lvl in saved.items():
        logging.getLogger(n).setLevel(lvl)


# --- parsing -----------------------------------------------------------------

def test_parse_empty_and_none_yield_nothing():
    assert parse_log_levels(None) == {}
    assert parse_log_levels("") == {}
    assert parse_log_levels("   ") == {}


def test_parse_one_pair():
    assert parse_log_levels("comfy.jobs=WARNING") == {"comfy.jobs": "WARNING"}


def test_parse_several_pairs_and_tolerates_whitespace():
    assert parse_log_levels(" a.b = debug , c.d=WARNING ") == {"a.b": "DEBUG", "c.d": "WARNING"}


def test_parse_SKIPS_a_malformed_pair_rather_than_raising():
    """One bad entry in a deployment env must not cost every other entry, and
    must not take down startup."""
    assert parse_log_levels("good.one=DEBUG,garbage,other=INFO") == {
        "good.one": "DEBUG", "other": "INFO",
    }


def test_parse_SKIPS_an_unknown_level_name():
    assert parse_log_levels("a=DEBUG,b=LOUD") == {"a": "DEBUG"}


def test_parse_accepts_the_root_logger_by_empty_key():
    assert parse_log_levels("=DEBUG") == {"": "DEBUG"}


# --- precedence --------------------------------------------------------------

def test_LOG_LEVEL_applies_to_the_tracking_loggers(monkeypatch):
    logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.delenv("LOG_LEVELS", raising=False)
    apply_runtime_levels()
    for name in LEVEL_TRACKING_LOGGERS:
        assert logging.getLogger(name).level == logging.DEBUG, name


def test_LOG_LEVEL_does_NOT_touch_a_declared_level(monkeypatch):
    """The Spring point. comfy.jobs is a declared default; a root-level change
    must not silently stomp it."""
    logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.delenv("LOG_LEVELS", raising=False)
    apply_runtime_levels()
    assert logging.getLogger("comfy.jobs").level == logging.DEBUG


def test_LOG_LEVELS_beats_a_declared_level(monkeypatch):
    logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
    monkeypatch.setenv("LOG_LEVELS", "comfy.jobs=WARNING")
    apply_runtime_levels()
    assert logging.getLogger("comfy.jobs").level == logging.WARNING


def test_LOG_LEVELS_beats_LOG_LEVEL(monkeypatch):
    logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("LOG_LEVELS", "uvicorn=DEBUG")
    apply_runtime_levels()
    assert logging.getLogger("uvicorn").level == logging.DEBUG
    assert logging.getLogger("uvicorn.error").level == logging.WARNING


def test_LOG_LEVELS_reaches_a_logger_that_does_not_exist_yet(monkeypatch):
    """getLogger CREATES it, so the level is waiting when the module imports.
    This is what a per-var name-mangling scheme could not do."""
    monkeypatch.setenv("LOG_LEVELS", "st.probe.child=DEBUG")
    apply_runtime_levels()
    assert logging.getLogger("st.probe.child").level == logging.DEBUG


# --- the dev-path bug, reproduced ---------------------------------------------

def test_a_STALE_BAKED_config_is_corrected_at_runtime(monkeypatch):
    """THE bug. The dev image materialises LOGGING_CONFIG to JSON at BUILD time,
    so the file carries whatever LOG_LEVEL was set during the build. Simulate a
    build at INFO, then start the container at DEBUG."""
    baked = json.loads(json.dumps(LOGGING_CONFIG))
    for name in LEVEL_TRACKING_LOGGERS:
        baked["loggers"][name]["level"] = "INFO"        # frozen at build
    logging.config.dictConfig(baked)
    assert logging.getLogger("").level == logging.INFO   # the bug, before

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")             # container start
    monkeypatch.delenv("LOG_LEVELS", raising=False)
    apply_runtime_levels()

    assert logging.getLogger("").level == logging.DEBUG
    assert logging.getLogger("uvicorn.error").level == logging.DEBUG
    assert logging.getLogger("comfy.jobs").level == logging.DEBUG  # declared, untouched


def test_an_INVALID_LOG_LEVEL_falls_back_to_INFO_not_to_the_BAKED_value(monkeypatch):
    """Caught in review. The precedence chain ends at INFO and that floor has to be
    real.

    For a TRACKING logger the baked level is a snapshot of the build environment,
    not a declaration — so 'on invalid input, leave it alone' preserves exactly the
    frozen state this issue exists to fix, in a costume. Baked at WARNING here
    precisely so that 'left alone' and 'fell back to INFO' are distinguishable;
    with a baked INFO the test would pass either way.
    """
    baked = json.loads(json.dumps(LOGGING_CONFIG))
    for name in LEVEL_TRACKING_LOGGERS:
        baked["loggers"][name]["level"] = "WARNING"     # frozen at build
    logging.config.dictConfig(baked)
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")
    monkeypatch.delenv("LOG_LEVELS", raising=False)

    apply_runtime_levels()

    for name in LEVEL_TRACKING_LOGGERS:
        assert logging.getLogger(name).level == logging.INFO, name
    # A DECLARED level is an intent and is untouched either way.
    assert logging.getLogger("comfy.jobs").level == logging.DEBUG


def test_an_ABSENT_LOG_LEVEL_also_resolves_to_INFO(monkeypatch):
    """Same floor, the other way in. logging_config.py's own default is
    os.getenv('LOG_LEVEL', 'INFO'); the override must not disagree with it."""
    baked = json.loads(json.dumps(LOGGING_CONFIG))
    for name in LEVEL_TRACKING_LOGGERS:
        baked["loggers"][name]["level"] = "WARNING"
    logging.config.dictConfig(baked)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_LEVELS", raising=False)

    apply_runtime_levels()

    assert logging.getLogger("").level == logging.INFO


def test_an_invalid_LOG_LEVELS_ENTRY_is_skipped_and_does_NOT_fall_back(monkeypatch):
    """The deliberate asymmetry with LOG_LEVEL. A per-logger override that cannot
    be parsed simply does not apply — it must not knock the logger to INFO."""
    logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("LOG_LEVELS", "comfy.jobs=LOUD")

    apply_runtime_levels()

    assert logging.getLogger("comfy.jobs").level == logging.DEBUG   # declared, kept


def test_applying_twice_is_idempotent(monkeypatch):
    """The prod path already has correct levels from run.py's dictConfig, so this
    runs as a no-op there. It must stay one."""
    logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    first = apply_runtime_levels()
    second = apply_runtime_levels()
    assert first == second


# --- it must never raise ------------------------------------------------------

def test_a_broken_environment_does_not_raise(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")
    monkeypatch.setenv("LOG_LEVELS", "=,==,a=,=b,x=NOPE")
    apply_runtime_levels()      # must simply return


def test_an_internal_failure_is_swallowed_and_LOGGED(monkeypatch):
    """It runs in lifespan (a raise fails server startup) and in the spawned
    child's bootstrap (a raise hangs a parent waiting on _READY). But a silent
    failure in a config path is its own bug — assert it reports."""
    monkeypatch.setattr(log_levels, "parse_log_levels",
                        lambda raw: (_ for _ in ()).throw(RuntimeError("boom")))
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    log_levels.logger.addHandler(handler)
    try:
        apply_runtime_levels()
    finally:
        log_levels.logger.removeHandler(handler)
    assert any(r.exc_info for r in records), "the failure was swallowed silently"
```

- [x] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_log_levels.py -q`
Expected: `ModuleNotFoundError: No module named 'server.log_levels'`.

- [x] **Step 3: Write the implementation**

```python
# server/log_levels.py
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
            applied[name] = root_level

        # Per-logger overrides win over everything, including a declared level.
        # Unparseable entries are SKIPPED rather than defaulted: an override that
        # cannot be read simply does not apply, leaving the declared or tracking
        # level intact. That asymmetry with LOG_LEVEL above is deliberate.
        applied.update(parse_log_levels(os.getenv("LOG_LEVELS")))

        for name, level in applied.items():
            # getLogger CREATES an unknown logger, which is deliberate: a level
            # set for a module that has not imported yet is waiting when it does.
            logging.getLogger(name).setLevel(level)
    except Exception:
        logger.warning("[log_levels] failed to apply runtime levels", exc_info=True)
    return applied
```

- [x] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_log_levels.py -q`
Expected: all pass (5 from Task 1 + ~14 here).

- [x] **Step 5: Commit**

```bash
git add server/log_levels.py tests/test_log_levels.py
git commit -m "feat(logging): runtime LOG_LEVEL and per-logger LOG_LEVELS overrides (STABL-ataigkdk) — next: Task 3, wire into lifespan and the child"
```

---

## Task 3: Wire it into both entry paths and the child

**Files:**
- Modify: `server/lcm_sr_server.py` (`lifespan`, `:390`)
- Modify: `backends/worker_handle_subprocess.py` (`_configure_child_logging`)
- Test: `tests/test_log_levels.py` (extend), `tests/test_worker_child_logging.py` (extend)

**Interfaces:** consumes `apply_runtime_levels()`.

`lifespan` is the seam because it fires on **both** entry paths — the dev CMD
imports the app, and prod's `uvicorn.run(app)` triggers it too — and it runs
*after* `dictConfig`, so re-setting levels there is exactly "baked default,
runtime wins".

**Known limit, state it rather than let it be found as a bug:** records emitted
*before* lifespan — import-time logging and uvicorn's own startup lines — keep the
baked level. The window is small and mostly uvicorn's own output.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_log_levels.py
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _calls_in(path: pathlib.Path, func_name: str):
    """Every function called anywhere inside `func_name`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == func_name), None)
    assert fn is not None, f"{func_name} not found in {path.name}"
    return {n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def test_lifespan_applies_runtime_levels():
    """Wiring, not behaviour: the unit tests above pass whether or not anything
    calls apply_runtime_levels()."""
    assert "apply_runtime_levels" in _calls_in(
        ROOT / "server" / "lcm_sr_server.py", "lifespan"
    )


def test_the_child_bootstrap_applies_runtime_levels():
    """Under WORKER_ISOLATION=subprocess the child is where generation happens.
    Its LOG_LEVEL is already live (fresh import after spawn), but the per-logger
    LOG_LEVELS half still needs applying."""
    assert "apply_runtime_levels" in _calls_in(
        ROOT / "backends" / "worker_handle_subprocess.py", "_configure_child_logging"
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_log_levels.py -q -k wiring or applies`
Expected: both new tests FAIL.

- [ ] **Step 3: Wire the lifespan**

In `server/lcm_sr_server.py`, `lifespan` currently opens:

```python
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI server lifespan...")
    logger.info(f"BACKEND={BACKEND}, NUM_WORKERS={NUM_WORKERS}, LOG_LEVEL={os.getenv('LOG_LEVEL', 'INFO')}")
```

Note that second line already *reports* the runtime `LOG_LEVEL` while not applying
it. Apply it **first**, so the report is true and so the two lines above it are the
only ones that miss the override:

```python
async def lifespan(app: FastAPI):
    # FIRST, before anything else logs. dictConfig has already installed the baked
    # config by now (run.py in prod, uvicorn --log-config in dev), so this is the
    # point where "baked default, runtime wins" is realisable — on BOTH entry
    # paths, since the dev CMD imports the app and prod's uvicorn.run(app) both
    # fire lifespan (STABL-ataigkdk).
    from server.log_levels import apply_runtime_levels
    _applied = apply_runtime_levels()

    logger.info("Starting FastAPI server lifespan...")
    logger.info(
        "BACKEND=%s, NUM_WORKERS=%s, LOG_LEVEL=%s, log level overrides applied: %s",
        BACKEND, NUM_WORKERS, os.getenv("LOG_LEVEL", "INFO"), _applied or "none",
    )
```

Imported inside the function, not at module top: `lcm_sr_server` is imported by
tests that do not want a logging side effect at import, and the cost is one lookup
per process.

- [ ] **Step 4: Wire the child bootstrap**

In `backends/worker_handle_subprocess.py`, `_configure_child_logging`, after the
existing `refresh_process_fields()`:

```python
        from server.log_levels import apply_runtime_levels

        logging.config.dictConfig(LOGGING_CONFIG)
        refresh_process_fields()        # this process's OWN pid and hostname
        # LOG_LEVEL is ALREADY live here — spawn re-imports logging_config in a
        # fresh process, so its module-level os.getenv reads the real
        # environment. This call is for the per-logger LOG_LEVELS half, which no
        # amount of re-importing provides.
        apply_runtime_levels()
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_log_levels.py tests/test_worker_child_logging.py -q`
Expected: all pass.

- [ ] **Step 6: Prove it end-to-end in a real spawned child**

Extend `tests/test_worker_child_logging.py`'s `_child` to set
`LOG_LEVELS=st.childprobe=DEBUG` before calling `_configure_child_logging()`, and
report `logging.getLogger("st.childprobe").level` back on the queue. Assert it is
`logging.DEBUG`. The existing test already proves the formatter crosses the spawn
boundary; this proves the levels do.

- [ ] **Step 7: Run the server and worker suites**

Run: `python -m pytest tests/test_ws_routes.py tests/test_subprocess_worker_handle.py tests/test_worker_handle.py tests/test_logging_wiring.py -q`
Expected: no new failures.

- [ ] **Step 8: Commit**

```bash
git add server/lcm_sr_server.py backends/worker_handle_subprocess.py tests/
git commit -m "feat(logging): apply runtime levels in lifespan and the spawned child (STABL-ataigkdk) — next: Task 4, document and close out"
```

---

## Task 4: Document, verify, close out

**Files:**
- Modify: `docs/observability-contract.md`

- [ ] **Step 1: Document the two variables**

Add to the **Structured logs** section of `docs/observability-contract.md`:

````markdown
### Log levels

The baked config in `server/logging_config.py` ships **declared defaults**; the
environment overrides them at startup. Precedence, highest first:

| source | scope |
|---|---|
| `LOG_LEVELS` | named loggers |
| declared level in `logging_config.py` | that logger (currently only `comfy.jobs`, at `DEBUG`) |
| `LOG_LEVEL` | root and the `uvicorn*` loggers |
| `INFO` | fallback |

```bash
LOG_LEVEL=WARNING
LOG_LEVELS="comfy.jobs=WARNING,backends.governor=DEBUG"
```

`LOG_LEVELS` takes the **logger name verbatim** — no name mangling, so
`server.ws_routes` is written exactly that way. A name that no module has imported
yet is valid: the level is waiting when it does.

Bad input never fails startup, but the two variables recover differently, and the
difference is deliberate:

| bad input | result |
|---|---|
| `LOG_LEVEL` absent or unrecognised | tracking loggers resolve to **`INFO`**, with a warning |
| a `LOG_LEVELS` entry malformed or unrecognised | **that entry is skipped**; the logger keeps its declared or tracking level |

An override that cannot be read simply does not apply. `LOG_LEVEL` is different
because the loggers it governs are *defined* as "whatever `LOG_LEVEL` says" — they
have to resolve to something, and falling back to the value baked at image-build
time would just reinstate a stale build environment.

Applied at FastAPI startup and in the spawned worker child. Records emitted before
that point — import-time logging and uvicorn's own startup lines — carry the baked
level.
````

- [ ] **Step 2: Live verification, all four precedence rungs**

```bash
LOG_LEVEL=WARNING LOG_LEVELS="comfy.jobs=ERROR,st.demo=DEBUG" python -c "
import json, logging, logging.config
from server.logging_config import LOGGING_CONFIG
from server.log_levels import apply_runtime_levels
logging.config.dictConfig(json.loads(json.dumps(LOGGING_CONFIG)))
print('applied:', apply_runtime_levels())
for n in ('', 'uvicorn.error', 'comfy.jobs', 'st.demo'):
    print(f'{n or \"(root)\":16}', logging.getLevelName(logging.getLogger(n).level))
"
```

Expected: root and `uvicorn.error` at `WARNING` (from `LOG_LEVEL`), `comfy.jobs` at
`ERROR` (`LOG_LEVELS` beating its declared `DEBUG`), `st.demo` at `DEBUG` (a logger
that did not exist).

Then run it again with **neither** variable set and confirm the baked defaults —
root `INFO`, `comfy.jobs` `DEBUG` — are what you get.

Finally, the floor, with the baked value set to something other than `INFO` so the
two outcomes are distinguishable:

```bash
LOG_LEVEL=NOT_A_LEVEL python -c "
import json, logging, logging.config
from server.logging_config import LOGGING_CONFIG, LEVEL_TRACKING_LOGGERS
from server.log_levels import apply_runtime_levels
baked = json.loads(json.dumps(LOGGING_CONFIG))
for n in LEVEL_TRACKING_LOGGERS: baked['loggers'][n]['level'] = 'WARNING'
logging.config.dictConfig(baked)
apply_runtime_levels()
print('root      :', logging.getLevelName(logging.getLogger('').level), '(expect INFO, NOT WARNING)')
print('comfy.jobs:', logging.getLevelName(logging.getLogger('comfy.jobs').level), '(expect DEBUG)')
"
```

- [ ] **Step 3: Check drift**

```bash
drift refs server/lcm_sr_server.py
drift refs server/logging_config.py
```

`logging_config.py` is not bound. `lcm_sr_server.py` is, across several docs.
**Baseline against `main` before relinking** — `drift check` is not clean there
(`STABL-qjbqzwpe`), so an absolute count is unreadable:

```bash
git worktree add /tmp/baseline main
(cd /tmp/baseline && drift check) | awk '/^docs\//{d=$1} /STALE/{print d" -> "$2}' | sort > /tmp/main.txt
drift check | awk '/^docs\//{d=$1} /STALE/{print d" -> "$2}' | sort > /tmp/branch.txt
comm -13 /tmp/main.txt /tmp/branch.txt
```

Review the prose behind each anchor that appears, then relink **per anchor** —
`drift link <doc> <anchor>`, never `drift link <doc>`, which refreshes every anchor
in the file including ones you did not review.

- [ ] **Step 4: Full suite**

```bash
python -m pytest -q 2>&1 | tail -3
```

Baseline on `main` is **1466 passed**, 9 skipped, 1 xfailed. Run it on the tree you
are about to summarise.

- [ ] **Step 5: Commit and close out**

```bash
git add docs/observability-contract.md docs/superpowers/plans/2026-08-07-runtime-log-levels.md
git commit -m "docs(logging): document LOG_LEVEL and LOG_LEVELS precedence (STABL-ataigkdk) — next: PR"
fp issue assign STABL-ataigkdk --rev <sha1>,<sha2>,...
fp comment STABL-ataigkdk "STOP: ... NEXT: ... DECIDED: ..."
```

---

## Deferred, NOT done here

| Item | Why |
|---|---|
| Spring-style `LOG_LEVEL_<NAME>` per-var overrides | The name mangling cannot round-trip logger names containing underscores, and the failure is silent. Rejected with reasoning in `STABL-ataigkdk`. |
| Regenerating `/app/logging_config.json` at build or entrypoint | Rejected on the record as a kludge. |
| Records emitted before `lifespan` | Import-time and uvicorn startup lines keep the baked level. Fixing it means configuring logging earlier than the app, which is the entrypoint approach already rejected. |
| `LOG_FORMAT` set in no deployment | `STABL-xqqqqvse` — a deployment decision with a `../continuous` precondition. |
| 18 inherited stale drift anchors on `main` | `STABL-qjbqzwpe`. |
