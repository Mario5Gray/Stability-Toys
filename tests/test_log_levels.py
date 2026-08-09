"""STABL-ataigkdk: runtime per-logger log levels."""
import json
import logging
import logging.config

import pytest

from server import log_levels, logging_config
from server.log_levels import apply_runtime_levels, parse_log_levels
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


# ---------------------------------------------------------------------------
# Task 2 — apply_runtime_levels()
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_logging():
    """dictConfig and setLevel are GLOBAL. Snapshot the levels this file touches
    and put them back, or every later test file inherits them."""
    names = ["", "comfy.jobs", "uvicorn", "uvicorn.error", "uvicorn.access",
             "st.probe", "st.probe.child", "st.demo"]
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
    monkeypatch.delenv("LOG_LEVEL", raising=False)
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
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("LOG_LEVELS", "st.probe.child=DEBUG")
    apply_runtime_levels()
    assert logging.getLogger("st.probe.child").level == logging.DEBUG


# --- the dev-path bug, reproduced ---------------------------------------------

def _bake(level):
    """A config as the dev image would materialise it, with the tracking loggers
    frozen at whatever LOG_LEVEL the BUILD happened to see."""
    baked = json.loads(json.dumps(LOGGING_CONFIG))
    for name in LEVEL_TRACKING_LOGGERS:
        baked["loggers"][name]["level"] = level
    return baked


def test_a_STALE_BAKED_config_is_corrected_at_runtime(monkeypatch):
    """THE bug. The dev image materialises LOGGING_CONFIG to JSON at BUILD time,
    so the file carries whatever LOG_LEVEL was set during the build. Simulate a
    build at INFO, then start the container at DEBUG."""
    logging.config.dictConfig(_bake("INFO"))
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
    logging.config.dictConfig(_bake("WARNING"))
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
    logging.config.dictConfig(_bake("WARNING"))
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


def test_the_return_value_reports_only_what_was_ACTUALLY_set(monkeypatch):
    """It feeds a startup log line. A line that claims levels it failed to apply
    is at its most misleading exactly when someone is debugging logging."""
    monkeypatch.setattr(log_levels, "parse_log_levels",
                        lambda raw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert apply_runtime_levels() == {}


# ---------------------------------------------------------------------------
# Task 3 — the wiring
#
# ast-based on purpose. EVERY unit test above passes whether or not anything
# actually calls apply_runtime_levels(), so behaviour tests cannot tell a wired
# system from an unwired one. Same trap as the child bootstrap in STABL-bpsfmoke.
# ---------------------------------------------------------------------------

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _calls_in(path: pathlib.Path, func_name: str):
    """Every function called by simple name anywhere inside `func_name`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == func_name), None)
    assert fn is not None, f"{func_name} not found in {path.name}"
    return {n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def test_lifespan_applies_runtime_levels():
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
