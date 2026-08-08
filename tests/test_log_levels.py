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
