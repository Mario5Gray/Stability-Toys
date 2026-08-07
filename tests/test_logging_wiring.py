"""STABL-bpsfmoke: the two entry paths that must both get the config.

- prod: server/run.py -> dictConfig(LOGGING_CONFIG) + uvicorn(log_config=...)
- dev:  docker/runtime/live-test.Dockerfile materializes LOGGING_CONFIG to
        /app/logging_config.json at BUILD time and passes --log-config

The dev path never sees the Python dict. Every test here therefore configures from
the JSON ROUND TRIP, not from the object.
"""
import json
import logging
import logging.config

import pytest

from server.log_format import StabilityFormatter
from server.logging_config import LOGGING_CONFIG


@pytest.fixture(autouse=True)
def _restore_logging():
    """dictConfig is GLOBAL. Without this, every test file collected after this
    one inherits whatever handler set the last test here installed."""
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    yield
    logging.config.dictConfig(LOGGING_CONFIG)
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_the_config_is_json_serialisable():
    """docker/runtime/live-test.Dockerfile does exactly this at BUILD time. A
    non-serialisable value here breaks the dev image, not the suite — so the suite
    has to check it."""
    json.dumps(LOGGING_CONFIG)


def test_a_ROUND_TRIPPED_config_still_builds_the_formatter():
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
