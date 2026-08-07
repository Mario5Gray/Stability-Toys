"""STABL-bpsfmoke: the JSON formatter and the LOG_FORMAT switch."""
import json
import logging
import logging.config
import pathlib
import re
import sys

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


def test_process_static_fields_are_merged():
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


# --- doc <-> code contract ---------------------------------------------------
#
# Bidirectional, like the metric-name contract in test_metrics.py. The reverse
# direction is the one that earns its keep: it caught invented series names in
# STABL-xmsrxvto, and prose drifts the same way for field names.

CONTRACT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "observability-contract.md"


def _documented_log_fields():
    """Field names from the contract's log-field table rows: | `name` | ... |"""
    section = CONTRACT.read_text().split("## Structured logs")[1].split("\n## ")[0]
    return set(re.findall(r"^\|\s*`([a-z_]+)`\s*\|", section, re.M))


def _emitted_log_fields():
    log_context.set_static_field("mode", "m")
    log_context.set_static_field("device_uuid", "GPU-x")
    try:
        with log_context.bind_job_id("j"):
            rec = logging.LogRecord("st.test", logging.ERROR, "/x.py", 1, "m", (), None)
            payload = json.loads(StabilityFormatter(log_format="json").format(rec))
    finally:
        log_context.set_static_field("mode", None)
        log_context.set_static_field("device_uuid", None)
    return set(payload)


def test_every_emitted_log_field_is_documented():
    undocumented = _emitted_log_fields() - _documented_log_fields()
    assert not undocumented, f"emitted but not in the contract: {sorted(undocumented)}"


def test_every_documented_log_field_is_actually_emitted():
    optional = {"exception", "stack"}   # only present on a failing record
    missing = _documented_log_fields() - _emitted_log_fields() - optional
    assert not missing, f"documented but never emitted: {sorted(missing)}"


def test_the_contract_documents_the_OPTIONAL_fields_too():
    """`exception` and `stack` are excluded from the reverse check above, so
    without this they could be dropped from the doc unnoticed."""
    documented = _documented_log_fields()
    assert {"exception", "stack"} <= documented
