"""Tracing facade gate tests (STABL-qnlaclof step 2)."""
import importlib.util

import pytest

from server import tracing as t


@pytest.fixture(autouse=True)
def _fresh():
    t.reset_tracing()
    yield
    t.reset_tracing()


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRACING_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert t.get_tracing().enabled is False


def test_disabled_facade_accepts_every_call_unchanged(monkeypatch):
    monkeypatch.delenv("TRACING_ENABLED", raising=False)
    tracer = t.get_tracer("tests.disabled")
    with tracer.start_as_current_span("demo") as span:
        span.set_attribute("mode", "SDXL")
        span.add_event("worker.recovered", {"oom": True})
        span.record_exception(RuntimeError("boom"))
        span.set_status(object())
        assert span.is_recording() is False


def test_proxy_endpoint_alone_does_not_enable_sdk_export(monkeypatch, caplog):
    monkeypatch.setenv("TRACING_ENABLED", "1")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_PROXY_ENDPOINT", "http://otel-collector:4318/v1/traces")

    with caplog.at_level("WARNING"):
        tracing = t.get_tracing()

    assert tracing.enabled is False
    assert any("OTEL_EXPORTER_OTLP_ENDPOINT" in r.message for r in caplog.records)


def test_missing_opentelemetry_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setenv("TRACING_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.setattr(
        t,
        "_import_opentelemetry",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )

    tracing = t.get_tracing()

    assert tracing.enabled is False
    with t.get_tracer("tests.missing").start_as_current_span("demo") as span:
        span.add_event("still.noop")


def test_module_does_not_import_opentelemetry_at_module_scope():
    spec = importlib.util.find_spec("server.tracing")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as fh:
        head = [ln for ln in fh if ln.startswith(("import ", "from "))]
    assert not any("opentelemetry" in ln for ln in head), (
        "opentelemetry must be imported lazily so a missing dep degrades"
    )


def test_a_QUOTED_flag_still_enables(monkeypatch):
    """Env files quote values, and `TRACING_ENABLED="1"` read literally is not
    "1" — it is '"1"'. STABL-voqsoicx built utils.env for exactly this after a
    quoted LOG_LEVEL crashed dictConfig, and STABL-xqqqqvse is the record of a
    pillar shipping dark because nothing set its flag. A gate that reads the
    environment by hand re-opens both.
    """
    monkeypatch.setenv("TRACING_ENABLED", '"1"')
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.setattr(
        t, "_import_opentelemetry",
        lambda: (_ for _ in ()).throw(ImportError("no sdk here")),
    )

    # Enablement is read BEFORE the SDK import, so the degrade path still proves
    # the flag was honoured: an unread flag never reaches the import at all.
    assert t._enabled_from_env() is True


def test_an_explicitly_false_flag_stays_off(monkeypatch):
    """The other direction, and the one that matters operationally: `off`,
    `FALSE` and a trailing space must not read as enabled."""
    for value in ("0", "false", "FALSE", "off", " no ", ""):
        monkeypatch.setenv("TRACING_ENABLED", value)
        assert t._enabled_from_env() is False, f"{value!r} enabled tracing"
