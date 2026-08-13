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


# ---------------------------------------------------------------------------
# Step 6: the SDK is real now
# ---------------------------------------------------------------------------

def test_the_signal_path_is_APPENDED_to_a_base_endpoint(monkeypatch):
    """MEASURED against opentelemetry-exporter-otlp-proto-http 1.27.0:

        OTLPSpanExporter(endpoint='http://c:4318')            -> http://c:4318
        OTLPSpanExporter() with the env var set               -> http://c:4318/v1/traces

    An EXPLICIT `endpoint=` is used verbatim; only the env-var path appends.
    The spec's step-6 note said SDK exporters "append the signal path
    themselves", which is true of the env var and false of the argument — and
    our facade passes the argument. Every span would have POSTed to the
    collector root and 404'd, with the gate reporting tracing as enabled.
    """
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    assert t._traces_endpoint() == "http://otel-collector:4318/v1/traces"


def test_a_trailing_slash_does_not_double_up(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318/")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    assert t._traces_endpoint() == "http://otel-collector:4318/v1/traces"


def test_an_explicit_TRACES_endpoint_is_used_VERBATIM(monkeypatch):
    """The signal-specific variable is already a full path by definition, so
    appending to it would produce /v1/traces/v1/traces — the exact failure the
    spec warned about for OTEL_PROXY_ENDPOINT, one variable over."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://elsewhere:4318/v1/traces")

    assert t._traces_endpoint() == "http://elsewhere:4318/v1/traces"


def test_the_sampler_is_PARENT_BASED(monkeypatch):
    """REGRESSION GUARD, not a driver: this passed the moment it was written,
    because the SDK's default sampler already IS ParentBased(ALWAYS_ON). Kept
    because the requirement is load-bearing and easy to break later — a bare
    `sampler=ALWAYS_ON` added for local debugging makes the child sample
    independently of its parent, producing traces whose parent span was never
    recorded. That reads as data loss, is a configuration error, and shows up
    only under WORKER_ISOLATION=subprocess, the deployed path.
    """
    from opentelemetry.sdk.trace.sampling import ParentBased

    monkeypatch.setenv("TRACING_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    t.reset_tracing()
    try:
        tracing = t.get_tracing()
        assert tracing.enabled is True
        assert isinstance(tracing._provider.sampler, ParentBased)
    finally:
        t.reset_tracing()


def test_the_service_name_is_what_tempo_will_group_on(monkeypatch):
    """Also a guard rather than a driver. service.name is the primary index in
    Tempo's search, and the SDK's default is 'unknown_service' — which would put
    every span in the fleet in one bucket."""
    from opentelemetry.sdk.resources import SERVICE_NAME

    monkeypatch.setenv("TRACING_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    t.reset_tracing()
    try:
        resource = t.get_tracing()._provider.resource
        assert resource.attributes[SERVICE_NAME] == "stability-toys"
    finally:
        t.reset_tracing()


def test_the_SIGNAL_SPECIFIC_endpoint_alone_ENABLES_tracing(monkeypatch):
    """Found at review of PR #71. The gate read the BASE variable while
    _traces_endpoint() honoured the signal-specific one, so the standard OTel
    config — set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, leave the base unset —
    resolved to a correct endpoint and was then discarded by a gate that could
    not see it. Silent no-op, with a warning naming a variable the operator had
    deliberately not used.
    """
    monkeypatch.setenv("TRACING_ENABLED", "1")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://otel-collector:4318/v1/traces")
    t.reset_tracing()
    try:
        assert t.get_tracing().enabled is True
    finally:
        t.reset_tracing()


def test_no_endpoint_at_all_yields_an_EMPTY_string_not_a_bare_signal_path(monkeypatch):
    """`"".rstrip("/") + "/v1/traces"` is `/v1/traces` — truthy, and a plausible
    enough string to be handed to an exporter. The absent case has to be falsy or
    the gate cannot use this function to decide anything."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    assert t._traces_endpoint() == ""


def test_the_unset_warning_names_BOTH_variables(monkeypatch, caplog):
    """An operator who set the signal-specific variable and is told the base one
    is unset will go looking in the wrong place."""
    monkeypatch.setenv("TRACING_ENABLED", "1")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    t.reset_tracing()
    try:
        with caplog.at_level("WARNING"):
            assert t.get_tracing().enabled is False
        message = " ".join(r.message for r in caplog.records)
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in message
        assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" in message
    finally:
        t.reset_tracing()
