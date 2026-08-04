"""Repo-local Prometheus facade (STABL-asawxgvp).

Plan: docs/superpowers/plans/2026-08-03-prometheus-substrate.md Tasks 1-2.
Spec: docs/superpowers/specs/2026-08-03-server-observability-seams-design.md
"""
import importlib.util

import pytest

from server import metrics as m


@pytest.fixture(autouse=True)
def _fresh():
    m.reset_metrics()
    yield
    m.reset_metrics()


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    assert m.get_metrics().enabled is False


def test_disabled_facade_accepts_every_call_unchanged(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    met = m.get_metrics()
    # Instrumentation code is unconditional: these must all be no-ops, not errors.
    met.job_terminal_total.labels(mode="SDXL", outcome="ok").inc()
    met.job_execution_seconds.labels(mode="SDXL").observe(1.5)
    met.device_free_bytes.labels(device_uuid="GPU-x").set(123)


def test_enabled_registers_and_renders(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    assert met.enabled is True
    met.job_terminal_total.labels(mode="SDXL", outcome="ok").inc()
    body, content_type = met.render()
    assert b"st_governor_job_terminal_total" in body
    assert "text/plain" in content_type


def test_disabled_render_is_empty(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    body, content_type = m.get_metrics().render()
    assert body == b""
    assert "text/plain" in content_type


def test_missing_prometheus_client_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    monkeypatch.setattr(m, "_import_prometheus", lambda: (_ for _ in ()).throw(ImportError()))
    met = m.get_metrics()
    assert met.enabled is False          # degraded, not crashed
    met.job_terminal_total.labels(mode="x", outcome="ok").inc()


def test_warns_when_multiple_web_workers(monkeypatch, caplog):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with caplog.at_level("WARNING"):
        m.get_metrics()
    assert any("WEB_CONCURRENCY" in r.message for r in caplog.records)


def test_module_does_not_import_prometheus_at_module_scope():
    src = importlib.util.find_spec("server.metrics").origin
    with open(src) as fh:
        head = [ln for ln in fh if ln.startswith(("import ", "from "))]
    assert not any("prometheus_client" in ln for ln in head), (
        "prometheus_client must be imported lazily so a missing dep degrades"
    )
