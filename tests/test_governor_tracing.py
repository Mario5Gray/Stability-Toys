"""Governor tracing guard tests (STABL-qnlaclof step 2)."""
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from backends.governor import Governor
from tests.test_governor import StubHandle, _make_mock_registry, _make_multi_mode_config, _resolve_by_path


def _mode_config(*names, default=None):
    config = _make_multi_mode_config(*names, default=default)
    config.list_modes.return_value = list(names)
    return config


def _make_governor(*names, default=None):
    return Governor(
        handle=StubHandle(),
        mode_config=_mode_config(*names, default=default),
        registry=_make_mock_registry(),
    )


@pytest.fixture
def governor_with_stub_handle():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _make_governor("mode-a", default="mode-a")
        try:
            yield gov
        finally:
            gov.shutdown()


def test_span_helper_swallows_tracing_failures_and_logs_traceback(governor_with_stub_handle, monkeypatch, caplog):
    class _ExplodesTracer:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr("backends.governor.tracing.get_tracer", lambda name: _ExplodesTracer())

    with caplog.at_level("DEBUG"):
        with governor_with_stub_handle._span("governor.dispatch") as span:
            span.set_attribute("job.id", "abc123")

    record = next(r for r in caplog.records if "[Governor] span governor.dispatch failed" in r.message)
    assert record.exc_info is not None


def test_span_helper_yields_a_noop_span_when_tracing_backend_breaks(governor_with_stub_handle, monkeypatch):
    @contextmanager
    def _explode(*args, **kwargs):
        raise RuntimeError("boom")
        yield

    class _ExplodesTracer:
        def start_as_current_span(self, *args, **kwargs):
            return _explode()

    monkeypatch.setattr("backends.governor.tracing.get_tracer", lambda name: _ExplodesTracer())

    with governor_with_stub_handle._span("governor.dispatch") as span:
        span.add_event("job.terminal", {"outcome": "ok"})
