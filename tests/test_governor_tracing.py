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


# --- the guard must protect span CREATION, never the caller's body ----------
#
# STABL-hdzggeir is the precedent the _span docstring itself cites: an error
# handler that can raise wedges the dispatch loop. A @contextmanager whose
# except clause yields a SECOND time does exactly that — contextlib raises
# "generator didn't stop after throw()" and the caller's real exception is gone.


def test_span_does_not_swallow_or_replace_the_callers_exception(governor_with_stub_handle):
    """The body's exception must arrive at the caller unchanged.

    If it does not, every downstream classification breaks at once:
    classify_exception() maps only CancelledError to CANCELLED, and the
    subprocess recovery branch tests `terminal_error_code == OOM`. Rewriting the
    exception type silently disables the reap AND the facet-3 kill+respawn.
    """
    with pytest.raises(ValueError, match="the caller's own failure"):
        with governor_with_stub_handle._span("governor.dispatch"):
            raise ValueError("the caller's own failure")


def test_span_lets_a_CancelledError_through_intact(governor_with_stub_handle):
    """Named separately because this is the one that costs a behaviour, not just
    a message: classify_exception() maps only CancelledError (or a class named
    CancelledError) to CANCELLED, so a rewritten type arrives as GENERIC and the
    STABL-jredufxb reap reports the wrong terminal."""
    from concurrent.futures import CancelledError

    with pytest.raises(CancelledError):
        with governor_with_stub_handle._span("governor.dispatch"):
            raise CancelledError("reaped")


def test_span_still_ends_the_span_when_the_body_raises(governor_with_stub_handle, monkeypatch):
    """Propagating the exception must not come at the cost of leaking the span —
    an unended span is never exported, so the failure it describes disappears."""
    exits = []

    class _RecordingCM:
        def __enter__(self):
            return "span"

        def __exit__(self, exc_type, exc, tb):
            exits.append(exc_type)
            return False

    class _Tracer:
        def start_as_current_span(self, *args, **kwargs):
            return _RecordingCM()

    monkeypatch.setattr("backends.governor.tracing.get_tracer", lambda name: _Tracer())

    with pytest.raises(ValueError):
        with governor_with_stub_handle._span("governor.dispatch"):
            raise ValueError("boom")

    assert exits == [ValueError], "the span was not ended with the body's exception"
