"""Governor tracing tests (STABL-qnlaclof steps 2 and 4)."""
import pickle
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from backends.governor import Governor
from backends.governor import GenerationJob, Governor
from tests.test_governor import (
    StubHandle,
    _drain_queue,
    _freeze_dispatch,
    _make_mock_registry,
    _make_multi_mode_config,
    _resolve_by_path,
)
from server.lcm_sr_server import GenerateRequest


def _req(prompt="hello"):
    return GenerateRequest(prompt=prompt, num_inference_steps=4, size="512x512")


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


# ---------------------------------------------------------------------------
# Step 4: lifecycle spans
# ---------------------------------------------------------------------------

class _RecordedSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.events = []
        self.exit_exc_type = None
        self.ended = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_exc_type = exc_type
        self.ended = True
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def update_name(self, name):
        self.name = name

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))

    def record_exception(self, exc):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def is_recording(self):
        return True


class _RecordingTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name, *args, **kwargs):
        span = _RecordedSpan(name)
        self.spans.append(span)
        return span


@pytest.fixture
def spans(monkeypatch):
    tracer = _RecordingTracer()
    monkeypatch.setattr("backends.governor.tracing.get_tracer", lambda name: tracer)
    return tracer


@pytest.fixture
def running_governor(spans):
    """A Governor that can actually run a job to completion.

    StubHandle exposes worker=None, so the dispatch loop takes the SUBPROCESS
    branch and _SubprocessFutureBridge UNPICKLES the blob. The result therefore
    has to be a pickle: a raw str raises "a bytes-like object is required" and
    raw bytes raise "unpickling stack underflow", both from inside the dispatch
    loop, and both present as a tracing failure while being nothing of the kind.
    """
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = Governor(
            handle=StubHandle(result=pickle.dumps(b"stub_result")),
            mode_config=_mode_config("mode-a", default="mode-a"),
            registry=_make_mock_registry(),
        )
        spans.spans.clear()          # discard the construction-time mode load
        try:
            yield gov
        finally:
            gov.shutdown()


def _named(tracer, name):
    return [s for s in tracer.spans if s.name == name]


def test_a_mode_load_is_traced_with_its_mode(spans, governor_with_stub_handle):
    spans.spans.clear()      # the Governor loads its default mode at construction
    governor_with_stub_handle._load_mode("mode-a")

    loads = _named(spans, "governor.mode_load")
    assert len(loads) == 1
    assert loads[0].attributes["mode"] == "mode-a"


def test_an_unload_carries_the_REASON(spans, governor_with_stub_handle):
    """`switch` is routine churn — _load_mode unloads the outgoing worker before
    every load. `idle_evict` is the one worth looking at. Without the reason the
    two are indistinguishable in a trace."""
    governor_with_stub_handle._load_mode("mode-a")
    spans.spans.clear()

    governor_with_stub_handle._unload_current_worker(reason="idle_evict")

    unloads = _named(spans, "governor.unload")
    assert len(unloads) == 1
    assert unloads[0].attributes["reason"] == "idle_evict"


def test_submit_is_traced(spans, governor_with_stub_handle):
    gov = governor_with_stub_handle
    _freeze_dispatch(gov)
    try:
        gov.submit_job(GenerationJob(req=_req(), resolution_epoch=gov._resolution_epoch))
    finally:
        _drain_queue(gov)

    assert len(_named(spans, "governor.submit")) == 1


def test_a_mode_switch_is_traced_with_its_target(spans, governor_with_stub_handle):
    gov = governor_with_stub_handle
    _freeze_dispatch(gov)
    try:
        gov.switch_mode("mode-a", force=True)
    finally:
        _drain_queue(gov)

    switches = _named(spans, "governor.mode_switch")
    assert len(switches) == 1
    assert switches[0].attributes["mode"] == "mode-a"


def test_a_cancel_is_traced(spans, governor_with_stub_handle):
    gov = governor_with_stub_handle
    _freeze_dispatch(gov)
    job = GenerationJob(req=_req(), resolution_epoch=gov._resolution_epoch)
    try:
        gov.submit_job(job)
        gov.cancel_job(job.job_id)
    finally:
        _drain_queue(gov)

    assert len(_named(spans, "governor.cancel")) == 1


def test_the_dispatch_span_records_the_terminal_OUTCOME(spans, running_governor):
    """The outcome is derived at ONE choke point (_observe_job_terminal) so a
    trace and the st_governor_job_terminal_total counter cannot disagree. Two
    derivations of "how did this job end" is how they drift."""
    gov = running_governor
    fut = gov.submit_job(GenerationJob(req=_req(), resolution_epoch=gov._resolution_epoch))
    fut.result(timeout=10)

    dispatches = _named(spans, "governor.dispatch")
    assert len(dispatches) == 1
    assert dispatches[0].attributes["job.outcome"] == "ok"
    assert dispatches[0].ended


def test_the_dispatch_span_is_ENDED_even_when_the_job_fails(spans, running_governor):
    """An unended span is never exported, so the failure it describes disappears
    — precisely the trace you would go looking for."""
    gov = running_governor

    # The failure has to come from the HANDLE, not from job.execute: StubHandle
    # exposes worker=None, so the dispatch loop takes the subprocess branch and
    # never calls job.execute at all. Overriding it there is dead code that
    # passes for a working test.
    def _fail(job):
        from backends.backplane.inproc import InProcBackplane
        from backends.backplane.frames import BackplaneError
        sink, pub = InProcBackplane(job.job_id).open()
        sink.error(BackplaneError.from_exc(RuntimeError("worker exploded")))
        return pub

    gov._handle.submit = _fail

    fut = gov.submit_job(GenerationJob(req=_req(), resolution_epoch=gov._resolution_epoch))
    with pytest.raises(Exception):
        fut.result(timeout=10)

    dispatches = _named(spans, "governor.dispatch")
    assert len(dispatches) == 1
    assert dispatches[0].ended
    assert dispatches[0].attributes["job.outcome"] == "error"


def test_the_dispatch_span_opens_once_PER_JOB(spans, running_governor):
    gov = running_governor
    for _ in range(3):
        gov.submit_job(
            GenerationJob(req=_req(), resolution_epoch=gov._resolution_epoch)
        ).result(timeout=10)

    assert len(_named(spans, "governor.dispatch")) == 3


def test_tracing_failures_never_reach_the_dispatch_loop(monkeypatch):
    """The whole reason _span exists. A broken tracer must cost a trace, not the
    queue: STABL-hdzggeir is the record of an error handler that could itself
    raise permanently deadening the dispatch thread."""
    class _ExplodesTracer:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError("tracer down")

    monkeypatch.setattr("backends.governor.tracing.get_tracer", lambda name: _ExplodesTracer())

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = Governor(
            handle=StubHandle(result=pickle.dumps(b"stub_result")),
            mode_config=_mode_config("mode-a", default="mode-a"),
            registry=_make_mock_registry(),
        )
        try:
            fut = gov.submit_job(
                GenerationJob(req=_req(), resolution_epoch=gov._resolution_epoch))
            assert fut.result(timeout=10) == b"stub_result"
        finally:
            gov.shutdown()
