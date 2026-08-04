"""Governor metrics instrumentation (STABL-asawxgvp).

Plan: docs/superpowers/plans/2026-08-03-prometheus-substrate.md Task 3.

The terminal counter is instrumented at ONE choke point — _finalize_job_record —
because every dispatch-loop terminal branch funnels through it and the job's
future is resolved by the time it is called.
"""
import time
from concurrent.futures import Future, CancelledError
from unittest.mock import Mock, patch

import pytest

from server import metrics as m
from backends.governor import Governor, GenerationJob, JobRecord, _terminal_outcome
from tests.test_governor import (
    StubHandle,
    _make_multi_mode_config,
    _make_mock_registry,
    _resolve_by_path,
    _freeze_dispatch,
)


@pytest.fixture(autouse=True)
def _metrics_on(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.reset_metrics()
    yield
    m.reset_metrics()


@pytest.fixture
def governor_with_stub_handle():
    """A Governor on a stub handle with one real loaded mode.

    resolve_model is patched for the fixture's whole life, not just construction:
    the Governor loads its default mode in __init__, and an unpatched load would
    leave _current_mode None so every observation would label mode="unknown".
    """
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = Governor(
            handle=StubHandle(),
            mode_config=_make_multi_mode_config("mode-a", default="mode-a"),
            registry=_make_mock_registry(),
        )
        try:
            yield gov
        finally:
            gov.shutdown()


def _sample(body: bytes, name: str) -> list[str]:
    return [ln for ln in body.decode().splitlines()
            if ln.startswith(name) and not ln.startswith("#")]


# --- JobRecord.enqueued_at ---

def test_register_job_stamps_enqueued_at():
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    rec = JobRecord(job_id=job.job_id, state="queued", job=job)
    assert hasattr(rec, "enqueued_at")
    assert rec.enqueued_at is None      # default; _register_job stamps it


# --- _terminal_outcome classification ---

def test_terminal_outcome_ok():
    fut = Future()
    fut.set_result("png")
    assert _terminal_outcome(fut) == "ok"


def test_terminal_outcome_cancelled_from_exception():
    fut = Future()
    fut.set_exception(CancelledError())
    assert _terminal_outcome(fut) == "cancelled"


def test_terminal_outcome_cancelled_future_does_not_raise():
    """fut.exception() RAISES CancelledError on a cancelled future, so the
    cancelled() check must come first."""
    fut = Future()
    fut.cancel()
    assert _terminal_outcome(fut) == "cancelled"


def test_terminal_outcome_oom():
    fut = Future()
    fut.set_exception(RuntimeError("CUDA out of memory. Tried to allocate 96.00 MiB"))
    assert _terminal_outcome(fut) == "oom"


def test_terminal_outcome_error():
    fut = Future()
    fut.set_exception(ValueError("nope"))
    assert _terminal_outcome(fut) == "error"


# --- _observe_job_terminal ---

def test_observe_emits_durations_and_outcome(governor_with_stub_handle):
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_result("png")
    now = time.monotonic()
    rec = JobRecord(job_id=job.job_id, state="running", job=job)
    rec.enqueued_at = now - 10.0
    rec.executing_since = now - 4.0

    gov._observe_job_terminal(rec)

    body, _ = m.get_metrics().render()
    terminals = _sample(body, "st_governor_job_terminal_total")
    assert terminals
    assert any('outcome="ok"' in ln for ln in terminals)
    assert _sample(body, "st_governor_job_queue_wait_seconds_count")
    assert _sample(body, "st_governor_job_execution_seconds_count")


def test_observe_skips_a_job_that_never_ran(governor_with_stub_handle):
    """The queue.Full rollback in submit_job finalizes a record whose future was
    never resolved. That is not a terminal and must not be counted."""
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    rec = JobRecord(job_id=job.job_id, state="queued", job=job)
    rec.enqueued_at = time.monotonic()

    gov._observe_job_terminal(rec)      # future not done

    body, _ = m.get_metrics().render()
    assert not _sample(body, "st_governor_job_terminal_total")


def test_observe_skips_when_enqueued_at_is_missing(governor_with_stub_handle):
    """A record built by an older path has no enqueue stamp; the arithmetic
    would raise on None. Skipping is correct and must not count a terminal."""
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_result("png")
    rec = JobRecord(job_id=job.job_id, state="running", job=job)
    assert rec.enqueued_at is None

    gov._observe_job_terminal(rec)

    body, _ = m.get_metrics().render()
    assert not _sample(body, "st_governor_job_terminal_total")


def test_observe_counts_terminal_even_without_executing_since(governor_with_stub_handle):
    """A job cancelled while still QUEUED never gets executing_since. It has no
    durations to report, but it is still a terminal and must be counted."""
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_exception(CancelledError())
    rec = JobRecord(job_id=job.job_id, state="cancelled", job=job)
    rec.enqueued_at = time.monotonic() - 2.0
    assert rec.executing_since is None

    gov._observe_job_terminal(rec)

    body, _ = m.get_metrics().render()
    assert any('outcome="cancelled"' in ln
               for ln in _sample(body, "st_governor_job_terminal_total"))
    assert not _sample(body, "st_governor_job_execution_seconds_count")


def test_observe_cannot_raise(governor_with_stub_handle, monkeypatch):
    """STABL-hdzggeir: nothing added here may be able to kill the dispatch loop."""
    gov = governor_with_stub_handle

    class _Explodes:
        def labels(self, **kw):
            raise RuntimeError("metrics backend died")

    met = m.get_metrics()
    monkeypatch.setattr(met, "job_terminal_total", _Explodes())
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_result("png")
    rec = JobRecord(job_id=job.job_id, state="running", job=job)
    rec.enqueued_at = time.monotonic() - 1
    rec.executing_since = time.monotonic() - 0.5

    gov._observe_job_terminal(rec)      # must return normally


def test_observe_cannot_raise_on_a_broken_record(governor_with_stub_handle):
    """The guard must survive a malformed record too, not just a broken backend."""
    gov = governor_with_stub_handle

    class _BrokenRecord:
        enqueued_at = 1.0
        executing_since = 2.0

        @property
        def job(self):
            raise RuntimeError("record is garbage")

    gov._observe_job_terminal(_BrokenRecord())   # must return normally


# --- _finalize_job_record wiring ---

def test_finalize_observes_once_and_pops(governor_with_stub_handle):
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_result("png")
    gov._register_job(job)
    rec = gov._get_job_record(job.job_id)
    assert rec is not None
    assert rec.enqueued_at is not None, "_register_job must stamp enqueued_at"
    rec.executing_since = time.monotonic()

    gov._finalize_job_record(job.job_id)

    assert gov._get_job_record(job.job_id) is None
    body, _ = m.get_metrics().render()
    assert len(_sample(body, "st_governor_job_terminal_total")) == 1


def test_finalize_on_an_unknown_job_id_is_a_noop(governor_with_stub_handle):
    """_finalize_job_record is called defensively in places; a missing record
    must not count a terminal or raise."""
    gov = governor_with_stub_handle

    gov._finalize_job_record("no-such-job")

    body, _ = m.get_metrics().render()
    assert not _sample(body, "st_governor_job_terminal_total")


def test_finalize_does_not_hold_the_job_lock_during_observation(governor_with_stub_handle):
    """The observation runs OUTSIDE _job_lock: the backplane's Subscriber<->lock
    invariant, and there is no reason to hold it across a metrics write."""
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_result("png")
    gov._register_job(job)
    rec = gov._get_job_record(job.job_id)
    assert rec is not None
    rec.executing_since = time.monotonic()

    seen = {}

    real_observe = gov._observe_job_terminal

    def _spy(record):
        # RLock is reentrant, so "can I acquire it" is not the question; ask
        # whether THIS thread is holding it via the internal recursion count.
        seen["depth"] = gov._job_lock._recursion_count()  # type: ignore[attr-defined]
        return real_observe(record)

    gov._observe_job_terminal = _spy  # type: ignore[method-assign]
    gov._finalize_job_record(job.job_id)

    assert seen["depth"] == 0, "observation ran while holding _job_lock"


def test_cancel_pending_counts_a_cancelled_terminal(governor_with_stub_handle):
    """End-to-end through a REAL production caller, not a direct _observe call.

    cancel_pending_generation_jobs calls fut.cancel() on a still-queued job
    (governor.py:683), which produces a genuinely CANCELLED future rather than one
    carrying a CancelledError. That is the concrete path where _terminal_outcome's
    cancelled()-before-exception() ordering matters: fut.exception() would raise.
    """
    gov = governor_with_stub_handle
    _freeze_dispatch(gov)                      # keep the job queued
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    gov.submit_job(job)

    cancelled = gov.cancel_pending_generation_jobs(reason="test")

    assert cancelled == [job.job_id]
    assert job.fut.cancelled(), "expected a cancelled future, not an exception"
    body, _ = m.get_metrics().render()
    assert any('outcome="cancelled"' in ln
               for ln in _sample(body, "st_governor_job_terminal_total"))
    # never executed, so no execution duration was observed
    assert not _sample(body, "st_governor_job_execution_seconds_count")
