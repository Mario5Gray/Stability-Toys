"""Governor metrics instrumentation (STABL-asawxgvp).

Plan: docs/superpowers/plans/2026-08-03-prometheus-substrate.md Tasks 3-4.

Task 3 — the terminal counter is instrumented at ONE choke point,
_finalize_job_record, because every dispatch-loop terminal branch funnels through
it and the job's future is resolved by the time it is called.

Task 4 — lifecycle counters (load, churn, recovery, wait expiry). Every one runs
inside Governor._metric, so no counter can raise into the dispatch loop.
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
    _drain_queue,
)


@pytest.fixture(autouse=True)
def _metrics_on(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.reset_metrics()
    yield
    m.reset_metrics()


def _mode_config(*names, default=None):
    """_make_multi_mode_config plus a real list_modes().

    The shared helper returns a Mock, whose .list_modes() would yield another Mock —
    not iterable. _publish_mode_active iterates it, so tests that assert on
    mode_active must supply the real list. (Production code survives the Mock
    because every metrics side effect runs inside Governor._metric; the hazard is
    pinned by test_mode_active_survives_a_mode_config_that_cannot_list_modes.)
    """
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
    """A Governor on a stub handle with one real loaded mode.

    resolve_model is patched for the fixture's whole life, not just construction:
    the Governor loads its default mode in __init__, and an unpatched load would
    leave _current_mode None so every observation would label mode="unknown".
    """
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _make_governor("mode-a", default="mode-a")
        try:
            yield gov
        finally:
            gov.shutdown()


@pytest.fixture
def two_mode_governor():
    """Two configured modes — the only way to prove mode_active reports 0 for the
    modes that are NOT loaded."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _make_governor("mode-a", "mode-b", default="mode-a")
        try:
            yield gov
        finally:
            _freeze_dispatch(gov)
            _drain_queue(gov)
            gov.shutdown()


def _sample(body: bytes, name: str) -> list[str]:
    return [ln for ln in body.decode().splitlines()
            if ln.startswith(name) and not ln.startswith("#")]


def _value(body: bytes, name: str, contains: str | None = None) -> float | None:
    """The numeric value of the first matching sample line."""
    for ln in _sample(body, name):
        if contains is None or contains in ln:
            return float(ln.rsplit(" ", 1)[1])
    return None


def _render() -> bytes:
    body, _ = m.get_metrics().render()
    return body


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


# ============================================================================
# Task 4: Governor lifecycle counters
# ============================================================================

# --- mode load ---

def test_mode_load_observes_a_duration(governor_with_stub_handle):
    """Counted per SUCCESSFUL load. The fixture's Governor already loaded once in
    __init__, so this asserts an increment rather than mere presence — otherwise
    the test would pass without _load_mode being instrumented at all."""
    gov = governor_with_stub_handle
    before = _value(_render(), "st_governor_mode_load_seconds_count") or 0.0

    gov._load_mode("mode-a")

    after = _value(_render(), "st_governor_mode_load_seconds_count")
    assert after == before + 1


def test_failed_load_observes_no_duration(governor_with_stub_handle):
    """A load that raises has no duration to report. Failure is visible as
    mode_active staying 0, not as a bogus timing sample."""
    gov = governor_with_stub_handle
    before = _value(_render(), "st_governor_mode_load_seconds_count") or 0.0

    with patch.object(gov._handle, "start", side_effect=RuntimeError("load boom")):
        with pytest.raises(Exception):
            gov._load_mode("mode-a")

    after = _value(_render(), "st_governor_mode_load_seconds_count") or 0.0
    assert after == before


def test_resolution_epoch_is_published(governor_with_stub_handle):
    gov = governor_with_stub_handle
    gov._load_mode("mode-a")
    assert _value(_render(), "st_governor_resolution_epoch") == gov._resolution_epoch


# --- mode_active over the whole configured set ---

def test_mode_active_is_one_for_loaded_and_zero_for_the_rest(two_mode_governor):
    """The claim the design rests on: a single gauge labelled with the current
    mode would leave a stale 1 on the previous mode forever."""
    gov = two_mode_governor
    gov._load_mode("mode-b")

    body = _render()
    assert _value(body, "st_governor_mode_active", 'mode="mode-b"') == 1.0
    assert _value(body, "st_governor_mode_active", 'mode="mode-a"') == 0.0


def test_mode_active_survives_a_mode_config_that_cannot_list_modes(governor_with_stub_handle):
    """_publish_mode_active iterates list_modes(). A Mock config returns a
    non-iterable Mock — which is what most existing Governor tests supply. That
    must degrade silently through Governor._metric, never break a load."""
    gov = governor_with_stub_handle
    gov._mode_config.list_modes.side_effect = TypeError("not iterable")

    gov._load_mode("mode-a")     # must not raise

    # the load itself is still timed; only the gauge publication was skipped
    assert _value(_render(), "st_governor_mode_load_seconds_count") is not None


# --- wait expiry ---

@pytest.mark.parametrize("budget", ["admission", "execution"])
def test_expire_counts_the_budget_that_blew(governor_with_stub_handle, budget):
    gov = governor_with_stub_handle
    with pytest.raises(TimeoutError):
        gov._expire(None, budget, 120.0, 121.0)

    assert _value(_render(), "st_governor_wait_expired_total",
                  f'budget="{budget}"') == 1.0


def test_expire_still_raises_when_the_counter_explodes(governor_with_stub_handle, monkeypatch):
    """The counter must never swallow the TimeoutError — the caller depends on it."""
    class _Explodes:
        def labels(self, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(m.get_metrics(), "wait_expired_total", _Explodes())
    with pytest.raises(TimeoutError):
        governor_with_stub_handle._expire(None, "execution", 1.0, 2.0)


# --- worker recovery ---

@pytest.mark.parametrize("oom,reason", [(True, "oom"), (False, "dead")])
def test_worker_recovery_counter_labels_the_cause(governor_with_stub_handle, oom, reason):
    governor_with_stub_handle._count_worker_recovery(oom=oom)
    assert _value(_render(), "st_governor_worker_recovery_total",
                  f'reason="{reason}"') == 1.0


# --- churn: switch, demand reload, unload ---

def test_switch_counts_the_target_mode(two_mode_governor):
    """Labelled by TARGET only — a {from,to} pair squares the cardinality for a
    transition matrix nobody asked for."""
    gov = two_mode_governor
    _freeze_dispatch(gov)                      # count happens at enqueue
    gov.switch_mode("mode-b")

    assert _value(_render(), "st_governor_mode_switch_total",
                  'mode="mode-b"') == 1.0


def test_demand_reload_counts(governor_with_stub_handle):
    gov = governor_with_stub_handle
    gov._reload_from_snapshot()
    assert _value(_render(), "st_governor_demand_reload_total",
                  'mode="mode-a"') == 1.0


def test_unload_counts_with_a_reason_and_clears_mode_active(two_mode_governor):
    gov = two_mode_governor
    assert _value(_render(), "st_governor_mode_active", 'mode="mode-a"') == 1.0

    gov.unload_current_model()

    body = _render()
    assert _value(body, "st_governor_unload_total", 'reason="explicit"') == 1.0
    assert _value(body, "st_governor_mode_active", 'mode="mode-a"') == 0.0
    assert _value(body, "st_governor_mode_active", 'mode="mode-b"') == 0.0


def test_unload_reason_distinguishes_callers(governor_with_stub_handle):
    """A hardcoded reason label would be dead weight. Idle eviction and an
    explicit unload are different operational events and must be separable."""
    gov = governor_with_stub_handle
    gov._unload_current_worker(reason="idle_evict")
    assert _value(_render(), "st_governor_unload_total",
                  'reason="idle_evict"') == 1.0


def test_first_ever_load_does_not_count_a_phantom_unload():
    """_load_mode unloads the outgoing worker before loading. On the very first
    load there is no outgoing mode, and counting an unload there would report
    churn that never happened."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _make_governor("mode-a", default="mode-a")
        try:
            assert _value(_render(), "st_governor_unload_total") is None
        finally:
            gov.shutdown()


# --- the guard ---

def test_lifecycle_counters_cannot_raise(governor_with_stub_handle, monkeypatch):
    """STABL-hdzggeir: every lifecycle counter goes through Governor._metric so a
    broken metrics backend cannot reach the dispatch loop."""
    class _Explodes:
        def labels(self, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(m.get_metrics(), "worker_recovery_total", _Explodes())
    governor_with_stub_handle._count_worker_recovery(oom=True)   # returns normally


def test_metric_helper_swallows_anything(governor_with_stub_handle):
    """_metric guards the whole side effect, not just the metric call."""
    governor_with_stub_handle._metric(lambda met: 1 / 0)          # returns normally
