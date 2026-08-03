"""STABL-jredufxb: the cancel predicate at the denoise step boundary."""
import pytest
from concurrent.futures import CancelledError

from backends.step_progress import inject_step_progress


class ModernPipe:
    """Pipeline exposing the modern `callback_on_step_end` hook."""

    def __init__(self):
        self.seen = []

    def __call__(self, callback_on_step_end=None, **kwargs):
        for step in range(10):
            self.seen.append(step)
            callback_on_step_end(self, step, None, {})
        return "done"


class LegacyPipe:
    """Pipeline exposing only the legacy `callback` / `callback_steps` pair."""

    def __init__(self):
        self.seen = []

    def __call__(self, callback=None, callback_steps=1, **kwargs):
        for step in range(10):
            self.seen.append(step)
            callback(step, None, None)
        return "done"


def test_predicate_stops_the_modern_loop():
    """The predicate flips mid-run; the loop must stop at the NEXT step, not run
    to completion and not stop before it was asked to."""
    pipe = ModernPipe()
    kwargs = {}

    def should_cancel():
        # Flip on the fourth interrogation, i.e. after three steps have run.
        should_cancel.calls += 1
        return should_cancel.calls > 3
    should_cancel.calls = 0

    inject_step_progress(pipe, kwargs, None, 10, should_cancel=should_cancel)

    with pytest.raises(CancelledError):
        pipe(**kwargs)
    assert pipe.seen == [0, 1, 2, 3]


def test_predicate_stops_the_legacy_loop():
    pipe = LegacyPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, None, 10, should_cancel=lambda: True)
    with pytest.raises(CancelledError):
        pipe(**kwargs)
    assert pipe.seen == [0]


def test_cancel_message_never_mentions_out_of_memory():
    """The dispatch loop's OOM test is a substring match on str(e) and runs
    BEFORE the cancel branch — a cancel that says 'out of memory' would route
    into OOM recovery."""
    pipe = ModernPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, None, 10, should_cancel=lambda: True)
    with pytest.raises(CancelledError) as excinfo:
        pipe(**kwargs)
    assert "out of memory" not in str(excinfo.value).lower()


def test_a_raising_progress_consumer_still_cannot_break_generation():
    """Regression guard on the _emit swallow — proves the cancel check sits
    OUTSIDE it."""
    def boom(step, total, stage):
        raise RuntimeError("bad consumer")

    pipe = ModernPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, boom, 10)
    assert pipe(**kwargs) == "done"
    assert pipe.seen == list(range(10))


def test_no_progress_and_no_predicate_installs_nothing():
    pipe = ModernPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, None, 10)
    assert kwargs == {}


def test_predicate_alone_installs_the_callback():
    """progress=None must no longer short-circuit: a reap with no progress
    consumer attached still needs the callback installed."""
    pipe = ModernPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, None, 10, should_cancel=lambda: False)
    assert "callback_on_step_end" in kwargs
