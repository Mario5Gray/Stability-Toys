import concurrent.futures
from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob
from backends.backplane.frames import (
    Ack, Progress, Result, BackplaneError, BackplaneErrorCode,
)
from backends.backplane.reactivestreams import Subscriber


class Recorder(Subscriber):
    def __init__(self, demand=1 << 62):
        self.frames, self.error, self.completed = [], None, False
        self._demand = demand
        self.sub = None

    def on_subscribe(self, subscription):
        self.sub = subscription
        subscription.request(self._demand)

    def on_next(self, value):
        self.frames.append(value)

    def on_error(self, error):
        self.error = error

    def on_complete(self):
        self.completed = True


def test_unbounded_demand_delivers_ack_result_complete_in_order():
    sink, pub = InProcBackplane("j1").open()
    rec = Recorder()
    pub.subscribe(rec)
    sink.ack()
    sink.result(seed=7, blob=InProcBlob(b"png"))
    sink.complete()
    assert [type(f).__name__ for f in rec.frames] == ["Ack", "Result"]
    assert rec.frames[1].seed == 7 and rec.completed is True


def test_error_carries_live_exception_instance():
    sink, pub = InProcBackplane("j1").open()
    rec = Recorder()
    pub.subscribe(rec)
    orig = RuntimeError("CUDA out of memory")
    sink.error(BackplaneError.from_exc(orig))
    assert rec.error.to_exception() is orig  # live instance preserved


def test_progress_conflates_under_zero_demand_result_never_dropped():
    sink, pub = InProcBackplane("j1").open()
    rec = Recorder(demand=0)  # no demand yet
    pub.subscribe(rec)
    sink.progress(1, 20)
    sink.progress(2, 20)
    sink.progress(3, 20)      # only the newest should survive
    sink.result(seed=9, blob=InProcBlob(b"x"))  # must-deliver, buffered
    rec.sub.request(1 << 62)  # open the floodgates
    progresses = [f for f in rec.frames if isinstance(f, Progress)]
    results = [f for f in rec.frames if isinstance(f, Result)]
    assert progresses == [Progress("j1", 3, 20)]  # conflated to latest
    assert len(results) == 1 and results[0].seed == 9  # never dropped


def test_result_delivered_synchronously_before_sink_result_returns():
    """The load-bearing invariant Task 4 depends on: with an attached, unbounded
    subscriber, sink.result() delivers on_next(Result) synchronously and returns
    only after it ran (matches today's fut.set_result semantics)."""
    delivered = []

    class ImmediateRecorder(Subscriber):
        def on_subscribe(self, subscription):
            subscription.request(1 << 62)  # unbounded, before any emit
        def on_next(self, value):
            delivered.append(value)
        def on_error(self, error):
            pass
        def on_complete(self):
            pass

    sink, pub = InProcBackplane("j1").open()
    pub.subscribe(ImmediateRecorder())  # attach BEFORE emit
    assert delivered == []              # nothing yet
    sink.result(seed=5, blob=InProcBlob(b"z"))
    # If delivery were async/buffered, this would still be empty here:
    assert len(delivered) == 1 and delivered[0].seed == 5


def test_cancel_sets_sink_cancelled():
    sink, pub = InProcBackplane("j1").open()
    rec = Recorder()
    pub.subscribe(rec)
    assert sink.cancelled is False
    rec.sub.cancel()
    assert sink.cancelled is True
