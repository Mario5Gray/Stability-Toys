from __future__ import annotations

from typing import Optional

from .frames import Ack, Progress, Result, BackplaneError
from .interface import JobSink
from .reactivestreams import Publisher, Subscriber, Subscription


class _InProcSubscription(Subscription):
    def __init__(self, channel: "_Channel"):
        self._channel = channel

    def request(self, n: int) -> None:
        self._channel.add_demand(n)

    def cancel(self) -> None:
        self._channel.cancel()


class _Channel:
    """Synchronous single-subscriber channel. Progress conflates while demand==0;
    Ack/Result/terminal are buffered must-deliver.

    ORDERING INVARIANT (load-bearing for the no-op facade, spec §3.3/§5):
    "synchronous must-deliver on return" holds ONLY when the subscriber is already
    attached with unbounded demand before any emit_* call. Under that ordering an
    emit_must_deliver / emit_terminal delivers to the subscriber synchronously and
    returns after on_next/on_complete/on_error has run — matching today's
    fut.set_result semantics. If a frame is emitted before subscribe()/request(),
    it is buffered and delivered on later demand instead (NOT synchronous). Task 4's
    submit_job MUST attach the compat Subscriber (request unbounded) before the job
    is enqueued, so the worker thread never emits into an unattached channel.

    Single terminal: emit exactly one of complete()/error(); do not emit after a
    terminal (the worker loop never does).
    """

    def __init__(self):
        self._subscriber: Optional[Subscriber] = None
        self._demand = 0
        self._cancelled = False
        self._pending_progress: Optional[Progress] = None
        self._buffer: list = []          # Ack / Result (must-deliver)
        self._terminal = None            # ("complete", None) | ("error", BackplaneError)

    # producer side -------------------------------------------------------
    def emit_must_deliver(self, frame) -> None:
        self._buffer.append(frame)
        self._drain()

    def emit_progress(self, frame: Progress) -> None:
        self._pending_progress = frame   # latest-wins conflation
        self._drain()

    def emit_terminal(self, terminal) -> None:
        self._terminal = terminal
        self._drain()

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # consumer side -------------------------------------------------------
    def attach(self, subscriber: Subscriber) -> None:
        self._subscriber = subscriber
        subscriber.on_subscribe(_InProcSubscription(self))

    def add_demand(self, n: int) -> None:
        self._demand += n
        self._drain()

    def _drain(self) -> None:
        if self._subscriber is None or self._cancelled:
            return
        # Ack/Result first (ordered, must-deliver), then conflated progress.
        while self._buffer and self._demand > 0:
            self._demand -= 1
            self._subscriber.on_next(self._buffer.pop(0))
        if self._pending_progress is not None and self._demand > 0:
            self._demand -= 1
            frame, self._pending_progress = self._pending_progress, None
            self._subscriber.on_next(frame)
        if not self._buffer and self._terminal is not None:
            kind, payload = self._terminal
            self._terminal = None
            if kind == "complete":
                self._subscriber.on_complete()
            else:
                self._subscriber.on_error(payload)


class _InProcJobSink(JobSink):
    def __init__(self, job_id: str, channel: _Channel):
        self._job_id = job_id
        self._channel = channel

    def ack(self, queued_position: int = 0) -> None:
        self._channel.emit_must_deliver(Ack(self._job_id, queued_position))

    def progress(self, step: int, total: int, stage: str = "denoise") -> None:
        self._channel.emit_progress(Progress(self._job_id, step, total, stage))

    def result(self, seed: int, blob) -> None:
        self._channel.emit_must_deliver(Result(self._job_id, seed, blob))

    def complete(self) -> None:
        self._channel.emit_terminal(("complete", None))

    def error(self, err: BackplaneError) -> None:
        self._channel.emit_terminal(("error", err))

    @property
    def cancelled(self) -> bool:
        return self._channel.cancelled


class _InProcPublisher(Publisher):
    def __init__(self, channel: _Channel):
        self._channel = channel

    def subscribe(self, subscriber: Subscriber) -> None:
        self._channel.attach(subscriber)


class InProcBackplane:
    def __init__(self, job_id: str):
        self._job_id = job_id

    def open(self):
        channel = _Channel()
        return _InProcJobSink(self._job_id, channel), _InProcPublisher(channel)
