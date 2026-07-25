"""Task 4 facade proof, isolated at the _FutureBridge level (no worker_pool fixtures).

The pool-level no-op is proven by the existing suite (tests/test_worker_pool.py,
tests/test_ws_routes.py, tests/test_model_routes.py) staying green unmodified with a
zero-byte server/ws_routes.py diff. This file pins the crux of the spec-§5
reconciliation — the compat Subscriber carries the worker result OPAQUELY and
reproduces fut.set_result / fut.set_exception verbatim — by driving _FutureBridge
directly through a synchronous InProcBackplane channel.
"""
import concurrent.futures
import pytest

from backends.worker_pool import _FutureBridge
from backends.backplane.inproc import InProcBackplane
from backends.backplane.blob import InProcBlob
from backends.backplane.frames import BackplaneError, BackplaneErrorCode


def _wire():
    fut: concurrent.futures.Future = concurrent.futures.Future()
    sink, pub = InProcBackplane("j1").open()
    pub.subscribe(_FutureBridge(fut))  # attach + request unbounded before any emit
    return fut, sink


def test_opaque_string_result_passes_through_verbatim():
    """A non-(png, seed) result (as test mocks return) survives unchanged — this is
    why the facade must NOT decompose (spec §5 reconciliation)."""
    fut, sink = _wire()
    sink.result(0, InProcBlob("test_result"))
    sink.complete()
    assert fut.result(timeout=1.0) == "test_result"


def test_png_seed_tuple_result_passes_through_verbatim():
    fut, sink = _wire()
    sink.result(0, InProcBlob((b"PNG", 321)))
    sink.complete()
    assert fut.result(timeout=1.0) == (b"PNG", 321)


def test_backend_exception_instance_is_preserved():
    fut, sink = _wire()
    boom = RuntimeError("backend exploded")
    sink.error(BackplaneError.from_exc(boom))
    with pytest.raises(RuntimeError) as ei:
        fut.result(timeout=1.0)
    assert ei.value is boom  # live instance, unwrapped — not a BackplaneError


def test_cancelled_terminal_reconstructs_cancelled_error():
    fut, sink = _wire()
    sink.error(BackplaneError(BackplaneErrorCode.CANCELLED, "cancelled"))
    with pytest.raises(concurrent.futures.CancelledError):
        fut.result(timeout=1.0)


def test_result_resolves_future_synchronously_on_emit():
    """Matches today's fut.set_result: the Future is done the instant the worker
    emits, because demand was requested unbounded at subscribe time."""
    fut, sink = _wire()
    assert not fut.done()
    sink.result(0, InProcBlob("x"))
    assert fut.done()  # resolved before sink.result() returned
