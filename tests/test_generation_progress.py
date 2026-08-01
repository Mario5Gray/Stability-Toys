"""STABL-zueslhah slice — Progress frames flow worker -> sink -> bridge.

Task 1 (transport): GenerationJob.execute threads a progress callback into
run_job, and _FutureBridge forwards Progress frames to an on_progress consumer
instead of discarding them, while still fulfilling the Future on the terminal.
The real diffusers step-callback wiring is Task 2; here a fake worker drives the
callback so the transport is proven in isolation.
"""
from concurrent.futures import Future

from backends.governor import _FutureBridge, GenerationJob
from backends.backplane.frames import Progress, Result
from backends.backplane.blob import InProcBlob


class _Sub:
    def request(self, n):
        pass


def test_future_bridge_forwards_progress_and_still_fulfills_future():
    fut = Future()
    got = []
    bridge = _FutureBridge(fut, on_progress=lambda step, total, stage: got.append((step, total, stage)))
    bridge.on_subscribe(_Sub())

    bridge.on_next(Progress("job1", 3, 30, "denoise"))
    bridge.on_next(Progress("job1", 6, 30, "denoise"))
    assert got == [(3, 30, "denoise"), (6, 30, "denoise")]
    assert not fut.done()  # progress is non-terminal

    bridge.on_next(Result("job1", 42, InProcBlob("PNG")))
    assert fut.result() == "PNG"


def test_future_bridge_without_consumer_still_works():
    """on_progress=None (Task 1 default until the WS consumer lands) drops
    progress silently and preserves today's Future behaviour exactly."""
    fut = Future()
    bridge = _FutureBridge(fut)  # no on_progress
    bridge.on_subscribe(_Sub())
    bridge.on_next(Progress("job1", 1, 10))  # must not raise
    bridge.on_next(Result("job1", 7, InProcBlob("BYTES")))
    assert fut.result() == "BYTES"


def test_generation_job_execute_threads_progress_to_run_job():
    seen = {}

    class _Worker:
        def run_job(self, job, progress=None):
            seen["progress"] = progress
            if progress is not None:
                progress(5, 10, "denoise")
            return ("png", 42)

    calls = []
    cb = lambda step, total, stage: calls.append((step, total, stage))
    job = GenerationJob(req=object(), resolution_epoch=0)

    result = job.execute(_Worker(), progress=cb)

    assert seen["progress"] is cb
    assert calls == [(5, 10, "denoise")]
    assert result == ("png", 42)
