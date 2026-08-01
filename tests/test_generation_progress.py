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


def test_subprocess_bridge_forwards_progress_and_fulfills_future():
    """Subprocess path: _SubprocessFutureBridge mirrors _FutureBridge — forwards
    Progress to on_progress, unpickles the opaque Result for the Future."""
    import pickle
    from backends.worker_handle_subprocess import _SubprocessFutureBridge

    fut = Future()
    got = []
    bridge = _SubprocessFutureBridge(fut, on_progress=lambda step, total, stage: got.append((step, total, stage)))
    bridge.on_subscribe(_Sub())

    bridge.on_next(Progress("j", 2, 20, "denoise"))
    assert got == [(2, 20, "denoise")]
    assert not fut.done()

    bridge.on_next(Result("j", 0, InProcBlob(pickle.dumps("PNG"))))
    assert fut.result() == "PNG"


def test_demand_reload_emits_load_stage_progress():
    """Model-load progress: when a generation job's own dispatch triggers a demand
    reload (worker evicted), emit a 'load' stage Progress via on_progress so the
    client sees loading instead of silence (STABL-zueslhah)."""
    import pickle
    from unittest.mock import Mock, patch
    from backends.governor import Governor, GenerationJob
    from backends.worker_handle import WorkerHealth
    from backends.conditioning.contracts import ConditioningConfig
    from backends.model_resolution import LocalModelBinding
    from backends.backplane.inproc import InProcBackplane
    from backends.backplane.blob import InProcBlob
    from tests.test_governor import StubHandle

    class _EvictableHandle(StubHandle):
        def __init__(self):
            super().__init__()
            self.available = True  # set True after the __init__ load's start()

        def start(self, resolved_mode, binding, mode):
            super().start(resolved_mode, binding, mode)
            self.available = True  # a (re)load restores availability

        def health(self):
            state = "ready" if self.available else "dead"
            return WorkerHealth(state=state, vram_free_bytes=0, vram_total_bytes=0, mode=None)

        def submit(self, job):
            sink, pub = InProcBackplane(job.job_id).open()
            sink.result(0, InProcBlob(pickle.dumps("R")))
            sink.complete()
            return pub

    def _resolve(model_path, mode):
        return Mock(), LocalModelBinding(model_path)

    with patch("backends.governor.resolve_model", side_effect=_resolve):
        mode = Mock()
        mode.model_path = "/models/test.safetensors"
        mode.loras = []
        mode.conditioning = ConditioningConfig()
        mode_config = Mock()
        mode_config.get_mode.return_value = mode
        mode_config.get_default_mode.return_value = "test-mode"
        registry = Mock()
        registry.get_used_vram.return_value = 0
        registry.get_allocated_vram.return_value = 0
        registry.get_total_vram.return_value = 8 * 1024**3
        registry.register_model = Mock()

        handle = _EvictableHandle()
        gov = Governor(worker_factory=Mock(), handle=handle,
                       mode_config=mode_config, registry=registry)
        handle.available = False  # evict: the next dispatch must demand-reload

        got = []
        job = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
        fut = gov.submit_job(job, on_progress=lambda s, t, st: got.append((s, t, st)))
        assert fut.result(timeout=2.0) == "R"
        gov.shutdown()

    stages = [st for (_s, _t, st) in got]
    assert "load" in stages  # the demand reload surfaced a load-stage frame


def test_subprocess_dispatch_threads_on_progress_to_bridge():
    """The Governor's SUBPROCESS dispatch path (handle.worker is None) must attach
    _SubprocessFutureBridge with the submit_job on_progress, so subprocess-mode
    generation streams progress to the WS consumer too (STABL-zueslhah follow-on)."""
    import pickle
    from unittest.mock import Mock, patch
    from backends.governor import Governor, GenerationJob
    from backends.conditioning.contracts import ConditioningConfig
    from backends.model_resolution import LocalModelBinding
    from backends.backplane.inproc import InProcBackplane
    from backends.backplane.blob import InProcBlob
    from tests.test_governor import StubHandle

    class _ProgressHandle(StubHandle):
        def submit(self, job):
            sink, pub = InProcBackplane(job.job_id).open()
            sink.progress(2, 20, "denoise")
            sink.result(0, InProcBlob(pickle.dumps("R")))
            sink.complete()
            return pub

    def _resolve(model_path, mode):
        return Mock(), LocalModelBinding(model_path)

    with patch("backends.governor.resolve_model", side_effect=_resolve):
        mode = Mock()
        mode.model_path = "/models/test.safetensors"
        mode.loras = []
        mode.conditioning = ConditioningConfig()
        mode_config = Mock()
        mode_config.get_mode.return_value = mode
        mode_config.get_default_mode.return_value = "test-mode"
        registry = Mock()
        registry.get_used_vram.return_value = 0
        registry.get_allocated_vram.return_value = 0
        registry.get_total_vram.return_value = 8 * 1024**3
        registry.register_model = Mock()

        gov = Governor(
            worker_factory=Mock(),
            handle=_ProgressHandle(),
            mode_config=mode_config,
            registry=registry,
        )
        got = []
        job = GenerationJob(req=Mock(), resolution_epoch=gov.current_resolution_epoch())
        fut = gov.submit_job(job, on_progress=lambda s, t, st: got.append((s, t, st)))
        assert fut.result(timeout=2.0) == "R"
        gov.shutdown()

    assert got == [(2, 20, "denoise")]


async def test_ws_progress_forwarder_schedules_job_progress_send(monkeypatch):
    """Task 4: the WS forwarder turns an on_progress call (on a worker thread)
    into a thread-safe hub.send of a job:progress frame on the loop."""
    import asyncio
    from server import ws_routes

    sent = []

    class _Hub:
        async def send(self, cid, msg):
            sent.append((cid, msg))

    monkeypatch.setattr(ws_routes, "hub", _Hub())

    scheduled = []

    class _Loop:
        def call_soon_threadsafe(self, fn, arg):
            scheduled.append((fn, arg))

    fwd = ws_routes.make_generation_progress_forwarder(_Loop(), "client-1", "job-9")
    fwd(3, 30, "denoise")

    assert len(scheduled) == 1
    fn, coro = scheduled[0]
    assert fn is asyncio.ensure_future
    await coro  # execute the scheduled hub.send coroutine
    assert sent == [("client-1", {
        "type": "job:progress", "jobId": "job-9",
        "step": 3, "total": 30, "stage": "denoise",
    })]


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
