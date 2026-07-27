"""SubprocessWorkerHandle isolation tests — M1 across a REAL spawn boundary.

Uses the module-level FaultWorker (tests/_fault_worker.py) so the spawn child can
import it by dotted ref; no real GPU. Proves start()->READY, submit()->Publisher
driven by the child, opaque-return pickle round-trip, and stop()->dead.
"""
from concurrent.futures import Future

from backends.worker_handle_subprocess import (
    SubprocessWorkerHandle,
    _SubprocessFutureBridge,
)
from backends.governor import GenerationJob
from server.lcm_sr_server import GenerateRequest


def _req(prompt="hello"):
    # Real GenerateRequest fields (server/lcm_sr_server.py:136): num_inference_steps
    # + size, NOT the plan's stale steps/width/height (same T0 correction).
    return GenerateRequest(prompt=prompt, num_inference_steps=4, size="512x512")


def test_subprocess_handle_runs_a_job_end_to_end():
    h = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    # start() args are pickled into the spawn child, so they must be picklable —
    # Mock() is not. FaultWorker ignores resolved/binding/mode, so None stands in.
    # Real resolved/binding/mode picklability is validated by Task 8 live acceptance.
    h.start(None, None, None)                   # spawns child, loads FaultWorker
    assert h.health().state == "ready"
    assert h.worker is None                     # no in-proc worker
    job = GenerationJob(req=_req("hello"), resolution_epoch=0)
    pub = h.submit(job)
    fut = Future()
    pub.subscribe(_SubprocessFutureBridge(fut))   # unpickles the opaque return
    assert fut.result(timeout=15) == b"PNG:hello"  # pickle(bytes) round-trips to bytes
    h.stop()
    assert h.health().state == "dead"
