"""STABL-zuhuxwvf: job_id must survive the spawn boundary.

STABL-bpsfmoke shipped job_id correlation and its closeout claimed both processes
that write to the container's stdout — the server, and the spawned child where
generation actually happens. Measured on live Loki over 24h, the child half does
not work: every `backends.cuda_worker` line came from a child pid and not one
carried a job_id, while the server pid did.

The parent binds a ContextVar in the dispatch loop (governor.py:1048). Under
WORKER_ISOLATION=subprocess the work runs in `_worker_main` in a DIFFERENT
process, and a ContextVar cannot cross a spawn boundary.

These tests run across a REAL spawn boundary through SubprocessWorkerHandle —
the production path. An in-proc test cannot fail here, which is exactly why the
bug survived a full suite.
"""
import ast
import json
import pathlib
from concurrent.futures import Future

import pytest

from backends.governor import GenerationJob
from backends.worker_handle_subprocess import (
    SubprocessWorkerHandle,
    _SubprocessFutureBridge,
)
from server.lcm_sr_server import GenerateRequest


def _req(prompt="hello"):
    return GenerateRequest(prompt=prompt, num_inference_steps=4, size="512x512")


def _run(handle, job):
    """Submit and unpickle the child's opaque return."""
    fut = Future()
    handle.submit(job).subscribe(_SubprocessFutureBridge(fut))
    return fut.result(timeout=30)


@pytest.fixture
def child(monkeypatch):
    """A live spawn child running LogLineWorker.

    LOG_FORMAT is set BEFORE the spawn because the child inherits the parent's
    environment at spawn time; setting it afterwards would not reach the child.
    """
    monkeypatch.setenv("LOG_FORMAT", "json")
    handle = SubprocessWorkerHandle("tests._fault_worker.make_log_line_worker")
    handle.start(None, None, None)
    try:
        yield handle
    finally:
        handle.stop()


def test_a_log_line_from_inside_the_child_carries_the_job_id(child):
    """The measured failure, as a test. Without the fix the line renders with no
    job_id field at all — which is what every backends.cuda_worker line in Loki
    looked like."""
    job = GenerationJob(req=_req(), resolution_epoch=0)

    payload = json.loads(_run(child, job).decode())

    assert payload["message"] == "generating in the child"
    assert payload.get("job_id") == job.job_id, (
        "a log line emitted from the child's job body carries no job_id; "
        "generation is where the interesting logs are and they are uncorrelated"
    )


def test_the_child_rebinds_PER_JOB_rather_than_once(child):
    """Binding once at child startup would satisfy the test above and still be
    wrong: every job after the first would log under the first job's id, which
    reads as real correlation and is worse than none."""
    first = GenerationJob(req=_req("one"), resolution_epoch=0)
    second = GenerationJob(req=_req("two"), resolution_epoch=0)
    assert first.job_id != second.job_id

    first_payload = json.loads(_run(child, first).decode())
    second_payload = json.loads(_run(child, second).decode())

    assert first_payload.get("job_id") == first.job_id
    assert second_payload.get("job_id") == second.job_id, (
        "the second job logged under the first job's id — the binding is not "
        "per-job"
    )


def test_the_childs_bind_is_RESET_on_every_exit_path():
    """Set-without-reset is the failure that matters, and no behavioural test can
    see it: a stale id is only observable on lines emitted BETWEEN jobs, and
    `_worker_main`'s loop logs nothing there. So this pins the structure — the
    reset must sit on a path that runs even when run_job raises.

    Same technique as test_worker_main_calls_the_bootstrap_FIRST, and the same
    reason: the wiring is the thing that can silently lapse.
    """
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "backends" / "worker_handle_subprocess.py"
    )
    tree = ast.parse(source.read_text(), filename=str(source))
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_worker_main"
    )

    def _guards_the_job_body(node) -> bool:
        """A `with` block, or a try whose finally runs regardless."""
        if isinstance(node, ast.With):
            return True
        return isinstance(node, ast.Try) and bool(node.finalbody)

    binds = [
        n for n in ast.walk(fn)
        if isinstance(n, (ast.With, ast.Try))
        and "job_id" in ast.dump(n)
        and _guards_the_job_body(n)
    ]
    assert binds, (
        "_worker_main binds job_id outside a `with` or a try/finally, so an "
        "exception from run_job leaves the id set for whatever the child logs next"
    )
