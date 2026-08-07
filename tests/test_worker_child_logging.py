"""STABL-bpsfmoke: the spawned child configures the same logging the parent has.

Under WORKER_ISOLATION=subprocess — the PRODUCTION path — generation happens in
the child. It inherits the parent's stdout but sees neither run.py's dictConfig
nor uvicorn's --log-config, so without this bootstrap the container's log stream
is JSON from the parent interleaved with default-formatted lines from the process
doing the actual work.
"""
import ast
import json
import logging
import multiprocessing as mp
import os
import pathlib


def _child(queue):
    """Runs in a REAL spawned process. Anything the parent configured is absent
    here — that absence is the bug under test.

    Every failure is REPORTED on the queue rather than left to kill the process. A
    child that dies silently turns every assertion into a queue timeout, so a
    one-line import error costs the whole timeout and reports nothing useful.
    """
    try:
        os.environ["LOG_FORMAT"] = "json"
        from backends.worker_handle_subprocess import _configure_child_logging

        _configure_child_logging()

        from server.log_format import StabilityFormatter

        handler = logging.getLogger().handlers[0]
        rec = logging.LogRecord("st.child", logging.INFO, "/x.py", 1, "from the child", (), None)
        queue.put(("ok", isinstance(handler.formatter, StabilityFormatter),
                   handler.formatter.format(rec)))
    except BaseException:
        import traceback
        queue.put(("error", False, traceback.format_exc()))


def test_the_child_gets_the_json_formatter_and_its_OWN_pid():
    ctx = mp.get_context("spawn")     # spawn, NOT fork — the facet-3 invariant
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q,))
    p.start()
    try:
        status, is_ours, payload_or_trace = q.get(timeout=120)
    finally:
        p.join(timeout=30)

    assert status == "ok", f"child failed:\n{payload_or_trace}"
    assert is_ours, "the child's root handler is not using StabilityFormatter"
    payload = json.loads(payload_or_trace)
    assert payload["message"] == "from the child"
    assert payload["pid"] == p.pid          # ITS pid, not the parent's
    assert payload["pid"] != os.getpid()


def test_worker_main_calls_the_bootstrap_FIRST():
    """The test above calls _configure_child_logging directly, so it would still
    pass if _worker_main stopped calling it at all. This pins the wiring AND the
    ordering: torch and diffusers log at import, and anything inserted above this
    call emits before the formatter is installed.
    """
    source = pathlib.Path(__file__).resolve().parent.parent / "backends" / "worker_handle_subprocess.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_worker_main"
    )

    body = fn.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]     # skip the docstring

    first = body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), (
        f"_worker_main's first statement is {ast.dump(first)[:80]}, not a call"
    )
    assert getattr(first.value.func, "id", None) == "_configure_child_logging"
