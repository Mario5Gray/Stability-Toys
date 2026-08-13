"""Trace context across the spawn boundary (STABL-qnlaclof step 5).

The same wall STABL-zuhuxwvf hit one layer down: a span opened in the parent
cannot be continued in the child, because nothing ambient survives spawn. job_id
crossed by riding the job envelope; trace context crosses the same way.

WHAT THESE TESTS DO NOT PROVE: that the child's span is really a CHILD of the
parent's. That is W3C traceparent semantics and needs the OTel SDK, which is not
a dependency until step 6. The end-to-end linkage test below is written and
skipped rather than omitted, so it starts running the moment the pin lands.
What IS proven here is the plumbing: the carrier is produced, survives the wire,
is refused when the version disagrees, and is consumed on the far side.
"""
import ast
import importlib.util
import pathlib
import pickle
from unittest.mock import Mock

import pytest

from backends.governor import GenerationJob
from backends.job_envelope import JOB_SCHEMA_VERSION, decode_job, encode_job
from server import tracing as t
from server.lcm_sr_server import GenerateRequest

HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None


def _job(prompt="hello"):
    return GenerationJob(
        req=GenerateRequest(prompt=prompt, num_inference_steps=4, size="512x512"),
        resolution_epoch=0,
    )


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------

def test_the_carrier_survives_the_wire():
    carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}

    decoded = decode_job(encode_job(_job(), trace_carrier=carrier))

    assert decoded.trace_carrier == carrier


def test_no_carrier_decodes_to_None_rather_than_an_empty_dict():
    """Tracing off is the default and must stay distinguishable from tracing on
    with nothing to say — the child branches on it to decide root vs child."""
    assert decode_job(encode_job(_job())).trace_carrier is None


def test_the_schema_version_was_BUMPED_for_this():
    """Adding a carried element is a version bump, not an additive change.
    decode_job REFUSES an unknown version rather than default-filling, and that
    refusal is the feature: STABL-spxwqlan was a field silently taking its
    default across this exact boundary."""
    assert JOB_SCHEMA_VERSION == 3


def test_a_v2_envelope_is_REFUSED_by_this_build():
    """A rolling deploy pairs a v2 parent with a v3 child. Default-filling the
    carrier would be harmless; default-filling is the HABIT that is not."""
    job = _job()
    v2 = bytes([2]) + pickle.dumps((job.req, job.job_id, 0, None, []))

    with pytest.raises(ValueError, match="schema_version"):
        decode_job(v2)


def test_the_job_payload_still_round_trips_unchanged():
    """The v3 bump must not disturb what v2 already carried — that regression
    would be STABL-spxwqlan a third time."""
    job = _job("a prompt")
    job.init_image = b"initbytes"

    decoded = decode_job(encode_job(job))

    assert decoded.req.prompt == "a prompt"
    assert decoded.job_id == job.job_id
    assert decoded.init_image == b"initbytes"


# ---------------------------------------------------------------------------
# The facade
# ---------------------------------------------------------------------------

def test_injecting_with_tracing_disabled_yields_None(monkeypatch):
    """Default off, and no OTel installed. This must be a quiet None, not an
    ImportError from the middle of a job submit."""
    monkeypatch.delenv("TRACING_ENABLED", raising=False)
    t.reset_tracing()
    try:
        assert t.inject_trace_context() is None
    finally:
        t.reset_tracing()


def test_an_ABSENT_carrier_yields_no_parent_context():
    """Which is what makes the child open a ROOT span. A child that DROPPED the
    span instead would be indistinguishable from a healthy idle worker — the
    failure mode this whole pillar exists to remove."""
    assert t.context_from_carrier(None) is None
    assert t.context_from_carrier({}) is None


def test_a_broken_propagator_degrades_instead_of_raising(monkeypatch):
    """This runs on the job submit path. A tracing failure must cost a trace,
    never a job."""
    monkeypatch.setattr(
        t, "_import_propagator",
        lambda: (_ for _ in ()).throw(ImportError("no api here")),
    )
    assert t.inject_trace_context() is None
    assert t.context_from_carrier({"traceparent": "whatever"}) is None


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------

def test_the_handle_puts_the_injected_carrier_ON_THE_WIRE(monkeypatch):
    """The PRODUCER half. Asserts on the bytes actually sent, because that is
    the only thing the child will ever see."""
    from backends.worker_handle_subprocess import SubprocessWorkerHandle

    carrier = {"traceparent": "00-11111111111111111111111111111111-2222222222222222-01"}
    monkeypatch.setattr(
        "backends.worker_handle_subprocess.inject_trace_context", lambda: carrier)

    sent = []
    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    handle._parent_conn = Mock(send_bytes=sent.append)
    handle._liveness = Mock()

    handle.submit(_job())

    assert len(sent) == 1
    assert decode_job(sent[0]).trace_carrier == carrier


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------

def test_the_child_opens_its_span_with_the_DECODED_carrier():
    """ast, for the same reason test_worker_main_calls_the_bootstrap_FIRST is:
    the wiring is what silently lapses. A child that opened a span while ignoring
    the carrier would produce a plausible, permanently orphaned trace.
    """
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "backends" / "worker_handle_subprocess.py"
    )
    fn = next(
        n for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_worker_main"
    )
    dumped = ast.dump(fn)

    assert "context_from_carrier" in dumped, (
        "_worker_main never consumes the carrier; the child's spans would be "
        "roots even when the parent sent a parent"
    )
    assert "trace_carrier" in dumped


def test_the_child_bootstraps_tracing_BEFORE_the_job_loop():
    """Same ordering argument as _configure_child_logging: the child inherits
    stdout but no SDK state, and a provider configured after the first job is a
    provider that missed it."""
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "backends" / "worker_handle_subprocess.py"
    )
    fn = next(
        n for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_worker_main"
    )
    body = fn.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]

    calls = [
        n.value.func.id for n in body[:3]
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
    ]
    assert "configure_child_tracing" in calls, (
        f"_worker_main's opening statements are {calls}; the tracing bootstrap "
        "must sit with the logging one, before anything heavy imports"
    )


def test_a_job_carrying_trace_context_still_runs_across_a_real_spawn():
    """The carrier must not break the boundary it rides. Cheap, and it is the
    only test here that exercises a real child."""
    from concurrent.futures import Future

    from backends.worker_handle_subprocess import (
        SubprocessWorkerHandle,
        _SubprocessFutureBridge,
    )

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    handle.start(None, None, None)
    try:
        job = _job("hi")
        fut = Future()
        handle.submit(job).subscribe(_SubprocessFutureBridge(fut))
        assert fut.result(timeout=30) == b"PNG:hi"
    finally:
        handle.stop()


@pytest.mark.skipif(not HAS_OTEL, reason="OTel SDK is not a dependency until step 6")
def test_the_childs_span_shares_the_parents_TRACE_ID():
    """THE acceptance for this step, and it cannot run yet.

    Written and skipped rather than omitted: everything above proves the carrier
    is produced, survives the wire and is consumed, and NONE of it proves the
    child's span is really a child. That is W3C semantics and needs the SDK.
    Turning on the step-6 pin turns this on.
    """
    pytest.fail(
        "unimplemented: assert the child's span trace_id equals the parent's "
        "across a real spawn, using an in-memory exporter in both processes"
    )
