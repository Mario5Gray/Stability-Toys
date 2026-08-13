"""Trace context across the spawn boundary (STABL-qnlaclof step 5).

The same wall STABL-zuhuxwvf hit one layer down: a span opened in the parent
cannot be continued in the child, because nothing ambient survives spawn. job_id
crossed by riding the job envelope; trace context crosses the same way.

Step 6 pinned the SDK, so the linkage test below now RUNS. It asserts the trace
id, the parent-child EDGE, and the PRODUCER/CONSUMER kinds — the last two because
a trace-id check alone passed while the boundary span was missing and the child
was mislabelled INTERNAL, which is what review of PR #70 found.
"""
import ast
import pathlib
import pickle
from unittest.mock import Mock

import pytest

from backends.governor import GenerationJob
from backends.job_envelope import JOB_SCHEMA_VERSION, decode_job, encode_job
from server import tracing as t
from server.lcm_sr_server import GenerateRequest

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


def test_the_parent_and_child_spans_share_one_TRACE_ID_and_are_a_hand_off():
    """THE acceptance for step 5, unskipped by the step-6 SDK pin.

    It asserts THREE things, and the review of PR #70 is why it is not just the
    first: a trace-id equality check passes even when the boundary span is
    missing entirely and the child's span is mislabelled INTERNAL. Both of those
    defects existed and both would have survived this test in its obvious form.

      1. the child's span carries the parent's trace id  (context crossed)
      2. the child's parent_span_id IS the boundary span (the edge exists)
      3. the kinds are PRODUCER -> CONSUMER               (it reads as a hand-off)

    In-process on both sides of a REAL propagator: injecting under a live parent
    span and extracting the carrier is exactly what the two processes do, and it
    needs no second process to be true. The spawn boundary itself is covered by
    test_a_job_carrying_trace_context_still_runs_across_a_real_spawn.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    # Parent side: the boundary span, with the carrier injected INSIDE it.
    with tracer.start_as_current_span(
            "worker.submit", **t.kind_kwargs("PRODUCER")) as producer:
        carrier = {}
        from opentelemetry.propagate import inject
        inject(carrier)
        producer_ctx = producer.get_span_context()

    assert "traceparent" in carrier, "nothing was injected to cross with"

    # Child side: a fresh context, exactly as a spawned process has.
    parent_ctx = t.context_from_carrier(carrier)
    with tracer.start_as_current_span(
            "worker.execute", context=parent_ctx, **t.kind_kwargs("CONSUMER")):
        pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    submit, execute = spans["worker.submit"], spans["worker.execute"]

    assert execute.context.trace_id == producer_ctx.trace_id, "not one trace"
    assert execute.parent is not None and \
        execute.parent.span_id == producer_ctx.span_id, (
            "the child hangs off something other than the boundary span; a "
            "shared trace id alone would not have caught this"
        )
    assert submit.kind is trace.SpanKind.PRODUCER
    assert execute.kind is trace.SpanKind.CONSUMER


# ---------------------------------------------------------------------------
# Span TOPOLOGY, not just carrier plumbing
#
# Found at review of the first cut: the carrier crossed correctly while the
# emitted trace was still the wrong SHAPE — no boundary span at all, and the
# child's span left at the default INTERNAL kind. A trace-id equality check
# passes in that state, which is exactly why these assertions exist separately.
# ---------------------------------------------------------------------------

class _KindRecordingSpan:
    def __init__(self, name, kwargs):
        self.name = name
        self.kwargs = kwargs
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_attribute(self, k, v):
        self.attributes[k] = v

    def add_event(self, *a, **k):
        pass

    def record_exception(self, e):
        pass

    def set_status(self, *a, **k):
        pass

    def update_name(self, n):
        self.name = n

    def is_recording(self):
        return True


class _KindRecordingTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name, *args, **kwargs):
        span = _KindRecordingSpan(name, kwargs)
        self.spans.append(span)
        return span


def test_the_handle_opens_a_worker_submit_span(monkeypatch):
    """The boundary span itself. Without it the child's span attaches straight to
    governor.dispatch and the hand-off — the most interesting edge in the trace —
    is invisible."""
    from backends.worker_handle_subprocess import SubprocessWorkerHandle

    tracer = _KindRecordingTracer()
    monkeypatch.setattr(
        "backends.worker_handle_subprocess.get_tracer", lambda name: tracer)

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    handle._parent_conn = Mock(send_bytes=lambda b: None)
    handle._liveness = Mock()
    handle.submit(_job())

    assert [s.name for s in tracer.spans] == ["worker.submit"]


def test_the_boundary_span_is_a_PRODUCER(monkeypatch):
    """Kind is not decoration: PRODUCER/CONSUMER is how a viewer knows the two
    spans are a hand-off across a boundary rather than an ordinary nesting."""
    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from server.tracing import kind_kwargs

    tracer = _KindRecordingTracer()
    monkeypatch.setattr(
        "backends.worker_handle_subprocess.get_tracer", lambda name: tracer)

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    handle._parent_conn = Mock(send_bytes=lambda b: None)
    handle._liveness = Mock()
    handle.submit(_job())

    assert tracer.spans[0].kwargs == kind_kwargs("PRODUCER")


def test_the_carrier_is_injected_INSIDE_the_boundary_span(monkeypatch):
    """Ordering is the whole point. Injected before the span opens, the carrier
    names governor.dispatch as the parent and the child hangs off the Governor,
    skipping the boundary span entirely — a plausible trace with the one edge
    that matters missing."""
    from backends.worker_handle_subprocess import SubprocessWorkerHandle

    order = []

    class _Tracer:
        def start_as_current_span(self, name, *a, **k):
            order.append(("span", name))
            return _KindRecordingSpan(name, k)

    monkeypatch.setattr(
        "backends.worker_handle_subprocess.get_tracer", lambda name: _Tracer())
    monkeypatch.setattr(
        "backends.worker_handle_subprocess.inject_trace_context",
        lambda: order.append(("inject", None)) or {"traceparent": "x"})

    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    handle._parent_conn = Mock(send_bytes=lambda b: None)
    handle._liveness = Mock()
    handle.submit(_job())

    assert order == [("span", "worker.submit"), ("inject", None)]


def test_the_childs_span_is_a_CONSUMER():
    """ast: the child's worker.execute must declare CONSUMER. Left at the default
    it is INTERNAL, which reads as ordinary nested work rather than the far side
    of a hand-off — and nothing about a trace-id check would notice."""
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "backends" / "worker_handle_subprocess.py"
    )
    fn = next(
        n for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_worker_main"
    )
    assert '"CONSUMER"' in ast.dump(fn) or "'CONSUMER'" in ast.dump(fn), (
        "_worker_main's span does not declare CONSUMER kind"
    )


def test_kind_kwargs_is_EMPTY_without_the_sdk(monkeypatch):
    """No OTel installed is the current state and the default state. Passing
    kind=None explicitly is not the same as omitting it, so this must be {}."""
    monkeypatch.setattr(
        t, "_import_span_kind",
        lambda: (_ for _ in ()).throw(ImportError("no api here")),
    )
    assert t.kind_kwargs("PRODUCER") == {}
