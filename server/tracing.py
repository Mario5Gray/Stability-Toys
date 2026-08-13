"""Repo-local tracing facade (STABL-qnlaclof).

Runtime code asks this module for a tracer and uses it unconditionally. When the
gate is off, or the SDK cannot be configured, the facade returns a no-op tracer
whose context manager yields a no-op span so call sites carry no branches.
"""
from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from typing import Optional

from utils.env import env_bool, env_str

logger = logging.getLogger(__name__)


def _import_opentelemetry():
    """Seam: patched in tests to prove the ImportError degrade path."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return Resource, TracerProvider, BatchSpanProcessor, OTLPSpanExporter


class _NoopSpan:
    def set_attribute(self, *args, **kwargs):
        pass

    def update_name(self, *args, **kwargs):
        """Real spans support renaming, so the no-op must too.

        Entry spans need it: neither the HTTP route template nor the WS message
        type is knowable when the span opens, and a name built from the raw path
        would be one operation per model id. A missing method here is an
        AttributeError on the DISABLED path only — the configuration nobody
        exercises before shipping.
        """
        pass

    def add_event(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def is_recording(self) -> bool:
        return False


class _NoopTracer:
    def start_as_current_span(self, *args, **kwargs):
        return nullcontext(_NoopSpan())


class Tracing:
    """Singleton facade for tracing configuration and tracer access."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._provider = None
        self._tracer_cache: dict[str, object] = {}

        if enabled:
            endpoint = _endpoint_from_env()
            if not endpoint:
                logger.warning(
                    "TRACING_ENABLED is set but OTEL_EXPORTER_OTLP_ENDPOINT is unset; "
                    "tracing degrades to no-op"
                )
                self.enabled = False
            else:
                try:
                    Resource, TracerProvider, BatchSpanProcessor, OTLPSpanExporter = (
                        _import_opentelemetry()
                    )
                except ImportError:
                    logger.warning(
                        "TRACING_ENABLED is set but opentelemetry is not installed; "
                        "tracing degrades to no-op"
                    )
                    self.enabled = False
                else:
                    resource = Resource.create({"service.name": "stability-toys"})
                    provider = TracerProvider(resource=resource)
                    provider.add_span_processor(
                        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
                    )
                    self._provider = provider

        if not self.enabled:
            self._noop_tracer = _NoopTracer()

    def get_tracer(self, name: str):
        if not self.enabled or self._provider is None:
            return self._noop_tracer
        tracer = self._tracer_cache.get(name)
        if tracer is None:
            tracer = self._provider.get_tracer(name)
            self._tracer_cache[name] = tracer
        return tracer


_tracing: Optional[Tracing] = None
_tracing_lock = threading.Lock()


def _enabled_from_env() -> bool:
    """Default OFF, and quote-tolerant.

    Goes through utils.env rather than reading os.environ by hand: env files
    quote values, so `TRACING_ENABLED="1"` read literally is '"1"' and would
    leave the pillar silently disabled. That is the STABL-voqsoicx accessor
    doing the job it was built for, and the failure it prevents is the
    STABL-xqqqqvse one — a feature implemented, tested, and enabled nowhere.
    """
    return env_bool("TRACING_ENABLED", False)


def _endpoint_from_env() -> str:
    return env_str("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()


def get_tracing() -> Tracing:
    global _tracing
    with _tracing_lock:
        if _tracing is None:
            _tracing = Tracing(_enabled_from_env())
        return _tracing


def get_tracer(name: str):
    return get_tracing().get_tracer(name)


def _import_propagator():
    """Seam: patched in tests to prove the ImportError degrade path.

    Separate from _import_opentelemetry because propagation lives in the API
    package while the exporter lives in the SDK. A child that only needs to
    CONTINUE a trace does not need an exporter configuration to do it.
    """
    from opentelemetry.propagate import extract, inject

    return inject, extract


def inject_trace_context() -> Optional[dict]:
    """The carrier to put on the wire, or None when there is nothing to say.

    None rather than {} on purpose: the child branches on it to choose root
    versus child, and "tracing is off" must stay distinguishable from "tracing
    is on and produced nothing".

    Runs on the job submit path, so every failure degrades to None. A tracing
    problem must cost a trace, never a job.
    """
    if not get_tracing().enabled:
        return None
    try:
        inject, _ = _import_propagator()
        carrier: dict = {}
        inject(carrier)
        return carrier or None
    except Exception:                       # noqa: BLE001 — degrade, never raise
        logger.debug("trace context injection failed", exc_info=True)
        return None


def context_from_carrier(carrier: Optional[dict]):
    """The parent context to open a span under, or None for a ROOT span.

    None is a legitimate, expected answer: tracing disabled parent-side, or a
    parent whose propagator produced nothing. The caller must still open a span
    — a DROPPED span is indistinguishable from a healthy idle worker, which is
    the failure this pillar exists to remove.

    NOT among the reasons: a version-mismatched envelope. decode_job REFUSES an
    unknown schema_version outright, so a v2 sender never reaches this function
    at all. An earlier draft of this docstring claimed otherwise and described a
    fallback that cannot execute.
    """
    if not carrier:
        return None
    try:
        _, extract = _import_propagator()
        return extract(carrier) or None
    except Exception:                       # noqa: BLE001 — degrade, never raise
        logger.debug("trace context extraction failed", exc_info=True)
        return None


def _import_span_kind():
    """Seam: patched in tests to prove the ImportError degrade path."""
    from opentelemetry.trace import SpanKind

    return SpanKind


def kind_kwargs(name: str) -> dict:
    """`{"kind": SpanKind.<name>}`, or `{}` when the API is unavailable.

    Returns kwargs rather than the enum because `kind=None` is NOT the same as
    omitting the argument — the SDK defaults to INTERNAL, and passing None
    explicitly is a type error rather than a default.

    Kind is not decoration on this pair. PRODUCER on the parent and CONSUMER on
    the child is how a viewer knows the two spans are a hand-off ACROSS A
    PROCESS BOUNDARY rather than ordinary nesting, and it is the one thing a
    trace-id equality check cannot tell you — the ids match either way.
    """
    try:
        return {"kind": getattr(_import_span_kind(), name)}
    except Exception:                       # noqa: BLE001 — degrade, never raise
        logger.debug("span kind %s unavailable", name, exc_info=True)
        return {}


def configure_child_tracing() -> None:
    """Bootstrap tracing inside a spawn child (mirrors _configure_child_logging).

    The child inherits the parent's stdout but none of its SDK state: it never
    runs the parent's provider setup, so without this every child span goes
    nowhere even when the carrier arrived intact.

    Wrapped and lazy for the same reason the logging bootstrap is — this runs
    before the worker exists, and a failure here must degrade to no tracing
    rather than kill a child the parent is blocked waiting on.

    NOTE for step 6: the provider this builds must be ParentBased. An independent
    sampler in the child produces traces whose parent span was never recorded,
    which looks like data loss and is a configuration error.
    """
    try:
        reset_tracing()
        get_tracing()
    except Exception:                       # noqa: BLE001 — a child that cannot
        pass                                # configure tracing must still run


def reset_tracing() -> None:
    global _tracing
    with _tracing_lock:
        _tracing = None
