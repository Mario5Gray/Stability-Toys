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


def reset_tracing() -> None:
    global _tracing
    with _tracing_lock:
        _tracing = None
