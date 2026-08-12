"""Repo-local tracing facade (STABL-qnlaclof).

Runtime code asks this module for a tracer and uses it unconditionally. When the
gate is off, or the SDK cannot be configured, the facade returns a no-op tracer
whose context manager yields a no-op span so call sites carry no branches.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import nullcontext
from typing import Optional

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
    return os.getenv("TRACING_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _endpoint_from_env() -> str:
    return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()


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
