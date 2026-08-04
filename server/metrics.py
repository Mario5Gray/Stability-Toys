"""Repo-local Prometheus facade (STABL-asawxgvp).

The ONLY module permitted to import prometheus_client. Runtime code calls
get_metrics() and touches metric objects unconditionally: when the gate is off
the facade returns no-op objects, so instrumentation sites carry no branches.

Imports nothing from backends/ — the sampler injects readers by callable.

Spec: docs/superpowers/specs/2026-08-03-server-observability-seams-design.md
"""
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

QUEUE_WAIT_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 900)
EXECUTION_BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600)
MODE_LOAD_BUCKETS = (1, 2, 5, 10, 20, 30, 60, 120, 300)

_PLAIN_TEXT = "text/plain; version=0.0.4; charset=utf-8"


def _import_prometheus():
    """Seam: patched in tests to prove the ImportError degrade path."""
    import prometheus_client
    return prometheus_client


class _NoopMetric:
    """Accepts every metric call and does nothing. Returns self from labels()
    so chained calls work identically to the real object."""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass


class Metrics:
    """Holds every declared family. Attribute names are stable API for call sites."""

    # NOTE ON `hostname`: deliberately not a label on any family (spec §5, amended
    # 2026-08-03). Prometheus attaches host identity as the `instance` label from
    # the scrape target; a second host label on every series would duplicate it and
    # go stale after a container move. hostname remains a structured-LOG field
    # (STABL-bpsfmoke). Recorded in docs/observability-contract.md so ../continuous
    # does not go looking for a label that is absent on purpose.

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._registry = None
        if enabled:
            try:
                prom = _import_prometheus()
            except ImportError:
                logger.warning(
                    "METRICS_ENABLED is set but prometheus_client is not installed; "
                    "metrics degrade to no-op"
                )
                self.enabled = False
            else:
                self._registry = prom.CollectorRegistry()
                self._declare(prom)
                return
        self._declare_noop()

    # --- family declarations (Task 2 fills these in) ---

    def _declare(self, prom):
        raise NotImplementedError  # Task 2

    def _declare_noop(self):
        raise NotImplementedError  # Task 2

    def render(self) -> tuple[bytes, str]:
        if not self.enabled or self._registry is None:
            return b"", _PLAIN_TEXT
        prom = _import_prometheus()
        return prom.generate_latest(self._registry), _PLAIN_TEXT


_metrics: Optional[Metrics] = None
_metrics_lock = threading.Lock()


def _enabled_from_env() -> bool:
    return os.getenv("METRICS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def get_metrics() -> Metrics:
    """Singleton accessor (get_device_memory pattern).

    Deliberately called at each site rather than cached on the caller: the sites
    are cold (once per job, once per load, once per sample interval), and caching
    a facade on a long-lived object is how patch targets drift out from under
    tests (see the STABL-anxqlxkm torch-binding class of failure).
    """
    global _metrics
    with _metrics_lock:
        if _metrics is None:
            enabled = _enabled_from_env()
            if enabled:
                try:
                    concurrency = int(os.getenv("WEB_CONCURRENCY", "1"))
                except ValueError:
                    concurrency = 1
                if concurrency > 1:
                    logger.warning(
                        "WEB_CONCURRENCY=%s with METRICS_ENABLED: prometheus_client's "
                        "registry is process-local, so /metrics reports whichever worker "
                        "answered the scrape and counters will appear to go backwards. "
                        "Run a single uvicorn worker or take multiprocess mode.",
                        concurrency,
                    )
            _metrics = Metrics(enabled)
        return _metrics


def reset_metrics() -> None:
    """Test seam (reset_device_memory pattern)."""
    global _metrics
    with _metrics_lock:
        _metrics = None
