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

    # --- family declarations ---

    def _declare(self, prom):
        C, G, H = prom.Counter, prom.Gauge, prom.Histogram
        r = self._registry

        # --- Governor: queue pressure ---
        self.queue_depth = G(
            "st_governor_queue_depth", "Jobs currently queued", registry=r)
        self.jobs_in_flight = G(
            "st_governor_jobs_in_flight", "Jobs currently executing", registry=r)
        self.job_queue_wait_seconds = H(
            "st_governor_job_queue_wait_seconds",
            "Seconds between enqueue and execution start",
            ["mode"], buckets=QUEUE_WAIT_BUCKETS, registry=r)

        # --- Governor: execution cost ---
        self.job_execution_seconds = H(
            "st_governor_job_execution_seconds",
            "Seconds spent executing a job",
            ["mode"], buckets=EXECUTION_BUCKETS, registry=r)
        self.mode_load_seconds = H(
            "st_governor_mode_load_seconds",
            "Seconds to load a mode's worker",
            ["mode"], buckets=MODE_LOAD_BUCKETS, registry=r)

        # --- Governor: failure paths ---
        self.job_terminal_total = C(
            "st_governor_job_terminal_total",
            "Jobs by terminal outcome (ok|cancelled|oom|error)",
            ["mode", "outcome"], registry=r)
        self.wait_expired_total = C(
            "st_governor_wait_expired_total",
            "Waiters that exceeded their budget (admission|execution)",
            ["budget"], registry=r)
        self.worker_recovery_total = C(
            "st_governor_worker_recovery_total",
            "Worker kill+respawn recoveries by cause (oom|dead)",
            ["reason"], registry=r)

        # --- Governor: churn ---
        self.mode_switch_total = C(
            "st_governor_mode_switch_total",
            "Mode switches, labelled by TARGET mode",
            ["mode"], registry=r)
        self.demand_reload_total = C(
            "st_governor_demand_reload_total",
            "Reloads from a retained snapshot after idle eviction",
            ["mode"], registry=r)
        self.unload_total = C(
            "st_governor_unload_total", "Model unloads by reason",
            ["mode", "reason"], registry=r)

        # --- Governor: authority ---
        self.mode_active = G(
            "st_governor_mode_active",
            "1 for the loaded mode, 0 for every other configured mode",
            ["mode"], registry=r)
        self.resolution_epoch = G(
            "st_governor_resolution_epoch", "Current resolution epoch", registry=r)

        # --- DeviceMemory: capacity truth ---
        self.device_total_bytes = G(
            "st_device_total_bytes", "Device total bytes",
            ["device_uuid"], registry=r)
        self.device_free_bytes = G(
            "st_device_free_bytes", "Driver-truth free bytes",
            ["device_uuid"], registry=r)
        self.device_used_bytes = G(
            "st_device_used_bytes", "Driver-truth used bytes",
            ["device_uuid"], registry=r)
        self.device_unattributed_bytes = G(
            "st_device_unattributed_bytes",
            "Used bytes not attributed to any registered consumer pool "
            "(CUDA contexts, non-torch workspaces, other processes)",
            ["device_uuid"], registry=r)
        self.consumer_reserved_bytes = G(
            "st_consumer_reserved_bytes", "Per-consumer framework pool, reserved",
            ["device_uuid", "consumer"], registry=r)
        self.consumer_allocated_bytes = G(
            "st_consumer_allocated_bytes", "Per-consumer framework pool, allocated",
            ["device_uuid", "consumer"], registry=r)
        self.device_snapshot_stale = G(
            "st_device_snapshot_stale",
            "1 when the last snapshot substituted last-known values after a "
            "consumer fan-out timeout (i.e. a consumer is not answering)",
            ["device_uuid"], registry=r)

    def _declare_noop(self):
        for name in (
            "queue_depth", "jobs_in_flight", "job_queue_wait_seconds",
            "job_execution_seconds", "job_terminal_total", "wait_expired_total",
            "mode_load_seconds", "mode_switch_total", "demand_reload_total",
            "unload_total", "worker_recovery_total", "mode_active",
            "resolution_epoch", "device_total_bytes", "device_free_bytes",
            "device_used_bytes", "device_unattributed_bytes",
            "consumer_reserved_bytes", "consumer_allocated_bytes",
            "device_snapshot_stale",
        ):
            setattr(self, name, _NoopMetric())

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
