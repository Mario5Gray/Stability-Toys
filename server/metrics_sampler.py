"""Background sampler — the ONLY thing that fans out to DeviceMemory consumers.

DeviceMemory.snapshot() round-trips every registered consumer, and under
subprocess isolation the worker consumer's pool_stats() is a request/reply over
the child's control pipe. Putting that on the /metrics render path would tie pipe
traffic to scrape cadence and put a blocking round-trip inside an ASGI handler —
the event-loop starvation shape that already makes /status time out during a job.
So: this thread samples on a fixed interval and writes gauges; /metrics renders
those gauges and touches nothing else.

Sampling DURING a job is safe by prior design: the control channel is a separate
pipe from the data pipe precisely because drain_to_subscriber reads the data pipe
concurrently while a job runs.

Imports nothing from backends/ — readers arrive as callables.

Spec: docs/superpowers/specs/2026-08-03-server-observability-seams-design.md §2
"""
import logging
import os
import threading
from typing import Callable, Optional, Protocol, Sequence

from server.metrics import get_metrics

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 15.0


class _ConsumerLike(Protocol):
    """Structural mirror of backends.device_memory.ConsumerMemory.

    Declared here rather than imported so this module stays free of backends/ —
    the readers arrive as callables and only their SHAPE matters. Note pid is
    absent on purpose: it exists on the real dataclass and must never reach a
    label, so leaving it out of the contract makes that explicit.
    """
    label: str
    allocated_bytes: int
    reserved_bytes: int
    stale: bool


class _SnapshotLike(Protocol):
    """Structural mirror of backends.device_memory.DeviceMemorySnapshot."""
    device_uuid: str
    total_bytes: int
    free_bytes: int
    consumers: Sequence[_ConsumerLike]

    @property
    def used_bytes(self) -> int: ...

    @property
    def unattributed_bytes(self) -> int: ...


def _coerce_interval(value, source: str) -> float:
    """Every interval — env or explicit argument — goes through here.

    Two rejections, for different reasons:

    - NOT A NUMBER: a typo in deployment config must not stop the server starting.
    - NOT POSITIVE: `Event.wait(0)` and `wait(<0)` return IMMEDIATELY, so a
      non-positive interval turns `_loop` into a tight loop calling `snapshot_fn`
      as fast as the GIL allows — measured at ~12,000 iterations in 0.2s. Since
      `snapshot_fn` is the child control-pipe round-trip, that is a self-inflicted
      denial of service on the worker, and it presents as "the server is busy"
      rather than as an error.

    Small positive values are deliberately allowed: the test suite runs the
    sampler at 0.01s and sub-second sampling is a legitimate deployment choice.
    """
    if value is None:
        return DEFAULT_INTERVAL_S
    try:
        interval = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not a number; using %.1fs", source, value, DEFAULT_INTERVAL_S)
        return DEFAULT_INTERVAL_S
    if interval <= 0:
        logger.warning(
            "%s=%r must be positive (a non-positive interval spins the sampler "
            "against the worker control pipe); using %.1fs",
            source, value, DEFAULT_INTERVAL_S,
        )
        return DEFAULT_INTERVAL_S
    return interval


class MetricsSampler:
    def __init__(
        self,
        snapshot_fn: Callable[[], _SnapshotLike],
        runtime_stats_fn: Optional[Callable[[], Optional[dict]]] = None,
        interval_s: Optional[float] = None,
    ):
        self._snapshot_fn = snapshot_fn
        self._runtime_stats_fn = runtime_stats_fn
        # Both paths are validated: the explicit argument bypasses the env parser,
        # so guarding only the env var would leave the caller-supplied value
        # unchecked.
        self.interval_s = (
            _coerce_interval(interval_s, "interval_s")
            if interval_s is not None
            else _coerce_interval(
                os.getenv("METRICS_SAMPLE_INTERVAL_S"), "METRICS_SAMPLE_INTERVAL_S")
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not get_metrics().enabled:
            logger.debug("[Metrics] sampler not started: metrics disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="metrics-sampler", daemon=True)
        self._thread.start()
        logger.info("[Metrics] sampler started (interval=%.1fs)", self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def sample_once(self) -> None:
        """One sampling pass. MUST NOT raise: a sampler that dies on one bad read
        stops every device metric permanently and does it silently.

        The two readers are guarded independently — a wedged device must not blank
        the queue gauges, and an unavailable runtime must not blank the device
        gauges.
        """
        met = get_metrics()
        if not met.enabled:
            # Not merely pointless: snapshot_fn round-trips the child's control
            # pipe, so an ungated sampler would keep paying for that on a server
            # that exports nothing.
            return

        try:
            snap = self._snapshot_fn()
        except Exception:
            logger.debug("[Metrics] device snapshot failed", exc_info=True)
            snap = None
        if snap is not None:
            try:
                uuid = snap.device_uuid
                met.device_total_bytes.labels(device_uuid=uuid).set(snap.total_bytes)
                met.device_free_bytes.labels(device_uuid=uuid).set(snap.free_bytes)
                met.device_used_bytes.labels(device_uuid=uuid).set(snap.used_bytes)
                met.device_unattributed_bytes.labels(device_uuid=uuid).set(
                    snap.unattributed_bytes)
                stale = any(c.stale for c in snap.consumers)
                met.device_snapshot_stale.labels(device_uuid=uuid).set(1 if stale else 0)
                for c in snap.consumers:
                    # Keyed on the consumer LABEL only. c.pid is deliberately
                    # dropped: the subprocess handle mints a new pid on every
                    # kill+respawn, so a pid label leaks a dead series per
                    # recovery (spec §5).
                    met.consumer_reserved_bytes.labels(
                        device_uuid=uuid, consumer=c.label).set(c.reserved_bytes)
                    met.consumer_allocated_bytes.labels(
                        device_uuid=uuid, consumer=c.label).set(c.allocated_bytes)
            except Exception:
                logger.debug("[Metrics] device gauge write failed", exc_info=True)

        if self._runtime_stats_fn is None:
            return
        try:
            stats = self._runtime_stats_fn()
            if stats:
                met.queue_depth.set(stats.get("queue_depth", 0))
                met.jobs_in_flight.set(stats.get("jobs_in_flight", 0))
        except Exception:
            logger.debug("[Metrics] runtime stats failed", exc_info=True)
