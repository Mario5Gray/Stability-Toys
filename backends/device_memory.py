"""DeviceMemory — backend-neutral device memory accounting.

Torch-free contract: driver truth (NVML-by-UUID | psutil) merged with a
registry of per-consumer framework pool stats. See
docs/superpowers/specs/2026-07-28-device-memory-design.md (STABL-hjldxurg).

Hard invariants (spec §2.2):
- available_for_load() MUST NOT fan out (liveness property).
- reclaim() is soft trim for live consumers only — NOT recovery (that is
  WorkerHandle kill+respawn), NOT teardown (that flushes inline in unload()).
- Registration.close() is idempotent + crash-safe; the parent is sole closer.
- pool_stats() MUST return stale=False; only snapshot()'s fan-out sets True.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

POOL_STATS_TIMEOUT_S = 0.5
_FANOUT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="devmem-fanout")


class MemoryTopology(Enum):
    DISCRETE = "discrete"   # separate device pool (CUDA)
    UNIFIED = "unified"     # shares host RAM (MLX, RKNN, CPU)
    UNKNOWN = "unknown"     # NullDeviceMemory degrade


@dataclass(frozen=True)
class ConsumerMemory:            # one GPU consumer's slice
    label: str                   # "worker", "superres", ...
    pid: Optional[int]
    allocated_bytes: int         # framework pool: live allocations
    reserved_bytes: int          # framework pool: full held pool
    stale: bool = False          # snapshot-substituted last-known (spec §2.3)


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    device_uuid: str             # ties to STABL-cchxvuhs
    topology: MemoryTopology
    total_bytes: int             # device total (DISCRETE) | host total (UNIFIED)
    free_bytes: int              # driver-truth free (NVML | psutil)
    consumers: tuple[ConsumerMemory, ...]

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    @property
    def unattributed_bytes(self) -> int:
        # Driver-observed usage not attributed to any registered consumer's
        # pool: CUDA contexts, non-torch workspaces (cuDNN/cuBLAS/xformers),
        # unregistered/other-process usage. RESERVED, not allocated: a
        # consumer's cached-but-free pool blocks belong to that consumer;
        # subtracting allocated would mislabel torch's cache as unexplained.
        # On DISCRETE single-consumer (enigma) ~= the CUDA context.
        # On UNIFIED this is host RAM incl. OS + unrelated processes:
        # INFORMATIONAL ONLY — never alert on it.
        return max(0, self.used_bytes - sum(c.reserved_bytes for c in self.consumers))


class MemoryConsumer(Protocol):
    def pool_stats(self) -> ConsumerMemory: ...   # MUST return stale=False
    def reclaim(self) -> None: ...                # soft pool-trim; no-op where N/A


class Registration(Protocol):
    def close(self) -> None: ...   # idempotent + crash-safe; deregisters


class DeviceMemory(Protocol):
    topology: MemoryTopology
    device_uuid: str

    @property
    def device_name(self) -> str: ...
    def register(self, c: MemoryConsumer) -> Registration: ...
    def snapshot(self) -> DeviceMemorySnapshot: ...       # fresh; refreshes cache
    def cached_snapshot(self) -> DeviceMemorySnapshot: ...  # last computed; NO fan-out
    def available_for_load(self) -> int: ...              # cheap; NO fan-out
    def reclaim(self) -> None: ...                        # fan out to live consumers


# --- registry internals --------------------------------------------------------

class _RegistrationImpl:
    """Idempotent + crash-safe close. Never touches the consumer — the parent
    closes what a SIGKILLed corpse never could (spec §2.2)."""

    def __init__(self, remover):
        self._remover = remover
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._remover()
        except Exception:
            logger.warning("[DeviceMemory] registration remover failed", exc_info=True)


class _ConsumerRegistry:
    """Shared consumer-registry + bounded fan-out + last-known cache, used by
    the real providers. snapshot() stores by rebinding a single reference to a
    new frozen DeviceMemorySnapshot — atomic under the GIL; concurrent readers
    see whole-old or whole-new, never torn. No lock on the read path."""

    def __init__(self, owner):
        self._owner = owner  # provides _driver_free_total(), device_uuid, topology
        self._consumers: list = []
        self._lock = threading.Lock()
        self._last_known: dict[int, ConsumerMemory] = {}
        self._cached: Optional[DeviceMemorySnapshot] = None

    def register(self, consumer) -> Registration:
        with self._lock:
            self._consumers.append(consumer)
        return _RegistrationImpl(lambda: self._remove(consumer))

    def _remove(self, consumer) -> None:
        with self._lock:
            if consumer in self._consumers:
                self._consumers.remove(consumer)
        self._last_known.pop(id(consumer), None)

    def snapshot(self) -> DeviceMemorySnapshot:
        free_b, total_b = self._owner._driver_free_total()
        with self._lock:
            consumers = list(self._consumers)
        stats = tuple(self._read_consumer(c) for c in consumers)
        snap = DeviceMemorySnapshot(
            device_uuid=self._owner.device_uuid,
            topology=self._owner.topology,
            total_bytes=total_b,
            free_bytes=free_b,
            consumers=stats,
        )
        residual = snap.used_bytes - sum(c.reserved_bytes for c in snap.consumers)
        if residual < 0:
            logger.debug("[DeviceMemory] consumer over-report: residual=%d bytes", residual)
        self._cached = snap  # atomic rebind
        return snap

    def _read_consumer(self, consumer) -> ConsumerMemory:
        fut = _FANOUT_EXECUTOR.submit(consumer.pool_stats)
        try:
            cm = fut.result(timeout=POOL_STATS_TIMEOUT_S)
        except Exception:
            label = getattr(consumer, "label", type(consumer).__name__)
            last = self._last_known.get(id(consumer))
            if last is None:
                last = ConsumerMemory(label=label, pid=None, allocated_bytes=0,
                                      reserved_bytes=0)
            logger.warning("[DeviceMemory] pool_stats failed for %s; "
                           "substituting last-known with stale=True", label, exc_info=True)
            return replace(last, stale=True)
        self._last_known[id(consumer)] = cm
        return cm

    def cached_snapshot(self) -> DeviceMemorySnapshot:
        if self._cached is None:
            self._cached = self.snapshot()  # pre-seed: driver truth, consumers=()
        return self._cached

    def reclaim(self) -> None:
        with self._lock:
            consumers = list(self._consumers)
        for c in consumers:
            try:
                c.reclaim()
            except Exception:
                logger.warning("[DeviceMemory] consumer reclaim failed", exc_info=True)


# --- providers ---------------------------------------------------------------

def _import_pynvml():
    import pynvml
    return pynvml


def _import_psutil():
    import psutil
    return psutil


class NullDeviceMemory:
    """Degrade, never borrow: NVML broken on a CUDA host must not fall back to
    mem_get_info() through a consumer's context (spec §3)."""
    topology = MemoryTopology.UNKNOWN
    device_uuid = "unknown"

    @property
    def device_name(self) -> str:
        return "Unknown"

    def register(self, c: MemoryConsumer) -> Registration:
        return _RegistrationImpl(lambda: None)

    def snapshot(self) -> DeviceMemorySnapshot:
        return DeviceMemorySnapshot(self.device_uuid, self.topology, 0, 0, ())

    def cached_snapshot(self) -> DeviceMemorySnapshot:
        return self.snapshot()

    def available_for_load(self) -> int:
        return 0

    def reclaim(self) -> None:
        return None


class UnifiedDeviceMemory:
    """UNIFIED topology: host RAM is the device pool (MLX/RKNN/CPU)."""
    topology = MemoryTopology.UNIFIED

    def __init__(self, _psutil=None):
        self._psutil = _psutil or _import_psutil()
        self.device_uuid = "host"
        import platform
        self._device_name = platform.node() or "unified-host"
        self._core = _ConsumerRegistry(self)

    @property
    def device_name(self) -> str:
        return self._device_name

    def _driver_free_total(self) -> tuple[int, int]:
        vm = self._psutil.virtual_memory()
        return int(vm.available), int(vm.total)

    def register(self, c): return self._core.register(c)
    def snapshot(self): return self._core.snapshot()
    def cached_snapshot(self): return self._core.cached_snapshot()
    def available_for_load(self) -> int: return self._driver_free_total()[0]
    def reclaim(self) -> None: return self._core.reclaim()


# --- singleton ---------------------------------------------------------------

_device_memory: Optional[DeviceMemory] = None
_device_memory_lock = threading.Lock()


def get_device_memory(uuid: Optional[str] = None) -> DeviceMemory:
    """Singleton accessor (get_worker_pool pattern). Provider chosen once.
    uuid=None resolves to the single/default device (v1); multi-device UUID
    resolution is STABL-cchxvuhs behind this same accessor."""
    global _device_memory
    with _device_memory_lock:
        if _device_memory is None:
            _device_memory = _select_provider(uuid)
        return _device_memory


def reset_device_memory() -> None:
    global _device_memory
    with _device_memory_lock:
        _device_memory = None


def _select_provider(uuid: Optional[str]) -> DeviceMemory:
    try:
        nvml = _import_pynvml()
    except ImportError:
        nvml = None
    if nvml is not None:
        try:
            nvml.nvmlInit()
            resolved = uuid or _first_device_uuid(nvml)
            dm = CudaDeviceMemory(resolved, _pynvml=nvml)  # defined in Task 3
            logger.info("[DeviceMemory] CUDA provider, device %s", resolved)
            return dm
        except Exception:
            logger.warning("[DeviceMemory] NVML present but unusable; degrading to Null",
                           exc_info=True)
            return NullDeviceMemory()
    try:
        _import_psutil()
        logger.info("[DeviceMemory] unified provider (host RAM)")
        return UnifiedDeviceMemory()
    except ImportError:
        logger.warning("[DeviceMemory] no NVML, no psutil; Null provider")
        return NullDeviceMemory()


def _first_device_uuid(nvml) -> str:
    count = nvml.nvmlDeviceGetCount()
    if count > 1:
        logger.warning("[DeviceMemory] v1 is single-device; using index 0 of %d", count)
    h = nvml.nvmlDeviceGetHandleByIndex(0)
    u = nvml.nvmlDeviceGetUUID(h)
    return u.decode() if isinstance(u, bytes) else str(u)
