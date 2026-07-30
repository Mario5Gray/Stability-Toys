# DeviceMemory v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace three ad-hoc direct-torch VRAM readers (ModelRegistry, WorkerHealth, Governor status) with one backend-neutral DeviceMemory authority: NVML-by-UUID driver truth merged with a consumer registry, per `docs/superpowers/specs/2026-07-28-device-memory-design.md`.

**Architecture:** New torch-free leaf module `backends/device_memory.py` (contract + 3 providers + singleton). ModelRegistry becomes a pure view over it; `InProcessWorkerHandle` injects it (health + Registration lifecycle); Governor reads status/load-measurement through it. v1 = one in-proc consumer ("worker"), single device.

**Tech Stack:** Python 3, pynvml (nvidia-ml-py, already in requirements.txt unpinned — Task 9 pins), psutil (already requirements.txt:17), pytest. Conda env: `conda activate stability-toys`, use `python` (not `python3`).

**FP:** STABL-hjldxurg (child of STABL-nvmieaxh). Branch: `feat/device-memory`.

**Spec-locked invariants (do not violate):**
1. `available_for_load()` MUST NOT fan out (driver truth only — liveness property).
2. `reclaim()` ≠ recovery AND ≠ teardown (soft trim, live consumers only; teardown flushes inline).
3. `Registration.close()` idempotent + crash-safe; parent (WorkerHandle) is sole closer.
4. `pool_stats()` MUST return `stale=False`; only `snapshot()` fan-out sets `stale=True`.
5. Registry reads via `cached_snapshot()`/`available_for_load()` — never `pool_stats()`, never fresh `snapshot()`.
6. Load-time VRAM measurement is the ONE exception: it reads a fresh `snapshot()` (not admission path).

**Post-gate additive contract extensions (approved in plan, additive only):**
- `DeviceMemory.device_name` property (Cuda→NVML name; Unified→host node; Null→"Unknown"). Replaces the registry's context-burning `torch.cuda.get_device_properties` detection (`model_registry.py:111-122`).
- `MemoryTopology.UNKNOWN = "unknown"` third member, reported by NullDeviceMemory only (spec §3 table says Null = "0 / UNKNOWN").

---

## File Structure

| File | Responsibility |
|---|---|
| `backends/device_memory.py` (NEW) | Torch-free contract (MemoryTopology, ConsumerMemory, DeviceMemorySnapshot, MemoryConsumer/Registration/DeviceMemory protocols), `_BaseDeviceMemory` (registry + bounded fan-out + cache), CudaDeviceMemory / UnifiedDeviceMemory / NullDeviceMemory, WorkerMemoryConsumer, `get_device_memory()`/`reset_device_memory()` singleton |
| `backends/model_registry.py` | ModelRegistry: drop all torch, delegate VRAM reads to DeviceMemory (pure view) |
| `backends/worker_handle.py` | InProcessWorkerHandle: inject DeviceMemory, health() rewire, register on start(), close on unload()/stop() |
| `backends/governor.py` | Inject DeviceMemory (param + both handle-construction sites); `_build_runtime_status` :513-514 swap + `stale` field; `_load_mode` :330-331/:348-352 fresh-snapshot measurement; `_cleanup_vram` :507 + `unload_current_model` :794 → `device_memory.reclaim()` |
| `backends/worker_pool.py` | Facade: pass-through `device_memory` param to Governor |
| `tests/test_device_memory.py` (NEW) | Contract math, providers, fan-out/timeout/stale, consumer, singleton selection |
| `tests/test_model_registry.py` | Repoint from torch-mock to DeviceMemory stub (mrrrbmjp vaccine) |
| `tests/test_worker_handle.py` | Registration lifecycle + health rewire |
| `requirements.txt` | Pin `nvidia-ml-py` (currently unpinned, line 6) |

---

### Task 1: Contract types + snapshot math

**Files:**
- Create: `backends/device_memory.py`
- Test: `tests/test_device_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_device_memory.py
"""DeviceMemory contract tests. Torch-free: this file must never import torch."""
from backends.device_memory import (
    ConsumerMemory,
    DeviceMemorySnapshot,
    MemoryTopology,
)


def _snap(consumers=(), total=24 * 1024**3, free=10 * 1024**3):
    return DeviceMemorySnapshot(
        device_uuid="GPU-test",
        topology=MemoryTopology.DISCRETE,
        total_bytes=total,
        free_bytes=free,
        consumers=consumers,
    )


def test_used_bytes_derived():
    assert _snap().used_bytes == 14 * 1024**3


def test_unattributed_subtracts_reserved_not_allocated():
    # reserved=5GB, allocated=3GB: the cached-but-free 2GB belongs to the
    # consumer; unattributed must be computed against reserved.
    c = ConsumerMemory(label="worker", pid=123, allocated_bytes=3 * 1024**3,
                       reserved_bytes=5 * 1024**3)
    assert _snap(consumers=(c,)).unattributed_bytes == 9 * 1024**3


def test_unattributed_clamps_at_zero():
    c = ConsumerMemory(label="worker", pid=1, allocated_bytes=20 * 1024**3,
                       reserved_bytes=20 * 1024**3)  # over-report vs used=14GB
    assert _snap(consumers=(c,)).unattributed_bytes == 0


def test_stale_defaults_false():
    c = ConsumerMemory(label="worker", pid=1, allocated_bytes=0, reserved_bytes=0)
    assert c.stale is False


def test_module_is_torch_free():
    import sys
    import backends.device_memory as dm
    assert "torch" not in vars(dm), "device_memory must not bind torch at module scope"
    assert "torch" not in [m for m in sys.modules if m == "torch"] or True  # env may have torch; scope check above is the invariant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate stability-toys && python -m pytest tests/test_device_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.device_memory'`

- [ ] **Step 3: Write minimal implementation**

```python
# backends/device_memory.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backends/device_memory.py tests/test_device_memory.py
git commit -m "feat(device-memory): contract types + snapshot math (STABL-hjldxurg)

MemoryTopology/ConsumerMemory/DeviceMemorySnapshot/protocols. unattributed_bytes
derived @property (reserved-not-allocated, clamped), stale defaults False.
Torch-free module."
```

---

### Task 2: Null + Unified providers + singleton selection

**Files:**
- Modify: `backends/device_memory.py` (append)
- Test: `tests/test_device_memory.py`

- [ ] **Step 1: Write the failing test**

```python
from backends.device_memory import (
    NullDeviceMemory, UnifiedDeviceMemory, get_device_memory, reset_device_memory,
)


class _FakePsutil:
    class _VM:
        available = 8 * 1024**3
        total = 32 * 1024**3
    @staticmethod
    def virtual_memory():
        return _FakePsutil._VM()


def test_null_degrades_to_zero_unknown():
    dm = NullDeviceMemory()
    snap = dm.snapshot()
    assert snap.total_bytes == 0 and snap.free_bytes == 0
    assert snap.topology == MemoryTopology.UNKNOWN
    assert dm.available_for_load() == 0
    assert dm.device_name == "Unknown"
    reg = dm.register(object())  # inert, returns a Registration
    reg.close(); reg.close()     # idempotent


def test_unified_reads_psutil_host_ram():
    dm = UnifiedDeviceMemory(_psutil=_FakePsutil())
    assert dm.available_for_load() == 8 * 1024**3
    snap = dm.cached_snapshot()  # pre-seeded at construction
    assert snap.total_bytes == 32 * 1024**3
    assert snap.topology == MemoryTopology.UNIFIED


def test_singleton_selects_unified_when_no_nvml(monkeypatch):
    reset_device_memory()
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "pynvml":
            raise ImportError("no nvml on this host")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    dm = get_device_memory()
    assert isinstance(dm, (UnifiedDeviceMemory, NullDeviceMemory))  # psutil present → Unified
    reset_device_memory()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: FAIL — `ImportError: cannot import name 'NullDeviceMemory'`

- [ ] **Step 3: Write minimal implementation** (append to `backends/device_memory.py`)

```python
# --- providers ---------------------------------------------------------------

def _import_pynvml():
    import pynvml
    return pynvml


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


def _import_psutil():
    import psutil
    return psutil


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
            dm = CudaDeviceMemory(resolved, _pynvml=nvml)
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
```

Note: `_ConsumerRegistry` and `_RegistrationImpl` are defined in Task 4 — for Task 3 to land green, define them now as part of this step's append (UnifiedDeviceMemory references `_ConsumerRegistry` at construction). Place directly above the providers section:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add backends/device_memory.py tests/test_device_memory.py
git commit -m "feat(device-memory): Null/Unified providers + singleton (STABL-hjldxurg)

NullDeviceMemory degrades without borrowing a context. UnifiedDeviceMemory reads
psutil host RAM. get_device_memory() singleton with topology selection;
reset_device_memory() for tests. _ConsumerRegistry: bounded fan-out, last-known
stale substitution, atomic cache rebind, pre-seed on first cached_snapshot()."
```

---

### Task 3: CudaDeviceMemory — NVML by UUID

**Files:**
- Modify: `backends/device_memory.py` (append, providers section)
- Test: `tests/test_device_memory.py`

- [ ] **Step 1: Write the failing test**

```python
from backends.device_memory import CudaDeviceMemory, ConsumerMemory


class _FakeNvml:
    """Fake NVML boundary. Provider unit tests stub THIS (the injected module),
    which is the one allowed pynvml seam; consumers never see pynvml."""
    class _Info:
        free = 20 * 1024**3
        total = 24 * 1024**3
    def __init__(self):
        self.inited = False
    def nvmlInit(self):
        self.inited = True
    def nvmlDeviceGetHandleByUUID(self, uuid):
        assert uuid == "GPU-fake-uuid"
        return "HANDLE"
    def nvmlDeviceGetMemoryInfo(self, handle):
        assert handle == "HANDLE"
        return self._Info()
    def nvmlDeviceGetName(self, handle):
        return b"NVIDIA GeForce RTX 3090"


def test_cuda_provider_nvml_by_uuid():
    nvml = _FakeNvml()
    dm = CudaDeviceMemory("GPU-fake-uuid", _pynvml=nvml)
    assert nvml.inited
    assert dm.device_uuid == "GPU-fake-uuid"
    assert dm.topology == MemoryTopology.DISCRETE
    assert dm.device_name == "NVIDIA GeForce RTX 3090"
    assert dm.available_for_load() == 20 * 1024**3
    snap = dm.cached_snapshot()  # pre-seeded at construction
    assert snap.total_bytes == 24 * 1024**3
    assert snap.free_bytes == 20 * 1024**3
    assert snap.consumers == ()


def test_available_for_load_does_not_fan_out():
    nvml = _FakeNvml()
    dm = CudaDeviceMemory("GPU-fake-uuid", _pynvml=nvml)

    class _ExplodingConsumer:
        def pool_stats(self):
            raise AssertionError("available_for_load() MUST NOT fan out")
        def reclaim(self):
            pass

    dm.register(_ExplodingConsumer())
    assert dm.available_for_load() == 20 * 1024**3  # no consumer round-trip
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: FAIL — `ImportError: cannot import name 'CudaDeviceMemory'`

- [ ] **Step 3: Write minimal implementation** (append to providers section)

```python
class CudaDeviceMemory:
    """DISCRETE provider: driver truth via NVML keyed by UUID. NVML is a driver
    query — it initializes NO CUDA context in the caller (spec §3)."""
    topology = MemoryTopology.DISCRETE

    def __init__(self, device_uuid: str, _pynvml=None):
        self.device_uuid = device_uuid
        self._nvml = _pynvml or _import_pynvml()
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByUUID(device_uuid)
        name = self._nvml.nvmlDeviceGetName(self._handle)
        self._device_name = name.decode() if isinstance(name, bytes) else str(name)
        self._core = _ConsumerRegistry(self)
        self._core.snapshot()  # pre-seed: driver truth from startup, consumers=()

    @property
    def device_name(self) -> str:
        return self._device_name

    def _driver_free_total(self) -> tuple[int, int]:
        info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        return int(info.free), int(info.total)

    def register(self, c): return self._core.register(c)
    def snapshot(self): return self._core.snapshot()
    def cached_snapshot(self): return self._core.cached_snapshot()
    def available_for_load(self) -> int: return self._driver_free_total()[0]
    def reclaim(self) -> None: return self._core.reclaim()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add backends/device_memory.py tests/test_device_memory.py
git commit -m "feat(device-memory): CudaDeviceMemory NVML-by-UUID (STABL-hjldxurg)

Driver truth via nvmlDeviceGetHandleByUUID + nvmlDeviceGetMemoryInfo — no CUDA
context in the caller. UUID keying seeds STABL-cchxvuhs. Pre-seeded cache.
available_for_load() proven no-fan-out by exploding-consumer test."
```

---

### Task 4: Fan-out semantics — timeout → stale, reclaim, Registration

**Files:**
- Modify: `backends/device_memory.py` (no new code expected — semantics landed in Task 2's `_ConsumerRegistry`; this task proves them)
- Test: `tests/test_device_memory.py`

- [ ] **Step 1: Write the failing test**

```python
import time
from backends.device_memory import POOL_STATS_TIMEOUT_S


def _cuda():
    from backends.device_memory import CudaDeviceMemory
    return CudaDeviceMemory("GPU-fake-uuid", _pynvml=_FakeNvml())


class _GoodConsumer:
    label = "worker"
    def __init__(self, reserved=5 * 1024**3):
        self._reserved = reserved
    def pool_stats(self):
        return ConsumerMemory(label=self.label, pid=123,
                              allocated_bytes=3 * 1024**3,
                              reserved_bytes=self._reserved)
    def reclaim(self):
        pass


def test_snapshot_merges_driver_truth_and_consumer_pools():
    dm = _cuda()
    dm.register(_GoodConsumer())
    snap = dm.snapshot()
    assert len(snap.consumers) == 1
    assert snap.consumers[0].stale is False
    # used = 24-20 = 4GB; consumer reserved = 5GB → residual -1GB → clamps to 0
    assert snap.unattributed_bytes == 0


def test_registration_close_deregisters_and_is_idempotent():
    dm = _cuda()
    reg = dm.register(_GoodConsumer())
    assert len(dm.snapshot().consumers) == 1
    reg.close()
    assert len(dm.snapshot().consumers) == 0
    reg.close()  # double-close = no-op
    assert len(dm.snapshot().consumers) == 0


def test_hung_consumer_substitutes_last_known_stale():
    dm = _cuda()

    class _Wedgie(_GoodConsumer):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def pool_stats(self):
            self.calls += 1
            if self.calls == 1:
                return super().pool_stats()
            time.sleep(POOL_STATS_TIMEOUT_S + 1.0)  # wedge
            raise AssertionError("unreachable")

    dm.register(_Wedgie())
    fresh = dm.snapshot()
    assert fresh.consumers[0].stale is False
    assert fresh.consumers[0].reserved_bytes == 5 * 1024**3
    stale = dm.snapshot()  # wedged → last-known with stale=True, never omitted
    assert len(stale.consumers) == 1
    assert stale.consumers[0].stale is True
    assert stale.consumers[0].reserved_bytes == 5 * 1024**3  # last-known, not zero


def test_reclaim_fans_out_to_live_consumers():
    dm = _cuda()

    class _Reclaimer(_GoodConsumer):
        def __init__(self):
            super().__init__()
            self.reclaimed = 0
        def reclaim(self):
            self.reclaimed += 1

    c = _Reclaimer()
    reg = dm.register(c)
    dm.reclaim()
    assert c.reclaimed == 1
    reg.close()
    dm.reclaim()  # closed consumer not called
    assert c.reclaimed == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: all 4 FAIL (import error / missing names if registry internals not yet importable) — or if Task 2/3 code already landed the semantics, run to verify they PASS immediately (that is the TDD "test first proves the contract" case; acceptable here because `_ConsumerRegistry` was written against this exact spec).

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: 14 passed. If any fail, fix `_ConsumerRegistry` until green (do not weaken tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_device_memory.py backends/device_memory.py
git commit -m "test(device-memory): prove fan-out semantics (STABL-hjldxurg)

Merge driver truth + pools; idempotent close; hung consumer -> last-known
stale=True (never omitted, would inflate unattributed_bytes); reclaim fans out
to live consumers only."
```

---

### Task 5: WorkerMemoryConsumer adapter

**Files:**
- Modify: `backends/device_memory.py` (append)
- Test: `tests/test_device_memory.py`

- [ ] **Step 1: Write the failing test**

```python
from backends.device_memory import WorkerMemoryConsumer


def test_worker_consumer_reports_torch_pool_stale_false(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    torch_mock = MagicMock()
    torch_mock.cuda.memory_allocated.return_value = 3 * 1024**3
    torch_mock.cuda.memory_reserved.return_value = 5 * 1024**3
    monkeypatch.setitem(sys.modules, "torch", torch_mock)

    c = WorkerMemoryConsumer(worker=object())
    cm = c.pool_stats()
    assert cm.label == "worker"
    assert cm.allocated_bytes == 3 * 1024**3
    assert cm.reserved_bytes == 5 * 1024**3
    assert cm.stale is False  # consumers can never self-declare staleness
    assert cm.pid is not None


def test_worker_consumer_reclaim_calls_empty_cache(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", torch_mock)

    WorkerMemoryConsumer(worker=object()).reclaim()
    torch_mock.cuda.empty_cache.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: FAIL — `ImportError: cannot import name 'WorkerMemoryConsumer'`

- [ ] **Step 3: Write minimal implementation** (append to `backends/device_memory.py`)

```python
class WorkerMemoryConsumer:
    """Adapter: the generation worker as a DeviceMemory consumer (spec §6).
    torch imported lazily inside methods — the module stays torch-free."""

    def __init__(self, worker, label: str = "worker"):
        self._worker = worker
        self.label = label

    def pool_stats(self) -> ConsumerMemory:
        import torch
        return ConsumerMemory(
            label=self.label,
            pid=os.getpid(),
            allocated_bytes=int(torch.cuda.memory_allocated()),
            reserved_bytes=int(torch.cuda.memory_reserved()),
            stale=False,
        )

    def reclaim(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_device_memory.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add backends/device_memory.py tests/test_device_memory.py
git commit -m "feat(device-memory): WorkerMemoryConsumer adapter (STABL-hjldxurg)

pool_stats() reports torch pool with stale=False always (consumers cannot
self-declare staleness). reclaim() = empty_cache, lazy torch import keeps
module torch-free."
```

---

### Task 6: ModelRegistry becomes a pure view

**Files:**
- Modify: `backends/model_registry.py`
- Test: `tests/test_model_registry.py`

- [ ] **Step 1: Write the failing test** (rewrite `tests/test_model_registry.py`'s fixture core — the mrrrbmjp vaccine: stub the DeviceMemory interface, never pynvml/torch call sites)

Replace the module-level torch sys.modules mock (lines ~8-21) and the `mock_cuda`/`registry` fixtures with:

```python
"""Tests for ModelRegistry as a pure DeviceMemory view."""
import pytest
from unittest.mock import Mock

from backends.device_memory import (
    ConsumerMemory, DeviceMemorySnapshot, MemoryTopology,
)
from backends.model_registry import ModelRegistry


def _snap(consumers=(), total=24 * 1024**3, free=10 * 1024**3):
    return DeviceMemorySnapshot(
        device_uuid="GPU-test", topology=MemoryTopology.DISCRETE,
        total_bytes=total, free_bytes=free, consumers=consumers,
    )


@pytest.fixture
def device_memory():
    dm = Mock()
    dm.device_name = "NVIDIA GeForce RTX 3090"
    dm.cached_snapshot.return_value = _snap()
    dm.available_for_load.return_value = 10 * 1024**3
    return dm


@pytest.fixture
def registry(device_memory):
    return ModelRegistry(device_memory=device_memory)


class TestDeviceMemoryView:
    def test_total_from_cached_snapshot(self, registry, device_memory):
        assert registry.get_total_vram() == 24 * 1024**3
        device_memory.cached_snapshot.assert_called()

    def test_available_from_available_for_load(self, registry):
        assert registry.get_available_vram() == 10 * 1024**3

    def test_allocated_reserved_from_worker_consumer_entry(self, registry, device_memory):
        worker = ConsumerMemory(label="worker", pid=1,
                                allocated_bytes=3 * 1024**3,
                                reserved_bytes=5 * 1024**3)
        device_memory.cached_snapshot.return_value = _snap(consumers=(worker,))
        assert registry.get_allocated_vram() == 3 * 1024**3
        assert registry.get_reserved_vram() == 5 * 1024**3
        assert registry.get_used_vram() == 5 * 1024**3  # reserved alias

    def test_no_worker_consumer_reads_zero(self, registry):
        assert registry.get_allocated_vram() == 0
        assert registry.get_reserved_vram() == 0

    def test_registry_never_calls_fresh_snapshot(self, registry, device_memory):
        registry.get_total_vram()
        registry.get_allocated_vram()
        registry.get_reserved_vram()
        device_memory.snapshot.assert_not_called()  # no fan-out from the view

    def test_registry_has_no_torch(self):
        import backends.model_registry as mr
        assert not hasattr(mr, "torch"), "registry must not import torch"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_registry.py -v`
Expected: FAIL — `TypeError: ModelRegistry.__init__() got an unexpected keyword argument 'device_memory'` (and collection errors from the old torch-mock fixtures once removed — rewrite the whole file around the new fixtures; keep the registration/lifecycle/estimate tests that don't touch VRAM sourcing, adapting them to the `registry` fixture above).

- [ ] **Step 3: Write minimal implementation**

In `backends/model_registry.py`:

1. Delete the torch import block (lines 16-19) and the GPU-detection block in `__init__` (lines 104-122, incl. `self._device_index`, `self._total_vram`, `self._device_name`).
2. New `__init__`:

```python
    def __init__(self, device_memory=None) -> None:
        """Initialize model registry as a pure view over DeviceMemory."""
        self._loaded: Dict[str, LoadedModel] = {}
        self._lock = Lock()
        if device_memory is None:
            from backends.device_memory import get_device_memory
            device_memory = get_device_memory()
        self._dm = device_memory
        logger.info(
            "[ModelRegistry] Device: %s (%.2f GB)",
            self._dm.device_name,
            self._dm.cached_snapshot().total_bytes / 1024**3,
        )
```

3. Replace the four VRAM readers (keep every other method byte-identical):

```python
    def _worker_entry(self):
        return next(
            (c for c in self._dm.cached_snapshot().consumers if c.label == "worker"),
            None,
        )

    def get_reserved_vram(self) -> int:
        """Allocator-reserved VRAM of the worker consumer (last snapshot)."""
        entry = self._worker_entry()
        return entry.reserved_bytes if entry else 0

    def get_used_vram(self) -> int:
        """Backward-compatible alias for allocator-reserved VRAM."""
        return self.get_reserved_vram()

    def get_allocated_vram(self) -> int:
        """Live torch allocations of the worker consumer (last snapshot)."""
        entry = self._worker_entry()
        return entry.allocated_bytes if entry else 0

    def get_total_vram(self) -> int:
        """Total device VRAM in bytes (last snapshot; correct from startup
        because the provider pre-seeds its cache)."""
        return self._dm.cached_snapshot().total_bytes

    def get_available_vram(self) -> int:
        """Driver-truth free VRAM — NVML free, no consumer fan-out."""
        return self._dm.available_for_load()
```

4. `can_fit`: delete the `if torch is None or not torch.cuda.is_available(): return False` guard (lines 252-253); the rest is unchanged (`available = self.get_available_vram()`; on Null provider available=0 → `estimated < 0 + 0` is False — same outcome as the old guard).
5. `get_vram_stats`: change `"device": self._device_name` to `"device": self._dm.device_name`. Everything else in the method flows through the view methods above.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_registry.py -v`
Expected: all passed (new view tests + surviving lifecycle/estimate tests)

- [ ] **Step 5: Commit**

```bash
git add backends/model_registry.py tests/test_model_registry.py
git commit -m "feat(device-memory): ModelRegistry becomes a pure DeviceMemory view (STABL-hjldxurg)

Drops all torch from the registry (incl. the context-burning
get_device_properties detection). get_total/get_available -> cached_snapshot/
available_for_load; get_allocated/get_reserved -> worker entry in cached
consumers[]; no fresh snapshot(), no fan-out from the view. Tests stub the
DeviceMemory interface — the mrrrbmjp vaccine: no pynvml/torch call-site mocks."
```

---

### Task 7: InProcessWorkerHandle — inject, health rewire, Registration lifecycle

**Files:**
- Modify: `backends/worker_handle.py`
- Test: `tests/test_worker_handle.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_worker_handle.py`; check existing fixtures first with `grep -n "def \|fixture" tests/test_worker_handle.py | head -20` and reuse their worker-factory/resolved/binding/mode stubs)

```python
from backends.device_memory import (
    ConsumerMemory, DeviceMemorySnapshot, MemoryTopology, get_device_memory,
    reset_device_memory,
)


def _stub_dm():
    from unittest.mock import Mock
    dm = Mock()
    dm.available_for_load.return_value = 20 * 1024**3
    dm.cached_snapshot.return_value = DeviceMemorySnapshot(
        device_uuid="GPU-test", topology=MemoryTopology.DISCRETE,
        total_bytes=24 * 1024**3, free_bytes=20 * 1024**3, consumers=(),
    )
    return dm


def test_health_reads_device_memory_not_torch():
    from backends.worker_handle import InProcessWorkerHandle
    dm = _stub_dm()
    h = InProcessWorkerHandle(lambda **kw: object(), device_memory=dm)
    health = h.health()
    assert health.vram_free_bytes == 20 * 1024**3
    assert health.vram_total_bytes == 24 * 1024**3


def test_start_registers_consumer_unload_closes():
    from backends.worker_handle import InProcessWorkerHandle
    from unittest.mock import Mock
    dm = _stub_dm()
    registration = Mock()
    dm.register.return_value = registration
    h = InProcessWorkerHandle(lambda **kw: object(), device_memory=dm)

    class _Mode:
        conditioning = Mock(requires_configurable_worker=lambda: False)

    h.start(resolved_mode=Mock(), binding=Mock(), mode=_Mode())
    dm.register.assert_called_once()
    consumer = dm.register.call_args.args[0]
    assert consumer.label == "worker"

    h.unload()
    registration.close.assert_called_once()


def test_double_unload_closes_once():
    from backends.worker_handle import InProcessWorkerHandle
    from unittest.mock import Mock
    dm = _stub_dm()
    registration = Mock()
    dm.register.return_value = registration
    h = InProcessWorkerHandle(lambda **kw: object(), device_memory=dm)

    class _Mode:
        conditioning = Mock(requires_configurable_worker=lambda: False)

    h.start(resolved_mode=Mock(), binding=Mock(), mode=_Mode())
    h.unload()
    h.unload()
    registration.close.assert_called_once()  # handle guards re-close
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_worker_handle.py -v`
Expected: FAIL — `TypeError: InProcessWorkerHandle.__init__() got an unexpected keyword argument 'device_memory'`

- [ ] **Step 3: Write minimal implementation**

In `backends/worker_handle.py`:

1. `InProcessWorkerHandle.__init__`:

```python
    def __init__(self, worker_factory: Callable, device_memory=None):
        self._worker_factory = worker_factory
        if device_memory is None:
            from backends.device_memory import get_device_memory
            device_memory = get_device_memory()
        self._dm = device_memory
        self._registration = None
        self._worker: Optional[PipelineWorker] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._state = "starting"
```

2. End of `start()` (after `self._state = "ready"`, i.e. worker live → register so `pool_stats()` is immediately valid):

```python
        from backends.device_memory import WorkerMemoryConsumer
        self._registration = self._dm.register(WorkerMemoryConsumer(self._worker))
```

3. `health()` — replace the torch block (lines 166-176) with:

```python
    def health(self) -> WorkerHealth:
        return WorkerHealth(
            state=self._state,
            vram_free_bytes=int(self._dm.available_for_load()),
            vram_total_bytes=int(self._dm.cached_snapshot().total_bytes),
            mode=None,  # mode is the Governor's authority; the handle doesn't track it
        )
```

4. `unload()` — close FIRST (deregister-before-teardown, spec §5), then existing teardown verbatim:

```python
    def unload(self) -> None:
        if self._registration is not None:
            self._registration.close()
            self._registration = None

        from backends.controlnet_cache import get_controlnet_cache
        dropped = get_controlnet_cache().clear()
        # ... rest of existing body unchanged (del worker, gc, empty_cache) ...
```

(`torch` import stays — teardown's inline `empty_cache` is teardown, not reclaim, per spec §2.2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_worker_handle.py tests/test_device_memory.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add backends/worker_handle.py tests/test_worker_handle.py
git commit -m "feat(device-memory): handle injects DeviceMemory, owns Registration (STABL-hjldxurg)

health() reads available_for_load()/cached_snapshot — the mem_get_info call at
:168 is gone. start() registers the worker consumer after READY; unload()/stop()
close first (deregister-before-teardown). Parent is sole closer, close guarded
against double-unload."
```

---

### Task 8: Governor + WorkerPool facade rewire

**Files:**
- Modify: `backends/governor.py` (:261-297 init, :320-376 _load_mode, :501-508 _cleanup_vram, :510-529 _build_runtime_status, :786-795 unload_current_model)
- Modify: `backends/worker_pool.py` (:59-73 facade init, :198-201 get_worker_pool)
- Test: `tests/test_governor.py`, `tests/test_worker_pool.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_governor.py`; reuse its existing registry/mode_config/worker_factory fixture pattern — `grep -n "def governor\|def registry\|fixture" tests/test_governor.py | head` first)

```python
def test_runtime_status_reads_worker_consumer_entry():
    """_build_runtime_status:513-514 direct-torch reads are gone; the vram
    block comes from the DeviceMemory snapshot's worker consumer entry."""
    from backends.device_memory import (
        ConsumerMemory, DeviceMemorySnapshot, MemoryTopology,
    )
    worker = ConsumerMemory(label="worker", pid=1, allocated_bytes=3 * 1024**3,
                            reserved_bytes=5 * 1024**3, stale=False)
    snap = DeviceMemorySnapshot(device_uuid="GPU-t", topology=MemoryTopology.DISCRETE,
                                total_bytes=24 * 1024**3, free_bytes=10 * 1024**3,
                                consumers=(worker,))
    # construct Governor with device_memory stub per existing fixture pattern:
    governor = make_governor(device_memory_snap=snap)  # adapt to existing fixture
    status = governor._build_runtime_status()
    assert status["vram"]["allocated_bytes"] == 3 * 1024**3
    assert status["vram"]["reserved_bytes"] == 5 * 1024**3
    assert status["vram"]["stale"] is False


def test_runtime_status_no_worker_reads_zero_not_hang():
    from backends.device_memory import DeviceMemorySnapshot, MemoryTopology
    snap = DeviceMemorySnapshot(device_uuid="GPU-t", topology=MemoryTopology.DISCRETE,
                                total_bytes=24 * 1024**3, free_bytes=24 * 1024**3,
                                consumers=())
    governor = make_governor(device_memory_snap=snap)  # adapt to existing fixture
    status = governor._build_runtime_status()
    assert status["vram"]["allocated_bytes"] == 0
    assert status["vram"]["stale"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_governor.py -k runtime_status -v`
Expected: FAIL — Governor has no `device_memory` param / status fields missing.

- [ ] **Step 3: Write minimal implementation**

In `backends/governor.py`:

1. `__init__` signature + body (:261-297):

```python
    def __init__(
        self,
        queue_max: int = 64,
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        worker_factory: Optional[WorkerFactory] = None,
        mode_config: Optional[ModeConfigManager] = None,
        registry: Optional[ModelRegistryProtocol] = None,
        handle: Optional[WorkerHandle] = None,
        device_memory=None,
    ):
```

after `self._registry = registry or get_model_registry()` add:

```python
        if device_memory is None:
            from backends.device_memory import get_device_memory
            device_memory = get_device_memory()
        self._dm = device_memory
```

and pass to BOTH handle-construction sites:

```python
        if handle is not None:
            self._handle = handle
        elif worker_factory is not None:
            self._handle = InProcessWorkerHandle(worker_factory, device_memory=self._dm)
        else:
            self._handle = InProcessWorkerHandle(self._default_worker_factory, device_memory=self._dm)
```

2. `_load_mode` measurement (:330-331 and :348-352) — the ONE fresh-snapshot exception (spec §4.1/MUST-FIX 2; not the admission path, fan-out permitted):

```python
        self._unload_current_worker()  # unregister old mode + tear down worker
        with self._job_lock:
            self._active_snapshot = None

        allocated_before = _worker_allocated(self._dm.snapshot())
```

```python
        vram_allocated = _worker_allocated(self._dm.snapshot())
        vram_used = max(0, vram_allocated - allocated_before)
        vram_total = self._registry.get_total_vram()
```

(The old `self._registry.get_used_vram()` at :330 and `vram_reserved = ...` at :348 are deleted — they were side-effect-free reads. Known delta: `allocated_before` is 0 when no worker is registered, where the old code subtracted any residual post-flush pool; teardown flushes inline so residual ~= 0 in practice. Within behavioral tolerance.)

Add module-level helper (near `_FutureBridge`):

```python
def _worker_allocated(snapshot) -> int:
    """Worker consumer's live allocations from a DeviceMemory snapshot.
    Load-time measurement reads a FRESH snapshot() (the one fan-out exception);
    /status-shaped readers use cached_snapshot() instead."""
    return next((c.allocated_bytes for c in snapshot.consumers if c.label == "worker"), 0)
```

3. `_build_runtime_status` (:510-529) — `/status` IS the cache-refresh point, so it calls fresh `snapshot()` once, then reads everything from it:

```python
    def _build_runtime_status(
        self, cancelled_jobs: Optional[list[str]] = None, *, status: str = "ok"
    ) -> dict:
        snap = self._dm.snapshot()  # refreshes the cache; /status is the refresh point
        worker = next((c for c in snap.consumers if c.label == "worker"), None)
        payload = {
            "status": status,
            "is_loaded": self.is_model_loaded(),
            "current_mode": self._current_mode,
            "queue_size": self.get_queue_size(),
            "vram": {
                "allocated_bytes": worker.allocated_bytes if worker else 0,
                "reserved_bytes": worker.reserved_bytes if worker else 0,
                "total_bytes": int(self._registry.get_total_vram()),
                "stale": worker.stale if worker else False,
            },
        }
        if cancelled_jobs is not None:
            payload["cancelled_jobs"] = cancelled_jobs
        return payload
```

4. `_cleanup_vram` (:507) and `unload_current_model` (:794): replace `torch.cuda.empty_cache()` with `self._dm.reclaim()`. (`gc.collect()` stays; the `import torch` at :17 stays for the `:344-345` load-error teardown path — that inline flush is teardown, per spec §2.2.)

5. `backends/worker_pool.py`: facade `__init__` and `get_worker_pool` gain `device_memory=None` and pass it through to `Governor(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_governor.py tests/test_worker_pool.py tests/test_worker_handle.py -v`
Expected: all passed. Then full suite: `python -m pytest tests/ -x -q` — expected: all green (baseline 1008 + new). Inert `patch("backends.worker_pool.torch.cuda.memory_allocated", ...)` sites in test_worker_pool.py (e.g. :144, :219, :1355) stay green (the attribute still exists on real torch) — remove them opportunistically ONLY in the tests this task already touches; do not churn the whole file.

- [ ] **Step 5: Commit**

```bash
git add backends/governor.py backends/worker_pool.py tests/test_governor.py tests/test_worker_pool.py
git commit -m "feat(device-memory): Governor status + load measurement via DeviceMemory (STABL-hjldxurg)

_build_runtime_status:513-514 direct-torch reads -> worker consumers[] entry +
stale field (/status refreshes the cache via fresh snapshot()). _load_mode
:330-331/:348-352 measure via fresh snapshot() — the one fan-out exception.
_cleanup_vram/:507 and unload_current_model/:794 empty_cache -> dm.reclaim().
Facade passes device_memory through."
```

---

### Task 9: Pin nvidia-ml-py + drift check

**Files:**
- Modify: `requirements.txt` (:6)
- Modify: `Dockerfile` (:107), `Dockerfile.test` (:103) if the pin must match (both install `nvidia-ml-py` bare)

- [ ] **Step 1: Discover the version to pin**

Run: `python -m pip index versions nvidia-ml-py 2>/dev/null || python -m pip install "nvidia-ml-py==" 2>&1 | head -3`
Then check what's installed in the env: `python -c "import pynvml; print(pynvml.__version__)"` (module name is pynvml; the distribution is nvidia-ml-py).
Pin to the installed env version if it is a released nvidia-ml-py version (matches container), else the latest stable from the index listing.

- [ ] **Step 2: Apply the pin**

Edit `requirements.txt:6` from `nvidia-ml-py` to `nvidia-ml-py==<version>` (e.g. `nvidia-ml-py==12.575.51` — use the version discovered in Step 1, not this example).
Edit `Dockerfile:107` and `Dockerfile.test:103` `pip install --no-cache-dir nvidia-ml-py` to the same pinned string.

- [ ] **Step 3: Verify import still works**

Run: `python -c "from backends.device_memory import get_device_memory, reset_device_memory; reset_device_memory(); dm = get_device_memory(); print(type(dm).__name__, dm.device_name)"`
Expected on darwin: `UnifiedDeviceMemory <hostname>`; on enigma/CUDA: `CudaDeviceMemory NVIDIA ...` — either is correct per host; NullDeviceMemory on a CUDA host is a FAIL (investigate NVML).

- [ ] **Step 4: drift check**

Run: `drift check`
Expected: no NEW stale anchors from this branch's edits (the pre-existing `conf/modes.yml` staleness from bd7b322 is not ours). If `docs/superpowers/plans/2026-04-13-explicit-backend-provider.md` prose goes stale against the new registry/governor code: update the prose FIRST, then `drift link` to refresh provenance, then re-run `drift check`. Never relink stale prose.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt Dockerfile Dockerfile.test
git commit -m "chore(device-memory): pin nvidia-ml-py (STABL-hjldxurg)

Driver-truth source for CudaDeviceMemory; was unpinned in requirements.txt.
Pinned in requirements + both Dockerfiles to the env/container version."
```

---

### Task 10: Behavioral acceptance on enigma (live-gated)

**This is the B-proof (spec §7 criterion 9): behavioral-equivalent-within-delta, NOT byte-identical.** Run on enigma (RTX 3090, CUDA host). Record all measurements in `fp comment STABL-hjldxurg`.

- [ ] **Step 1: Pre-load delta measurement**

Start the server on this branch with NO model load triggered beyond startup. Before any generation:
- `nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv`
- `curl -s localhost:8000/models/status | python -m json.tool` (or the status route the deployment exposes)
- Record: NVML free (via /status `vram` block) vs nvidia-smi free vs the OLD source's expectation (`mem_get_info` free). Document the delta and why it is safe: admission is observability + `can_fit` (a 5%-slack estimate gate, `model_registry.py:242-265`), not a hard barrier. Expected idle delta: small (tens of MiB).

- [ ] **Step 2: Admission equivalence replay**

Record the registry's `"[ModelRegistry] VRAM check: need ... available ..., fits: ..."` log line across a recorded sequence on `main` (or from enigma logs): mode loads of sd15/sdxl/hunyuandit. Replay the same load sequence on this branch; assert **identical fits/admit-deny outcomes** at each step. Any divergence = investigate before proceeding (delta bigger than expected, or a cold-cache bug).

- [ ] **Step 3: /status tolerance + stale field**

After loading a mode and running one generation: `/models/status` `vram` fields within the Step-1-documented ±delta of the pre-branch values for the same model state; `vram.stale` present and `false`; `unattributed_bytes`-shaped gap (nvidia-smi used minus torch reserved) visible and ≈ the CUDA context + workspaces (~0.5–1.5 GB range documented in umbrella finding #2).

- [ ] **Step 4: free-vram + mode-switch smoke**

`st conflate off; st gen ...` smoke per repo norms, then hit the free-vram route and a mode switch: memory returns to baseline ± context residual; no exception in Governor recovery paths; `/status` after each transition stays coherent (no 0-total flash — pre-seed guarantee).

- [ ] **Step 5: Record + close the loop**

```bash
fp comment STABL-hjldxurg "Behavioral acceptance on enigma: delta=<measured>, admission replay <N>/<N> identical, /status within ±<delta>, stale=false present, free-vram baseline OK. v1 = behavioral-within-delta, not 0-byte."
fp issue assign STABL-hjldxurg --rev <branch HEAD sha>
```

---

## Self-Review

**Spec coverage:**
- §2 contract → Tasks 1, 2, 4. `cached_snapshot()` + pre-seed + atomicity → Task 2 (`_ConsumerRegistry`). `unattributed_bytes` docstring/clamp/log → Tasks 1+2. `stale` semantics → Tasks 2, 4, 5.
- §3 providers + Null-degrade + UUID seeding cchxvuhs → Tasks 2, 3. Dep pin → Task 9.
- §4.1 registry view + load-time exception → Tasks 6, 8. §4.2 WorkerHealth → Task 7. §4.3 Governor status → Task 8.
- §5 Registration lifecycle (4 events in-proc: start→register, unload/stop→close, double-close) → Task 7 (subprocess reap events are ptoicrho, not v1).
- §6 wiring (singleton, injection, consumer) → Tasks 2, 5, 6, 7, 8. §6.1 multi-device: forward-proof `get_device_memory(uuid=None)` → Task 2; aggregator itself is cchxvuhs, not built.
- §7 acceptance: criteria 1-8, 10-12 → Tasks 1-9; criterion 9 (behavioral no-op) → Task 10.
- §8 scope: v1 builds vs deferred — respected; no subprocess/superres/multi-GPU code anywhere in this plan.

**Placeholder scan:** Task 9 Step 1's pin version is discovered-by-command (not a placeholder — the value is host-dependent; the command + decision rule are given). Task 7/8 say "adapt to existing fixture" with the grep command to locate it — fixture names in those files are test-internal; the adaptation points are explicit. No TBD/TODO/"add error handling" anywhere.

**Type consistency:** `ConsumerMemory(label, pid, allocated_bytes, reserved_bytes, stale)`, `DeviceMemorySnapshot(device_uuid, topology, total_bytes, free_bytes, consumers)`, `_ConsumerRegistry`, `_RegistrationImpl`, `WorkerMemoryConsumer(worker, label="worker")`, `get_device_memory(uuid=None)`/`reset_device_memory()` — identical across Tasks 1-8. `_worker_allocated` helper defined and used only in Task 8 (both shown). Governor/Registry/Handle param name `device_memory` consistent; internal attr `self._dm` consistent.

**Known accepted deltas (recorded, not hidden):**
- `_load_mode` `allocated_before` reads 0 when no consumer registered (old code subtracted residual post-flush pool ≈ 0) — within behavioral tolerance, noted in Task 8.
- Inert `patch("...torch.cuda.memory_allocated")` sites in test_worker_pool.py stay green; opportunistic removal only where already touched.
- `MemoryTopology.UNKNOWN` + `device_name` are the two additive post-gate contract extensions (header, on record).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-28-device-memory.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration (superpowers:subagent-driven-development)

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
