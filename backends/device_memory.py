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
