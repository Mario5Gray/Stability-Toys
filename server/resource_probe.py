"""OS resource counts for the leak watch (STABL-cxbwwgly).

STABL-nstyyrhh established that the server leaks one POSIX named semaphore per
MODEL LOAD - linear, never reclaimed - and accepted that risk specifically
BECAUSE it is cheap to watch. This is the watch.

`None` means "could not be read here" and is never rendered as a metric. `0`
means "read it, found none". Conflating the two would report a healthy-looking
zero on any host without /dev/shm.

Imports nothing from backends/ and nothing from server/metrics - it is a pure
measurement, and the sampler decides what to do with it.
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SHM_ROOT = "/dev/shm"
_SEM_PREFIX = "sem."


@dataclass(frozen=True)
class ResourceCounts:
    leaked_semaphores: Optional[int]
    shm_segments: Optional[int]
    open_fds: Optional[int]


def _count_fds() -> Optional[int]:
    try:
        import psutil

        return int(psutil.Process().num_fds())
    except Exception:
        return None


def _count_shm(shm_root: str) -> tuple[Optional[int], Optional[int]]:
    """Return (semaphores, segments). Both None when the root cannot be listed."""
    try:
        entries = os.listdir(shm_root)
    except Exception:
        return None, None

    sems = sum(1 for entry in entries if entry.startswith(_SEM_PREFIX))
    return sems, len(entries) - sems


def probe_resources(shm_root: str = DEFAULT_SHM_ROOT) -> ResourceCounts:
    """One measurement. Must never raise."""
    # These two guards catch the UNEXPECTED — _count_shm and _count_fds already
    # return None for every failure they anticipate. Logging matters here for the
    # same reason the whole module exists: an absent series otherwise reads
    # identically whether the source is legitimately unavailable (no /dev/shm) or
    # the probe itself broke, and an observability component that hides its own
    # failure is the worst kind.
    try:
        sems, segments = _count_shm(shm_root)
    except Exception:
        logger.debug("[ResourceProbe] shm count failed unexpectedly", exc_info=True)
        sems, segments = None, None

    try:
        fds = _count_fds()
    except Exception:
        logger.debug("[ResourceProbe] fd count failed unexpectedly", exc_info=True)
        fds = None

    return ResourceCounts(
        leaked_semaphores=sems,
        shm_segments=segments,
        open_fds=fds,
    )
