from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class LivenessSource(Protocol):
    """Transport-agnostic liveness seam the Governor consumes only through
    `health()`. Subprocess impl = process-alive + heartbeat staleness; a future
    rsocket/remote handle backs the same protocol with KEEPALIVE — no Governor
    change. Keep this Protocol clean of any subprocess/multiprocessing specifics."""

    def state(self) -> str: ...          # "live" | "dead"

    def note_heartbeat(self) -> None: ...


class SubprocessLiveness:
    """Liveness = process alive AND last heartbeat within stale_after_s. When the
    transport becomes rsocket, a KeepaliveLiveness backs the same protocol with no
    Governor change.

    Note (recon #5B): M1 wires this with stale_after_s=float("inf") — the staleness
    window is disabled until the periodic-heartbeat follow-on (spec §7 defers it),
    so an idle-but-alive subprocess is never falsely dead. The default arg stays
    10.0; do not hardcode the 10s anywhere as load-bearing, and do not wire
    heartbeats here — that is the caller's (Task 5) responsibility."""

    def __init__(self, process, stale_after_s: float = 10.0):
        self._process = process
        self._stale_after_s = stale_after_s
        self._last_heartbeat = time.monotonic()

    def note_heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def state(self) -> str:
        if not self._process.is_alive():
            return "dead"
        if time.monotonic() - self._last_heartbeat > self._stale_after_s:
            return "dead"
        return "live"

