import time
import uuid
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any


@dataclass
class AssetEntry:
    ref: str
    data: bytes
    kind: str
    created_at: float
    last_accessed: float
    byte_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    pin_count: int = 0


class AssetStore:
    _ALLOWED_KINDS = {"upload", "control_map"}

    def __init__(self, byte_budget: int = 512 * 1024 * 1024) -> None:
        self._entries: dict[str, AssetEntry] = {}
        self._byte_budget = byte_budget
        self._total_bytes = 0
        self._lock = RLock()

    def insert(self, kind: str, data: bytes, metadata: dict[str, Any] | None = None) -> str:
        if kind not in self._ALLOWED_KINDS:
            raise ValueError(f"unknown asset kind {kind!r}")

        ref = uuid.uuid4().hex
        now = time.time()
        entry = AssetEntry(
            ref=ref,
            data=data,
            kind=kind,
            created_at=now,
            last_accessed=now,
            byte_size=len(data),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._entries[ref] = entry
            self._total_bytes += entry.byte_size
            self._evict_to_budget()
        return ref

    def resolve(self, ref: str) -> AssetEntry:
        with self._lock:
            entry = self._entries.get(ref)
            if entry is None:
                raise KeyError(f"asset ref {ref!r} not found or evicted")
            entry.last_accessed = time.time()
            return replace(entry, metadata=dict(entry.metadata))

    def cleanup_expired(self, ttl_s: float = 300.0) -> list[str]:
        """Evict unpinned upload entries by age since insertion, not last access time."""
        now = time.time()
        with self._lock:
            expired = [
                ref
                for ref, entry in self._entries.items()
                if entry.kind == "upload" and entry.pin_count == 0 and (now - entry.created_at) > ttl_s
            ]
            for ref in expired:
                entry = self._entries.pop(ref)
                self._total_bytes -= entry.byte_size
            return expired

    def pin(self, ref: str) -> None:
        with self._lock:
            entry = self._entries.get(ref)
            if entry is None:
                raise KeyError(f"asset ref {ref!r} not found or evicted")
            entry.pin_count += 1

    def unpin(self, ref: str) -> None:
        with self._lock:
            entry = self._entries.get(ref)
            if entry is None:
                raise KeyError(f"asset ref {ref!r} not found or evicted")
            if entry.pin_count == 0:
                raise ValueError("pin_count is already 0")
            entry.pin_count -= 1

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def _evict_to_budget(self) -> None:
        # Caller must hold self._lock.
        while self._total_bytes > self._byte_budget:
            candidates = [entry for entry in self._entries.values() if entry.pin_count == 0]
            if not candidates:
                break
            oldest = min(candidates, key=lambda entry: entry.last_accessed)
            del self._entries[oldest.ref]
            self._total_bytes -= oldest.byte_size


_DEFAULT_STORE = AssetStore()


def get_store() -> AssetStore:
    return _DEFAULT_STORE
