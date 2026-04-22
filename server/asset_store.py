import time
import uuid
from dataclasses import dataclass, field
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
    def __init__(self, byte_budget: int = 512 * 1024 * 1024) -> None:
        self._entries: dict[str, AssetEntry] = {}
        self._byte_budget = byte_budget

    def insert(self, kind: str, data: bytes, metadata: dict[str, Any] | None = None) -> str:
        ref = uuid.uuid4().hex
        now = time.time()
        self._entries[ref] = AssetEntry(
            ref=ref,
            data=data,
            kind=kind,
            created_at=now,
            last_accessed=now,
            byte_size=len(data),
            metadata=metadata or {},
        )
        self._evict_to_budget()
        return ref

    def resolve(self, ref: str) -> AssetEntry:
        entry = self._entries.get(ref)
        if entry is None:
            raise KeyError(f"asset ref {ref!r} not found or evicted")
        entry.last_accessed = time.time()
        return entry

    def cleanup_expired(self, ttl_s: float = 300.0) -> list[str]:
        now = time.time()
        expired = [
            ref
            for ref, entry in self._entries.items()
            if entry.kind == "upload" and entry.pin_count == 0 and (now - entry.created_at) > ttl_s
        ]
        for ref in expired:
            del self._entries[ref]
        return expired

    def pin(self, ref: str) -> None:
        entry = self._entries.get(ref)
        if entry is not None:
            entry.pin_count += 1

    def unpin(self, ref: str) -> None:
        entry = self._entries.get(ref)
        if entry is not None and entry.pin_count > 0:
            entry.pin_count -= 1

    @property
    def total_bytes(self) -> int:
        return sum(entry.byte_size for entry in self._entries.values())

    def _evict_to_budget(self) -> None:
        while self.total_bytes > self._byte_budget:
            candidates = [entry for entry in self._entries.values() if entry.pin_count == 0]
            if not candidates:
                break
            oldest = min(candidates, key=lambda entry: entry.last_accessed)
            del self._entries[oldest.ref]


_DEFAULT_STORE = AssetStore()


def get_store() -> AssetStore:
    return _DEFAULT_STORE
