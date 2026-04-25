from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable


@dataclass
class CacheEntry:
    model_id: str
    model_path: str
    model: Any
    pin_count: int = 0


class ControlNetModelCache:
    def __init__(self, max_entries: int = 4) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()

    def acquire(self, model_id: str, model_path: str, *, loader: Callable[[str], Any]) -> Any:
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                entry = CacheEntry(
                    model_id=model_id,
                    model_path=model_path,
                    model=loader(model_path),
                )
                self._entries[model_id] = entry
            else:
                self._entries.move_to_end(model_id)
            entry.pin_count += 1
            self._evict_if_needed()
            return entry.model

    def release(self, model_id: str) -> None:
        with self._lock:
            entry = self._entries[model_id]
            entry.pin_count -= 1

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            victim_id, victim = next(iter(self._entries.items()))
            if victim.pin_count > 0:
                break
            self._entries.pop(victim_id)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {"entries": list(self._entries.keys())}


_CACHE: ControlNetModelCache | None = None


def get_controlnet_cache() -> ControlNetModelCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = ControlNetModelCache()
    return _CACHE


def reset_controlnet_cache() -> None:
    global _CACHE
    _CACHE = None
