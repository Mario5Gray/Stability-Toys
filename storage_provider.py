# storage_provider.py
from __future__ import annotations

import os
import uuid
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

STORAGE_MAX_ITEMS = int(os.environ.get("STORAGE_MAX_ITEMS", "512"))
STORAGE_TTL_IMAGE = int(os.environ.get("STORAGE_TTL_IMAGE", "3600"))
STORAGE_ENABLE_HTTP = os.environ.get("STORAGE_ENABLE_HTTP", "0") in ("1", "true", "True")


@dataclass
class StorageItem:
    key: str
    value: bytes
    content_type: str
    meta: Dict[str, Any]
    created_at: float
    expires_at: Optional[float] = None


class StorageProvider:
    def put(
        self,
        key: str,
        value: bytes,
        *,
        content_type: str = "application/octet-stream",
        meta: Optional[Dict[str, Any]] = None,
        ttl_s: Optional[int] = None,
    ) -> StorageItem:
        raise NotImplementedError

    def get(self, key: str) -> Optional[StorageItem]:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @staticmethod
    def make_storage_provider_from_env() -> StorageProvider:
        provider = os.environ.get("STORAGE_PROVIDER", "DISABLED").upper()
        print("storage:", provider)
        if provider == "MEMORY":
            return InMemoryStorageProvider(max_items=STORAGE_MAX_ITEMS)
        elif provider == "DISABLED":
            return None
        elif provider == "REDIS":
            from redis_provider import RedisStorageProvider
            return RedisStorageProvider()
        elif provider in ("FILESYSTEM", "FS"):
            from filesystem_provider import FilesystemStorageProvider
            return FilesystemStorageProvider()
        else:
            raise RuntimeError(f"Unknown STORAGE_PROVIDER={provider}")
        

    @staticmethod
    def _new_key(prefix: str) -> str:
      return f"{prefix}:{uuid.uuid4()}"

class InMemoryStorageProvider(StorageProvider):
    def __init__(self, max_items: int = 512):
        self.max_items = int(max_items)
        self._lock = threading.Lock()
        self._items: Dict[str, StorageItem] = {}

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [k for k, it in self._items.items() if it.expires_at is not None and it.expires_at <= now]
        for k in expired:
            self._items.pop(k, None)

    def put(self, key: str, value: bytes, *, content_type: str = "application/octet-stream",
            meta: Optional[Dict[str, Any]] = None, ttl_s: Optional[int] = None) -> StorageItem:
        with self._lock:
            self._purge_expired_locked()

            # crude backpressure: drop oldest if over max_items
            if len(self._items) >= self.max_items:
                oldest_key = min(self._items.items(), key=lambda kv: kv[1].created_at)[0]
                self._items.pop(oldest_key, None)

            now = time.time()
            expires_at = (now + int(ttl_s)) if ttl_s is not None else None
            item = StorageItem(
                key=key,
                value=value,
                content_type=content_type,
                meta=dict(meta or {}),
                created_at=now,
                expires_at=expires_at,
            )
            self._items[key] = item
            return item

    def get(self, key: str) -> Optional[StorageItem]:
        with self._lock:
            self._purge_expired_locked()
            it = self._items.get(key)
            if not it:
                return None
            if it.expires_at is not None and it.expires_at <= time.time():
                self._items.pop(key, None)
                return None
            return it

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._items.pop(key, None) is not None