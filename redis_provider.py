# redis_provider.py
"""
Redis storage provider for LCM/SR server.

Implements the same StorageProvider API as InMemoryStorageProvider, using redis-py.

Design:
- Value bytes stored at:     <prefix>:<key>
- Metadata stored as hash at <prefix>:<key>:meta
- TTL is applied to BOTH keys (value + meta) to keep them in sync.

Dependencies:
  pip install redis

Typical env wiring (your server can read these):
  REDIS_URL=redis://localhost:6379/0
  REDIS_PREFIX=lcm
  REDIS_SOCKET_TIMEOUT=2
  REDIS_CONNECT_TIMEOUT=2
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional, Dict, Any

import redis  # redis-py

# Import your contract/types from your project
# Adjust these names if your storage_provider.py differs.
from storage_provider import StorageProvider, StorageItem


class RedisStorageProvider(StorageProvider):
    """
    Redis-backed provider.

    Notes:
    - We don't enforce max_items here (Redis eviction policy should be used if needed).
    - TTL is the primary control for cache retention.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        prefix: Optional[str] = None,
        socket_timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        decode_responses: bool = False,
    ):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.prefix = (prefix or os.environ.get("REDIS_PREFIX", "lcm")).strip() or "lcm"

        st = socket_timeout if socket_timeout is not None else float(os.environ.get("REDIS_SOCKET_TIMEOUT", "2"))
        ct = connect_timeout if connect_timeout is not None else float(os.environ.get("REDIS_CONNECT_TIMEOUT", "2"))

        # decode_responses=False => bytes in/out (what we want for image blobs)
        self.client = redis.Redis.from_url(
            self.redis_url,
            socket_timeout=st,
            socket_connect_timeout=ct,
            decode_responses=decode_responses,
        )

    # -------------------------
    # internal key helpers
    # -------------------------
    def _k(self, key: str) -> str:
        key = str(key).strip()
        if not key:
            raise ValueError("key must be non-empty")
        return f"{self.prefix}:{key}"

    def _km(self, key: str) -> str:
        return f"{self._k(key)}:meta"

    # -------------------------
    # StorageProvider API
    # -------------------------
    def put(
        self,
        key: str,
        value: bytes,
        *,
        content_type: str = "application/octet-stream",
        meta: Optional[Dict[str, Any]] = None,
        ttl_s: Optional[int] = None,
    ) -> None:
        if value is None:
            raise ValueError("value must not be None")
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("value must be bytes-like")

        if ttl is None
          ttl = STORAGE_TTL_IMAGE

        ttl = int(ttl_s)
        if ttl is not None and ttl <= 0:
            # treat non-positive TTL as "expire immediately"
            ttl = 1

        now = time.time()
        meta_obj: Dict[str, Any] = dict(meta or {})
        meta_obj.update(
            {
                "content_type": str(content_type or "application/octet-stream"),
                "created_at": now,
            }
        )

        k = self._k(key)
        km = self._km(key)

        # Pipeline so value + meta are consistent
        pipe = self.client.pipeline(transaction=True)
        pipe.set(k, bytes(value))
        pipe.hset(
            km,
            mapping={
                "content_type": meta_obj["content_type"],
                "created_at": str(meta_obj["created_at"]),
                "meta_json": json.dumps(meta_obj, separators=(",", ":"), ensure_ascii=False),
            },
        )
        if ttl is not None:
            pipe.expire(k, ttl)
            pipe.expire(km, ttl)
        pipe.execute()

    def get(self, key: str) -> Optional[StorageItem]:
        k = self._k(key)
        km = self._km(key)

        pipe = self.client.pipeline(transaction=False)
        pipe.get(k)
        pipe.hgetall(km)
        val, mh = pipe.execute()

        if val is None:
            return None

        # hgetall returns dict[bytes,bytes] when decode_responses=False
        if not mh:
            content_type = "application/octet-stream"
            created_at = time.time()
            meta = {}
        else:
            def _b2s(x):
                return x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)

            mh2 = { _b2s(a): _b2s(b) for a, b in mh.items() }
            content_type = mh2.get("content_type") or "application/octet-stream"
            try:
                created_at = float(mh2.get("created_at") or time.time())
            except Exception:
                created_at = time.time()
            try:
                meta = json.loads(mh2.get("meta_json") or "{}")
            except Exception:
                meta = {}

        # StorageItem contract (adjust if your dataclass differs)
        return StorageItem(
            key=str(key),
            value=val if isinstance(val, (bytes, bytearray)) else bytes(val),
            content_type=content_type,
            meta=meta,
            created_at=created_at,
        )

    def delete(self, key: str) -> bool:
        k = self._k(key)
        km = self._km(key)
        n = self.client.delete(k, km)
        return n > 0

    def health(self) -> Dict[str, Any]:
        """
        Light health probe that won't spam INFO unless you want it.
        """
        try:
            pong = self.client.ping()
            return {
                "ok": bool(pong),
                "provider": "redis",
                "redis_url": self.redis_url,
                "prefix": self.prefix,
            }
        except Exception as e:
            return {
                "ok": False,
                "provider": "redis",
                "redis_url": self.redis_url,
                "prefix": self.prefix,
                "error": str(e),
            }

    def close(self) -> None:
        try:
            # redis-py: disconnect pooled conns
            self.client.connection_pool.disconnect()
        except Exception:
            pass
