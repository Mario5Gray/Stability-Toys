# filesystem_provider.py
"""
Filesystem storage provider for LCM/SR server.

Stores images as files on disk with metadata sidecars.

Directory structure:
  {base_dir}/
    {shard[0:2]}/
      {key}.bin       # Image data
      {key}.meta.json # Metadata JSON

Env vars:
  FS_STORAGE_DIR=/data/image-cache
  FS_STORAGE_TTL_S=604800  (7 days default)
  FS_STORAGE_CLEANUP_INTERVAL_S=3600  (hourly sweep)
"""

from __future__ import annotations

import json
import os
import time
import threading
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from storage_provider import StorageProvider, StorageItem, STORAGE_TTL_IMAGE


class FilesystemStorageProvider(StorageProvider):
    """
    Filesystem-backed storage provider with TTL expiration.

    Images stored as binary files with JSON metadata sidecars.
    Background thread periodically cleans up expired entries.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        *,
        default_ttl_s: Optional[int] = None,
        cleanup_interval_s: Optional[int] = None,
    ):
        self.base_dir = Path(
            base_dir
            or os.environ.get("FS_STORAGE_DIR", "/data/image-cache")
        )
        self.default_ttl_s = int(
            default_ttl_s
            if default_ttl_s is not None
            else os.environ.get("FS_STORAGE_TTL_S", str(STORAGE_TTL_IMAGE))
        )
        self.cleanup_interval_s = int(
            cleanup_interval_s
            if cleanup_interval_s is not None
            else os.environ.get("FS_STORAGE_CLEANUP_INTERVAL_S", "3600")
        )

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Background cleanup thread
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="fs-storage-cleanup",
        )
        self._cleanup_thread.start()

        print(f"[FSStorage] Initialized: dir={self.base_dir}, ttl={self.default_ttl_s}s, cleanup_interval={self.cleanup_interval_s}s")

    def _shard(self, key: str) -> str:
        """Extract 2-char shard from key UUID portion."""
        # Keys are like "lcm_image:a3f5c2d9-..."
        parts = key.split(":")
        uuid_part = parts[-1] if len(parts) > 1 else key
        return uuid_part[:2].lower()

    def _data_path(self, key: str) -> Path:
        shard = self._shard(key)
        # Sanitize key for filesystem (replace : with _)
        safe_key = key.replace(":", "_")
        return self.base_dir / shard / f"{safe_key}.bin"

    def _meta_path(self, key: str) -> Path:
        shard = self._shard(key)
        safe_key = key.replace(":", "_")
        return self.base_dir / shard / f"{safe_key}.meta.json"

    def put(
        self,
        key: str,
        value: bytes,
        *,
        content_type: str = "application/octet-stream",
        meta: Optional[Dict[str, Any]] = None,
        ttl_s: Optional[int] = None,
    ) -> StorageItem:
        if value is None:
            raise ValueError("value must not be None")

        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        now = time.time()
        expires_at = now + ttl if ttl else None

        data_path = self._data_path(key)
        meta_path = self._meta_path(key)

        # Ensure shard directory exists
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write - data file
        with tempfile.NamedTemporaryFile(
            dir=data_path.parent,
            delete=False,
        ) as tmp:
            tmp.write(value)
            tmp_data = tmp.name
        os.replace(tmp_data, data_path)

        # Atomic write - metadata file
        meta_obj = {
            "key": key,
            "content_type": content_type,
            "meta": dict(meta or {}),
            "created_at": now,
            "expires_at": expires_at,
            "size": len(value),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=meta_path.parent,
            delete=False,
            suffix=".json",
        ) as tmp:
            json.dump(meta_obj, tmp)
            tmp_meta = tmp.name
        os.replace(tmp_meta, meta_path)

        return StorageItem(
            key=key,
            value=value,
            content_type=content_type,
            meta=dict(meta or {}),
            created_at=now,
            expires_at=expires_at,
        )

    def get(self, key: str) -> Optional[StorageItem]:
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)

        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r") as f:
                meta_obj = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        # Check expiration
        expires_at = meta_obj.get("expires_at")
        if expires_at and time.time() > expires_at:
            # Async cleanup (don't block the read)
            threading.Thread(
                target=self._delete_files,
                args=(data_path, meta_path),
                daemon=True,
            ).start()
            return None

        # Read data file
        if not data_path.exists():
            return None

        try:
            with open(data_path, "rb") as f:
                value = f.read()
        except IOError:
            return None

        return StorageItem(
            key=meta_obj.get("key", key),
            value=value,
            content_type=meta_obj.get("content_type", "application/octet-stream"),
            meta=meta_obj.get("meta", {}),
            created_at=meta_obj.get("created_at", 0),
            expires_at=expires_at,
        )

    def delete(self, key: str) -> bool:
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)
        return self._delete_files(data_path, meta_path)

    def _delete_files(self, data_path: Path, meta_path: Path) -> bool:
        deleted = False
        for p in (data_path, meta_path):
            try:
                p.unlink(missing_ok=True)
                deleted = True
            except Exception:
                pass
        return deleted

    def _cleanup_loop(self):
        """Background thread to clean up expired files."""
        while not self._stop_event.wait(self.cleanup_interval_s):
            self._cleanup_expired()

    def _cleanup_expired(self):
        """Scan all meta files and delete expired entries."""
        now = time.time()
        deleted = 0

        try:
            for meta_file in self.base_dir.rglob("*.meta.json"):
                try:
                    with open(meta_file, "r") as f:
                        meta_obj = json.load(f)
                    expires_at = meta_obj.get("expires_at")
                    if expires_at and now > expires_at:
                        key = meta_obj.get("key")
                        if key:
                            data_path = self._data_path(key)
                            self._delete_files(data_path, meta_file)
                            deleted += 1
                except Exception:
                    pass
        except Exception as e:
            print(f"[FSStorage] Cleanup error: {e}")

        if deleted > 0:
            print(f"[FSStorage] Cleanup: deleted {deleted} expired entries")

    def health(self) -> Dict[str, Any]:
        """Health check with stats."""
        try:
            meta_count = sum(1 for _ in self.base_dir.rglob("*.meta.json"))
            total_bytes = sum(
                f.stat().st_size
                for f in self.base_dir.rglob("*.bin")
            )
            return {
                "ok": True,
                "provider": "filesystem",
                "base_dir": str(self.base_dir),
                "entries": meta_count,
                "total_bytes": total_bytes,
                "default_ttl_s": self.default_ttl_s,
            }
        except Exception as e:
            return {
                "ok": False,
                "provider": "filesystem",
                "error": str(e),
            }

    def close(self):
        """Stop background cleanup thread."""
        self._stop_event.set()
        self._cleanup_thread.join(timeout=2)
