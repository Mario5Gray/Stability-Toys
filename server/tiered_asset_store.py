"""tiered_asset_store.py — TieredAssetStore: memory hot tier + StorageProvider
persistent tier behind the AssetStore Protocol.

Provider selection is dedicated to the asset store (ASSET_STORE_PROVIDER) and
deliberately decoupled from STORAGE_PROVIDER, which drives the separate
/storage/* provider. Redis and other backends are intentionally out of scope
in v1 (see docs/superpowers/specs/2026-07-05-tiered-asset-store-persistence-design.md).
"""

import os
from dataclasses import replace
from typing import Any

from persistence.storage_provider import (
    StorageProvider,
    InMemoryStorageProvider,
    STORAGE_MAX_ITEMS,
)
from server.asset_store import AssetEntry, InMemoryAssetStore, prepare_promotion
from server.asset_codec import decode, encode


def make_asset_store_provider_from_env() -> StorageProvider | None:
    """Dedicated provider selector for the asset-store persistence tier."""
    kind = os.environ.get("ASSET_STORE_PROVIDER", "DISABLED").upper()
    if kind == "DISABLED":
        return None
    if kind == "MEMORY":
        return InMemoryStorageProvider(max_items=STORAGE_MAX_ITEMS)
    if kind in ("FILESYSTEM", "FS"):
        from persistence.filesystem_provider import FilesystemStorageProvider
        return FilesystemStorageProvider()
    raise RuntimeError(
        f"ASSET_STORE_PROVIDER={kind} is out of scope for the asset-store persistence tier "
        f"(v1 supports DISABLED, MEMORY, FILESYSTEM). Redis and other backends are excluded."
    )


class TieredAssetStore:
    """AssetStore implementation composing a memory hot tier with an optional
    persistent StorageProvider. Memory is a write-through cache for persisted
    buckets; resolve rehydrates from the provider on a memory miss."""

    def __init__(self, memory: InMemoryAssetStore, provider: StorageProvider | None) -> None:
        self._memory = memory
        self._provider = provider

    def write(self, bucket: str, data: bytes, metadata: dict[str, Any] | None = None) -> str:
        ref = self._memory.write(bucket, data, metadata)
        policy = self._memory.policy(bucket)
        if policy.persist and self._provider is not None:
            enc = encode(self._memory.resolve(ref), policy)
            try:
                self._provider.put(
                    enc.key, enc.value,
                    content_type=enc.content_type, meta=enc.meta, ttl_s=enc.ttl_s,
                )
            except Exception:
                self._memory.discard(ref)  # strict rollback: remove the just-admitted ref
                raise
        return ref

    def resolve(self, ref: str) -> AssetEntry:
        try:
            return self._memory.resolve(ref)
        except KeyError:
            if self._provider is None:
                raise
        item = self._provider.get(ref)
        if item is None:
            raise KeyError(f"asset ref {ref!r} not found or evicted")
        entry = decode(item)
        try:
            self._memory.admit(entry)  # best-effort re-cache
        except ValueError:
            pass  # memory tier full/infeasible; still return the resolved value
        return replace(entry, metadata=dict(entry.metadata))

    def promote(self, ref: str, target_bucket: str) -> str:
        self._memory.policy(target_bucket)  # validate target bucket up front
        entry = self.resolve(ref)           # rehydrates from provider if evicted
        merged = prepare_promotion(entry.data, entry.metadata, ref)
        return self.write(target_bucket, entry.data, merged)

    def close(self) -> None:
        if self._provider is not None:
            self._provider.close()

    def pin(self, ref: str) -> None:
        self._memory.pin(ref)

    def unpin(self, ref: str) -> None:
        self._memory.unpin(ref)

    def cleanup_expired(self) -> list[str]:
        return self._memory.cleanup_expired()

    def bucket_bytes(self, bucket: str) -> int:
        return self._memory.bucket_bytes(bucket)

    def total_bytes(self) -> int:
        return self._memory.total_bytes()

    def buckets(self) -> list[str]:
        return self._memory.buckets()
