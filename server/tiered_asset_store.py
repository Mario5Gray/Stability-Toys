"""tiered_asset_store.py — TieredAssetStore: memory hot tier + StorageProvider
persistent tier behind the AssetStore Protocol.

Provider selection is dedicated to the asset store (ASSET_STORE_PROVIDER) and
deliberately decoupled from STORAGE_PROVIDER, which drives the separate
/storage/* provider. Redis and other backends are intentionally out of scope
in v1 (see docs/superpowers/specs/2026-07-05-tiered-asset-store-persistence-design.md).
"""

import os

from persistence.storage_provider import (
    StorageProvider,
    InMemoryStorageProvider,
    STORAGE_MAX_ITEMS,
)


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
