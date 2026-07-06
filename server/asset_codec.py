"""asset_codec.py — the single shared seam between the semantic asset layer
(AssetEntry, server/asset_store.py) and the generic provider layer
(StorageItem, persistence/storage_provider.py).

Pure functions, no I/O. Runtime-only fields (pin_count, last_accessed) are never
persisted; bucket and created_at travel inside provider meta and are restored to
first-class AssetEntry fields on decode.
"""

import time
from dataclasses import dataclass
from typing import Any

from server.asset_store import AssetEntry, BucketPolicy
from persistence.storage_provider import StorageItem


@dataclass(frozen=True)
class EncodedAsset:
    key: str
    value: bytes
    content_type: str
    meta: dict[str, Any]
    ttl_s: int | None


def encode(entry: AssetEntry, policy: BucketPolicy) -> EncodedAsset:
    ttl = int(policy.persistence_ttl_s) if policy.persistence_ttl_s is not None else None
    return EncodedAsset(
        key=entry.ref,
        value=entry.data,
        content_type=entry.metadata.get("media_type", "application/octet-stream"),
        meta={**entry.metadata, "bucket": entry.bucket, "created_at": entry.created_at},
        ttl_s=ttl,
    )


def decode(item: StorageItem) -> AssetEntry:
    meta = dict(item.meta)
    bucket = meta.pop("bucket", None)
    if bucket is None:
        raise ValueError(f"storage item {item.key!r} missing bucket metadata")
    created_at = meta.pop("created_at", item.created_at)
    return AssetEntry(
        ref=item.key,
        data=item.value,
        bucket=bucket,
        created_at=created_at,
        last_accessed=time.time(),
        byte_size=len(item.value),
        metadata=meta,
        pin_count=0,
    )
