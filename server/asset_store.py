import io
import time
import uuid
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, Protocol

from PIL import Image

MB = 1024 * 1024


@dataclass(frozen=True)
class BucketPolicy:
    name: str
    byte_budget: int
    ttl_s: float | None
    pinnable: bool = True
    persist: bool = False
    persistence_ttl_s: float | None = None


@dataclass
class AssetEntry:
    ref: str
    data: bytes
    bucket: str
    created_at: float
    last_accessed: float
    byte_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    pin_count: int = 0


_DEFAULT_BUCKETS: dict[str, BucketPolicy] = {
    "upload": BucketPolicy("upload", byte_budget=128 * MB, ttl_s=300),
    "control_map": BucketPolicy(
        "control_map", byte_budget=256 * MB, ttl_s=None, persist=True, persistence_ttl_s=3600
    ),
    "ref_image": BucketPolicy(
        "ref_image", byte_budget=128 * MB, ttl_s=None, persist=True, persistence_ttl_s=None
    ),
}


def prepare_promotion(data: bytes, source_metadata: dict[str, Any], source_ref: str) -> dict[str, Any]:
    """Validate `data` decodes as an image, then return source metadata merged forward
    with promotion fields overlaid. Raises ValueError if `data` is not a decodable image."""
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception as exc:
        raise ValueError("asset is not a decodable image") from exc

    # verify() leaves the image unusable; reopen to read format/size.
    img = Image.open(io.BytesIO(data))
    fmt = img.format or "PNG"
    media_type = Image.MIME.get(fmt, f"image/{fmt.lower()}")
    width, height = img.size

    return {
        **source_metadata,
        "origin": "promoted",
        "source_asset_ref": source_ref,
        "media_type": media_type,
        "width": width,
        "height": height,
    }


class AssetStore(Protocol):
    def write(self, bucket: str, data: bytes, metadata: dict[str, Any] | None = None) -> str: ...
    def resolve(self, ref: str) -> AssetEntry: ...
    def promote(self, ref: str, target_bucket: str) -> str: ...
    def pin(self, ref: str) -> None: ...
    def unpin(self, ref: str) -> None: ...
    def cleanup_expired(self) -> list[str]: ...
    def bucket_bytes(self, bucket: str) -> int: ...
    def total_bytes(self) -> int: ...
    def buckets(self) -> list[str]: ...


class InMemoryAssetStore:
    def __init__(self, buckets: dict[str, BucketPolicy] | None = None) -> None:
        self._policies: dict[str, BucketPolicy] = (
            dict(buckets) if buckets is not None else dict(_DEFAULT_BUCKETS)
        )
        self._entries: dict[str, AssetEntry] = {}
        self._bucket_bytes: dict[str, int] = {name: 0 for name in self._policies}
        self._lock = RLock()

    def _policy(self, bucket: str) -> BucketPolicy:
        policy = self._policies.get(bucket)
        if policy is None:
            raise ValueError(f"unknown bucket {bucket!r}")
        return policy

    def _require(self, ref: str) -> AssetEntry:
        entry = self._entries.get(ref)
        if entry is None:
            raise KeyError(f"asset ref {ref!r} not found or evicted")
        return entry

    def _remove(self, ref: str) -> None:
        # Caller holds self._lock.
        entry = self._entries.pop(ref)
        self._bucket_bytes[entry.bucket] -= entry.byte_size

    def write(self, bucket: str, data: bytes, metadata: dict[str, Any] | None = None) -> str:
        policy = self._policy(bucket)
        byte_size = len(data)
        if byte_size > policy.byte_budget:
            raise ValueError(
                f"asset exceeds bucket budget: {byte_size} > {policy.byte_budget} "
                f"for bucket {bucket!r}"
            )
        ref = uuid.uuid4().hex
        now = time.time()
        entry = AssetEntry(
            ref=ref,
            data=data,
            bucket=bucket,
            created_at=now,
            last_accessed=now,
            byte_size=byte_size,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._entries[ref] = entry
            self._bucket_bytes[bucket] += byte_size
            self._evict_to_budget(bucket, protect=ref)
            return ref

    def resolve(self, ref: str) -> AssetEntry:
        with self._lock:
            entry = self._require(ref)
            entry.last_accessed = time.time()
            return replace(entry, metadata=dict(entry.metadata))

    def pin(self, ref: str) -> None:
        with self._lock:
            entry = self._require(ref)
            if not self._policies[entry.bucket].pinnable:
                raise ValueError(f"bucket {entry.bucket!r} is not pinnable")
            entry.pin_count += 1

    def unpin(self, ref: str) -> None:
        with self._lock:
            entry = self._require(ref)
            if not self._policies[entry.bucket].pinnable:
                raise ValueError(f"bucket {entry.bucket!r} is not pinnable")
            if entry.pin_count == 0:
                raise ValueError("pin_count is already 0")
            entry.pin_count -= 1

    def cleanup_expired(self) -> list[str]:
        now = time.time()
        with self._lock:
            expired: list[str] = []
            for ref, entry in self._entries.items():
                ttl = self._policies[entry.bucket].ttl_s
                if ttl is None or entry.pin_count > 0:
                    continue
                if (now - entry.created_at) > ttl:
                    expired.append(ref)
            for ref in expired:
                self._remove(ref)
            return expired

    def promote(self, ref: str, target_bucket: str) -> str:
        self._policy(target_bucket)  # validate target bucket up front
        with self._lock:
            src = self._require(ref)
            data = src.data
            src_meta = dict(src.metadata)
        merged = prepare_promotion(data, src_meta, ref)
        return self.write(target_bucket, data, metadata=merged)

    def bucket_bytes(self, bucket: str) -> int:
        with self._lock:
            self._policy(bucket)
            return self._bucket_bytes[bucket]

    def total_bytes(self) -> int:
        with self._lock:
            return sum(self._bucket_bytes.values())

    def buckets(self) -> list[str]:
        return list(self._policies)

    def policy(self, bucket: str) -> BucketPolicy:
        return self._policy(bucket)

    def admit(self, entry: AssetEntry) -> None:
        policy = self._policy(entry.bucket)
        if entry.byte_size > policy.byte_budget:
            raise ValueError(
                f"asset exceeds bucket budget: {entry.byte_size} > {policy.byte_budget} "
                f"for bucket {entry.bucket!r}"
            )
        with self._lock:
            self._entries[entry.ref] = entry
            self._bucket_bytes[entry.bucket] += entry.byte_size
            self._evict_to_budget(entry.bucket, protect=entry.ref)

    def discard(self, ref: str) -> None:
        with self._lock:
            if ref in self._entries:
                self._remove(ref)

    def _evict_to_budget(self, bucket: str, protect: str) -> None:
        # Caller holds self._lock. `protect` is the just-admitted ref, which must
        # never evict itself.
        budget = self._policies[bucket].byte_budget
        if self._bucket_bytes[bucket] <= budget:
            return

        # Feasibility first: the new entry fits iff the bucket's pinned bytes plus
        # the new entry fit the budget, since every unpinned neighbour is evictable.
        # If it does not fit, fail closed and roll back to the pre-admission state —
        # evict nothing, remove only `protect`.
        protect_size = self._entries[protect].byte_size
        pinned_bytes = sum(
            e.byte_size
            for e in self._entries.values()
            if e.bucket == bucket and e.pin_count > 0
        )
        if pinned_bytes + protect_size > budget:
            self._remove(protect)
            raise ValueError(
                f"bucket {bucket!r} has insufficient evictable capacity "
                f"for {protect_size} bytes"
            )

        # Feasible: evict LRU unpinned neighbours (never `protect`) until within budget.
        while self._bucket_bytes[bucket] > budget:
            candidates = [
                e
                for e in self._entries.values()
                if e.bucket == bucket and e.pin_count == 0 and e.ref != protect
            ]
            oldest = min(candidates, key=lambda e: e.last_accessed)
            self._remove(oldest.ref)


_DEFAULT_STORE: AssetStore = InMemoryAssetStore()


def get_store() -> AssetStore:
    return _DEFAULT_STORE
