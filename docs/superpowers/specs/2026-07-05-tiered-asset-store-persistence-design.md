# Tiered AssetStore Persistence — Design

**Date:** 2026-07-05
**Status:** Design (approved decisions; ready for spec review)
**Canonical brainstorm:** `fp://brainstorm?id=cdvgrmvgkkpqflzbtpzsfmumtiijfdzt` (v2 — body is source of truth)
**Builds on:** `docs/superpowers/specs/2026-07-04-asset-store-bucketed-interface-design.md`

## Motivation

The bucketed `AssetStore` is memory-only: refs vanish on restart. This design adds a
persistent backing tier behind the existing `AssetStore` API without leaking asset
semantics (buckets, pinning, promotion) into the generic `StorageProvider` layer.

`TieredAssetStore` composes the existing in-memory store (hot cache) with a
`StorageProvider` (durable blob store), joined by a small, independently testable codec.
The `AssetStore` Protocol and all callers are unchanged.

## Locked decisions (from canonical brainstorm v2)

| Decision | Choice |
|---|---|
| Persist scope (v1) | `ref_image` and `control_map` persist; `upload` never persists. Per-bucket `persist` flag retained (not hardcoded to names). |
| Persist-failure contract | Strict write-through for all persisted buckets: `put` failure rolls back the memory admission, then raises. |
| Durability | Backend-selected, not a `BucketPolicy` field. Stronger durability = swap the provider. |
| Codec placement | Own module `server/asset_codec.py`. |
| Memory↔disk relationship | Memory is a write-through **cache**; budget eviction / memory-TTL expiry removes only the hot copy; `resolve` rehydrates on a memory miss. |
| TTL ownership | Tier-local: memory `ttl_s` by `cleanup_expired()`; `persistence_ttl_s` by the provider (`expires_at` + provider cleanup). Memory expiry/eviction never deletes the persistent copy. |
| Config | No `AssetStoreConfig`/`MemoryTierConfig` tree in v1 (YAGNI). Two new `BucketPolicy` fields + existing env provider factory. |

## Component structure

- **Modify `server/asset_store.py`:**
  - `BucketPolicy` gains `persist: bool = False` and `persistence_ttl_s: float | None = None`.
  - `InMemoryAssetStore` gains three small extensions used by the tier (below).
  - Extract the promotion image-decode + metadata-merge into a module-level helper reused by both stores.
- **New `server/asset_codec.py`:** pure `encode`/`decode` between `AssetEntry` and `StorageItem`. No I/O.
- **New `server/tiered_asset_store.py`:** `TieredAssetStore` implementing the `AssetStore` Protocol.
- **Modify `get_store()`** (in `server/asset_store.py`) to return a `TieredAssetStore` composing the singleton `InMemoryAssetStore` + `make_storage_provider_from_env()`.

## Provider TTL-seam contract

The tier passes `persistence_ttl_s` to the provider as `put(..., ttl_s=...)`:

- `persistence_ttl_s` is a number → `ttl_s = int(persistence_ttl_s)`; the provider sets `expires_at = created_at + ttl_s`. Honored consistently across providers.
- `persistence_ttl_s is None` → `ttl_s=None` is passed, meaning **provider-default retention**, which is backend-defined:
  - `FilesystemStorageProvider`: `default_ttl_s` (`FS_STORAGE_TTL_S`, 7 days default).
  - `InMemoryStorageProvider`: no expiry.

`None` therefore means "defer to the backend," matching the canonical body's "bounded by
backend/backpressure" note for `ref_image`. It is **not** a guarantee of permanence.
Implementers must not treat `persistence_ttl_s=None` as "store forever" — permanence is a
backend property.

## BucketPolicy additions + v1 defaults

```python
@dataclass(frozen=True)
class BucketPolicy:
    name: str
    byte_budget: int
    ttl_s: float | None
    pinnable: bool = True
    persist: bool = False
    persistence_ttl_s: float | None = None
```

Seeded `_DEFAULT_BUCKETS` (memory `byte_budget` values unchanged from the bucketed-store spec):

| Bucket | `ttl_s` (memory) | `persist` | `persistence_ttl_s` |
|---|---:|---|---:|
| `upload` | 300 | False | None |
| `control_map` | None | True | 3600 |
| `ref_image` | None | True | None |

Registry remains constructor-injectable.

## `InMemoryAssetStore` extensions

Three additions, each small and lock-guarded (reusing the existing `RLock`). None change
existing method behavior.

```python
def policy(self, bucket: str) -> BucketPolicy:
    """Public read of a bucket's policy (raises ValueError on unknown bucket)."""
    return self._policy(bucket)

def admit(self, entry: AssetEntry) -> None:
    """Insert a fully-formed entry under its existing ref (used to rehydrate from the
    persistent tier). Runs per-bucket fail-closed eviction with protect=entry.ref, so an
    infeasible admission raises ValueError and mutates nothing (same contract as write)."""
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
    """Remove a ref if present; no error if absent (used for write rollback)."""
    with self._lock:
        if ref in self._entries:
            self._remove(ref)
```

Promotion helper extraction (module-level, reused by `InMemoryAssetStore.promote` and
`TieredAssetStore.promote`):

```python
def prepare_promotion(data: bytes, source_metadata: dict, source_ref: str) -> dict:
    """Validate `data` decodes as an image (PIL verify), then return metadata merged
    forward: source_metadata overlaid with origin='promoted', source_asset_ref,
    media_type, width, height. Raises ValueError('asset is not a decodable image')."""
```

`InMemoryAssetStore.promote` is refactored to `merged = prepare_promotion(...); return self.write(target_bucket, data, merged)` — behavior identical, tests unchanged.

## `server/asset_codec.py`

Pure functions; the single shared seam. No provider or store references beyond the two
dataclasses.

```python
@dataclass(frozen=True)
class EncodedAsset:
    key: str
    value: bytes
    content_type: str
    meta: dict
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
    now = time.time()
    return AssetEntry(
        ref=item.key,
        data=item.value,
        bucket=bucket,
        created_at=created_at,
        last_accessed=now,     # runtime-only, reset on rehydrate
        byte_size=len(item.value),
        metadata=meta,          # caller metadata (media_type/width/height survive here)
        pin_count=0,            # runtime-only, reset on rehydrate
    )
```

`bucket` and `created_at` are carried in `meta` on the way out and stripped back into
first-class `AssetEntry` fields on the way in, so the persisted `metadata` round-trips to
exactly the caller-facing metadata. `pin_count` and `last_accessed` are never persisted.

## `TieredAssetStore`

Composes memory + provider; implements the `AssetStore` Protocol.

```python
class TieredAssetStore:
    def __init__(self, memory: InMemoryAssetStore, provider: StorageProvider | None):
        self._memory = memory
        self._provider = provider
```

**write(bucket, data, metadata) -> str**
1. `ref = self._memory.write(bucket, data, metadata)` — memory admission (existing oversize / fail-closed logic; raises propagate untouched).
2. `policy = self._memory.policy(bucket)`.
3. If `policy.persist and self._provider is not None`:
   - `enc = encode(self._memory.resolve(ref), policy)`
   - `try: self._provider.put(enc.key, enc.value, content_type=enc.content_type, meta=enc.meta, ttl_s=enc.ttl_s)`
   - `except Exception: self._memory.discard(ref); raise` — **strict rollback** (Tiered write is not cross-tier atomic under concurrency, but rollback removes exactly this unique ref).
4. Return `ref`.

**resolve(ref) -> AssetEntry**
1. `try: return self._memory.resolve(ref)` (memory hit — unchanged snapshot semantics).
2. On `KeyError`: if `self._provider is None`, re-raise `KeyError`.
3. `item = self._provider.get(ref)`; if `None`, raise `KeyError` (not found or expired on disk).
4. `entry = decode(item)`.
5. Best-effort re-cache: `try: self._memory.admit(entry) except ValueError: pass` — a full/infeasible memory tier does not fail the resolve.
6. Return a snapshot (`replace(entry, metadata=dict(entry.metadata))`).

**promote(ref, target_bucket) -> str**
`entry = self.resolve(ref)` (rehydrates if needed) → `merged = prepare_promotion(entry.data, entry.metadata, ref)` → `return self.write(target_bucket, entry.data, merged)`. Persistence into `target_bucket` happens automatically via `write`.

**pin / unpin / cleanup_expired** — delegate to `self._memory`. Pinning operates on the
resident (memory) copy; a ref evicted to disk must be `resolve`d (rehydrated) before
pinning. `cleanup_expired()` stays memory-tier only; the provider self-expires
`persistence_ttl_s` via `expires_at`.

**bucket_bytes / total_bytes / buckets** — delegate to `self._memory` (report the hot tier).

**close()** — `self._provider.close()` if a provider is present (stops the FS cleanup thread).

## Wiring

```python
_DEFAULT_MEMORY = InMemoryAssetStore()
_DEFAULT_STORE: AssetStore = TieredAssetStore(_DEFAULT_MEMORY, make_storage_provider_from_env())

def get_store() -> AssetStore:
    return _DEFAULT_STORE
```

With `STORAGE_PROVIDER=DISABLED` (the default), `make_storage_provider_from_env()` returns
`None`, so `TieredAssetStore` degrades to memory-only — behaviorally identical to today.
Existing tests that construct `InMemoryAssetStore` directly remain valid.

## Error handling & edge cases

- **Unknown bucket / oversize / fail-closed:** raised by the memory tier before any provider call. No partial disk state.
- **Provider put failure:** memory admission rolled back via `discard`; original exception re-raised.
- **Resolve miss with no provider:** `KeyError` (unchanged from memory-only).
- **Disk-expired ref:** `provider.get` returns `None` → `resolve` raises `KeyError`; caller re-preprocesses (control_map) or treats as gone (ref_image).
- **Rehydrate into a full memory tier:** `admit` may raise `ValueError`; caught and swallowed — resolve still returns the value, just uncached.
- **Corrupt/missing bucket in stored meta:** `decode` raises `ValueError`; surfaces as a resolve failure (does not silently mint a bucket-less entry).

## Testing

Provider fakes: use `InMemoryStorageProvider` (already exists) and a failing fake for the
rollback path.

- **Codec:** `encode` maps ref→key, data→value, media_type→content_type, bucket+created_at into meta, `persistence_ttl_s`→`ttl_s` (None passthrough). `decode` round-trips metadata exactly, strips bucket/created_at back to fields, resets `pin_count`/`last_accessed`. `decode` on bucketless meta raises.
- **write persist:** persisted bucket calls `provider.put` with the right `ttl_s`; `upload` never calls `put`; `provider=None` never calls `put`.
- **strict rollback:** failing provider → `write` raises and `bucket_bytes` returns to pre-write value (ref not resident, not on disk).
- **resolve rehydrate:** evict from memory (budget pressure) → `resolve` returns the entry from the provider with `pin_count=0`, fresh `last_accessed`, preserved `created_at`/metadata; the entry is re-admitted to memory.
- **rehydrate best-effort:** resolve into a pin-saturated bucket returns the value without raising.
- **promote persistence:** promote into `ref_image` writes through to the provider; source untouched.
- **TTL seam:** `control_map` put receives `ttl_s=3600`; `ref_image` put receives `ttl_s=None`.
- **degradation:** `TieredAssetStore(memory, None)` behaves as the memory store for write/resolve/promote/pin/cleanup.

## Out of scope (v1)

- `AssetStoreConfig`/`MemoryTierConfig`/`PersistenceTierConfig` object tree.
- Cross-tier delete API / explicit disk eviction from `AssetStore`.
- Cross-session or cross-user isolation.
- Write-back / async persistence (v1 is synchronous write-through).
- Sharding/layout/schema-version strategy beyond what `FilesystemStorageProvider` already does.
- Auto-rehydrate-on-pin (pin operates on the resident copy in v1).
