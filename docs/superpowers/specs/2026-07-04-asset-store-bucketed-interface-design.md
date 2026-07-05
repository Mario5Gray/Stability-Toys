# AssetStore — Bucketed, Swappable Interface

**Date:** 2026-07-04
**Status:** Design (approved for spec review)
**Supersedes v1 backing-store notes in:** `docs/superpowers/specs/2026-04-18-controlnet-design.md` §3

## Motivation

The current `server/asset_store.py` is a single concrete `AssetStore` class with a
flat `dict` keyed by ref, a two-value `_ALLOWED_KINDS` set (`upload`, `control_map`),
and one global byte budget. The 2026-04-18 controlnet design already treats this store
as process-scoped and swappable "in principle."

Two forces make the interface real now:

1. **Reusable image assets.** A promoted, controlnet-agnostic `ref_image` bucket lets
   any non-evicted image be reused as a generation reference, independent of how it
   entered the store.
2. **Swappable backend.** The store should be a `Protocol` with an in-memory
   implementation today, so a disk / object-store backend can land later without
   reshaping callers.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Abstraction | `AssetStore` **Protocol** + `InMemoryAssetStore` impl. `get_store()` returns the Protocol type. |
| Partition model | **Flat named buckets.** `bucket` is the single partition key. No hierarchy, no `kind`/`bucket` duality. |
| Promote semantics | **Copy → new ref.** Original untouched; a new ref is minted in the target bucket. Two independent lifetimes, no aliasing. |
| Budget / eviction | **Per-bucket byte budgets.** Each bucket LRU-evicts within itself. No global cap. |
| TTL cleanup | **Per-bucket TTL policy.** `cleanup_expired()` walks bucket policies; no `ttl_s` argument. |
| Naming | Canonical field is **`bucket`**. `insert` → `write`. **No compatibility shims or aliases** — rename in favor of the new shape. |
| Registry source | **Seeded in code**, injectable via constructor. Config wiring (`conf/`) deferred; not coupled into this step. |

## Data model

### `BucketPolicy`

```python
@dataclass(frozen=True)
class BucketPolicy:
    name: str
    byte_budget: int            # per-bucket cap; LRU-evicted within the bucket
    ttl_s: float | None          # None = no age expiry (survives on pin + LRU only)
    pinnable: bool = True        # False → pin()/unpin() on this bucket's refs raise ValueError
```

### `AssetEntry`

The `kind` field is **renamed to `bucket`**. No alias is kept.

```python
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
```

### Seeded registry

Default registry, injectable at construction. Budgets are seeded defaults, not final
tuning; they carry no config coupling in this step.

```python
MB = 1024 * 1024

_DEFAULT_BUCKETS: dict[str, BucketPolicy] = {
    "upload":      BucketPolicy("upload",      byte_budget=128 * MB, ttl_s=300,  pinnable=True),
    "control_map": BucketPolicy("control_map", byte_budget=256 * MB, ttl_s=None, pinnable=True),
    "ref_image":   BucketPolicy("ref_image",   byte_budget=128 * MB, ttl_s=None, pinnable=True),
}
```

Registering additional buckets later (whatever names future asset classes need) is a flat
registry addition — no protocol or code-shape change. New buckets register exactly like
the entries above.

## Protocol

```python
class AssetStore(Protocol):
    def write(self, bucket: str, data: bytes, metadata: dict | None = None) -> str: ...
    def resolve(self, ref: str) -> AssetEntry: ...
    def promote(self, ref: str, target_bucket: str) -> str: ...
    def pin(self, ref: str) -> None: ...
    def unpin(self, ref: str) -> None: ...
    def cleanup_expired(self) -> list[str]: ...
    def bucket_bytes(self, bucket: str) -> int: ...
    def total_bytes(self) -> int: ...
    def buckets(self) -> list[str]: ...
```

- `write(bucket, data, metadata)` — ingest **bytes** into a bucket, return a fresh ref.
- `promote(ref, target_bucket)` — copy an existing ref's bytes into `target_bucket`,
  return a **new** ref. This is the "write an existing ref into a destination bucket"
  primitive; `target_bucket` uses the same vocabulary as `write` — no `target_kind`.
- `resolve(ref)` — return a snapshot `AssetEntry` (deep-copied metadata), bump
  `last_accessed`. `KeyError` if missing/evicted.
- `cleanup_expired()` — take **no** argument; walk every bucket that declares a `ttl_s`
  and evict unpinned entries older than that bucket's TTL. Returns removed refs.
- `bucket_bytes(bucket)` / `total_bytes()` — per-bucket and summed introspection.
- `buckets()` — registered bucket names.

## Behavioral contracts

### Unknown bucket
`write` / `promote` / `bucket_bytes` against an unregistered bucket raise
`ValueError("unknown bucket …")`.

### Oversize write — fail closed
If a single asset's `byte_size` exceeds the **target bucket's** `byte_budget`,
`write` (and `promote`, since the copy would be oversize) raise
`ValueError("asset exceeds bucket budget …")` **before** inserting. No insert-then-thrash:
the store never admits an entry it must immediately evict, and never churns LRU/eviction
state on an impossible admission.

### Eviction — per bucket, fail closed under pinned pressure

On admission, `_evict_to_budget(bucket)` runs **within that bucket only**: while the
bucket's total bytes exceed its budget, evict the oldest unpinned entry (`last_accessed`
LRU). Pinned entries are never evicted.

If evicting every unpinned entry still leaves the bucket over budget — i.e. pinned
entries alone exceed the budget — admission **fails closed**: `write` / `promote` raise
`ValueError("bucket … has insufficient evictable capacity")` and the store rolls back to
its pre-admission state (the candidate entry is not retained, no LRU state is mutated).
The store never evicts the entry it just admitted, and never admits an entry it cannot
keep. Oversize-single-asset (previous section) is the degenerate case of this rule.

### Pinning — respects `pinnable`
- `pin(ref)` / `unpin(ref)` on a ref whose bucket has `pinnable=False` raise
  `ValueError("bucket … is not pinnable")`.
- `pin` / `unpin` on a missing ref raise `KeyError`.
- `unpin` at `pin_count == 0` raises `ValueError("pin_count is already 0")` (unchanged).
- All default buckets are `pinnable=True` in v1; the non-pinnable contract is exercised
  via an injected custom registry in tests, so the field is not dead config.

### TTL cleanup — per bucket
`cleanup_expired()` evicts unpinned entries in TTL-bearing buckets whose age
(`now - created_at`, **not** `last_accessed`) exceeds the bucket's `ttl_s`. Buckets with
`ttl_s=None` (`control_map`, `ref_image`) are never age-expired; they rely on pinning +
per-bucket LRU.

### Promotion validation — decode as image
`promote` validates the source is a real image by **decoding it** at promotion time
(`PIL.Image.open(BytesIO(data))` then `.verify()`), not by trusting `metadata.media_type`.
Rationale: today's uploads (`upload_routes.py`) store raw bytes with no image metadata, so
a metadata-only check would be effectively permissive. On decode failure, raise
`ValueError("asset is not a decodable image")`.

**Metadata is merged forward, not replaced.** The promoted entry starts as a copy of the
source entry's metadata (preserving provenance and any bucket-specific annotations), then
overlays the promotion fields: `origin="promoted"`, `source_asset_ref=<ref>`, and the
decoded `media_type` / `width` / `height`. Overlay keys win on collision; all other source
keys survive.

## Implementation notes (`InMemoryAssetStore`)

- Constructor: `InMemoryAssetStore(buckets: dict[str, BucketPolicy] | None = None)`,
  defaulting to `_DEFAULT_BUCKETS`. Per-bucket byte counters maintained on
  write/promote/evict/cleanup. One `RLock` guards all mutation (unchanged concurrency model).
- `get_store() -> AssetStore` returns the module singleton, **typed as the Protocol** so
  callers depend on the interface, not the impl.
- `resolve` continues to return a snapshot (`replace(entry, metadata=dict(...))`) — callers
  cannot mutate live state.

## Migration surface

No shims; callers move to the new shape directly.

- `server/upload_routes.py`
  - `get_store().insert("upload", data)` → `get_store().write("upload", data)`.
  - `cleanup_uploads_loop`: `cleanup_expired(ttl_s=TTL_S)` → `cleanup_expired()`
    (TTL now lives in the `upload` bucket policy). The module-level `TTL_S = 300` moves
    into the seeded `upload` `BucketPolicy`; the loop only calls `cleanup_expired()`.
- Controlnet control-map path: `insert("control_map", …)` → `write("control_map", …)`.
- New ref-image reuse path: `promote(ref, "ref_image")`.
- `tests/test_asset_store.py`: rewrite off `insert` / `entry.kind` / global
  `AssetStore(byte_budget=…)` onto `write` / `entry.bucket` / injected per-bucket
  `BucketPolicy` registries. Add coverage for: oversize-write fail-closed,
  per-bucket eviction isolation, promote copy + new ref, promote image-decode rejection,
  non-pinnable bucket raise, per-bucket TTL cleanup.
- Run `drift refs server/asset_store.py` and `drift refs server/upload_routes.py` before
  editing; update bound prose (notably the 2026-04-18 controlnet §3 backing-store notes)
  before relinking.

## Out of scope (v1)

- Cross-session / cross-user isolation (arrives with auth).
- Persistence across restart (store remains process-scoped).
- Config-driven bucket registry (constructor injection is the seam; `conf/` wiring later).
- Hierarchical / sub-partition buckets (flat model is sufficient; `*_map` buckets are flat).
