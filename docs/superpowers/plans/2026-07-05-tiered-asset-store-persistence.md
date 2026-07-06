# Tiered AssetStore Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (Project policy forbids subagent-driven development — execute inline.)

**Goal:** Add a filesystem-backed persistent tier behind the bucketed `AssetStore` via a new `TieredAssetStore` that composes the existing in-memory store with a `StorageProvider`, joined by a pure codec.

**Architecture:** `TieredAssetStore` implements the `AssetStore` Protocol, wrapping `InMemoryAssetStore` (hot cache) + an optional `StorageProvider` (durable tier). Persisted buckets write through strictly (rollback-then-raise on provider failure); `resolve` rehydrates from the provider on a memory miss. A dedicated `ASSET_STORE_PROVIDER` selector (filesystem/in-memory only) keeps Redis out of scope. Lifecycle is handled at the edges via `close_store()`.

**Tech Stack:** Python 3.11+, `dataclasses`, `threading.RLock`, Pillow (`PIL.Image`), `pytest`. Existing `persistence/` provider layer (`StorageProvider`, `InMemoryStorageProvider`, `FilesystemStorageProvider`).

**Spec:** `docs/superpowers/specs/2026-07-05-tiered-asset-store-persistence-design.md`
**Canonical brainstorm:** `fp://brainstorm?id=cdvgrmvgkkpqflzbtpzsfmumtiijfdzt` (v2)

## Global Constraints

- **No shims/aliases.** `bucket` field, `write`, per-bucket budgets — unchanged from the bucketed store.
- **Strict write-through:** on `provider.put` failure, discard the just-admitted ref, then raise. Do **not** attempt to restore hot entries evicted during admission.
- **Redis out of scope.** The tier uses its own `ASSET_STORE_PROVIDER` selector (`DISABLED`/`MEMORY`/`FILESYSTEM`); Redis/unknown values raise `RuntimeError`. Do not route through `StorageProvider.make_storage_provider_from_env()`.
- **`persistence_ttl_s=None` → provider default retention** (backend-defined); not a permanence guarantee.
- **Runtime-only fields never persisted:** `pin_count`, `last_accessed` reset on rehydrate; `created_at` survives via stored meta.
- **Lifecycle at the edges:** `close_store()` wired into the server lifespan shutdown; tests close providers in teardown.
- Run tests via Miniforge base: `source /Users/darkbit1001/miniforge3/bin/activate base` once per shell, then `python -m pytest ...`.

## File Structure

- **Modify `server/asset_store.py`:** `BucketPolicy` +2 fields; `_DEFAULT_BUCKETS` persistence defaults; module-level `prepare_promotion`; `InMemoryAssetStore.policy/admit/discard`; lazy `get_store()` + `close_store()`.
- **Create `server/asset_codec.py`:** `EncodedAsset`, `encode`, `decode`. Pure, no I/O.
- **Create `server/tiered_asset_store.py`:** `make_asset_store_provider_from_env`, `TieredAssetStore`.
- **Modify `server/lcm_sr_server.py`:** `_close_providers(app)` helper + call it in the lifespan shutdown; import `close_store`.
- **Modify `tests/test_upload_routes.py`, `tests/test_ws_routes.py`:** reset fixtures poke `get_store()._memory` internals now.
- **Create tests:** `tests/test_asset_codec.py`, `tests/test_tiered_asset_store.py`, `tests/test_server_shutdown.py`. Add to `tests/test_asset_store.py`.

---

### Task 1: `BucketPolicy` persistence fields + defaults

**Files:**
- Modify: `server/asset_store.py:13-18` (`BucketPolicy`), `:33-37` (`_DEFAULT_BUCKETS`)
- Test: `tests/test_asset_store.py`

**Interfaces:**
- Produces: `BucketPolicy(..., persist: bool = False, persistence_ttl_s: float | None = None)`; `_DEFAULT_BUCKETS` with persistence defaults.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_asset_store.py`:

```python
# --- persistence policy fields ---

def test_bucket_policy_persist_defaults():
    p = BucketPolicy("x", byte_budget=10, ttl_s=None)
    assert p.persist is False
    assert p.persistence_ttl_s is None


def test_default_buckets_persistence_policy():
    from server.asset_store import _DEFAULT_BUCKETS
    assert _DEFAULT_BUCKETS["upload"].persist is False
    assert _DEFAULT_BUCKETS["upload"].persistence_ttl_s is None
    assert _DEFAULT_BUCKETS["control_map"].persist is True
    assert _DEFAULT_BUCKETS["control_map"].persistence_ttl_s == 3600
    assert _DEFAULT_BUCKETS["ref_image"].persist is True
    assert _DEFAULT_BUCKETS["ref_image"].persistence_ttl_s is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_asset_store.py -q -k "persist or persistence_policy"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'persist'` / attribute errors.

- [ ] **Step 3: Add the fields and defaults**

In `server/asset_store.py`, replace the `BucketPolicy` dataclass:

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

Replace `_DEFAULT_BUCKETS`:

```python
_DEFAULT_BUCKETS: dict[str, BucketPolicy] = {
    "upload": BucketPolicy("upload", byte_budget=128 * MB, ttl_s=300),
    "control_map": BucketPolicy(
        "control_map", byte_budget=256 * MB, ttl_s=None, persist=True, persistence_ttl_s=3600
    ),
    "ref_image": BucketPolicy(
        "ref_image", byte_budget=128 * MB, ttl_s=None, persist=True, persistence_ttl_s=None
    ),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asset_store.py -q`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "feat(asset-store): BucketPolicy persist + persistence_ttl_s fields; v1 defaults (control_map/ref_image persist) — next: extract prepare_promotion"
```

---

### Task 2: Extract `prepare_promotion` helper

**Files:**
- Modify: `server/asset_store.py:139-165` (`InMemoryAssetStore.promote`)
- Test: `tests/test_asset_store.py`

**Interfaces:**
- Produces: module-level `prepare_promotion(data: bytes, source_metadata: dict, source_ref: str) -> dict` — PIL-validates the image, returns merge-forward metadata. `InMemoryAssetStore.promote` refactored to use it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_asset_store.py`:

```python
# --- prepare_promotion helper ---

def test_prepare_promotion_merges_and_validates():
    from server.asset_store import prepare_promotion
    merged = prepare_promotion(_png(), {"provenance": "u", "origin": "ingested"}, "srcref")
    assert merged["origin"] == "promoted"        # overlay wins
    assert merged["source_asset_ref"] == "srcref"
    assert merged["provenance"] == "u"           # source key preserved
    assert merged["media_type"] == "image/png"
    assert merged["width"] == 8 and merged["height"] == 8


def test_prepare_promotion_rejects_non_image():
    from server.asset_store import prepare_promotion
    with pytest.raises(ValueError, match="not a decodable image"):
        prepare_promotion(b"not an image", {}, "r")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_asset_store.py -q -k prepare_promotion`
Expected: FAIL — `ImportError: cannot import name 'prepare_promotion'`.

- [ ] **Step 3: Extract the helper and refactor `promote`**

In `server/asset_store.py`, add a module-level function after `_DEFAULT_BUCKETS` (before the `AssetStore` Protocol):

```python
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
```

Replace `InMemoryAssetStore.promote` body with:

```python
    def promote(self, ref: str, target_bucket: str) -> str:
        self._policy(target_bucket)  # validate target bucket up front
        with self._lock:
            src = self._require(ref)
            data = src.data
            src_meta = dict(src.metadata)
        merged = prepare_promotion(data, src_meta, ref)
        return self.write(target_bucket, data, metadata=merged)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asset_store.py -q`
Expected: PASS (new helper tests + all existing promote tests unchanged in behavior).

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "refactor(asset-store): extract prepare_promotion (image-decode + merge-forward) for reuse by tier — next: memory-tier extensions"
```

---

### Task 3: `InMemoryAssetStore.policy` / `admit` / `discard`

**Files:**
- Modify: `server/asset_store.py` (add three methods to `InMemoryAssetStore`)
- Test: `tests/test_asset_store.py`

**Interfaces:**
- Produces:
  - `InMemoryAssetStore.policy(bucket: str) -> BucketPolicy`
  - `InMemoryAssetStore.admit(entry: AssetEntry) -> None` — insert a prebuilt entry under its own ref, running fail-closed eviction (protect=entry.ref). Raises `ValueError` on unknown bucket / oversize / insufficient capacity.
  - `InMemoryAssetStore.discard(ref: str) -> None` — remove if present, no error if absent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_asset_store.py`:

```python
# --- tier-facing extensions ---

def test_policy_accessor():
    store = _store()
    assert store.policy("upload").name == "upload"
    with pytest.raises(ValueError, match="unknown bucket"):
        store.policy("nope")


def test_admit_inserts_entry_under_its_ref():
    store = _store()
    entry = AssetEntry(
        ref="fixed123", data=b"abc", bucket="ref_image",
        created_at=1.0, last_accessed=1.0, byte_size=3, metadata={"k": "v"},
    )
    store.admit(entry)
    got = store.resolve("fixed123")
    assert got.data == b"abc" and got.bucket == "ref_image" and got.metadata["k"] == "v"
    assert store.bucket_bytes("ref_image") == 3


def test_admit_oversize_raises_and_admits_nothing():
    store = _store(b=BucketPolicy("b", byte_budget=4, ttl_s=None))
    big = AssetEntry(
        ref="x", data=b"aaaaa", bucket="b",
        created_at=1.0, last_accessed=1.0, byte_size=5, metadata={},
    )
    with pytest.raises(ValueError, match="exceeds bucket budget"):
        store.admit(big)
    assert store.bucket_bytes("b") == 0


def test_admit_unknown_bucket_raises():
    store = _store()
    entry = AssetEntry(
        ref="r", data=b"x", bucket="nope",
        created_at=1.0, last_accessed=1.0, byte_size=1, metadata={},
    )
    with pytest.raises(ValueError, match="unknown bucket"):
        store.admit(entry)


def test_discard_removes_present_and_ignores_absent():
    store = _store()
    ref = store.write("upload", b"hi")
    store.discard(ref)
    with pytest.raises(KeyError):
        store.resolve(ref)
    store.discard("absent-ref")  # no error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_asset_store.py -q -k "policy_accessor or admit or discard"`
Expected: FAIL — `AttributeError: 'InMemoryAssetStore' object has no attribute 'policy'`.

- [ ] **Step 3: Add the three methods**

In `server/asset_store.py`, add to `InMemoryAssetStore` (place after `buckets`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asset_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "feat(asset-store): policy/admit/discard on InMemoryAssetStore for the persistence tier — next: asset_codec"
```

---

### Task 4: `server/asset_codec.py` (encode/decode)

**Files:**
- Create: `server/asset_codec.py`
- Test: `tests/test_asset_codec.py`

**Interfaces:**
- Consumes: `AssetEntry`, `BucketPolicy` (from `server.asset_store`), `StorageItem` (from `persistence.storage_provider`).
- Produces:
  - `EncodedAsset(key: str, value: bytes, content_type: str, meta: dict, ttl_s: int | None)`
  - `encode(entry: AssetEntry, policy: BucketPolicy) -> EncodedAsset`
  - `decode(item: StorageItem) -> AssetEntry`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_asset_codec.py`:

```python
import pytest

from server.asset_store import AssetEntry, BucketPolicy
from server.asset_codec import EncodedAsset, encode, decode
from persistence.storage_provider import StorageItem


def _entry(**over):
    base = dict(
        ref="r1", data=b"bytes", bucket="ref_image", created_at=123.0,
        last_accessed=999.0, byte_size=5,
        metadata={"media_type": "image/png", "width": 8, "height": 8},
    )
    base.update(over)
    return AssetEntry(**base)


def test_encode_maps_fields():
    policy = BucketPolicy("ref_image", 10, None, persist=True, persistence_ttl_s=None)
    enc = encode(_entry(), policy)
    assert isinstance(enc, EncodedAsset)
    assert enc.key == "r1"
    assert enc.value == b"bytes"
    assert enc.content_type == "image/png"
    assert enc.meta["bucket"] == "ref_image"
    assert enc.meta["created_at"] == 123.0
    assert enc.meta["width"] == 8
    assert enc.ttl_s is None


def test_encode_ttl_from_policy():
    policy = BucketPolicy("control_map", 10, None, persist=True, persistence_ttl_s=3600)
    assert encode(_entry(bucket="control_map"), policy).ttl_s == 3600


def test_encode_default_content_type_when_missing():
    policy = BucketPolicy("upload", 10, 300)
    enc = encode(_entry(metadata={}), policy)
    assert enc.content_type == "application/octet-stream"


def test_decode_round_trips_metadata_exactly():
    item = StorageItem(
        key="r1", value=b"bytes", content_type="image/png",
        meta={"bucket": "ref_image", "created_at": 123.0,
              "width": 8, "height": 8, "media_type": "image/png"},
        created_at=500.0, expires_at=None,
    )
    entry = decode(item)
    assert entry.ref == "r1"
    assert entry.data == b"bytes"
    assert entry.bucket == "ref_image"
    assert entry.created_at == 123.0
    assert entry.byte_size == 5
    assert entry.pin_count == 0
    assert entry.metadata == {"width": 8, "height": 8, "media_type": "image/png"}


def test_decode_missing_bucket_raises():
    item = StorageItem(key="r", value=b"x", content_type="image/png", meta={}, created_at=1.0)
    with pytest.raises(ValueError, match="missing bucket"):
        decode(item)


def test_decode_created_at_falls_back_to_item():
    item = StorageItem(
        key="r", value=b"x", content_type="image/png",
        meta={"bucket": "ref_image"}, created_at=777.0,
    )
    assert decode(item).created_at == 777.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_asset_codec.py -q`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'server.asset_codec'`.

- [ ] **Step 3: Create the codec**

Create `server/asset_codec.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asset_codec.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/asset_codec.py tests/test_asset_codec.py
git commit -m "feat(asset-codec): pure AssetEntry<->StorageItem encode/decode seam — next: provider selector"
```

---

### Task 5: `make_asset_store_provider_from_env` selector

**Files:**
- Create: `server/tiered_asset_store.py` (selector only; class added in Task 6)
- Test: `tests/test_tiered_asset_store.py`

**Interfaces:**
- Produces: `make_asset_store_provider_from_env() -> StorageProvider | None` — reads `ASSET_STORE_PROVIDER` (`DISABLED`→None, `MEMORY`→`InMemoryStorageProvider`, `FILESYSTEM`/`FS`→`FilesystemStorageProvider`); raises `RuntimeError` on `REDIS`/unknown.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tiered_asset_store.py`:

```python
import pytest

from persistence.storage_provider import InMemoryStorageProvider
from persistence.filesystem_provider import FilesystemStorageProvider
from server.tiered_asset_store import make_asset_store_provider_from_env


def test_selector_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("ASSET_STORE_PROVIDER", raising=False)
    assert make_asset_store_provider_from_env() is None


def test_selector_explicit_disabled(monkeypatch):
    monkeypatch.setenv("ASSET_STORE_PROVIDER", "DISABLED")
    assert make_asset_store_provider_from_env() is None


def test_selector_memory(monkeypatch):
    monkeypatch.setenv("ASSET_STORE_PROVIDER", "MEMORY")
    assert isinstance(make_asset_store_provider_from_env(), InMemoryStorageProvider)


def test_selector_filesystem(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSET_STORE_PROVIDER", "FS")
    monkeypatch.setenv("FS_STORAGE_DIR", str(tmp_path))
    p = make_asset_store_provider_from_env()
    assert isinstance(p, FilesystemStorageProvider)
    p.close()


def test_selector_rejects_redis(monkeypatch):
    monkeypatch.setenv("ASSET_STORE_PROVIDER", "REDIS")
    with pytest.raises(RuntimeError, match="out of scope"):
        make_asset_store_provider_from_env()


def test_selector_rejects_unknown(monkeypatch):
    monkeypatch.setenv("ASSET_STORE_PROVIDER", "S3")
    with pytest.raises(RuntimeError, match="out of scope"):
        make_asset_store_provider_from_env()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tiered_asset_store.py -q`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'server.tiered_asset_store'`.

- [ ] **Step 3: Create the selector**

Create `server/tiered_asset_store.py`:

```python
import os

from persistence.storage_provider import (
    StorageProvider,
    InMemoryStorageProvider,
    STORAGE_MAX_ITEMS,
)


def make_asset_store_provider_from_env() -> StorageProvider | None:
    """Dedicated provider selector for the asset-store persistence tier. Decoupled from
    STORAGE_PROVIDER (which drives the separate /storage/* provider). Redis and other
    backends are intentionally out of scope in v1."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tiered_asset_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/tiered_asset_store.py tests/test_tiered_asset_store.py
git commit -m "feat(tiered-store): dedicated ASSET_STORE_PROVIDER selector (FS/MEMORY/DISABLED; Redis rejected) — next: TieredAssetStore core"
```

---

### Task 6: `TieredAssetStore` core (write, resolve, delegations)

**Files:**
- Modify: `server/tiered_asset_store.py` (add the class)
- Test: `tests/test_tiered_asset_store.py`

**Interfaces:**
- Consumes: `InMemoryAssetStore`, `BucketPolicy`, `AssetEntry` (asset_store); `encode`, `decode` (asset_codec); `StorageProvider` (persistence).
- Produces:
  - `TieredAssetStore(memory: InMemoryAssetStore, provider: StorageProvider | None)`
  - `.write(bucket, data, metadata=None) -> str` (strict write-through + rollback)
  - `.resolve(ref) -> AssetEntry` (rehydrate best-effort)
  - `.pin/.unpin/.cleanup_expired/.bucket_bytes/.total_bytes/.buckets` (delegate to memory)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tiered_asset_store.py`:

```python
import io
from PIL import Image

from server.asset_store import InMemoryAssetStore, BucketPolicy, MB
from server.tiered_asset_store import TieredAssetStore
from persistence.storage_provider import StorageProvider, StorageItem, InMemoryStorageProvider


def _png(color=(255, 0, 0), size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mem(**buckets) -> InMemoryAssetStore:
    return InMemoryAssetStore(buckets=buckets) if buckets else InMemoryAssetStore()


class FailingProvider(StorageProvider):
    def put(self, key, value, *, content_type="application/octet-stream", meta=None, ttl_s=None):
        raise IOError("disk full")

    def get(self, key):
        return None

    def delete(self, key):
        return False


class FailAfter(StorageProvider):
    """Delegates to an in-memory provider; raises on puts after the Nth."""
    def __init__(self, n: int):
        self.n = n
        self.calls = 0
        self.inner = InMemoryStorageProvider()

    def put(self, key, value, *, content_type="application/octet-stream", meta=None, ttl_s=None):
        self.calls += 1
        if self.calls > self.n:
            raise IOError("boom")
        return self.inner.put(key, value, content_type=content_type, meta=meta, ttl_s=ttl_s)

    def get(self, key):
        return self.inner.get(key)

    def delete(self, key):
        return self.inner.delete(key)


def test_write_persists_persisted_bucket():
    prov = InMemoryStorageProvider()
    store = TieredAssetStore(_mem(), prov)
    ref = store.write("ref_image", _png(), metadata={"media_type": "image/png"})
    assert prov.get(ref) is not None


def test_write_upload_not_persisted():
    prov = InMemoryStorageProvider()
    store = TieredAssetStore(_mem(), prov)
    ref = store.write("upload", b"hi")
    assert prov.get(ref) is None


def test_write_no_provider_degrades_to_memory():
    store = TieredAssetStore(_mem(), None)
    ref = store.write("ref_image", _png(), metadata={"media_type": "image/png"})
    assert store.resolve(ref).bucket == "ref_image"


def test_write_strict_rollback_no_eviction():
    store = TieredAssetStore(
        _mem(rf=BucketPolicy("rf", byte_budget=10 * MB, ttl_s=None, persist=True)),
        FailingProvider(),
    )
    with pytest.raises(IOError):
        store.write("rf", _png(), metadata={"media_type": "image/png"})
    assert store.bucket_bytes("rf") == 0  # new ref discarded


def test_write_rollback_does_not_restore_evicted_entry():
    png = _png()
    size = len(png)
    store = TieredAssetStore(
        _mem(rf=BucketPolicy("rf", byte_budget=size + 1, ttl_s=None, persist=True)),
        FailAfter(1),  # first put ok, second fails
    )
    store.write("rf", png, metadata={"media_type": "image/png"})   # persists, resident
    with pytest.raises(IOError):
        store.write("rf", png, metadata={"media_type": "image/png"})  # evicts a, put fails, discard new
    # memory holds neither the evicted entry (not restored) nor the new ref
    assert store.bucket_bytes("rf") == 0


def test_resolve_rehydrates_after_memory_eviction():
    png = _png()
    size = len(png)
    prov = InMemoryStorageProvider()
    store = TieredAssetStore(
        _mem(rf=BucketPolicy("rf", byte_budget=size + 1, ttl_s=None, persist=True)),
        prov,
    )
    a = store.write("rf", png, metadata={"media_type": "image/png"})
    store.write("rf", png, metadata={"media_type": "image/png"})  # evicts a from memory
    got = store.resolve(a)  # memory miss -> rehydrate from provider
    assert got.data == png
    assert got.bucket == "rf"
    assert got.pin_count == 0


def test_resolve_miss_no_provider_raises_keyerror():
    store = TieredAssetStore(_mem(), None)
    with pytest.raises(KeyError):
        store.resolve("nope")


def test_resolve_miss_with_provider_absent_raises_keyerror():
    store = TieredAssetStore(_mem(), InMemoryStorageProvider())
    with pytest.raises(KeyError):
        store.resolve("nope")


def test_delegations():
    store = TieredAssetStore(_mem(), None)
    ref = store.write("upload", b"hi")
    store.pin(ref)
    store.unpin(ref)
    assert set(store.buckets()) == {"upload", "control_map", "ref_image"}
    assert store.total_bytes() == 2
    assert store.bucket_bytes("upload") == 2
    store._memory._entries[ref].created_at = 0.0
    assert ref in store.cleanup_expired()


def test_ttl_seam_passes_persistence_ttl():
    captured = {}

    class CapProv(StorageProvider):
        def put(self, key, value, *, content_type="application/octet-stream", meta=None, ttl_s=None):
            captured[meta["bucket"]] = ttl_s
            return StorageItem(key, value, content_type, dict(meta or {}), 0.0)

        def get(self, key):
            return None

        def delete(self, key):
            return False

    store = TieredAssetStore(_mem(), CapProv())
    store.write("control_map", _png(), metadata={"media_type": "image/png"})
    store.write("ref_image", _png(), metadata={"media_type": "image/png"})
    assert captured["control_map"] == 3600
    assert captured["ref_image"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tiered_asset_store.py -q -k "write or resolve or delegations or ttl_seam"`
Expected: FAIL — `ImportError: cannot import name 'TieredAssetStore'`.

- [ ] **Step 3: Add the class**

Append to `server/tiered_asset_store.py` (and extend the import at the top):

```python
from dataclasses import replace

from server.asset_store import AssetStore, AssetEntry, InMemoryAssetStore
from server.asset_codec import encode, decode
```

```python
class TieredAssetStore:
    def __init__(self, memory: InMemoryAssetStore, provider: StorageProvider | None) -> None:
        self._memory = memory
        self._provider = provider

    def write(self, bucket: str, data: bytes, metadata: dict | None = None) -> str:
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

    def promote(self, ref: str, target_bucket: str) -> str:  # completed in Task 7
        raise NotImplementedError

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
```

Note: `promote` and `close` are stubbed/added in Task 7; the Task 6 tests do not call them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tiered_asset_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/tiered_asset_store.py tests/test_tiered_asset_store.py
git commit -m "feat(tiered-store): TieredAssetStore write (strict write-through + rollback), resolve (rehydrate), delegations — next: promote + close + wiring"
```

---

### Task 7: `TieredAssetStore.promote` + `close`; wire `get_store`/`close_store`

**Files:**
- Modify: `server/tiered_asset_store.py` (promote, close)
- Modify: `server/asset_store.py:214-218` (lazy `get_store`, add `close_store`)
- Modify: `tests/test_upload_routes.py`, `tests/test_ws_routes.py` (reset fixtures → `._memory`)
- Test: `tests/test_tiered_asset_store.py`

**Interfaces:**
- Consumes: `prepare_promotion` (asset_store), `make_asset_store_provider_from_env` (this module).
- Produces:
  - `TieredAssetStore.promote(ref, target_bucket) -> str` (rehydrate source → validate → write into target)
  - `TieredAssetStore.close() -> None`
  - `server.asset_store.get_store() -> AssetStore` (lazy singleton = `TieredAssetStore`)
  - `server.asset_store.close_store() -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tiered_asset_store.py`:

```python
def test_promote_persists_into_target_bucket():
    prov = InMemoryStorageProvider()
    store = TieredAssetStore(_mem(), prov)
    src = store.write("upload", _png(), metadata={"media_type": "image/png"})
    dst = store.promote(src, "ref_image")
    assert dst != src
    assert prov.get(dst) is not None          # persisted target
    assert store.resolve(dst).bucket == "ref_image"
    assert store.resolve(src).bucket == "upload"


def test_promote_rehydrates_evicted_source():
    png = _png()
    size = len(png)
    prov = InMemoryStorageProvider()
    store = TieredAssetStore(
        _mem(
            control_map=BucketPolicy("control_map", byte_budget=size + 1, ttl_s=None, persist=True),
            ref_image=BucketPolicy("ref_image", byte_budget=10 * MB, ttl_s=None, persist=True),
        ),
        prov,
    )
    src = store.write("control_map", png, metadata={"media_type": "image/png"})
    store.write("control_map", png, metadata={"media_type": "image/png"})  # evict src from memory
    dst = store.promote(src, "ref_image")  # resolve rehydrates src, then promotes
    assert store.resolve(dst).data == png


def test_close_closes_provider():
    closed = {"v": False}

    class ClosProv(StorageProvider):
        def put(self, *a, **k):
            raise NotImplementedError

        def get(self, key):
            return None

        def delete(self, key):
            return False

        def close(self):
            closed["v"] = True

    TieredAssetStore(_mem(), ClosProv()).close()
    assert closed["v"] is True


def test_close_none_provider_is_noop():
    TieredAssetStore(_mem(), None).close()  # no error


def test_get_store_returns_tiered_singleton(monkeypatch):
    monkeypatch.delenv("ASSET_STORE_PROVIDER", raising=False)
    import server.asset_store as m
    m._DEFAULT_STORE = None  # reset lazy singleton
    s = m.get_store()
    assert isinstance(s, TieredAssetStore)
    assert m.get_store() is s
    m.close_store()
    m._DEFAULT_STORE = None  # leave clean for other tests
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tiered_asset_store.py -q -k "promote or close or singleton"`
Expected: FAIL — `NotImplementedError` (promote) / `AttributeError` (close) / `close_store` missing.

- [ ] **Step 3: Implement promote + close**

In `server/tiered_asset_store.py`, extend the asset_store import and replace the `promote` stub, add `close`:

```python
from server.asset_store import AssetStore, AssetEntry, InMemoryAssetStore, prepare_promotion
```

```python
    def promote(self, ref: str, target_bucket: str) -> str:
        self._memory.policy(target_bucket)  # validate target bucket up front
        entry = self.resolve(ref)           # rehydrates from provider if evicted
        merged = prepare_promotion(entry.data, entry.metadata, ref)
        return self.write(target_bucket, entry.data, merged)

    def close(self) -> None:
        if self._provider is not None:
            self._provider.close()
```

- [ ] **Step 4: Rewire `get_store` (lazy) + add `close_store`**

In `server/asset_store.py`, replace the tail (lines ~214-218):

```python
_DEFAULT_STORE = None  # type: ignore[var-annotated]


def get_store() -> AssetStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        # Lazy to avoid an import cycle (tiered_asset_store imports this module).
        from server.tiered_asset_store import (
            TieredAssetStore,
            make_asset_store_provider_from_env,
        )
        _DEFAULT_STORE = TieredAssetStore(
            InMemoryAssetStore(), make_asset_store_provider_from_env()
        )
    return _DEFAULT_STORE


def close_store() -> None:
    """Release the asset-store singleton's provider (e.g. the FS cleanup thread)."""
    if _DEFAULT_STORE is not None:
        _DEFAULT_STORE.close()
```

- [ ] **Step 5: Migrate the singleton-reset fixtures**

In `tests/test_upload_routes.py` and `tests/test_ws_routes.py`, the `_clear_store` fixture now operates on the wrapped memory tier. Replace both fixtures' bodies so every `store._X` becomes `store._memory._X`:

```python
@pytest.fixture(autouse=True)
def _clear_store():
    store = get_store()
    with store._memory._lock:
        store._memory._entries.clear()
        store._memory._bucket_bytes = {name: 0 for name in store._memory._policies}
    yield
    with store._memory._lock:
        store._memory._entries.clear()
        store._memory._bucket_bytes = {name: 0 for name in store._memory._policies}
```

(These tests run with `ASSET_STORE_PROVIDER` unset → provider `None`, so resetting the memory tier is sufficient; no disk state.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_tiered_asset_store.py tests/test_upload_routes.py tests/test_ws_routes.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add server/tiered_asset_store.py server/asset_store.py tests/test_upload_routes.py tests/test_ws_routes.py tests/test_tiered_asset_store.py
git commit -m "feat(tiered-store): promote + close; lazy get_store()/close_store() singleton; migrate reset fixtures to memory tier — next: server shutdown hook"
```

---

### Task 8: Server lifespan shutdown hook

**Files:**
- Modify: `server/lcm_sr_server.py` (import `close_store`; add `_close_providers(app)`; call it in lifespan shutdown at ~438-442)
- Test: `tests/test_server_shutdown.py`

**Interfaces:**
- Consumes: `close_store` (asset_store), `app.state.storage`.
- Produces: `server.lcm_sr_server._close_providers(app) -> None` — closes both the `/storage/*` provider and the asset-store singleton, tolerating errors from either.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_shutdown.py`:

```python
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import server.lcm_sr_server as srv


def test_close_providers_closes_both(monkeypatch):
    calls = []

    class Storage:
        def close(self):
            calls.append("storage")

    class App:
        class state:
            storage = Storage()

    monkeypatch.setattr(srv, "close_store", lambda: calls.append("store"))
    srv._close_providers(App)
    assert "storage" in calls
    assert "store" in calls


def test_close_providers_tolerates_storage_error(monkeypatch):
    calls = []

    class BadStorage:
        def close(self):
            raise IOError("boom")

    class App:
        class state:
            storage = BadStorage()

    monkeypatch.setattr(srv, "close_store", lambda: calls.append("store"))
    srv._close_providers(App)  # must not raise
    assert "store" in calls  # asset store still closed despite storage error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_server_shutdown.py -q`
Expected: FAIL — `AttributeError: module 'server.lcm_sr_server' has no attribute '_close_providers'`.

- [ ] **Step 3: Add the import and helper**

In `server/lcm_sr_server.py`, add to the imports near the other `server.` imports:

```python
from server.asset_store import close_store
```

Add a module-level helper (place it just above the `lifespan` definition at line ~351):

```python
def _close_providers(app: FastAPI) -> None:
    """Close both provider lifecycles at shutdown: the /storage/* provider on
    app.state.storage and the asset-store singleton. Each is best-effort."""
    try:
        app.state.storage.close()
    except Exception as e:
        logger.error(f"Error closing storage: {e}", exc_info=True)
    try:
        close_store()
    except Exception as e:
        logger.error(f"Error closing asset store: {e}", exc_info=True)
```

- [ ] **Step 4: Call it from the lifespan shutdown**

In `server/lcm_sr_server.py`, replace the existing shutdown close block (currently at ~438-442):

```python
    try:
        app.state.storage.close()  # type: ignore[union-attr]
    except Exception as e:
        logger.error(f"Error closing storage: {e}", exc_info=True)
```

with:

```python
    _close_providers(app)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_server_shutdown.py -q`
Expected: PASS.

- [ ] **Step 6: Full-suite regression check**

Run: `python -m pytest -q --ignore=tests/test_controlnet_preprocessors.py`
Expected: PASS except the 4 known pre-existing worker-ordering-pollution failures (`test_cuda_worker_controlnet` ×3, `test_sdxl_worker::test_cuda_available`). No new failures. Confirm no leaked filesystem cleanup threads (tests use `ASSET_STORE_PROVIDER` unset → no provider, or close providers they construct).

- [ ] **Step 7: Commit**

```bash
git add server/lcm_sr_server.py tests/test_server_shutdown.py
git commit -m "feat(server): close asset-store singleton at lifespan shutdown via _close_providers — persistence tier lifecycle at the edge"
```

---

## Self-Review

**Spec coverage:**
- `BucketPolicy` persist fields + defaults → Task 1. ✓
- Provider TTL-seam (`persistence_ttl_s`→`ttl_s`, None passthrough) → Task 4 (encode) + Task 6 (ttl_seam test). ✓
- Codec (`AssetEntry`↔`StorageItem`, runtime fields reset, bucket/created_at round-trip) → Task 4. ✓
- `InMemoryAssetStore` extensions (`policy`/`admit`/`discard`) → Task 3. ✓
- `prepare_promotion` extraction → Task 2. ✓
- Dedicated selector, Redis rejected → Task 5. ✓
- `TieredAssetStore` write (strict rollback), resolve (rehydrate best-effort), promote, delegations, degradation → Tasks 6–7. ✓
- Wiring `get_store` (lazy, tiered) + `close_store` → Task 7. ✓
- Reset-fixture migration to memory tier → Task 7. ✓
- Lifecycle shutdown hook (`_close_providers` + `close_store`) → Task 8. ✓
- Out-of-scope items (config tree, cross-tier delete, write-back) → not implemented (correct). ✓

**Placeholder scan:** Task 6 intentionally stubs `promote` with `NotImplementedError`, completed in Task 7 (noted). No `TBD`/`add error handling`/bare "write tests" — all steps carry real code and commands.

**Type consistency:** `EncodedAsset(key,value,content_type,meta,ttl_s)`, `encode(entry,policy)`, `decode(item)`, `TieredAssetStore(memory,provider)`, `make_asset_store_provider_from_env()`, `policy/admit/discard`, `prepare_promotion(data,source_metadata,source_ref)`, `get_store`/`close_store`, `_close_providers(app)` — names/signatures match across tasks and the reset-fixture `._memory` access matches the `TieredAssetStore._memory` attribute. `persistence_ttl_s=3600` for `control_map`, `None` for `ref_image` consistent between Task 1 defaults, Task 4 encode, and Task 6 ttl-seam test.
