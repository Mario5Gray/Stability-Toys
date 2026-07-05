# AssetStore Bucketed Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (Project policy forbids subagent-driven development — execute inline.)

**Goal:** Replace the flat concrete `AssetStore` with an `AssetStore` Protocol + `InMemoryAssetStore` implementation providing flat named buckets, per-bucket byte budgets, copy-promotion, and per-bucket TTL cleanup.

**Architecture:** `server/asset_store.py` is rewritten. `AssetStore` becomes a `typing.Protocol`; `InMemoryAssetStore` is the in-memory implementation, constructed with an injectable `dict[str, BucketPolicy]` registry (seeded defaults). `bucket` replaces `kind` everywhere with **no compatibility alias**. `insert` → `write`. All callers and tests migrate to the new shape.

**Tech Stack:** Python 3.11+, `dataclasses`, `threading.RLock`, Pillow (`PIL.Image`) for promotion image-decode validation, `pytest`.

**Spec:** `docs/superpowers/specs/2026-07-04-asset-store-bucketed-interface-design.md`

## Global Constraints

- **No shims, no aliases.** Rename directly to the new shape. Do not keep `insert`, `kind`, or a global `byte_budget` alive.
- **Canonical field is `bucket`** on `AssetEntry` and everywhere it surfaces.
- `AssetStore` is a `Protocol`; `InMemoryAssetStore` is the impl; `get_store()` is annotated to return `AssetStore`.
- Bucket registry is **seeded in code** (`_DEFAULT_BUCKETS`) and **injectable** via `InMemoryAssetStore(buckets=...)`. No `conf/` wiring in this work.
- **Per-bucket byte budgets.** No global cap. Eviction runs within one bucket.
- **Admission fails closed.** The store never evicts the entry it just admitted and never admits an entry it cannot keep.
- Store remains process-scoped (lost on restart).

## Sequencing note (suite goes red mid-plan)

Because the rename carries no compatibility alias, the moment Task 1 lands the new module, test files that still use the old API fail to import/collect. This is expected. **Each task's Run commands target only that task's own files.** The full test suite is green only after Task 6, which has a final step running everything.

Run tests via the repo's Python (Miniforge base): `source /Users/darkbit1001/miniforge3/bin/activate base` once per shell, then `python -m pytest ...`.

## File Structure

- **Rewrite:** `server/asset_store.py` — `BucketPolicy`, `AssetEntry` (with `bucket`), `AssetStore` Protocol, `InMemoryAssetStore`, `_DEFAULT_BUCKETS`, `get_store()`.
- **Rewrite:** `tests/test_asset_store.py` — new tests for the bucketed store (accumulated across Tasks 1–4).
- **Modify:** `server/upload_routes.py` — `write("upload", …)`, `cleanup_expired()` (no `ttl_s`), drop `TTL_S`, docstring.
- **Modify:** `server/controlnet_preprocessing.py:56` — `write("control_map", …)`.
- **Modify (test migration):** `tests/test_upload_routes.py`, `tests/test_ws_routes.py`, `tests/test_controlnet_preprocessing.py`, `tests/test_controlnet_execution.py`, `tests/test_controlnet_http_contract.py`, `tests/test_controlnet_success_contract.py`, `tests/test_controlnet_acceptance.py`.

---

### Task 1: Protocol, data model, `write`/`resolve`, module singleton

**Files:**
- Rewrite: `server/asset_store.py`
- Rewrite: `tests/test_asset_store.py`

**Interfaces:**
- Produces:
  - `BucketPolicy(name: str, byte_budget: int, ttl_s: float | None, pinnable: bool = True)` (frozen dataclass)
  - `AssetEntry(ref, data, bucket, created_at, last_accessed, byte_size, metadata, pin_count)`
  - `class AssetStore(Protocol)` with `write`, `resolve`, `promote`, `pin`, `unpin`, `cleanup_expired`, `bucket_bytes`, `total_bytes`, `buckets`
  - `InMemoryAssetStore(buckets: dict[str, BucketPolicy] | None = None)`
  - `InMemoryAssetStore.write(bucket: str, data: bytes, metadata: dict | None = None) -> str`
  - `InMemoryAssetStore.resolve(ref: str) -> AssetEntry`
  - `_DEFAULT_BUCKETS: dict[str, BucketPolicy]` (`upload` 128 MB/ttl 300, `control_map` 256 MB/ttl None, `ref_image` 128 MB/ttl None)
  - `get_store() -> AssetStore`

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_asset_store.py` with:

```python
import io
import time

import pytest
from PIL import Image

from server.asset_store import (
    AssetEntry,
    BucketPolicy,
    InMemoryAssetStore,
    get_store,
)

MB = 1024 * 1024


def _png(color=(255, 0, 0), size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _store(**buckets: BucketPolicy) -> InMemoryAssetStore:
    if buckets:
        return InMemoryAssetStore(buckets=buckets)
    return InMemoryAssetStore()


# --- write / resolve ---

def test_write_returns_hex_ref():
    store = _store()
    ref = store.write("upload", b"hello")
    assert isinstance(ref, str) and len(ref) == 32


def test_resolve_returns_entry_with_bucket():
    store = _store()
    ref = store.write("upload", b"hello world")
    entry = store.resolve(ref)
    assert entry.data == b"hello world"
    assert entry.bucket == "upload"
    assert entry.byte_size == len(b"hello world")


def test_write_stores_metadata():
    store = _store()
    ref = store.write("control_map", b"pixels", metadata={"control_type": "canny"})
    assert store.resolve(ref).metadata["control_type"] == "canny"


def test_resolve_missing_raises_key_error():
    store = _store()
    with pytest.raises(KeyError, match="not found"):
        store.resolve("nonexistent")


def test_resolve_returns_snapshot_not_live_entry():
    store = _store()
    ref = store.write("upload", b"hi", metadata={"source": "user"})
    entry = store.resolve(ref)
    entry.pin_count = 99
    entry.metadata["source"] = "mutated"
    fresh = store.resolve(ref)
    assert fresh.pin_count == 0
    assert fresh.metadata["source"] == "user"


# --- bucket registry ---

def test_write_unknown_bucket_raises_value_error():
    store = _store()
    with pytest.raises(ValueError, match="unknown bucket"):
        store.write("uploads", b"hello")


def test_buckets_lists_registered_names():
    store = _store()
    assert set(store.buckets()) == {"upload", "control_map", "ref_image"}


# --- oversize fail-closed ---

def test_write_oversize_asset_raises_and_admits_nothing():
    store = _store(tiny=BucketPolicy("tiny", byte_budget=4, ttl_s=None))
    with pytest.raises(ValueError, match="exceeds bucket budget"):
        store.write("tiny", b"aaaaa")  # 5 > 4
    assert store.bucket_bytes("tiny") == 0
    assert store.total_bytes() == 0


# --- module singleton ---

def test_get_store_returns_shared_singleton():
    assert get_store() is get_store()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_asset_store.py -q`
Expected: FAIL at import — `ImportError: cannot import name 'BucketPolicy'` (module still has the old shape).

- [ ] **Step 3: Rewrite `server/asset_store.py`**

Replace the entire file with:

```python
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
    "control_map": BucketPolicy("control_map", byte_budget=256 * MB, ttl_s=None),
    "ref_image": BucketPolicy("ref_image", byte_budget=128 * MB, ttl_s=None),
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

    def bucket_bytes(self, bucket: str) -> int:
        with self._lock:
            self._policy(bucket)
            return self._bucket_bytes[bucket]

    def total_bytes(self) -> int:
        with self._lock:
            return sum(self._bucket_bytes.values())

    def buckets(self) -> list[str]:
        return list(self._policies)

    def _evict_to_budget(self, bucket: str, protect: str) -> None:
        # Caller holds self._lock. `protect` is the just-admitted ref, which must
        # never evict itself; if only pinned entries plus `protect` remain over
        # budget, admission fails closed and `protect` is rolled back.
        budget = self._policies[bucket].byte_budget
        while self._bucket_bytes[bucket] > budget:
            candidates = [
                e
                for e in self._entries.values()
                if e.bucket == bucket and e.pin_count == 0 and e.ref != protect
            ]
            if not candidates:
                oversize = self._entries[protect].byte_size
                self._remove(protect)
                raise ValueError(
                    f"bucket {bucket!r} has insufficient evictable capacity "
                    f"for {oversize} bytes"
                )
            oldest = min(candidates, key=lambda e: e.last_accessed)
            self._remove(oldest.ref)


_DEFAULT_STORE: AssetStore = InMemoryAssetStore()


def get_store() -> AssetStore:
    return _DEFAULT_STORE
```

Note: `promote`, `pin`, `unpin`, `cleanup_expired` are declared on the Protocol but not yet implemented on `InMemoryAssetStore` — Tasks 2–4 add them. The Task 1 tests do not call them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asset_store.py -q`
Expected: PASS (all Task 1 tests green).

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "feat(asset-store): Protocol + InMemoryAssetStore write/resolve, buckets, oversize fail-closed (STABL — asset-store bucketed interface) — next: Task 2 pin/unpin + per-bucket eviction"
```

---

### Task 2: `pin`/`unpin` (pinnable contract) + per-bucket eviction + `bucket_bytes`

**Files:**
- Modify: `server/asset_store.py`
- Modify: `tests/test_asset_store.py`

**Interfaces:**
- Consumes: `InMemoryAssetStore`, `BucketPolicy`, `_evict_to_budget` from Task 1.
- Produces:
  - `InMemoryAssetStore.pin(ref: str) -> None`
  - `InMemoryAssetStore.unpin(ref: str) -> None`
  - Eviction guarantees: LRU by `last_accessed` within the target bucket; pinned entries survive; fail-closed when pinned entries + new admission exceed budget.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_asset_store.py`:

```python
# --- pinning ---

def test_pin_missing_raises_key_error():
    store = _store()
    with pytest.raises(KeyError, match="not found"):
        store.pin("missing")


def test_unpin_missing_raises_key_error():
    store = _store()
    with pytest.raises(KeyError, match="not found"):
        store.unpin("missing")


def test_unpin_zero_pin_count_raises_value_error():
    store = _store()
    ref = store.write("upload", b"hi")
    with pytest.raises(ValueError, match="already 0"):
        store.unpin(ref)


def test_pin_on_non_pinnable_bucket_raises():
    store = _store(fixed=BucketPolicy("fixed", byte_budget=MB, ttl_s=None, pinnable=False))
    ref = store.write("fixed", b"data")
    with pytest.raises(ValueError, match="not pinnable"):
        store.pin(ref)


def test_unpin_on_non_pinnable_bucket_raises():
    store = _store(fixed=BucketPolicy("fixed", byte_budget=MB, ttl_s=None, pinnable=False))
    ref = store.write("fixed", b"data")
    with pytest.raises(ValueError, match="not pinnable"):
        store.unpin(ref)


# --- per-bucket eviction ---

def test_evicts_lru_within_bucket_when_budget_exceeded():
    store = _store(b=BucketPolicy("b", byte_budget=10, ttl_s=None))
    ref_a = store.write("b", b"aaaaaaa")  # 7
    ref_b = store.write("b", b"bbbb")     # 4 -> total 11 > 10, evict a
    with pytest.raises(KeyError):
        store.resolve(ref_a)
    assert store.resolve(ref_b).data == b"bbbb"


def test_eviction_is_isolated_per_bucket():
    store = _store(
        a=BucketPolicy("a", byte_budget=10, ttl_s=None),
        b=BucketPolicy("b", byte_budget=10, ttl_s=None),
    )
    a_ref = store.write("a", b"aaaaaaa")  # 7 in bucket a
    b1 = store.write("b", b"bbbbbbb")     # 7 in bucket b
    b2 = store.write("b", b"cccc")        # 4 -> bucket b over budget, evicts b1
    assert store.resolve(a_ref).data == b"aaaaaaa"  # bucket a untouched
    with pytest.raises(KeyError):
        store.resolve(b1)
    assert store.resolve(b2).data == b"cccc"


def test_evicts_oldest_unpinned_by_last_accessed():
    store = _store(b=BucketPolicy("b", byte_budget=15, ttl_s=None))
    ref_a = store.write("b", b"a" * 6)
    ref_b = store.write("b", b"b" * 6)
    store.resolve(ref_b)  # bumps ref_b last_accessed -> ref_a now oldest
    ref_c = store.write("b", b"c" * 6)  # total 18 > 15, evict ref_a
    with pytest.raises(KeyError):
        store.resolve(ref_a)
    assert store.resolve(ref_b).data == b"b" * 6
    assert store.resolve(ref_c).data == b"c" * 6


def test_admission_fails_closed_when_pins_exceed_budget():
    store = _store(b=BucketPolicy("b", byte_budget=10, ttl_s=None))
    ref_a = store.write("b", b"aaaaaaa")  # 7
    store.pin(ref_a)
    with pytest.raises(ValueError, match="insufficient evictable capacity"):
        store.write("b", b"bbbb")  # 7 pinned + 4 = 11 > 10, cannot evict pin
    # rolled back: only ref_a remains
    assert store.resolve(ref_a).data == b"aaaaaaa"
    assert store.bucket_bytes("b") == 7


def test_unpin_allows_later_eviction():
    store = _store(b=BucketPolicy("b", byte_budget=10, ttl_s=None))
    ref_a = store.write("b", b"aaaaaaa")
    store.pin(ref_a)
    store.unpin(ref_a)
    ref_b = store.write("b", b"bbbb")  # a now unpinned -> evicted
    with pytest.raises(KeyError):
        store.resolve(ref_a)
    assert store.resolve(ref_b).data == b"bbbb"


def test_bucket_bytes_tracks_admission_and_eviction():
    store = _store(b=BucketPolicy("b", byte_budget=10, ttl_s=None))
    store.write("b", b"aaaaaaa")   # 7
    store.write("b", b"bbbb")      # evicts a -> 4
    assert store.bucket_bytes("b") == 4
    assert store.total_bytes() == 4
```

Delete the placeholder test `test_pinned_ref_survives_budget_pressure` (it documents the scenario but asserts nothing) — replace it entirely with `test_admission_fails_closed_when_pins_exceed_budget` above. Do not keep both.

Corrected: remove the `test_pinned_ref_survives_budget_pressure` block from the paste; keep only `test_admission_fails_closed_when_pins_exceed_budget`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_asset_store.py -q -k "pin or evict or admission or bucket_bytes"`
Expected: FAIL — `AttributeError: 'InMemoryAssetStore' object has no attribute 'pin'`.

- [ ] **Step 3: Add `pin`/`unpin` to `InMemoryAssetStore`**

Insert these methods after `resolve` in `server/asset_store.py`:

```python
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
```

(Per-bucket eviction and `bucket_bytes` already exist from Task 1; these tests exercise them.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asset_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "feat(asset-store): pin/unpin with pinnable contract, per-bucket LRU eviction, fail-closed under pin pressure — next: Task 3 per-bucket TTL cleanup"
```

---

### Task 3: `cleanup_expired` — per-bucket TTL

**Files:**
- Modify: `server/asset_store.py`
- Modify: `tests/test_asset_store.py`

**Interfaces:**
- Consumes: `InMemoryAssetStore`, `_remove`, `BucketPolicy.ttl_s`.
- Produces: `InMemoryAssetStore.cleanup_expired() -> list[str]` — no arguments; evicts unpinned entries older than their bucket's `ttl_s`; buckets with `ttl_s is None` are never age-expired.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_asset_store.py`:

```python
# --- per-bucket TTL cleanup ---

def test_cleanup_expired_removes_old_upload():
    store = _store()
    ref = store.write("upload", b"old")
    store._entries[ref].created_at = time.time() - 400  # upload ttl = 300
    removed = store.cleanup_expired()
    assert ref in removed
    with pytest.raises(KeyError):
        store.resolve(ref)


def test_cleanup_expired_preserves_ttl_none_bucket():
    store = _store()
    ref = store.write("control_map", b"cmap")  # ttl None
    store._entries[ref].created_at = time.time() - 9999
    removed = store.cleanup_expired()
    assert ref not in removed
    assert store.resolve(ref).data == b"cmap"


def test_cleanup_expired_ignores_pinned_entries():
    store = _store()
    ref = store.write("upload", b"pinned-old")
    store.pin(ref)
    store._entries[ref].created_at = time.time() - 400
    removed = store.cleanup_expired()
    assert ref not in removed
    assert store.resolve(ref).data == b"pinned-old"


def test_cleanup_expired_uses_created_at_not_last_accessed():
    store = _store()
    ref = store.write("upload", b"old")
    store.resolve(ref)  # bumps last_accessed only
    store._entries[ref].created_at = time.time() - 400
    store._entries[ref].last_accessed = time.time()
    removed = store.cleanup_expired()
    assert ref in removed


def test_cleanup_expired_reduces_bucket_bytes():
    store = _store()
    old = store.write("upload", b"abcd")
    keep = store.write("upload", b"xyz")
    store._entries[old].created_at = time.time() - 400
    removed = store.cleanup_expired()
    assert removed == [old]
    assert store.resolve(keep).data == b"xyz"
    assert store.bucket_bytes("upload") == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_asset_store.py -q -k cleanup`
Expected: FAIL — `AttributeError: 'InMemoryAssetStore' object has no attribute 'cleanup_expired'`.

- [ ] **Step 3: Add `cleanup_expired`**

Insert after `unpin` in `server/asset_store.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asset_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "feat(asset-store): cleanup_expired walks per-bucket TTL policies (no ttl_s arg) — next: Task 4 promote"
```

---

### Task 4: `promote` — copy, image-decode validation, merge-forward metadata

**Files:**
- Modify: `server/asset_store.py`
- Modify: `tests/test_asset_store.py`

**Interfaces:**
- Consumes: `InMemoryAssetStore.write`, `_require`, `PIL.Image`.
- Produces: `InMemoryAssetStore.promote(ref: str, target_bucket: str) -> str` — resolves the source, validates it decodes as an image, writes a **copy** into `target_bucket` under a **new ref**, merging source metadata forward under `origin="promoted"`, `source_asset_ref`, decoded `media_type`/`width`/`height`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_asset_store.py`:

```python
# --- promote ---

def test_promote_copies_into_target_bucket_with_new_ref():
    store = _store()
    src = store.write("upload", _png())
    dst = store.promote(src, "ref_image")
    assert dst != src
    assert store.resolve(dst).bucket == "ref_image"
    assert store.resolve(src).bucket == "upload"  # original untouched
    assert store.resolve(dst).data == store.resolve(src).data


def test_promote_merges_source_metadata_forward():
    store = _store()
    src = store.write("upload", _png(), metadata={"provenance": "user-upload", "origin": "ingested"})
    dst = store.promote(src, "ref_image")
    meta = store.resolve(dst).metadata
    assert meta["provenance"] == "user-upload"       # source key preserved
    assert meta["origin"] == "promoted"              # overlay wins on collision
    assert meta["source_asset_ref"] == src
    assert meta["media_type"] == "image/png"
    assert meta["width"] == 8 and meta["height"] == 8


def test_promote_missing_ref_raises_key_error():
    store = _store()
    with pytest.raises(KeyError, match="not found"):
        store.promote("missing", "ref_image")


def test_promote_unknown_target_bucket_raises_value_error():
    store = _store()
    src = store.write("upload", _png())
    with pytest.raises(ValueError, match="unknown bucket"):
        store.promote(src, "nope")


def test_promote_non_image_raises_value_error():
    store = _store()
    src = store.write("upload", b"this is not an image")
    with pytest.raises(ValueError, match="not a decodable image"):
        store.promote(src, "ref_image")


def test_promoted_and_source_have_independent_lifetimes():
    store = _store(
        upload=BucketPolicy("upload", byte_budget=MB, ttl_s=300),
        ref_image=BucketPolicy("ref_image", byte_budget=MB, ttl_s=None),
    )
    src = store.write("upload", _png())
    dst = store.promote(src, "ref_image")
    store._entries[src].created_at = time.time() - 400  # expire the upload
    store.cleanup_expired()
    with pytest.raises(KeyError):
        store.resolve(src)               # source gone
    assert store.resolve(dst).bucket == "ref_image"  # copy survives
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_asset_store.py -q -k promote`
Expected: FAIL — `AttributeError: 'InMemoryAssetStore' object has no attribute 'promote'`.

- [ ] **Step 3: Add `promote`**

Insert after `cleanup_expired` in `server/asset_store.py`:

```python
    def promote(self, ref: str, target_bucket: str) -> str:
        self._policy(target_bucket)  # validate target bucket up front
        with self._lock:
            src = self._require(ref)
            data = src.data
            src_meta = dict(src.metadata)

        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception as exc:
            raise ValueError("asset is not a decodable image") from exc

        # verify() leaves the image unusable; reopen to read format/size.
        img = Image.open(io.BytesIO(data))
        media_type = Image.MIME.get(img.format, f"image/{(img.format or 'png').lower()}")
        width, height = img.size

        merged = {
            **src_meta,
            "origin": "promoted",
            "source_asset_ref": ref,
            "media_type": media_type,
            "width": width,
            "height": height,
        }
        return self.write(target_bucket, data, metadata=merged)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asset_store.py -q`
Expected: PASS (full `test_asset_store.py` green — the store is now complete).

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "feat(asset-store): promote(ref, target_bucket) copy + image-decode validation + merge-forward metadata — next: Task 5 migrate production callers"
```

---

### Task 5: Migrate production callers

**Files:**
- Modify: `server/upload_routes.py`
- Modify: `server/controlnet_preprocessing.py:56`
- Modify: `tests/test_upload_routes.py` (singleton reset + `entry.bucket`)

**Interfaces:**
- Consumes: `get_store().write`, `get_store().cleanup_expired`, `entry.bucket`.

- [ ] **Step 1: Update the `test_upload_routes.py` singleton-reset fixture and assertions**

In `tests/test_upload_routes.py`, replace the two `store._total_bytes = 0` lines inside `_clear_store` (lines 13 and 17) with a per-bucket counter reset. The fixture becomes:

```python
@pytest.fixture(autouse=True)
def _clear_store():
    store = get_store()
    with store._lock:
        store._entries.clear()
        store._bucket_bytes = {name: 0 for name in store._policies}
    yield
    with store._lock:
        store._entries.clear()
        store._bucket_bytes = {name: 0 for name in store._policies}
```

Then change the assertion at line 63 from `entry.kind == "upload"` to `entry.bucket == "upload"`.

- [ ] **Step 2: Run the upload-routes tests to confirm they fail against old prod code**

Run: `python -m pytest tests/test_upload_routes.py -q`
Expected: FAIL — `upload_routes.py` still calls `get_store().insert(...)`, which no longer exists (`AttributeError`).

- [ ] **Step 3: Migrate `server/upload_routes.py`**

Apply these edits:

Docstring line 7 — replace:
```
Upload entries have kind="upload" and a 5-minute TTL enforced by cleanup_uploads_loop.
```
with:
```
Upload entries live in the "upload" bucket; the bucket policy carries the 5-minute TTL enforced by cleanup_uploads_loop.
```

Delete the module-level `TTL_S = 300` constant (line 21) — the TTL now lives in the `upload` `BucketPolicy`.

Line 30 — replace `ref = get_store().insert("upload", data)` with:
```python
    ref = get_store().write("upload", data)
```

Lines 40–46 — replace the loop body's cleanup call. The function becomes:
```python
async def cleanup_uploads_loop():
    """Background task that purges expired upload entries every 30s."""
    while True:
        await asyncio.sleep(30)
        expired = get_store().cleanup_expired()
        if expired:
            logger.debug("Cleaned %d expired uploads", len(expired))
```

- [ ] **Step 4: Migrate `server/controlnet_preprocessing.py`**

Line 56 — replace `new_ref = store.insert("control_map", result.image_bytes, metadata)` with:
```python
        new_ref = store.write("control_map", result.image_bytes, metadata)
```

- [ ] **Step 5: Run the affected tests to verify they pass**

Run: `python -m pytest tests/test_upload_routes.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/upload_routes.py server/controlnet_preprocessing.py tests/test_upload_routes.py
git commit -m "refactor(asset-store): migrate upload_routes + controlnet_preprocessing to write()/cleanup_expired(); bucket field — next: Task 6 migrate remaining test suite"
```

---

### Task 6: Migrate remaining test suite + full-suite green

**Files:**
- Modify: `tests/test_ws_routes.py`
- Modify: `tests/test_controlnet_preprocessing.py`
- Modify: `tests/test_controlnet_execution.py`
- Modify: `tests/test_controlnet_http_contract.py`
- Modify: `tests/test_controlnet_success_contract.py`
- Modify: `tests/test_controlnet_acceptance.py`

**Interfaces:**
- Consumes: `InMemoryAssetStore`, `BucketPolicy`, `write`, `.bucket`, `total_bytes()`.

Migration rules (apply mechanically across the files below):
1. `from server.asset_store import AssetStore` used to **construct** a store → import `InMemoryAssetStore` and construct that instead. `AssetStore` is now a Protocol and is not instantiable.
2. `AssetStore()` → `InMemoryAssetStore()`.
3. `AssetStore(byte_budget=N)` → a store with a custom registry sized for the test. For a control-map eviction test needing budget `N`: `InMemoryAssetStore(buckets={"upload": BucketPolicy("upload", N, ttl_s=300), "control_map": BucketPolicy("control_map", N, ttl_s=None)})`. Import `BucketPolicy`.
4. `store.insert("upload", …)` / `store.insert("control_map", …)` → `store.write("upload", …)` / `store.write("control_map", …)`.
5. `entry.kind` / `.resolve(x).kind` → `.bucket`.
6. `store.total_bytes` (property) → `store.total_bytes()` (call it).
7. Singleton-reset fixtures using `store._total_bytes = 0` → `store._bucket_bytes = {name: 0 for name in store._policies}`.

- [ ] **Step 1: Confirm the current failure surface**

Run: `python -m pytest tests/test_ws_routes.py tests/test_controlnet_preprocessing.py tests/test_controlnet_execution.py tests/test_controlnet_http_contract.py tests/test_controlnet_success_contract.py tests/test_controlnet_acceptance.py -q`
Expected: FAIL/ERROR — collection or attribute errors referencing `insert`, `kind`, `byte_budget`, or `total_bytes`.

- [ ] **Step 2: Migrate `tests/test_ws_routes.py`**

In the `_clear_store` fixture (lines ~56–61) replace both `store._total_bytes = 0` with `store._bucket_bytes = {name: 0 for name in store._policies}`. Change `get_store().insert("upload", _solid_png_bytes())` (lines 249, 476) to `get_store().write("upload", _solid_png_bytes())`. Change `get_store().resolve(emitted_ref).kind == "control_map"` (lines 292, 548) to `.bucket == "control_map"`.

- [ ] **Step 3: Migrate `tests/test_controlnet_preprocessing.py`**

Change `store = AssetStore()` occurrences to `store = InMemoryAssetStore()` (update the import at line 8 to `from server.asset_store import InMemoryAssetStore`). Change `store.insert("control_map", b"existing-map")` (line 153) and `store.insert("upload", …)` calls to `store.write(...)`. Change `entry.kind == "control_map"` (line 84) to `entry.bucket`.

- [ ] **Step 4: Migrate `tests/test_controlnet_execution.py`**

Update the local import (lines 48, 75) to `from server.asset_store import InMemoryAssetStore`; `store = AssetStore()` → `InMemoryAssetStore()`; `store.insert("control_map", …)` (lines 53, 80, 81) → `store.write("control_map", …)`.

- [ ] **Step 5: Migrate `tests/test_controlnet_http_contract.py` and `tests/test_controlnet_success_contract.py`**

Both construct `AssetStore(byte_budget=64 * 1024 * 1024)` (http_contract lines 65/68/134/137; success_contract lines 83/86) and then `store.insert("upload", _make_png())`. Replace the import with `from server.asset_store import InMemoryAssetStore, BucketPolicy` and the construction with:
```python
    store = InMemoryAssetStore(buckets={
        "upload": BucketPolicy("upload", 64 * 1024 * 1024, ttl_s=300),
        "control_map": BucketPolicy("control_map", 64 * 1024 * 1024, ttl_s=None),
    })
```
Change `store.insert("upload", …)` → `store.write("upload", …)`.

- [ ] **Step 6: Migrate `tests/test_controlnet_acceptance.py`**

Update import (line 20) to `from server.asset_store import InMemoryAssetStore, BucketPolicy`. Replace:
- `store = AssetStore()` → `store = InMemoryAssetStore()`
- `store = AssetStore(byte_budget=25)` (line 132) and `AssetStore(byte_budget=15)` (line 171) → custom-registry stores where the exercised bucket has that budget, e.g. for the control-map eviction test:
```python
    store = InMemoryAssetStore(buckets={
        "upload": BucketPolicy("upload", 25, ttl_s=300),
        "control_map": BucketPolicy("control_map", 25, ttl_s=None),
    })
```
  (use `15` for line 171's store). Match the budget to whichever bucket the test writes into.
- `store.insert("upload"|"control_map", …)` → `store.write(...)`
- `entry.kind == "control_map"` (lines 72, 95) → `.bucket`
- `store.total_bytes` (lines 162, 186) → `store.total_bytes()`

Note: line 133/172 comments say `# 3 bytes` for `b"src"`; those tests insert into `"upload"`. Ensure the custom registry's `upload` budget matches the test's intent (the budget value from the old `byte_budget` arg applies to the bucket under test).

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS (entire suite green). If any failure remains, it is a missed old-API reference — grep for it: `grep -rn --include='*.py' '\.insert(\|\.kind\b\|byte_budget\|\.total_bytes\b\|_total_bytes' server/ tests/` and migrate per the rules above.

- [ ] **Step 8: Drift check on bound files**

Run: `drift refs server/asset_store.py` and `drift refs server/upload_routes.py`. If either is bound and the prose is stale (it references `kind`, `insert`, or a global budget — notably the 2026-04-18 controlnet spec §3 backing-store notes), update the prose first, then `drift link ...`, then `drift check`.

- [ ] **Step 9: Commit**

```bash
git add tests/
git commit -m "refactor(asset-store): migrate remaining test suite to InMemoryAssetStore/write/bucket/total_bytes(); full suite green"
```

---

## Self-Review

**Spec coverage:**
- Protocol + `InMemoryAssetStore` + `get_store` returns Protocol → Task 1. ✓
- Flat buckets, `bucket == kind`, no alias → Tasks 1 (field), 5–6 (migration). ✓
- Copy-promote, new ref → Task 4. ✓
- Per-bucket budgets, per-bucket LRU → Tasks 1 (`_evict_to_budget`), 2 (tests). ✓
- Oversize fail-closed + pinned-pressure fail-closed → Tasks 1, 2. ✓
- Per-bucket TTL `cleanup_expired()` (no arg) → Task 3. ✓
- `pinnable` raise-contract → Task 2. ✓
- Promotion image-decode + merge-forward metadata → Task 4. ✓
- Seeded, injectable registry → Task 1 (`_DEFAULT_BUCKETS`, constructor). ✓
- Migration surface (prod + tests) → Tasks 5–6. ✓
- Drift on bound files → Task 6 Step 8. ✓

**Placeholder scan:** Task 2 Step 1 intentionally flags and removes a no-assert scenario test; all other steps carry real code. No TBD/TODO.

**Type consistency:** `write`/`resolve`/`promote`/`pin`/`unpin`/`cleanup_expired`/`bucket_bytes`/`total_bytes`/`buckets` names and signatures match between the Protocol (Task 1), the impl, and all call sites. `total_bytes()` is a method (called with `()`) consistently in tests and migration. `bucket` (not `kind`) used consistently. `InMemoryAssetStore(buckets=...)` construction consistent across Tasks 5–6.
