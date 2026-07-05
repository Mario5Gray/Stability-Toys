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
