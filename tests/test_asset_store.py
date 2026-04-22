import time

import pytest

from server.asset_store import AssetEntry, AssetStore


def test_insert_upload_returns_ref():
    store = AssetStore()
    ref = store.insert("upload", b"hello")
    assert isinstance(ref, str) and len(ref) == 32


def test_resolve_returns_entry():
    store = AssetStore()
    ref = store.insert("upload", b"hello world")
    entry = store.resolve(ref)
    assert entry.data == b"hello world"
    assert entry.kind == "upload"
    assert entry.byte_size == len(b"hello world")


def test_resolve_missing_raises_key_error():
    store = AssetStore()
    with pytest.raises(KeyError, match="not found"):
        store.resolve("nonexistent")


def test_insert_control_map_stores_metadata():
    store = AssetStore()
    meta = {"control_type": "canny", "source_asset_ref": "abc"}
    ref = store.insert("control_map", b"pixels", metadata=meta)
    entry = store.resolve(ref)
    assert entry.kind == "control_map"
    assert entry.metadata["control_type"] == "canny"


def test_cleanup_expired_removes_old_uploads():
    store = AssetStore()
    ref = store.insert("upload", b"old")
    store._entries[ref].created_at = time.time() - 400
    removed = store.cleanup_expired(ttl_s=300)
    assert ref in removed
    with pytest.raises(KeyError):
        store.resolve(ref)


def test_cleanup_expired_preserves_control_maps():
    store = AssetStore()
    ref = store.insert("control_map", b"cmap")
    store._entries[ref].created_at = time.time() - 9999
    removed = store.cleanup_expired(ttl_s=300)
    assert ref not in removed
    assert store.resolve(ref).data == b"cmap"


def test_total_bytes_sums_all_entries():
    store = AssetStore()
    store.insert("upload", b"ab")
    store.insert("upload", b"cde")
    assert store.total_bytes == 5


def test_resolve_returns_snapshot_not_live_entry():
    store = AssetStore()
    ref = store.insert("upload", b"hello", metadata={"source": "user"})
    entry = store.resolve(ref)
    entry.pin_count = 99
    entry.metadata["source"] = "mutated"

    fresh = store.resolve(ref)
    assert fresh.pin_count == 0
    assert fresh.metadata["source"] == "user"


def test_pin_missing_raises_key_error():
    store = AssetStore()
    with pytest.raises(KeyError, match="not found"):
        store.pin("missing")


def test_unpin_missing_raises_key_error():
    store = AssetStore()
    with pytest.raises(KeyError, match="not found"):
        store.unpin("missing")


def test_insert_rejects_unknown_kind():
    store = AssetStore()
    with pytest.raises(ValueError, match="unknown asset kind"):
        store.insert("uploads", b"hello")


def test_cleanup_expired_uses_created_at_not_last_accessed():
    store = AssetStore()
    ref = store.insert("upload", b"old")
    store.resolve(ref)
    store._entries[ref].created_at = time.time() - 400
    store._entries[ref].last_accessed = time.time()

    removed = store.cleanup_expired(ttl_s=300)

    assert ref in removed
