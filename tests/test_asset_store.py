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
