import io

import pytest
from PIL import Image

from persistence.storage_provider import (
    StorageProvider,
    StorageItem,
    InMemoryStorageProvider,
)
from persistence.filesystem_provider import FilesystemStorageProvider
from server.asset_store import InMemoryAssetStore, BucketPolicy, MB
from server.tiered_asset_store import TieredAssetStore, make_asset_store_provider_from_env


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
