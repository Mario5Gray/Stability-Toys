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
