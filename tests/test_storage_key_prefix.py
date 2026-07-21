"""The generated storage-key prefix is part of the product's visible surface.

Keys appear in `/storage/<key>` URLs, in `storage_key` on generation responses,
and in `st read` output, so the prefix should say `st`, not the retired `lcm`
name. Old objects keep working: providers resolve a key by exact match and shard
on the UUID segment, so nothing parses the prefix.
"""

import re

from persistence.storage_provider import StorageProvider
from server import lcm_sr_server


KEY_RE = re.compile(r"^st_image:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def test_image_key_prefix_constant_is_st_image():
    assert lcm_sr_server.IMAGE_KEY_PREFIX == "st_image"


def test_generated_image_keys_use_the_st_prefix():
    key = StorageProvider._new_key(lcm_sr_server.IMAGE_KEY_PREFIX)
    assert KEY_RE.match(key), key


def test_no_lcm_image_keys_are_generated_in_server_source():
    from pathlib import Path

    src = Path(lcm_sr_server.__file__).read_text(encoding="utf-8")
    assert '_new_key("lcm_image")' not in src
    assert "_new_key('lcm_image')" not in src


def test_legacy_keys_still_resolve_through_a_provider():
    # A prefix change must not orphan already-stored artifacts.
    provider = __import__(
        "persistence.storage_provider", fromlist=["InMemoryStorageProvider"]
    ).InMemoryStorageProvider()
    legacy = "lcm_image:f42da25b-d51e-421a-ba24-56f787976e96"
    provider.put(legacy, b"old-bytes", content_type="image/png")
    got = provider.get(legacy)
    assert got is not None, "legacy keys must remain retrievable"
