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
