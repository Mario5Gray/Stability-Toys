import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.upload_routes import upload_router, resolve_file_ref
from server.asset_store import get_store


@pytest.fixture(autouse=True)
def _clear_store():
    get_store()._entries.clear()
    yield
    get_store()._entries.clear()


app = FastAPI()
app.include_router(upload_router)
client = TestClient(app)


def test_upload_returns_file_ref():
    resp = client.post(
        "/v1/upload",
        files={"file": ("test.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png")},
    )
    assert resp.status_code == 200
    ref = resp.json()["fileRef"]
    assert isinstance(ref, str) and len(ref) == 32


def test_resolve_file_ref_returns_bytes():
    resp = client.post(
        "/v1/upload",
        files={"file": ("img.png", b"imagedata", "image/png")},
    )
    ref = resp.json()["fileRef"]
    assert resolve_file_ref(ref) == b"imagedata"


def test_resolve_missing_ref_raises():
    with pytest.raises(KeyError, match="not found"):
        resolve_file_ref("nosuchref")


def test_upload_stores_as_upload_kind():
    resp = client.post(
        "/v1/upload",
        files={"file": ("x.png", b"bytes", "image/png")},
    )
    ref = resp.json()["fileRef"]
    entry = get_store().resolve(ref)
    assert entry.kind == "upload"
