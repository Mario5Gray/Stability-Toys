import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.upload_routes import upload_router, resolve_file_ref
from server.asset_store import get_store


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


app = FastAPI()
app.include_router(upload_router)
_client_cm = TestClient(app)
client = _client_cm.__enter__()


@pytest.fixture(scope="module", autouse=True)
def _close_test_client():
    yield
    _client_cm.__exit__(None, None, None)


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
    assert entry.bucket == "upload"


import io
from PIL import Image


def _png(w=8, h=6):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(type_label, data, filename="m.png"):
    files = {"file": (filename, data, "image/png")}
    fields = {"type": type_label} if type_label is not None else None
    return client.post("/v1/upload", files=files, data=fields)


def test_canny_routes_to_control_map_with_dimensions():
    resp = _upload("canny", _png(8, 6))
    assert resp.status_code == 200
    body = resp.json()
    assert body["bucket"] == "control_map"
    assert body["width"] == 8 and body["height"] == 6
    assert get_store().resolve(body["fileRef"]).bucket == "control_map"


@pytest.mark.parametrize("type_label, bucket", [
    ("canny", "control_map"),
    ("depth", "control_map"),
    ("pose", "control_map"),
    ("image", "ref_image"),
    ("ref", "ref_image"),
])
def test_routed_types_map_to_expected_bucket(type_label, bucket):
    resp = _upload(type_label, _png())
    assert resp.status_code == 200
    assert resp.json()["bucket"] == bucket
    assert get_store().resolve(resp.json()["fileRef"]).bucket == bucket


def test_unknown_type_falls_back_to_upload_bucket():
    resp = _upload("wat", _png())
    assert resp.status_code == 200
    body = resp.json()
    assert body["bucket"] == "upload"
    assert "width" not in body
    assert "height" not in body
    assert get_store().resolve(body["fileRef"]).bucket == "upload"


def test_no_type_uses_upload_bucket_backcompat():
    resp = _upload(None, b"not-an-image-bytes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["bucket"] == "upload"
    assert isinstance(body["fileRef"], str)
    assert "width" not in body and "height" not in body


def test_control_map_rejects_non_image():
    resp = _upload("canny", b"this is not an image")
    assert resp.status_code == 400
    assert "control_map" in resp.json()["detail"]


def test_empty_upload_still_400():
    resp = _upload("canny", b"")
    assert resp.status_code == 400
    assert "Empty" in resp.json()["detail"]
