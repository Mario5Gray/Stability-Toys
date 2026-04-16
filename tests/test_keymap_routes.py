from fastapi.testclient import TestClient
from fastapi import FastAPI
from server.keymap_routes import router as keymap_router


def test_get_keymap_defaults_returns_mapping():
    app = FastAPI()
    app.include_router(keymap_router)
    client = TestClient(app)

    resp = client.get("/api/keymap/defaults")
    assert resp.status_code == 200
    body = resp.json()

    assert "keymap" in body
    assert body["keymap"]["delete"]["code"] == "Backspace"
    assert body["keymap"]["next"]["code"] == "ArrowRight"
    assert body["keymap"]["open_new_tab"]["code"] == "Space"
    assert body["keymap"]["zoom"]["code"] == "Enter"


def test_get_keymap_defaults_handles_missing_file(tmp_path, monkeypatch):
    from server import keymap_routes

    monkeypatch.setattr(keymap_routes, "KEYMAP_CONFIG_PATH", str(tmp_path / "missing.yml"))
    app = FastAPI()
    app.include_router(keymap_routes.router)
    client = TestClient(app)

    resp = client.get("/api/keymap/defaults")
    assert resp.status_code == 200
    assert resp.json()["keymap"] == {}
