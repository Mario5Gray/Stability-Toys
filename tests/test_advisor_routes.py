from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.advisor_routes import router as advisor_router


def _make_app():
    app = FastAPI()
    app.include_router(advisor_router)
    return app


def test_advisor_digest_route_returns_digest_payload():
    app = _make_app()
    client = TestClient(app)

    payload = {
        "gallery_id": "gal_1",
        "evidence": {"version": 1, "gallery_id": "gal_1", "items": []},
        "digest_text": "digest text",
        "evidence_fingerprint": "sha256:abc",
        "mode": "sdxl-general",
        "length_limit": 120,
    }

    with patch("server.advisor_routes.generate_digest", new=AsyncMock(return_value=payload)):
        res = client.post(
            "/api/advisors/digest",
            json={
                "gallery_id": "gal_1",
                "evidence": {"version": 1, "gallery_id": "gal_1", "items": []},
            },
        )

    assert res.status_code == 200
    assert res.json()["digest_text"] == "digest text"


def test_advisor_digest_route_returns_400_on_validation_failure():
    app = _make_app()
    client = TestClient(app)

    with patch("server.advisor_routes.generate_digest", new=AsyncMock(side_effect=ValueError("bad request"))):
        res = client.post(
            "/api/advisors/digest",
            json={
                "gallery_id": "gal_1",
                "evidence": {"version": 1, "gallery_id": "gal_1", "items": []},
            },
        )

    assert res.status_code == 400
    assert "bad request" in res.json()["detail"]
