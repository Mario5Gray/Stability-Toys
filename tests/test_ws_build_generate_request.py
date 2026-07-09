from server.ws_routes import _build_generate_request


def test_build_generate_request_passes_through_controlnets():
    params = {
        "prompt": "a cat",
        "controlnets": [
            {
                "attachment_id": "cn_1",
                "control_type": "canny",
                "map_asset_ref": "asset_a",
            }
        ],
    }
    req = _build_generate_request(params)
    assert req.controlnets is not None
    assert len(req.controlnets) == 1
    assert req.controlnets[0].control_type == "canny"


def test_build_generate_request_default_controlnets_is_none():
    req = _build_generate_request({"prompt": "a cat"})
    assert req.controlnets is None


def test_build_generate_request_passes_through_init_image_ref_and_controlnets_together():
    params = {
        "prompt": "a cat",
        "init_image_ref": "abc123",
        "denoise_strength": 0.6,
        "controlnets": [
            {
                "attachment_id": "cn_1",
                "control_type": "canny",
                "map_asset_ref": "asset_a",
            }
        ],
    }
    req = _build_generate_request(params)
    assert req.denoise_strength == 0.6
    assert req.controlnets is not None
    assert len(req.controlnets) == 1
    # init_image_ref itself is WS-only and never reaches GenerateRequest (see
    # server/ws_routes.py handle_job_submit, which reads params.get("init_image_ref")
    # directly) — asserting its absence here documents that seam rather than
    # exercising a bug.
    assert not hasattr(req, "init_image_ref")
