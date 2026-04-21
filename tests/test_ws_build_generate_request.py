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
