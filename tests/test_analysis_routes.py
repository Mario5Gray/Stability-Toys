"""Tests for POST /v1/describe."""
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.analysis_routes import router as analysis_router
from server.mode_config import ModeConfigManager

BASE_YAML = textwrap.dedent("""\
    model_root: /tmp/models
    lora_root: /tmp/loras
    default_mode: SDXL
    resolution_sets:
      default:
        - size: 512x512
          aspect_ratio: "1:1"
    analysis_connections:
      local_vlm:
        endpoint: "http://node2.lan:8080/v1"
      local_detector:
        endpoint: "http://node2.lan:8090"
    analysis_delegates:
      vlm_caption:
        connection: local_vlm
        kind: caption
        model: qwen2.5-vl
      yolo_detect:
        connection: local_detector
        kind: detect
        model: yolo11x
    analysis_profiles:
      default:
        task_routes:
          caption: vlm_caption
          detect: yolo_detect
    modes:
      SDXL:
        model: sdxl/model.safetensors
        analysis_profile: default
""")

CAPTION_ONLY_YAML = BASE_YAML.replace(
    "      detect: yolo_detect\n",
    "",
)

NO_PROFILE_YAML = BASE_YAML.replace("    analysis_profile: default\n", "")


def _manager(tmp_path, yaml_text, subdir):
    d = tmp_path / subdir
    d.mkdir()
    (d / "modes.yml").write_text(yaml_text)
    return ModeConfigManager(str(d))


def _app():
    app = FastAPI()
    app.include_router(analysis_router)
    app.state.generation_runtime = SimpleNamespace(get_current_mode=lambda: "SDXL")
    return app


def _request_body(tasks=None):
    return {
        "targets": [
            {"id": "t1", "url": "http://example.com/a.png"},
            {"id": "t2", "url": "http://example.com/b.png"},
        ],
        "tasks": tasks or [
            {"id": "caption", "kind": "caption", "caption": {}},
            {"id": "detect", "kind": "detect", "detect": {}},
        ],
    }


def _post(mgr, body):
    with patch("server.analysis_routes.get_mode_config", return_value=mgr):
        with TestClient(_app()) as client:
            return client.post("/v1/describe", json=body)


def test_describe_happy_path_pins_run_order(tmp_path):
    res = _post(_manager(tmp_path, BASE_YAML, "base"), _request_body())
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert [(r["task_id"], r["target_id"]) for r in data["runs"]] == [
        ("caption", "t1"),
        ("caption", "t2"),
        ("detect", "t1"),
        ("detect", "t2"),
    ]
    assert all(r["status"] == "succeeded" for r in data["runs"])
    assert [(o["task_id"], o["target_id"]) for o in data["observations"]] == [
        ("caption", "t1"),
        ("caption", "t2"),
        ("detect", "t1"),
        ("detect", "t2"),
    ]


def test_describe_partial_when_kind_unrouted(tmp_path):
    res = _post(_manager(tmp_path, CAPTION_ONLY_YAML, "cap"), _request_body())
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "partial"
    detect_runs = [r for r in data["runs"] if r["task_id"] == "detect"]
    assert all(r["status"] == "skipped" for r in detect_runs)
    assert all(r["error"]["code"] == "analysis_no_supported_delegate" for r in detect_runs)


def test_describe_malformed_json_body_is_analysis_invalid_request(tmp_path):
    mgr = _manager(tmp_path, BASE_YAML, "base")
    with patch("server.analysis_routes.get_mode_config", return_value=mgr):
        with TestClient(_app()) as client:
            res = client.post(
                "/v1/describe",
                content=b"not json",
                headers={"content-type": "application/json"},
            )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_invalid_request"


def test_describe_parse_error_returns_code(tmp_path):
    res = _post(_manager(tmp_path, BASE_YAML, "base"), {"targets": [], "tasks": []})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_invalid_request"


def test_describe_binding_error_returns_code(tmp_path):
    body = _request_body(
        tasks=[
            {
                "id": "caption",
                "kind": "caption",
                "caption": {},
                "target_ids": ["nope"],
            },
        ]
    )
    res = _post(_manager(tmp_path, BASE_YAML, "base"), body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_target_binding_invalid"


def test_describe_unknown_request_mode_is_mode_not_found(tmp_path):
    body = _request_body()
    body["mode"] = "NOPE"
    res = _post(_manager(tmp_path, BASE_YAML, "base"), body)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_mode_not_found"


def test_describe_mode_without_profile_is_profile_not_found(tmp_path):
    res = _post(_manager(tmp_path, NO_PROFILE_YAML, "noprof"), _request_body())
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "analysis_profile_not_found"


def test_describe_reflects_reloaded_config_without_restart(tmp_path):
    managers = {"current": _manager(tmp_path, BASE_YAML, "base")}
    with patch("server.analysis_routes.get_mode_config", side_effect=lambda: managers["current"]):
        with TestClient(_app()) as client:
            first = client.post("/v1/describe", json=_request_body())
            managers["current"] = _manager(tmp_path, CAPTION_ONLY_YAML, "cap")
            second = client.post("/v1/describe", json=_request_body())
    assert first.json()["status"] == "ok"
    assert second.json()["status"] == "partial"


def test_describe_unexpected_exception_is_analysis_internal(tmp_path):
    mgr = _manager(tmp_path, BASE_YAML, "base")
    with patch("server.analysis_routes.get_mode_config", return_value=mgr), patch(
        "server.analysis_routes.build_providers",
        side_effect=RuntimeError("boom"),
    ):
        with TestClient(_app(), raise_server_exceptions=False) as client:
            res = client.post("/v1/describe", json=_request_body())
    assert res.status_code == 500
    assert res.json()["error"]["code"] == "analysis_internal"


def test_describe_serialization_exception_is_analysis_internal(tmp_path):
    mgr = _manager(tmp_path, BASE_YAML, "base")
    with patch("server.analysis_routes.get_mode_config", return_value=mgr), patch(
        "server.analysis_routes.response_to_dict",
        side_effect=ValueError("bad response"),
    ):
        with TestClient(_app(), raise_server_exceptions=False) as client:
            res = client.post("/v1/describe", json=_request_body())
    assert res.status_code == 500
    assert res.json()["error"]["code"] == "analysis_internal"
