"""Current-scope decisions for the MPS HunyuanDiT acceptance clients (STABL-dvozpehl).

The clients must not assert refusals that the active platform cell no longer
makes. These tests pin the pure decisions and the exact platform message text.
"""

import io
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

SPIKES = Path(__file__).resolve().parents[1] / "spikes"
if str(SPIKES) not in sys.path:
    sys.path.insert(0, str(SPIKES))

import mps_hunyuan_canny_acceptance as canny_client  # noqa: E402 - spikes/ is not a package
import mps_hunyuan_depth_acceptance as depth_client  # noqa: E402 - spikes/ is not a package
import mps_hunyuan_pose_acceptance as pose_client  # noqa: E402 - spikes/ is not a package
import mps_hunyuan_scope as scope  # noqa: E402 - spikes/ is not a package


def _status(
    *,
    backend="mps",
    current_mode="HunyuanDiT",
    allowed=("canny", "depth", "pose"),
    max_attachments: int | None = 1,
    combined=False,
    include_scope=True,
    include_combined=True,
):
    capabilities = {}
    if include_scope:
        capabilities["controlnet_scope"] = {
            "allowed_control_types": None if allowed is None else list(allowed),
            "max_attachments": max_attachments,
        }
    if include_combined:
        capabilities["supports_img2img_and_controlnet"] = combined
    return {
        "backend": backend,
        "current_mode": current_mode,
        "capabilities": capabilities,
    }


def _scope(allowed=("canny", "depth", "pose"), max_attachments: int | None = 1):
    return {
        "allowed_control_types": None if allowed is None else list(allowed),
        "max_attachments": max_attachments,
    }


# --- validate_status ---


def test_a_valid_hunyuandit_status_returns_its_scope_and_combined_flag():
    status = _status(combined=False)

    scope_values, combined = scope.validate_status(status, "HunyuanDiT")

    assert scope_values == _scope()
    assert combined is False


def test_a_different_active_mode_is_rejected():
    status = _status(current_mode="lcm-general")

    with pytest.raises(scope.StatusError, match="current_mode"):
        scope.validate_status(status, "HunyuanDiT")


def test_an_unknown_client_mode_is_rejected_before_any_control_decision():
    status = _status(current_mode="UnknownMode")

    with pytest.raises(scope.StatusError, match="no family mapping"):
        scope.validate_status(status, "UnknownMode")


def test_a_non_mps_backend_is_rejected():
    status = _status(backend="cuda")

    with pytest.raises(scope.StatusError, match="backend"):
        scope.validate_status(status, "HunyuanDiT")


def test_a_missing_scope_is_rejected():
    status = _status(include_scope=False)

    with pytest.raises(scope.StatusError, match="controlnet_scope"):
        scope.validate_status(status, "HunyuanDiT")


def test_a_malformed_scope_is_rejected():
    status = _status()
    status["capabilities"]["controlnet_scope"]["allowed_control_types"] = "canny"

    with pytest.raises(scope.StatusError, match="allowed_control_types"):
        scope.validate_status(status, "HunyuanDiT")


def test_a_scope_without_max_attachments_is_rejected():
    status = _status()
    del status["capabilities"]["controlnet_scope"]["max_attachments"]

    with pytest.raises(scope.StatusError, match="max_attachments"):
        scope.validate_status(status, "HunyuanDiT")


def test_a_missing_combined_flag_is_rejected():
    status = _status(include_combined=False)

    with pytest.raises(scope.StatusError, match="supports_img2img_and_controlnet"):
        scope.validate_status(status, "HunyuanDiT")


def test_null_scope_fields_mean_no_platform_narrowing():
    status = _status(allowed=None, max_attachments=None)

    scope_values, combined = scope.validate_status(status, "HunyuanDiT")

    assert scope_values == _scope(allowed=None, max_attachments=None)
    assert scope.evaluate(scope_values, combined, "depth") == "not_applicable"
    assert scope.evaluate(scope_values, combined, "pose") == "not_applicable"
    assert scope.evaluate(scope_values, combined, "two_attachments") == "not_applicable"


# --- evaluate ---


@pytest.mark.parametrize(
    ("allowed", "max_attachments", "combined", "control", "expected"),
    [
        (("canny",), 1, False, "depth", "applies"),
        (("canny",), 1, False, "pose", "applies"),
        (("canny",), 1, False, "two_attachments", "applies"),
        (("canny",), 1, False, "combined_pose", "applies"),
        (("canny", "depth", "pose"), 1, False, "depth", "not_applicable"),
        (("canny", "depth", "pose"), 1, False, "pose", "not_applicable"),
        (("canny", "depth", "pose"), 1, False, "two_attachments", "applies"),
        (("canny", "depth", "pose"), 1, False, "combined_pose", "applies"),
        (("canny", "depth", "pose"), 2, True, "depth", "not_applicable"),
        (("canny", "depth", "pose"), 2, True, "pose", "not_applicable"),
        (("canny", "depth", "pose"), 2, True, "two_attachments", "not_applicable"),
        (("canny", "depth", "pose"), 2, True, "combined_pose", "not_applicable"),
    ],
)
def test_evaluate_decides_each_control_from_the_active_cell(
    allowed, max_attachments, combined, control, expected
):
    assert (
        scope.evaluate(_scope(allowed, max_attachments), combined, control) == expected
    )


def test_evaluate_rejects_an_unknown_control():
    with pytest.raises(scope.StatusError, match="unknown control"):
        scope.evaluate(_scope(), False, "anything")


def test_evaluate_rejects_a_scope_missing_the_field_it_needs():
    incomplete = {"allowed_control_types": ["canny"]}

    with pytest.raises(scope.StatusError, match="max_attachments"):
        scope.evaluate(incomplete, False, "two_attachments")


# --- expected_message ---


def test_expected_message_names_the_live_allowed_types():
    assert scope.expected_message(_scope(allowed=("canny",)), "depth", "hunyuandit") == (
        "backend 'mps' does not support control_type 'depth' for family "
        "'hunyuandit'; allowed: canny"
    )


def test_expected_message_names_the_live_attachment_limit():
    assert scope.expected_message(_scope(max_attachments=1), "two_attachments", "hunyuandit") == (
        "backend 'mps' admits at most 1 controlnet attachment(s) for family "
        "'hunyuandit', got 2"
    )


def test_expected_message_builds_the_combined_operation_refusal():
    assert scope.expected_message(_scope(), "combined_pose", "hunyuandit") == (
        "family 'hunyuandit' does not support the img2img+controlnet operation"
    )


def test_expected_message_follows_the_supplied_family():
    message = scope.expected_message(_scope(), "combined_pose", "someotherfamily")

    assert "family 'someotherfamily' does not support" in message
    assert "hunyuandit" not in message


def test_expected_message_rejects_an_unknown_control():
    with pytest.raises(scope.StatusError, match="unknown control"):
        scope.expected_message(_scope(), "anything", "hunyuandit")


# --- client fixtures ---


def test_the_canny_fixture_follows_its_control_type():
    entry = canny_client.attachment(source_ref="src", control_type="depth")

    assert entry["control_type"] == "depth"
    assert entry["model_id"] == "hunyuandit-depth"
    assert entry["preprocess"]["id"] == "depth"


def test_the_canny_fixture_keeps_its_default():
    assert canny_client.attachment(source_ref="src") == {
        "attachment_id": "cn_1",
        "control_type": "canny",
        "model_id": "hunyuandit-canny",
        "strength": 1.0,
        "source_asset_ref": "src",
        "preprocess": {
            "id": "canny",
            "options": {"low_threshold": 100, "high_threshold": 200},
        },
    }


def test_the_canny_fixture_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="control_type"):
        canny_client.attachment(source_ref="src", control_type="pose")


def test_the_depth_fixture_follows_its_control_type():
    entry = depth_client.attachment(source_ref="src", control_type="pose")

    assert entry["control_type"] == "pose"
    assert entry["model_id"] == "hunyuandit-pose"
    assert entry["preprocess"]["id"] == "pose"


def test_the_depth_fixture_keeps_its_default():
    assert depth_client.attachment(source_ref="src") == {
        "attachment_id": "cn_1",
        "control_type": "depth",
        "model_id": "hunyuandit-depth",
        "strength": 1.0,
        "source_asset_ref": "src",
        "preprocess": {"id": "depth", "options": {}},
    }


def test_the_depth_fixture_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="control_type"):
        depth_client.attachment(source_ref="src", control_type="canny")


# --- control orchestration ---

TWO_ATTACHMENTS_MESSAGE = (
    "backend 'mps' admits at most 1 controlnet attachment(s) for family "
    "'hunyuandit', got 2"
)
COMBINED_MESSAGE = (
    "family 'hunyuandit' does not support the img2img+controlnet operation"
)
CURRENT_CELL = _scope()
WIDENED_CELL = _scope(max_attachments=2)


def _refusal(message):
    return {"ok": False, "terminal": {"type": "job:error", "error": message}}


def _current_cell_messages():
    return {
        "depth": "backend 'mps' does not support control_type 'depth' for family "
        "'hunyuandit'; allowed: canny, depth, pose",
        "pose": "backend 'mps' does not support control_type 'pose' for family "
        "'hunyuandit'; allowed: canny, depth, pose",
        "two_attachments": TWO_ATTACHMENTS_MESSAGE,
        "combined_pose": COMBINED_MESSAGE,
    }


def test_run_controls_skips_widened_controls_and_keeps_them_visible():
    submitted = []
    messages = _current_cell_messages()

    def submit(name, params):
        submitted.append(name)
        return _refusal(messages[name])

    report = scope.run_controls(
        scope_cell=CURRENT_CELL,
        supports_combined=False,
        controls={"depth": {}, "pose": {}, "two_attachments": {}, "combined_pose": {}},
        submit=submit,
        family="hunyuandit",
    )

    assert report["depth"]["decision"] == "not_applicable"
    assert report["pose"]["decision"] == "not_applicable"
    assert report["depth"]["reason"] == "active platform cell does not exclude this control"
    assert submitted == ["two_attachments", "combined_pose"]
    assert report["two_attachments"]["refused_by_the_platform"] is True
    assert report["combined_pose"]["refused_by_the_platform"] is True


def test_run_controls_marks_every_control_not_applicable_on_a_widened_cell():
    report = scope.run_controls(
        scope_cell=WIDENED_CELL,
        supports_combined=True,
        controls={"depth": {}, "pose": {}, "two_attachments": {}, "combined_pose": {}},
        submit=lambda name, params: pytest.fail("a widened control was submitted"),
        family="hunyuandit",
    )

    assert {name: entry["decision"] for name, entry in report.items()} == {
        "depth": "not_applicable",
        "pose": "not_applicable",
        "two_attachments": "not_applicable",
        "combined_pose": "not_applicable",
    }


def test_run_controls_requires_the_whole_platform_message():
    report = scope.run_controls(
        scope_cell=CURRENT_CELL,
        supports_combined=False,
        controls={"two_attachments": {}},
        submit=lambda name, params: _refusal(TWO_ATTACHMENTS_MESSAGE + " (retry)"),
        family="hunyuandit",
    )

    assert report["two_attachments"]["refused_by_the_platform"] is False


def test_run_controls_rejects_a_mode_policy_message():
    report = scope.run_controls(
        scope_cell=CURRENT_CELL,
        supports_combined=False,
        controls={"combined_pose": {}},
        submit=lambda name, params: _refusal(
            "controlnet_policy permits at most 1 attachment"
        ),
        family="hunyuandit",
    )

    assert report["combined_pose"]["refused_by_the_platform"] is False


def test_run_controls_rejects_a_non_job_error_frame():
    report = scope.run_controls(
        scope_cell=CURRENT_CELL,
        supports_combined=False,
        controls={"two_attachments": {}},
        submit=lambda name, params: _refusal(TWO_ATTACHMENTS_MESSAGE)
        | {"terminal": {"type": "error", "error": TWO_ATTACHMENTS_MESSAGE}},
        family="hunyuandit",
    )

    assert report["two_attachments"]["refused_by_the_platform"] is False


def test_run_controls_rejects_a_timeout():
    report = scope.run_controls(
        scope_cell=CURRENT_CELL,
        supports_combined=False,
        controls={"two_attachments": {}},
        submit=lambda name, params: {"ok": False},
        family="hunyuandit",
    )

    assert report["two_attachments"]["refused_by_the_platform"] is False


# --- client orchestration against the live status ---


def _noise_png():
    image = Image.frombytes("RGB", (1024, 1024), os.urandom(1024 * 1024 * 3))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _positive_result(png):
    return {
        "ok": True,
        "png": png,
        "wall_s": 0.1,
        "seed": 4242,
        "backend": "mps",
        "artifacts": [],
        "progress_frames": 0,
    }


def _run_client_main(monkeypatch, client, status, out):
    png = _noise_png()
    calls = []

    def fake_render(base_url, ws_url, params):
        calls.append("render")
        attachments = params.get("controlnets") or []
        if len(attachments) == 2:
            return _refusal(TWO_ATTACHMENTS_MESSAGE)
        return _positive_result(png)

    def fake_get_json(url):
        calls.append("status")
        return status

    monkeypatch.setattr(client, "render", fake_render)
    monkeypatch.setattr(client, "get_json", fake_get_json)
    monkeypatch.setattr(
        client, "post_multipart", lambda *args, **kwargs: {"fileRef": "ref-1"}
    )
    monkeypatch.setattr(sys, "argv", ["spike", "--url", "http://test", "--out", str(out)])

    return client.main(), calls


def test_a_client_reads_scope_after_its_first_successful_render(monkeypatch, tmp_path, capsys):
    rc, calls = _run_client_main(monkeypatch, canny_client, _status(), tmp_path / "out")
    report = json.loads(capsys.readouterr().out)

    assert calls[0] == "render"
    assert calls.index("status") > calls.index("render")
    assert report["models_status"]["current_mode"] == "HunyuanDiT"
    assert report["active_scope"]["allowed_control_types"] == [
        "canny",
        "depth",
        "pose",
    ]
    assert report["supports_img2img_and_controlnet"] is False
    assert report["scope"]["depth"]["decision"] == "not_applicable"
    assert report["scope"]["two_attachments"]["refused_by_the_platform"] is True
    assert "scope_refuses_two" in report["checks"]
    assert calls.count("render") == 6, "the widened Depth control was submitted"
    assert rc == 1, "the fake renders do not satisfy the steering checks"


def test_a_client_fails_when_the_active_family_is_not_hunyuan(monkeypatch, tmp_path, capsys):
    rc, calls = _run_client_main(
        monkeypatch, canny_client, _status(current_mode="lcm-general"), tmp_path / "out"
    )
    report = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert calls[0] == "render"
    assert report["models_status"]["current_mode"] == "lcm-general"
    assert "scope" not in report or report["scope"] == {}


@pytest.mark.parametrize(
    ("client", "applies"),
    [
        (canny_client, ["two_attachments"]),
        (depth_client, ["two_attachments"]),
        (pose_client, ["two_attachments", "combined_pose"]),
    ],
    ids=["canny", "depth", "pose"],
)
def test_each_client_control_set_follows_the_current_cell(client, applies):
    submitted = []

    def submit(name, params):
        submitted.append(name)
        return _refusal(_current_cell_messages()[name])

    report = scope.run_controls(
        scope_cell=CURRENT_CELL,
        supports_combined=False,
        controls=client.build_refusals("src", "alt"),
        submit=submit,
        family="hunyuandit",
    )

    assert [
        name for name, entry in report.items() if entry["decision"] == "applies"
    ] == applies
    assert submitted == applies
