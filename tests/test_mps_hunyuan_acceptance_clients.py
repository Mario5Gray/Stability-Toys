"""Current-scope decisions for the MPS HunyuanDiT acceptance clients (STABL-dvozpehl).

The clients must not assert refusals that the active platform cell no longer
makes. These tests pin the pure decisions and the exact platform message text.
"""

import sys
from pathlib import Path

import pytest

SPIKES = Path(__file__).resolve().parents[1] / "spikes"
if str(SPIKES) not in sys.path:
    sys.path.insert(0, str(SPIKES))

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
