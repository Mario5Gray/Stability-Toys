"""Pure admission-scope decisions for the MPS HunyuanDiT acceptance clients.

The clients must not assert a refusal that the active platform cell no longer
makes. This module turns one `/api/models/status` payload into the decisions
and the exact platform error text. It imports nothing from the server, so the
rules stay testable without a live host. STABL-dvozpehl.
"""

from typing import Any, Callable, Literal

FAMILY_BY_MODE = {"HunyuanDiT": "hunyuandit"}
BACKEND_ID = "mps"
TWO_ATTACHMENTS = 2

CONTROLS = frozenset({"depth", "pose", "two_attachments", "combined_pose"})

Decision = Literal["applies", "not_applicable"]


class StatusError(ValueError):
    """The status payload cannot support a scope decision."""


def validate_status(status: Any, mode: str) -> tuple[dict, bool]:
    """Validate the active-family status and return (scope cell, combined flag).

    A wrong backend, an unmapped mode, a different active mode, or a missing
    field is an error, never "not applicable". A null scope field is valid and
    means the platform does not narrow that dimension.
    """
    if not isinstance(status, dict):
        raise StatusError("status payload is not an object")

    backend = status.get("backend")
    if backend != BACKEND_ID:
        raise StatusError(f"status backend is {backend!r}, expected {BACKEND_ID!r}")

    if mode not in FAMILY_BY_MODE:
        raise StatusError(f"no family mapping for mode {mode!r}")

    current = status.get("current_mode")
    if current != mode:
        raise StatusError(f"current_mode is {current!r}, expected {mode!r}")

    capabilities = status.get("capabilities")
    if not isinstance(capabilities, dict):
        raise StatusError("capabilities is missing or malformed")

    cell = capabilities.get("controlnet_scope")
    if not isinstance(cell, dict):
        raise StatusError("capabilities.controlnet_scope is missing or malformed")

    if "allowed_control_types" not in cell:
        raise StatusError("controlnet_scope.allowed_control_types is missing")
    allowed = cell["allowed_control_types"]
    if allowed is not None and (
        not isinstance(allowed, list)
        or not all(isinstance(entry, str) for entry in allowed)
    ):
        raise StatusError(
            "controlnet_scope.allowed_control_types must be null or a list of strings"
        )

    if "max_attachments" not in cell:
        raise StatusError("controlnet_scope.max_attachments is missing")
    limit = cell["max_attachments"]
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool)
    ):
        raise StatusError(
            "controlnet_scope.max_attachments must be null or an integer"
        )

    if "supports_img2img_and_controlnet" not in capabilities:
        raise StatusError(
            "capabilities.supports_img2img_and_controlnet is missing"
        )
    combined = capabilities["supports_img2img_and_controlnet"]
    if not isinstance(combined, bool):
        raise StatusError(
            "capabilities.supports_img2img_and_controlnet must be a boolean"
        )

    return dict(cell), combined


def evaluate(cell: dict, supports_combined: bool, control: str) -> Decision:
    """Decide whether a negative control still exercises a platform refusal."""
    if control not in CONTROLS:
        raise StatusError(f"unknown control {control!r}")

    if control in ("depth", "pose"):
        if "allowed_control_types" not in cell:
            raise StatusError("controlnet_scope.allowed_control_types is missing")
        allowed = cell["allowed_control_types"]
        if allowed is None:
            return "not_applicable"
        return "applies" if control not in allowed else "not_applicable"

    if control == "two_attachments":
        if "max_attachments" not in cell:
            raise StatusError("controlnet_scope.max_attachments is missing")
        limit = cell["max_attachments"]
        if limit is None:
            return "not_applicable"
        return "applies" if limit < TWO_ATTACHMENTS else "not_applicable"

    return "applies" if not supports_combined else "not_applicable"


def expected_message(cell: dict, control: str, family: str) -> str:
    """Build the exact admission error text for an applicable control.

    The allowed list and the attachment limit come from the live payload, so a
    widened or narrowed scope changes the expected text with it. The family
    comes from the client's mode mapping, because the status payload has none.
    """
    if control not in CONTROLS:
        raise StatusError(f"unknown control {control!r}")

    if control in ("depth", "pose"):
        allowed = cell.get("allowed_control_types")
        if not isinstance(allowed, list):
            raise StatusError(
                "controlnet_scope.allowed_control_types has no refusal text"
            )
        return (
            f"backend '{BACKEND_ID}' does not support control_type '{control}' "
            f"for family '{family}'; allowed: {', '.join(allowed)}"
        )

    if control == "two_attachments":
        limit = cell.get("max_attachments")
        if limit is None:
            raise StatusError("controlnet_scope.max_attachments has no refusal text")
        return (
            f"backend '{BACKEND_ID}' admits at most {limit} controlnet attachment(s) "
            f"for family '{family}', got {TWO_ATTACHMENTS}"
        )

    return f"family '{family}' does not support the img2img+controlnet operation"


def run_controls(
    *,
    scope_cell: dict,
    supports_combined: bool,
    controls: dict,
    submit: Callable[[str, dict], dict],
    family: str,
) -> dict:
    """Submit each applicable control and record a decision for every control.

    A control the active cell no longer excludes is recorded as
    `not_applicable` and is never submitted. A submitted control counts only
    when the terminal frame is `job:error` and its text equals the exact
    platform message.
    """
    report: dict = {}
    for name, params in controls.items():
        decision = evaluate(scope_cell, supports_combined, name)
        if decision == "not_applicable":
            report[name] = {
                "decision": decision,
                "reason": "active platform cell does not exclude this control",
            }
            continue
        result = submit(name, params)
        terminal = result.get("terminal") or {}
        message = terminal.get("error", "")
        expected = expected_message(scope_cell, name, family)
        report[name] = {
            "decision": decision,
            "expected": expected,
            "terminal_type": terminal.get("type"),
            "message": message,
            "refused_by_the_platform": (
                not result.get("ok")
                and terminal.get("type") == "job:error"
                and message == expected
            ),
        }
    return report
