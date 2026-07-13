import pytest

from backends.analysis import (
    AnalysisValidationError,
    DescribeStatus,
    RunStatus,
    TaskKind,
    parse_describe_request,
    response_to_dict,
)
from backends.analysis.contracts import (
    Box,
    DescribeObservation,
    DescribeResponse,
    DescribeRun,
    DetectionObservation,
    RunError,
    TextObservation,
)


def valid_payload():
    return {
        "targets": [{"id": "t1", "asset_ref": "asset-1"}],
        "tasks": [{"id": "cap1", "kind": "caption", "caption": {}}],
    }


def test_parse_valid_request():
    req = parse_describe_request(valid_payload())
    assert req.mode is None
    assert req.targets[0].id == "t1"
    assert req.targets[0].asset_ref == "asset-1"
    assert req.tasks[0].kind == TaskKind.CAPTION
    # params materialize as the typed block matching kind, never a raw dict
    from backends.analysis import CaptionParams
    assert req.tasks[0].caption == CaptionParams()
    assert req.tasks[0].detect is None


def test_parse_typed_detect_params():
    payload = valid_payload()
    payload["tasks"] = [{
        "id": "det1", "kind": "detect",
        "detect": {"labels": ["owl"], "min_confidence": 0.5},
    }]
    from backends.analysis import DetectParams
    req = parse_describe_request(payload)
    assert req.tasks[0].detect == DetectParams(labels=("owl",), min_confidence=0.5)


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda p: p.update(targets=[]), "analysis_invalid_request"),
        (lambda p: p.update(tasks=[]), "analysis_invalid_request"),
        (lambda p: p["targets"][0].update(url="http://x/i.png"), "analysis_invalid_request"),
        (lambda p: p["targets"][0].pop("asset_ref"), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(kind="segment"), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(detect={}), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(target_ids=["nope"]), "analysis_target_binding_invalid"),
        (lambda p: p["targets"][0].update(role="reference"), "analysis_target_binding_invalid"),
    ],
)
def test_parse_rejects_invalid(mutate, code):
    payload = valid_payload()
    mutate(payload)
    with pytest.raises(AnalysisValidationError) as exc:
        parse_describe_request(payload)
    assert exc.value.code == code


@pytest.mark.parametrize(
    "mutate,code",
    [
        # malformed scalar/block types must become analysis_invalid_request,
        # never AttributeError (review blocker)
        (lambda p: p["targets"][0].update(id=123), "analysis_invalid_request"),
        (lambda p: p["targets"][0].update(role=42), "analysis_invalid_request"),
        (lambda p: p["targets"][0].update(asset_ref=7), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(id=123), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(caption="x"), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(target_ids="t1"), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(target_ids=[1, 2]), "analysis_invalid_request"),
        (lambda p: p.update(mode=42), "analysis_invalid_request"),
        (lambda p: p["tasks"][0].update(caption={"prompt": 5}), "analysis_invalid_request"),
    ],
)
def test_parse_rejects_malformed_types(mutate, code):
    payload = valid_payload()
    mutate(payload)
    with pytest.raises(AnalysisValidationError) as exc:
        parse_describe_request(payload)
    assert exc.value.code == code


def test_parse_rejects_malformed_detect_params():
    payload = valid_payload()
    payload["tasks"] = [{"id": "det1", "kind": "detect", "detect": {"labels": "owl"}}]
    with pytest.raises(AnalysisValidationError) as exc:
        parse_describe_request(payload)
    assert exc.value.code == "analysis_invalid_request"


def test_response_to_dict_rejects_unknown_kinds():
    resp = DescribeResponse(
        status=DescribeStatus.OK,
        observations=(
            DescribeObservation(task_id="t", target_id="t1", kind="mask"),
        ),
        artifacts=(),
        runs=(
            DescribeRun(task_id="t", target_id="t1", delegate="d",
                        status=RunStatus.SUCCEEDED),
        ),
    )
    with pytest.raises(ValueError):
        response_to_dict(resp)


def test_explicit_binding_to_non_primary_is_allowed():
    payload = valid_payload()
    payload["targets"][0]["role"] = "reference"
    payload["tasks"][0]["target_ids"] = ["t1"]
    req = parse_describe_request(payload)
    assert req.tasks[0].target_ids == ("t1",)


def test_response_to_dict_wire_shape():
    resp = DescribeResponse(
        status=DescribeStatus.PARTIAL,
        summary=None,
        observations=(
            DescribeObservation(
                task_id="cap1", target_id="t1", kind="text",
                text=TextObservation(content="an owl"),
            ),
            DescribeObservation(
                task_id="det1", target_id="t1", kind="detection",
                detection=DetectionObservation(
                    label="owl", confidence=0.93, box=Box(x=0.1, y=0.2, w=0.3, h=0.4),
                ),
            ),
        ),
        artifacts=(),
        runs=(
            DescribeRun(task_id="cap1", target_id="t1", delegate="vlm_caption",
                        status=RunStatus.SUCCEEDED),
            DescribeRun(task_id="det1", target_id="t1", delegate="",
                        status=RunStatus.SKIPPED,
                        error=RunError(code="analysis_no_supported_delegate",
                                       message="no route for kind detect")),
        ),
    )
    d = response_to_dict(resp)
    assert d["status"] == "partial"
    assert "summary" not in d
    obs0 = d["observations"][0]
    assert obs0 == {"task_id": "cap1", "target_id": "t1", "kind": "text",
                    "text": {"content": "an owl"}}
    obs1 = d["observations"][1]
    assert obs1["detection"]["box"] == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    run1 = d["runs"][1]
    assert run1["status"] == "skipped"
    assert run1["error"]["code"] == "analysis_no_supported_delegate"
    run0 = d["runs"][0]
    assert "error" not in run0
