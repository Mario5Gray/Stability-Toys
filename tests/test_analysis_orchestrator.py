import pytest

from backends.analysis import parse_describe_request
from backends.analysis.orchestrator import RunPlan, expand_runs

ROUTES = {"caption": "vlm_caption", "detect": "yolo_detect"}


def two_target_payload():
    return {
        "targets": [
            {"id": "t1", "asset_ref": "asset-1"},
            {"id": "t2", "asset_ref": "asset-2", "role": "reference"},
        ],
        "tasks": [
            {"id": "cap1", "kind": "caption", "caption": {}},
            {"id": "det1", "kind": "detect", "target_ids": ["t2"], "detect": {}},
        ],
    }


def test_expand_binds_omitted_target_ids_to_primary_only():
    req = parse_describe_request(two_target_payload())
    runs = expand_runs(req, ROUTES)
    assert (
        RunPlan(task_id="cap1", target_id="t1", delegate="vlm_caption", skip_error=None)
        in runs
    )
    assert not any(r.task_id == "cap1" and r.target_id == "t2" for r in runs)


def test_expand_explicit_target_ids_bind_verbatim():
    req = parse_describe_request(two_target_payload())
    runs = expand_runs(req, ROUTES)
    det = [r for r in runs if r.task_id == "det1"]
    assert det == [RunPlan(task_id="det1", target_id="t2", delegate="yolo_detect", skip_error=None)]


def test_expand_unrouted_kind_produces_skip_plan():
    payload = two_target_payload()
    payload["tasks"].append({"id": "ocr1", "kind": "ocr", "ocr": {}})
    req = parse_describe_request(payload)
    runs = expand_runs(req, ROUTES)
    ocr = [r for r in runs if r.task_id == "ocr1"]
    assert len(ocr) == 1
    assert ocr[0].delegate is None
    assert ocr[0].skip_error.code == "analysis_no_supported_delegate"


def test_expand_is_deterministic_task_major_order():
    req = parse_describe_request(two_target_payload())
    assert expand_runs(req, ROUTES) == expand_runs(req, ROUTES)
    assert [r.task_id for r in expand_runs(req, ROUTES)] == ["cap1", "det1"]
