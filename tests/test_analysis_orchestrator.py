import asyncio

import pytest

from backends.analysis import (
    AnalysisValidationError,
    CaptionParams,
    DescribeRequest,
    DescribeStatus,
    DescribeTarget,
    DescribeTask,
    DetectParams,
    RunStatus,
    TaskKind,
    parse_describe_request,
)
from backends.analysis.orchestrator import AnalysisOrchestrator, RunPlan, expand_runs
from backends.analysis.providers import StubProvider

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


class ExplodingProvider:
    def supports(self, task):
        return True

    async def run(self, provider_run):
        raise RuntimeError("backend unreachable")


def run_describe(orchestrator, payload):
    req = parse_describe_request(payload)
    return asyncio.run(orchestrator.describe(req))


def single_caption_payload():
    return {
        "targets": [{"id": "t1", "asset_ref": "asset-1"}],
        "tasks": [{"id": "cap1", "kind": "caption", "caption": {}}],
    }


def test_all_success_is_ok_with_correlated_observations():
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption"},
        providers={"vlm_caption": StubProvider(kind="caption")},
    )
    resp = run_describe(orch, single_caption_payload())
    assert resp.status == DescribeStatus.OK
    assert resp.summary is None  # orchestrator-owned; unset in v1
    assert len(resp.runs) == 1
    run = resp.runs[0]
    assert (run.task_id, run.target_id, run.delegate, run.status) == (
        "cap1", "t1", "vlm_caption", RunStatus.SUCCEEDED,
    )
    obs = resp.observations[0]
    assert (obs.task_id, obs.target_id, obs.kind) == ("cap1", "t1", "text")
    assert obs.text.content  # stub emits non-empty text


def test_provider_exception_isolates_to_failed_run_partial_status():
    payload = single_caption_payload()
    payload["tasks"].append({"id": "det1", "kind": "detect", "detect": {}})
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption", "detect": "yolo_detect"},
        providers={
            "vlm_caption": StubProvider(kind="caption"),
            "yolo_detect": ExplodingProvider(),
        },
    )
    resp = run_describe(orch, payload)
    assert resp.status == DescribeStatus.PARTIAL
    by_task = {r.task_id: r for r in resp.runs}
    assert by_task["cap1"].status == RunStatus.SUCCEEDED
    failed = by_task["det1"]
    assert failed.status == RunStatus.FAILED
    assert failed.error is not None
    assert failed.error.code == "analysis_run_failed"
    assert "backend unreachable" in failed.error.message
    # sibling isolation: the caption observation still landed
    assert any(o.task_id == "cap1" for o in resp.observations)


def test_all_failed_status_failed():
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption"},
        providers={"vlm_caption": ExplodingProvider()},
    )
    resp = run_describe(orch, single_caption_payload())
    assert resp.status == DescribeStatus.FAILED
    assert resp.observations == ()


def test_describe_rejects_directly_constructed_invalid_request():
    # Review blocker: bypassing parse_describe_request must not yield
    # OK-with-empty-runs; the orchestrator revalidates at its boundary.
    orch = AnalysisOrchestrator(task_routes={}, providers={})
    empty = DescribeRequest(targets=(), tasks=())
    with pytest.raises(AnalysisValidationError) as exc:
        asyncio.run(orch.describe(empty))
    assert exc.value.code == "analysis_invalid_request"


def test_describe_rejects_zero_run_binding_request():
    from backends.analysis import CaptionParams, TaskKind

    orch = AnalysisOrchestrator(task_routes={"caption": "d"}, providers={})
    req = DescribeRequest(
        targets=(DescribeTarget(id="t1", asset_ref="a", role="reference"),),
        tasks=(DescribeTask(id="cap1", kind=TaskKind.CAPTION, caption=CaptionParams()),),
    )
    with pytest.raises(AnalysisValidationError) as exc:
        asyncio.run(orch.describe(req))
    assert exc.value.code == "analysis_target_binding_invalid"


@pytest.mark.parametrize(
    "targets,tasks",
    [
        # Review blocker: direct dataclass construction with malformed value
        # types must fail validation, not slip through describe().
        (  # non-str target id
            (DescribeTarget(id=1, asset_ref="a"),),
            None,  # use default valid task
        ),
        (  # params block of the wrong type entirely
            (DescribeTarget(id="t1", asset_ref="a"),),
            "caption_str",
        ),
        (  # non-str entries in target_ids
            (DescribeTarget(id="t1", asset_ref="a"),),
            "bad_target_ids",
        ),
        (  # non-str role
            (DescribeTarget(id="t1", asset_ref="a", role=42),),
            None,
        ),
    ],
)
def test_describe_rejects_malformed_dataclass_values(targets, tasks):
    from backends.analysis import CaptionParams, TaskKind

    if tasks is None:
        tasks = (DescribeTask(id="cap1", kind=TaskKind.CAPTION, caption=CaptionParams()),)
    elif tasks == "caption_str":
        tasks = (DescribeTask(id="cap1", kind=TaskKind.CAPTION, caption="bad"),)
    elif tasks == "bad_target_ids":
        tasks = (DescribeTask(id="cap1", kind=TaskKind.CAPTION, caption=CaptionParams(),
                              target_ids=(1,)),)
    orch = AnalysisOrchestrator(task_routes={"caption": "d"}, providers={})
    req = DescribeRequest(targets=targets, tasks=tasks)
    with pytest.raises(AnalysisValidationError) as exc:
        asyncio.run(orch.describe(req))
    assert exc.value.code == "analysis_invalid_request"


def _valid_target():
    return DescribeTarget(id="t1", asset_ref="a")


def _valid_task(**overrides):
    kwargs = dict(id="cap1", kind=TaskKind.CAPTION, caption=CaptionParams())
    kwargs.update(overrides)
    return DescribeTask(**kwargs)


@pytest.mark.parametrize(
    "build_request",
    [
        # Review blocker round 3: deep value validation on direct dataclasses.
        pytest.param(
            lambda: DescribeRequest(
                targets=(_valid_target(),), tasks=(_valid_task(),), mode=1
            ),
            id="int-mode",
        ),
        pytest.param(
            lambda: DescribeRequest(
                targets=(_valid_target(),),
                tasks=(_valid_task(caption=CaptionParams(prompt=1)),),
            ),
            id="caption-prompt-int",
        ),
        pytest.param(
            lambda: DescribeRequest(
                targets=(_valid_target(),),
                tasks=(
                    DescribeTask(
                        id="det1", kind=TaskKind.DETECT,
                        detect=DetectParams(labels="person"),
                    ),
                ),
            ),
            id="detect-labels-str-container",
        ),
        pytest.param(
            lambda: DescribeRequest(
                targets=(_valid_target(),),
                tasks=(
                    DescribeTask(
                        id="det1", kind=TaskKind.DETECT,
                        detect=DetectParams(labels=(1,)),
                    ),
                ),
            ),
            id="detect-labels-int-entry",
        ),
        pytest.param(
            lambda: DescribeRequest(
                targets=(_valid_target(),),
                tasks=(
                    DescribeTask(
                        id="det1", kind=TaskKind.DETECT,
                        detect=DetectParams(min_confidence="hi"),
                    ),
                ),
            ),
            id="detect-min-confidence-str",
        ),
        pytest.param(
            lambda: DescribeRequest(
                targets=(_valid_target(),),
                tasks=(_valid_task(target_ids="t1"),),
            ),
            id="target-ids-str-container",
        ),
        # Review round 4: container shapes must fail validation, not leak
        # AttributeError from field access on non-dataclass elements.
        pytest.param(
            lambda: DescribeRequest(targets="bad", tasks=(_valid_task(),)),
            id="targets-str-container",
        ),
        pytest.param(
            lambda: DescribeRequest(targets=(_valid_target(),), tasks="bad"),
            id="tasks-str-container",
        ),
        pytest.param(
            lambda: DescribeRequest(targets=(object(),), tasks=(_valid_task(),)),
            id="targets-non-dataclass-element",
        ),
        pytest.param(
            lambda: DescribeRequest(targets=(_valid_target(),), tasks=(object(),)),
            id="tasks-non-dataclass-element",
        ),
    ],
)
def test_describe_rejects_malformed_nested_values(build_request):
    orch = AnalysisOrchestrator(task_routes={"caption": "d", "detect": "d"}, providers={})
    with pytest.raises(AnalysisValidationError) as exc:
        asyncio.run(orch.describe(build_request()))
    assert exc.value.code == "analysis_invalid_request"


class UnsupportingProvider:
    def supports(self, task):
        return False

    async def run(self, provider_run):
        raise AssertionError("must not be dispatched when supports() is false")


def test_unsupporting_provider_is_skipped_not_dispatched():
    payload = single_caption_payload()
    payload["tasks"].append({"id": "det1", "kind": "detect", "detect": {}})
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption", "detect": "yolo_detect"},
        providers={
            "vlm_caption": StubProvider(kind="caption"),
            "yolo_detect": UnsupportingProvider(),
        },
    )
    resp = run_describe(orch, payload)
    assert resp.status == DescribeStatus.PARTIAL
    by_task = {r.task_id: r for r in resp.runs}
    skipped = by_task["det1"]
    assert skipped.status == RunStatus.SKIPPED
    assert skipped.error.code == "analysis_no_supported_delegate"


def test_unrouted_kind_yields_skipped_run_and_partial():
    payload = single_caption_payload()
    payload["tasks"].append({"id": "ocr1", "kind": "ocr", "ocr": {}})
    orch = AnalysisOrchestrator(
        task_routes={"caption": "vlm_caption"},
        providers={"vlm_caption": StubProvider(kind="caption")},
    )
    resp = run_describe(orch, payload)
    assert resp.status == DescribeStatus.PARTIAL
    skipped = [r for r in resp.runs if r.status == RunStatus.SKIPPED]
    assert len(skipped) == 1
    assert skipped[0].error.code == "analysis_no_supported_delegate"
