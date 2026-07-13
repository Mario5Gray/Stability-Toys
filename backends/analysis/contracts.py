"""Wire contracts for the describe/analysis capability.

Spec: docs/superpowers/specs/2026-07-11-describe-analysis-interface-design.md
Closed enums; exactly one typed params block per task; zero-run binding is a
request validation error so every DescribeResponse carries non-empty runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

PRIMARY_ROLE = "primary"


class AnalysisValidationError(ValueError):
    """Request/config validation failure with an operator-facing analysis_* code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class TaskKind(str, Enum):
    CAPTION = "caption"
    DETECT = "detect"
    OCR = "ocr"
    POSE = "pose"
    EMBED = "embed"


class ObservationKind(str, Enum):
    TEXT = "text"
    DETECTION = "detection"
    ATTRIBUTE = "attribute"
    KEYPOINTS = "keypoints"


class ArtifactKind(str, Enum):
    EMBEDDING_REF = "embedding_ref"


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class DescribeStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class DescribeTarget:
    id: str
    asset_ref: Optional[str] = None
    url: Optional[str] = None
    role: str = ""


def effective_role(target: DescribeTarget) -> str:
    return target.role or PRIMARY_ROLE


# v1-minimal typed params, mirroring the Go contract field-for-field.
@dataclass(frozen=True)
class CaptionParams:
    prompt: Optional[str] = None


@dataclass(frozen=True)
class DetectParams:
    labels: Tuple[str, ...] = ()
    min_confidence: Optional[float] = None


@dataclass(frozen=True)
class OcrParams:
    pass


@dataclass(frozen=True)
class PoseParams:
    pass


@dataclass(frozen=True)
class EmbedParams:
    pass


@dataclass(frozen=True)
class DescribeTask:
    id: str
    kind: TaskKind
    target_ids: Tuple[str, ...] = ()
    # Exactly one typed params block is set, matching `kind`; parse enforces it.
    caption: Optional[CaptionParams] = None
    detect: Optional[DetectParams] = None
    ocr: Optional[OcrParams] = None
    pose: Optional[PoseParams] = None
    embed: Optional[EmbedParams] = None


@dataclass(frozen=True)
class DescribeRequest:
    targets: Tuple[DescribeTarget, ...]
    tasks: Tuple[DescribeTask, ...]
    mode: Optional[str] = None


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class TextObservation:
    content: str


@dataclass(frozen=True)
class DetectionObservation:
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class AttributeObservation:
    name: str
    value: str
    confidence: Optional[float] = None
    box: Optional[Box] = None


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    name: str = ""
    confidence: Optional[float] = None


@dataclass(frozen=True)
class KeypointsObservation:
    points: Tuple[Keypoint, ...]
    skeleton: str = ""


@dataclass(frozen=True)
class DescribeObservation:
    task_id: str
    target_id: str
    # ObservationKind (str values accepted; validated against the closed
    # enum at serialization time in response_to_dict).
    kind: "ObservationKind | str"
    text: Optional[TextObservation] = None
    detection: Optional[DetectionObservation] = None
    attribute: Optional[AttributeObservation] = None
    keypoints: Optional[KeypointsObservation] = None


@dataclass(frozen=True)
class DescribeArtifact:
    task_id: str
    target_id: str
    # ArtifactKind (embedding_ref only in v1); validated at serialization.
    kind: "ArtifactKind | str"
    ref: str
    dims: Optional[int] = None


@dataclass(frozen=True)
class RunError:
    code: str
    message: str


@dataclass(frozen=True)
class DescribeRun:
    task_id: str
    target_id: str
    delegate: str
    status: RunStatus
    error: Optional[RunError] = None
    # Opaque provider payload; the contract deliberately does not type it.
    # Must be JSON-serializable; serialized verbatim, never restructured.
    raw_output: Optional[Any] = None


@dataclass(frozen=True)
class DescribeResponse:
    status: DescribeStatus
    observations: Tuple[DescribeObservation, ...]
    runs: Tuple[DescribeRun, ...]
    artifacts: Tuple[DescribeArtifact, ...] = ()
    summary: Optional[str] = None


_PARAM_KEYS = {
    TaskKind.CAPTION: "caption",
    TaskKind.DETECT: "detect",
    TaskKind.OCR: "ocr",
    TaskKind.POSE: "pose",
    TaskKind.EMBED: "embed",
}
_ALL_PARAM_KEYS = set(_PARAM_KEYS.values())


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise _invalid(f"{what} must be a string")
    return value


def _optional_str(value: Any, what: str) -> Optional[str]:
    if value is None:
        return None
    return _require_str(value, what)


def _parse_params(kind: TaskKind, task_id: str, raw: Any):
    if not isinstance(raw, Mapping):
        raise _invalid(f"task '{task_id}' params block for kind '{kind.value}' must be an object")
    if kind == TaskKind.CAPTION:
        return CaptionParams(
            prompt=_optional_str(raw.get("prompt"), f"task '{task_id}' caption.prompt"),
        )
    if kind == TaskKind.DETECT:
        labels = raw.get("labels")
        if labels is None:
            labels = ()
        if isinstance(labels, str) or not isinstance(labels, (list, tuple)):
            raise _invalid(f"task '{task_id}' detect.labels must be a list of strings")
        labels = tuple(
            _require_str(label, f"task '{task_id}' detect.labels entry") for label in labels
        )
        min_confidence = raw.get("min_confidence")
        if min_confidence is not None and not isinstance(min_confidence, (int, float)):
            raise _invalid(f"task '{task_id}' detect.min_confidence must be a number")
        return DetectParams(labels=labels, min_confidence=min_confidence)
    if kind == TaskKind.OCR:
        return OcrParams()
    if kind == TaskKind.POSE:
        return PoseParams()
    return EmbedParams()


def _invalid(message: str) -> AnalysisValidationError:
    return AnalysisValidationError("analysis_invalid_request", message)


def _binding_invalid(message: str) -> AnalysisValidationError:
    return AnalysisValidationError("analysis_target_binding_invalid", message)


def parse_describe_request(payload: Mapping[str, Any]) -> DescribeRequest:
    if not isinstance(payload, Mapping):
        raise _invalid("request body must be an object")
    raw_targets = payload.get("targets") or []
    raw_tasks = payload.get("tasks") or []
    if not raw_targets or not raw_tasks:
        raise _invalid("targets and tasks must be non-empty")

    targets = []
    roles: Dict[str, str] = {}
    primary_count = 0
    for raw in raw_targets:
        if not isinstance(raw, Mapping):
            raise _invalid("each target must be an object")
        target_id = (_optional_str(raw.get("id"), "target id") or "").strip()
        if not target_id:
            raise _invalid("target id must be set")
        if target_id in roles:
            raise _invalid(f"duplicate target id '{target_id}'")
        asset_ref = _optional_str(raw.get("asset_ref"), f"target '{target_id}' asset_ref")
        url = _optional_str(raw.get("url"), f"target '{target_id}' url")
        if bool(asset_ref) == bool(url):
            raise _invalid(f"target '{target_id}' must set exactly one of asset_ref or url")
        target = DescribeTarget(
            id=target_id,
            asset_ref=asset_ref,
            url=url,
            role=(_optional_str(raw.get("role"), f"target '{target_id}' role") or "").strip(),
        )
        roles[target_id] = effective_role(target)
        if roles[target_id] == PRIMARY_ROLE:
            primary_count += 1
        targets.append(target)

    tasks = []
    seen_task_ids = set()
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise _invalid("each task must be an object")
        task_id = (_optional_str(raw.get("id"), "task id") or "").strip()
        if not task_id:
            raise _invalid("task id must be set")
        if task_id in seen_task_ids:
            raise _invalid(f"duplicate task id '{task_id}'")
        seen_task_ids.add(task_id)
        try:
            kind = TaskKind(raw.get("kind"))
        except ValueError:
            raise _invalid(f"task '{task_id}' has unknown kind '{raw.get('kind')}'")
        set_blocks = [k for k in _ALL_PARAM_KEYS if raw.get(k) is not None]
        if set_blocks != [_PARAM_KEYS[kind]]:
            raise _invalid(
                f"task '{task_id}' must set exactly one params block matching kind '{kind.value}'"
            )
        raw_target_ids = raw.get("target_ids")
        if raw_target_ids is None:
            raw_target_ids = ()
        if isinstance(raw_target_ids, str) or not isinstance(raw_target_ids, (list, tuple)):
            raise _invalid(f"task '{task_id}' target_ids must be a list of strings")
        target_ids = tuple(
            _require_str(tid, f"task '{task_id}' target_ids entry") for tid in raw_target_ids
        )
        for tid in target_ids:
            if tid not in roles:
                raise _binding_invalid(f"task '{task_id}' references unknown target '{tid}'")
        if not target_ids and primary_count == 0:
            raise _binding_invalid(
                f"task '{task_id}' binds to zero targets: no primary targets declared"
            )
        params_kwargs: Dict[str, Any] = {
            _PARAM_KEYS[kind]: _parse_params(kind, task_id, raw.get(_PARAM_KEYS[kind]))
        }
        tasks.append(
            DescribeTask(
                id=task_id,
                kind=kind,
                target_ids=target_ids,
                **params_kwargs,
            )
        )

    mode = _optional_str(payload.get("mode"), "mode")
    return DescribeRequest(targets=tuple(targets), tasks=tuple(tasks), mode=mode)


def validate_describe_request(request: DescribeRequest) -> None:
    """Dataclass-level contract validation.

    parse_describe_request enforces these rules while building from wire
    payloads; this function re-enforces them for directly constructed
    requests at the orchestrator boundary, so bypassing the parser can never
    yield a zero-run or malformed execution.
    """
    if not request.targets or not request.tasks:
        raise _invalid("targets and tasks must be non-empty")
    roles: Dict[str, str] = {}
    primary_count = 0
    for target in request.targets:
        if not target.id:
            raise _invalid("target id must be set")
        if target.id in roles:
            raise _invalid(f"duplicate target id '{target.id}'")
        if bool(target.asset_ref) == bool(target.url):
            raise _invalid(f"target '{target.id}' must set exactly one of asset_ref or url")
        roles[target.id] = effective_role(target)
        if roles[target.id] == PRIMARY_ROLE:
            primary_count += 1
    seen_task_ids = set()
    for task in request.tasks:
        if not task.id:
            raise _invalid("task id must be set")
        if task.id in seen_task_ids:
            raise _invalid(f"duplicate task id '{task.id}'")
        seen_task_ids.add(task.id)
        try:
            kind = TaskKind(task.kind)
        except ValueError:
            raise _invalid(f"task '{task.id}' has unknown kind '{task.kind}'")
        set_blocks = [k for k in _ALL_PARAM_KEYS if getattr(task, k) is not None]
        if set_blocks != [_PARAM_KEYS[kind]]:
            raise _invalid(
                f"task '{task.id}' must set exactly one params block matching kind '{kind.value}'"
            )
        for tid in task.target_ids:
            if tid not in roles:
                raise _binding_invalid(f"task '{task.id}' references unknown target '{tid}'")
        if not task.target_ids and primary_count == 0:
            raise _binding_invalid(
                f"task '{task.id}' binds to zero targets: no primary targets declared"
            )


def _drop_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def response_to_dict(resp: DescribeResponse) -> Dict[str, Any]:
    """Serialize to the wire shape pinned by the stclient contract tests."""

    def obs_dict(o: DescribeObservation) -> Dict[str, Any]:
        # Coercion validates the closed enum: unknown kinds (e.g. a future
        # "mask" constructed prematurely) fail here rather than reaching wire.
        kind = ObservationKind(o.kind).value
        d: Dict[str, Any] = {"task_id": o.task_id, "target_id": o.target_id, "kind": kind}
        if o.text is not None:
            d["text"] = {"content": o.text.content}
        if o.detection is not None:
            d["detection"] = {
                "label": o.detection.label,
                "confidence": o.detection.confidence,
                "box": vars(o.detection.box).copy(),
            }
        if o.attribute is not None:
            d["attribute"] = _drop_nones({
                "name": o.attribute.name,
                "value": o.attribute.value,
                "confidence": o.attribute.confidence,
                "box": vars(o.attribute.box).copy() if o.attribute.box else None,
            })
        if o.keypoints is not None:
            d["keypoints"] = _drop_nones({
                "skeleton": o.keypoints.skeleton or None,
                "points": [
                    _drop_nones({"name": p.name or None, "x": p.x, "y": p.y,
                                 "confidence": p.confidence})
                    for p in o.keypoints.points
                ],
            })
        return d

    def run_dict(r: DescribeRun) -> Dict[str, Any]:
        return _drop_nones({
            "task_id": r.task_id,
            "target_id": r.target_id,
            "delegate": r.delegate,
            "status": r.status.value,
            "error": {"code": r.error.code, "message": r.error.message} if r.error else None,
            "raw_output": r.raw_output,  # opaque passthrough, never restructured
        })

    out: Dict[str, Any] = {
        "status": resp.status.value,
        "observations": [obs_dict(o) for o in resp.observations],
        "runs": [run_dict(r) for r in resp.runs],
    }
    if resp.summary is not None:
        out["summary"] = resp.summary
    if resp.artifacts:
        out["artifacts"] = [
            _drop_nones({
                "task_id": a.task_id, "target_id": a.target_id,
                "kind": ArtifactKind(a.kind).value, "ref": a.ref, "dims": a.dims,
            })
            for a in resp.artifacts
        ]
    return out
