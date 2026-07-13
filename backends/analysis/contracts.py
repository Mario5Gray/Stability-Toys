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
    kind: str  # ObservationKind value
    text: Optional[TextObservation] = None
    detection: Optional[DetectionObservation] = None
    attribute: Optional[AttributeObservation] = None
    keypoints: Optional[KeypointsObservation] = None


@dataclass(frozen=True)
class DescribeArtifact:
    task_id: str
    target_id: str
    kind: str  # "embedding_ref" only in v1
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


def _parse_params(kind: TaskKind, raw: Mapping[str, Any]):
    if kind == TaskKind.CAPTION:
        return CaptionParams(prompt=raw.get("prompt"))
    if kind == TaskKind.DETECT:
        return DetectParams(
            labels=tuple(raw.get("labels") or ()),
            min_confidence=raw.get("min_confidence"),
        )
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
        target_id = (raw.get("id") or "").strip()
        if not target_id:
            raise _invalid("target id must be set")
        if target_id in roles:
            raise _invalid(f"duplicate target id '{target_id}'")
        asset_ref = raw.get("asset_ref")
        url = raw.get("url")
        if bool(asset_ref) == bool(url):
            raise _invalid(f"target '{target_id}' must set exactly one of asset_ref or url")
        target = DescribeTarget(
            id=target_id,
            asset_ref=asset_ref,
            url=url,
            role=(raw.get("role") or "").strip(),
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
        task_id = (raw.get("id") or "").strip()
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
        target_ids = tuple(raw.get("target_ids") or ())
        for tid in target_ids:
            if tid not in roles:
                raise _binding_invalid(f"task '{task_id}' references unknown target '{tid}'")
        if not target_ids and primary_count == 0:
            raise _binding_invalid(
                f"task '{task_id}' binds to zero targets: no primary targets declared"
            )
        params_kwargs: Dict[str, Any] = {
            _PARAM_KEYS[kind]: _parse_params(kind, raw.get(_PARAM_KEYS[kind]) or {})
        }
        tasks.append(
            DescribeTask(
                id=task_id,
                kind=kind,
                target_ids=target_ids,
                **params_kwargs,
            )
        )

    mode = payload.get("mode")
    return DescribeRequest(targets=tuple(targets), tasks=tuple(tasks), mode=mode)


def _drop_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def response_to_dict(resp: DescribeResponse) -> Dict[str, Any]:
    """Serialize to the wire shape pinned by the stclient contract tests."""

    def obs_dict(o: DescribeObservation) -> Dict[str, Any]:
        d: Dict[str, Any] = {"task_id": o.task_id, "target_id": o.target_id, "kind": o.kind}
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
                "kind": a.kind, "ref": a.ref, "dims": a.dims,
            })
            for a in resp.artifacts
        ]
    return out
