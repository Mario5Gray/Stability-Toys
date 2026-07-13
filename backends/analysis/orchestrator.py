"""Async orchestration for the describe/analysis capability.

Chain shape mirrors backends/conditioning: validate -> resolve profile ->
expand runs -> dispatch -> normalize -> assemble. Providers stay simple; the
composition lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .contracts import (
    PRIMARY_ROLE,
    DescribeRequest,
    RunError,
    effective_role,
)


@dataclass(frozen=True)
class RunPlan:
    """One concrete (task, target) execution unit produced by expansion.

    delegate is None only for skip plans, which always carry skip_error.
    """
    task_id: str
    target_id: str
    delegate: Optional[str]
    skip_error: Optional[RunError] = None


def expand_runs(request: DescribeRequest, task_routes: Mapping[str, str]) -> Tuple[RunPlan, ...]:
    """Expand tasks x bound targets into RunPlans, task-major order.

    Requests reaching here already passed parse_describe_request, so every
    task binds to >=1 target; an unrouted kind yields one skip plan per
    bound target rather than a validation error (spec: RunStatus skipped).
    """
    primary_ids = [t.id for t in request.targets if effective_role(t) == PRIMARY_ROLE]
    plans = []
    for task in request.tasks:
        bound = list(task.target_ids) if task.target_ids else primary_ids
        delegate = task_routes.get(task.kind.value)
        for target_id in bound:
            if delegate is None:
                plans.append(RunPlan(
                    task_id=task.id,
                    target_id=target_id,
                    delegate=None,
                    skip_error=RunError(
                        code="analysis_no_supported_delegate",
                        message=f"no route for kind {task.kind.value}",
                    ),
                ))
            else:
                plans.append(RunPlan(task_id=task.id, target_id=target_id, delegate=delegate))
    return tuple(plans)
