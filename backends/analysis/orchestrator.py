"""Async orchestration for the describe/analysis capability.

Chain shape mirrors backends/conditioning: validate -> resolve profile ->
expand runs -> dispatch -> normalize -> assemble. Providers stay simple; the
composition lives here.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .contracts import (
    PRIMARY_ROLE,
    DescribeArtifact,
    DescribeObservation,
    DescribeRequest,
    DescribeResponse,
    DescribeRun,
    DescribeStatus,
    RunError,
    RunStatus,
    effective_role,
    validate_describe_request,
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


class AnalysisOrchestrator:
    """Owns validation, routing, dispatch, normalization, and assembly.

    Providers are keyed by delegate name. Runs against distinct delegates
    execute concurrently; per-run failure is isolated and degrades the
    response status rather than aborting siblings.
    """

    def __init__(self, task_routes, providers):
        self._task_routes = dict(task_routes)
        self._providers = dict(providers)

    async def describe(self, request: DescribeRequest) -> DescribeResponse:
        from .providers import ProviderRun  # local import: providers imports RunPlan from here

        # Boundary validation: directly constructed requests get the same
        # contract enforcement as wire payloads (non-empty runs invariant).
        validate_describe_request(request)

        plans = expand_runs(request, self._task_routes)
        tasks_by_id = {t.id: t for t in request.tasks}
        targets_by_id = {t.id: t for t in request.targets}

        async def execute(plan: RunPlan):
            if plan.delegate is None:
                return plan, None, plan.skip_error, RunStatus.SKIPPED
            provider = self._providers.get(plan.delegate)
            if provider is None:
                return plan, None, RunError(
                    code="analysis_delegate_not_found",
                    message=f"no provider registered for delegate '{plan.delegate}'",
                ), RunStatus.FAILED
            task = tasks_by_id[plan.task_id]
            try:
                supported = provider.supports(task)
            except Exception as exc:  # provider misbehavior stays per-run
                return plan, None, RunError(
                    code="analysis_run_failed",
                    message=f"{type(exc).__name__}: {exc}",
                ), RunStatus.FAILED
            if not supported:
                # Never dispatched -> skipped per spec's RunStatus semantics.
                return plan, None, RunError(
                    code="analysis_no_supported_delegate",
                    message=f"delegate '{plan.delegate}' does not support kind {task.kind.value}",
                ), RunStatus.SKIPPED
            provider_run = ProviderRun(
                plan=plan,
                task=task,
                target=targets_by_id[plan.target_id],
            )
            try:
                result = await provider.run(provider_run)
                return plan, result, None, RunStatus.SUCCEEDED
            except Exception as exc:  # per-run isolation is the contract
                return plan, None, RunError(
                    code="analysis_run_failed",
                    message=f"{type(exc).__name__}: {exc}",
                ), RunStatus.FAILED

        outcomes = await asyncio.gather(*(execute(p) for p in plans))

        observations: list[DescribeObservation] = []
        artifacts: list[DescribeArtifact] = []
        runs: list[DescribeRun] = []
        for plan, result, error, status in outcomes:
            if status == RunStatus.SUCCEEDED:
                assert result is not None  # execute() invariant: succeeded => result
                observations.extend(result.observations)
                artifacts.extend(result.artifacts)
                runs.append(DescribeRun(
                    task_id=plan.task_id, target_id=plan.target_id,
                    delegate=plan.delegate or "",
                    status=status, raw_output=result.raw_output,
                ))
            else:
                runs.append(DescribeRun(
                    task_id=plan.task_id, target_id=plan.target_id,
                    delegate=plan.delegate or "",
                    status=status, error=error,
                ))

        succeeded = sum(1 for r in runs if r.status == RunStatus.SUCCEEDED)
        if succeeded == len(runs):
            status = DescribeStatus.OK
        elif succeeded > 0:
            status = DescribeStatus.PARTIAL
        else:
            status = DescribeStatus.FAILED

        return DescribeResponse(
            status=status,
            observations=tuple(observations),
            artifacts=tuple(artifacts),
            runs=tuple(runs),
            summary=None,  # orchestrator-owned; deliberately unset in v1
        )
