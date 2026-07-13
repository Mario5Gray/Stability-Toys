"""Provider protocol and the v1 stub provider."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Tuple

from .contracts import (
    DescribeArtifact,
    DescribeObservation,
    DescribeTarget,
    DescribeTask,
    TextObservation,
)
from .orchestrator import RunPlan


@dataclass(frozen=True)
class ProviderRun:
    """Everything a provider needs for one concrete run."""
    plan: RunPlan
    task: DescribeTask
    target: DescribeTarget


@dataclass(frozen=True)
class ProviderResult:
    observations: Tuple[DescribeObservation, ...] = ()
    artifacts: Tuple[DescribeArtifact, ...] = ()
    # Opaque, JSON-serializable provider payload; passed through verbatim.
    raw_output: Optional[Any] = None


class DescribeProvider(Protocol):
    def supports(self, task: DescribeTask) -> bool: ...
    async def run(self, provider_run: ProviderRun) -> ProviderResult: ...


ObservationFactory = Callable[[ProviderRun], Tuple[DescribeObservation, ...]]


def _default_text_observation(provider_run: ProviderRun) -> Tuple[DescribeObservation, ...]:
    return (
        DescribeObservation(
            task_id=provider_run.plan.task_id,
            target_id=provider_run.plan.target_id,
            kind="text",
            text=TextObservation(content=f"stub:{provider_run.task.kind.value}"),
        ),
    )


@dataclass(frozen=True)
class StubProvider:
    """Deterministic in-process provider for contract tests."""
    kind: str
    observation_factory: ObservationFactory = field(default=_default_text_observation)

    def supports(self, task: DescribeTask) -> bool:
        return task.kind.value == self.kind

    async def run(self, provider_run: ProviderRun) -> ProviderResult:
        return ProviderResult(
            observations=self.observation_factory(provider_run),
            raw_output={"stub": True, "kind": self.kind},
        )
