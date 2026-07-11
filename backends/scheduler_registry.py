"""Canonical scheduler registry for Diffusers-backed generation."""

from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class SchedulerSpec:
    class_path: str
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_kwargs", MappingProxyType(dict(self.extra_kwargs)))


SCHEDULER_SPECS = {
    "ddim": SchedulerSpec("diffusers.schedulers.scheduling_ddim.DDIMScheduler"),
    "euler": SchedulerSpec(
        "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler"
    ),
    "euler_a": SchedulerSpec(
        "diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler"
    ),
    "dpmpp_2m": SchedulerSpec(
        "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler"
    ),
    "dpmpp_2m_karras": SchedulerSpec(
        "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
        {"use_karras_sigmas": True},
    ),
    "dpmpp_sde": SchedulerSpec(
        "diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler"
    ),
    "dpmpp_sde_karras": SchedulerSpec(
        "diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler",
        {"use_karras_sigmas": True},
    ),
    "lcm": SchedulerSpec("diffusers.schedulers.scheduling_lcm.LCMScheduler"),
}


def normalize_scheduler_id(scheduler_id: str) -> str:
    return str(scheduler_id).strip().lower()


def list_scheduler_ids() -> list[str]:
    return sorted(SCHEDULER_SPECS.keys())


def _get_scheduler_spec(scheduler_id: str) -> SchedulerSpec:
    normalized = normalize_scheduler_id(scheduler_id)
    spec = SCHEDULER_SPECS.get(normalized)
    if spec is None:
        raise ValueError(
            f"Unknown scheduler_id '{scheduler_id}'. Available: {', '.join(list_scheduler_ids())}"
        )
    return spec


def get_scheduler_class(scheduler_id: str) -> Any:
    spec = _get_scheduler_spec(scheduler_id)
    module_name, class_name = spec.class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_scheduler(scheduler_id: str, config: Any) -> Any:
    spec = _get_scheduler_spec(scheduler_id)
    scheduler_cls = get_scheduler_class(scheduler_id)
    return scheduler_cls.from_config(deepcopy(config), **spec.extra_kwargs)
