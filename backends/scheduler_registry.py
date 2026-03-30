"""Canonical scheduler registry for Diffusers-backed generation."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any


SCHEDULER_IMPORTS = {
    "ddim": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
    "euler": "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
    "euler_a": "diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler",
    "dpmpp_2m": "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
    "dpmpp_sde": "diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler",
    "lcm": "diffusers.schedulers.scheduling_lcm.LCMScheduler",
}


def normalize_scheduler_id(scheduler_id: str) -> str:
    return str(scheduler_id).strip().lower()


def list_scheduler_ids() -> list[str]:
    return sorted(SCHEDULER_IMPORTS.keys())


def get_scheduler_class(scheduler_id: str) -> Any:
    normalized = normalize_scheduler_id(scheduler_id)
    target = SCHEDULER_IMPORTS.get(normalized)
    if target is None:
        raise ValueError(
            f"Unknown scheduler_id '{scheduler_id}'. Available: {', '.join(list_scheduler_ids())}"
        )

    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_scheduler(scheduler_id: str, config: Any) -> Any:
    scheduler_cls = get_scheduler_class(scheduler_id)
    return scheduler_cls.from_config(deepcopy(config))
