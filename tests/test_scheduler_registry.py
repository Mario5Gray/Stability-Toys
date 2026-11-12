"""Tests for scheduler registry resolution and policy semantics."""

import sys
from types import SimpleNamespace

import pytest


class _FakeScheduler:
    @classmethod
    def from_config(cls, config, **kwargs):
        return {
            "scheduler": cls.__name__,
            "config": config,
            "kwargs": kwargs,
        }


def test_build_scheduler_known_id_without_extra_kwargs(monkeypatch):
    from backends import scheduler_registry

    module_name = "diffusers.schedulers.scheduling_euler_discrete"
    fake_module = SimpleNamespace(EulerDiscreteScheduler=_FakeScheduler)
    monkeypatch.setattr(
        scheduler_registry,
        "SCHEDULER_SPECS",
        {
            "euler": scheduler_registry.SchedulerSpec(
                f"{module_name}.EulerDiscreteScheduler"
            )
        },
    )
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    built = scheduler_registry.build_scheduler("euler", {"beta": "value"})

    assert built == {
        "scheduler": "_FakeScheduler",
        "config": {"beta": "value"},
        "kwargs": {},
    }


def test_build_scheduler_forwards_spec_kwargs_and_deepcopies_config(monkeypatch):
    from backends import scheduler_registry

    module_name = "diffusers.schedulers.scheduling_dpmsolver_singlestep"
    fake_module = SimpleNamespace(DPMSolverSinglestepScheduler=_FakeScheduler)
    monkeypatch.setattr(
        scheduler_registry,
        "SCHEDULER_SPECS",
        {
            "dpmpp_sde_karras": scheduler_registry.SchedulerSpec(
                f"{module_name}.DPMSolverSinglestepScheduler",
                {"use_karras_sigmas": True},
            )
        },
    )
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    config = {"nested": {"beta": "value"}}

    built = scheduler_registry.build_scheduler("DPMpp_SDE_Karras", config)

    assert built["kwargs"] == {"use_karras_sigmas": True}
    assert built["config"] == config
    assert built["config"] is not config
    assert built["config"]["nested"] is not config["nested"]


def test_list_scheduler_ids_includes_karras_variants():
    from backends.scheduler_registry import list_scheduler_ids

    assert {"dpmpp_2m_karras", "dpmpp_sde_karras"} <= set(list_scheduler_ids())


def test_scheduler_spec_extra_kwargs_are_immutable():
    from backends.scheduler_registry import SCHEDULER_SPECS

    spec = SCHEDULER_SPECS["dpmpp_sde_karras"]

    with pytest.raises(TypeError):
        spec.extra_kwargs["use_karras_sigmas"] = False

    assert spec.extra_kwargs == {"use_karras_sigmas": True}


def test_build_scheduler_ddpm_returns_ddpm_scheduler():
    """STABL-wsebfcjp: ddpm must build from a non-DDPM baseline config."""
    from diffusers import DDPMScheduler, LCMScheduler

    from backends.scheduler_registry import build_scheduler

    built = build_scheduler("ddpm", LCMScheduler().config)

    assert isinstance(built, DDPMScheduler)


def test_build_scheduler_unknown_id_raises():
    from backends import scheduler_registry

    with pytest.raises(ValueError, match="Unknown scheduler_id"):
        scheduler_registry.build_scheduler("not-a-scheduler", {})


def test_flow_match_euler_is_registered():
    from backends.scheduler_registry import get_scheduler_class, list_scheduler_ids

    assert "flow_match_euler" in list_scheduler_ids()
    assert get_scheduler_class("flow_match_euler").__name__ == "FlowMatchEulerDiscreteScheduler"


@pytest.mark.parametrize("shift", [3.0, 6.0])
def test_flow_match_euler_keeps_checkpoint_shift(shift):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from backends.scheduler_registry import build_scheduler

    baseline = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False
    )
    rebuilt = build_scheduler("flow_match_euler", baseline.config)
    assert rebuilt.config.shift == shift
    assert rebuilt.config.use_dynamic_shifting is False
