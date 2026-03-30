"""Tests for scheduler registry resolution and policy semantics."""

import sys
from types import SimpleNamespace

import pytest


class _FakeScheduler:
    @classmethod
    def from_config(cls, config):
        return {"scheduler": cls.__name__, "config": config}


def test_build_scheduler_known_id():
    from backends import scheduler_registry

    fake_module = SimpleNamespace(EulerDiscreteScheduler=_FakeScheduler)
    original = scheduler_registry.SCHEDULER_IMPORTS.copy()

    try:
        scheduler_registry.SCHEDULER_IMPORTS.clear()
        scheduler_registry.SCHEDULER_IMPORTS.update(
            {"euler": "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler"}
        )
        sys.modules["diffusers.schedulers.scheduling_euler_discrete"] = fake_module

        built = scheduler_registry.build_scheduler("euler", {"beta": "value"})

        assert built == {"scheduler": "_FakeScheduler", "config": {"beta": "value"}}
    finally:
        scheduler_registry.SCHEDULER_IMPORTS.clear()
        scheduler_registry.SCHEDULER_IMPORTS.update(original)
        sys.modules.pop("diffusers.schedulers.scheduling_euler_discrete", None)


def test_build_scheduler_unknown_id_raises():
    from backends import scheduler_registry

    with pytest.raises(ValueError, match="Unknown scheduler_id"):
        scheduler_registry.build_scheduler("not-a-scheduler", {})
