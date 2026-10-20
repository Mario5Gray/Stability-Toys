"""Unit tests for the Hunyuan-only img2img component.

Covers the strength-to-timestep selection contract used by the Hunyuan
denoising loop, independent of any diffusers/torch import.
"""

import pytest

from backends.hunyuandit_img2img import select_timesteps


class FakeScheduler:
    order = 1

    def set_timesteps(self, steps, device=None):
        self.timesteps = list(range(steps - 1, -1, -1))

    def add_noise(self, latent, noise, timestep):
        return latent + noise


class MissingSetTimesteps:
    def add_noise(self, latent, noise, timestep):
        return latent + noise


class MissingAddNoise:
    order = 1

    def set_timesteps(self, steps, device=None):
        self.timesteps = list(range(steps - 1, -1, -1))


def test_select_timesteps_partial_strength():
    timesteps, effective_steps = select_timesteps(FakeScheduler(), 30, 0.25, "cpu")
    assert timesteps == list(range(6, -1, -1))
    assert effective_steps == 7


def test_select_timesteps_full_strength_selects_all_steps():
    timesteps, effective_steps = select_timesteps(FakeScheduler(), 30, 1.0, "cpu")
    assert effective_steps == 30
    assert timesteps == list(range(29, -1, -1))


def test_select_timesteps_minimum_strength_selects_at_least_one():
    timesteps, effective_steps = select_timesteps(FakeScheduler(), 30, 0.01, "cpu")
    assert effective_steps == 1
    assert len(timesteps) == 1


@pytest.mark.parametrize("steps", [0, -1])
def test_select_timesteps_rejects_non_positive_steps(steps):
    with pytest.raises(ValueError):
        select_timesteps(FakeScheduler(), steps, 0.5, "cpu")


def test_select_timesteps_rejects_scheduler_missing_set_timesteps():
    with pytest.raises(ValueError, match="MissingSetTimesteps"):
        select_timesteps(MissingSetTimesteps(), 30, 0.5, "cpu")


def test_select_timesteps_rejects_scheduler_missing_add_noise():
    with pytest.raises(ValueError, match="MissingAddNoise"):
        select_timesteps(MissingAddNoise(), 30, 0.5, "cpu")
