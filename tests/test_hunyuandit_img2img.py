"""Unit tests for the Hunyuan-only img2img component.

Covers the strength-to-timestep selection contract and source-latent
preparation used by the Hunyuan denoising loop.
"""

import pytest
import torch

from backends.hunyuandit_img2img import prepare_source_latents, select_timesteps


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


class FakeImageProcessor:
    def __init__(self):
        self.calls = []

    def preprocess(self, image, height, width):
        self.calls.append((image, height, width))
        return torch.zeros(1, 3, height, width)


class FakeLatentDist:
    def __init__(self, value):
        self._value = value

    def sample(self, generator=None):
        return self._value


class FakeVaeOutput:
    def __init__(self, latent_dist):
        self.latent_dist = latent_dist


class FakeVaeConfig:
    scaling_factor = 0.18215


class FakeVae:
    def __init__(self):
        self.config = FakeVaeConfig()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.encode_calls = []

    def encode(self, tensor):
        self.encode_calls.append(tensor)
        value = torch.full((1, 4, tensor.shape[-2] // 8, tensor.shape[-1] // 8), 2.0)
        return FakeVaeOutput(FakeLatentDist(value))


class FakeAddNoiseScheduler:
    def __init__(self):
        self.add_noise_calls = []

    def add_noise(self, latent, noise, timestep):
        self.add_noise_calls.append((latent, noise, timestep))
        return latent + noise


class FakePipe:
    def __init__(self):
        self.image_processor = FakeImageProcessor()
        self.vae = FakeVae()
        self.scheduler = FakeAddNoiseScheduler()


def test_prepare_source_latents_preprocesses_encodes_and_noises():
    pipe = FakePipe()
    generator = torch.Generator(device="cpu").manual_seed(42)
    timestep = torch.tensor([17])

    result = prepare_source_latents(
        pipe,
        "source-image",
        width=64,
        height=64,
        timestep=timestep,
        generator=generator,
        dtype=torch.float16,
        device="cpu",
    )

    assert pipe.image_processor.calls == [("source-image", 64, 64)]
    assert len(pipe.vae.encode_calls) == 1
    assert pipe.vae.encode_calls[0].shape == (1, 3, 64, 64)
    assert len(pipe.scheduler.add_noise_calls) == 1
    noised_source, noise, used_timestep = pipe.scheduler.add_noise_calls[0]
    assert used_timestep is timestep
    expected_scaled = torch.full((1, 4, 8, 8), 2.0 * FakeVaeConfig.scaling_factor)
    assert torch.allclose(noised_source.float(), expected_scaled.float(), atol=1e-4)
    assert noise.shape == noised_source.shape
    assert result.dtype == torch.float16
    assert result.device == torch.device("cpu")


def test_prepare_source_latents_same_seed_reproduces_noise():
    pipe_a = FakePipe()
    pipe_b = FakePipe()
    timestep = torch.tensor([17])

    result_a = prepare_source_latents(
        pipe_a,
        "source-image",
        width=64,
        height=64,
        timestep=timestep,
        generator=torch.Generator(device="cpu").manual_seed(7),
        dtype=torch.float32,
        device="cpu",
    )
    result_b = prepare_source_latents(
        pipe_b,
        "source-image",
        width=64,
        height=64,
        timestep=timestep,
        generator=torch.Generator(device="cpu").manual_seed(7),
        dtype=torch.float32,
        device="cpu",
    )

    assert torch.equal(result_a, result_b)


def test_prepare_source_latents_different_seed_differs():
    pipe_a = FakePipe()
    pipe_b = FakePipe()
    timestep = torch.tensor([17])

    result_a = prepare_source_latents(
        pipe_a,
        "source-image",
        width=64,
        height=64,
        timestep=timestep,
        generator=torch.Generator(device="cpu").manual_seed(7),
        dtype=torch.float32,
        device="cpu",
    )
    result_b = prepare_source_latents(
        pipe_b,
        "source-image",
        width=64,
        height=64,
        timestep=timestep,
        generator=torch.Generator(device="cpu").manual_seed(8),
        dtype=torch.float32,
        device="cpu",
    )

    assert not torch.equal(result_a, result_b)


@pytest.mark.parametrize("width, height", [(0, 64), (64, 0), (-1, 64)])
def test_prepare_source_latents_rejects_invalid_dimensions_before_vae_encode(width, height):
    pipe = FakePipe()
    with pytest.raises(ValueError):
        prepare_source_latents(
            pipe,
            "source-image",
            width=width,
            height=height,
            timestep=torch.tensor([17]),
            generator=torch.Generator(device="cpu").manual_seed(0),
            dtype=torch.float32,
            device="cpu",
        )
    assert pipe.vae.encode_calls == []
