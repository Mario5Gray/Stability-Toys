"""Unit tests for the Hunyuan-only img2img component.

Covers the strength-to-timestep selection contract and source-latent
preparation used by the Hunyuan denoising loop.
"""

import concurrent.futures

import pytest
import torch
from PIL import Image

from backends.hunyuandit_img2img import (
    prepare_source_latents,
    run_hunyuan_img2img,
    select_timesteps,
)


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


# --- Task 3: Hunyuan denoising component -----------------------------------
#
# Fakes below stand in for the installed HunyuanDiTPipeline's bound
# `encode_prompt`, the HunyuanDiT2DModel transformer, a DDPMScheduler-shaped
# scheduler, and the loaded VAE+image-processor, so the test checks this
# module's own call contract against the Diffusers 0.39 reference
# (diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py) without needing CUDA
# weights.

HIDDEN_SIZE = 8
LATENT_CHANNELS = 4
PATCH_SIZE = 2
INNER_DIM = 8
NUM_HEADS = 2


class FakeImageProcessor3:
    def preprocess(self, image, height, width):
        return torch.zeros(1, 3, height, width)

    def postprocess(self, image, output_type="pil", do_denormalize=None):
        return [Image.new("RGB", (image.shape[-1], image.shape[-2]))]


class FakeVae3:
    class _Config:
        scaling_factor = 0.18215

    def __init__(self):
        self.config = self._Config()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, tensor):
        self.encode_calls.append(tensor)
        value = torch.full((1, LATENT_CHANNELS, tensor.shape[-2] // 8, tensor.shape[-1] // 8), 2.0)

        class _Dist:
            def sample(self, generator=None):
                return value

        class _Out:
            latent_dist = _Dist()

        return _Out()

    def decode(self, latents, return_dict=False):
        self.decode_calls.append(latents)
        return (latents,)


class FakeDenoiseScheduler:
    order = 1

    def __init__(self):
        self.scale_calls = []
        self.step_calls = []
        self.last_latents = None

    def set_timesteps(self, steps, device=None):
        self.timesteps = list(range(steps - 1, -1, -1))

    def add_noise(self, latent, noise, timestep):
        return latent + noise

    def scale_model_input(self, latent_model_input, t):
        self.scale_calls.append(t)
        return latent_model_input

    def step(self, noise_pred, t, latents, return_dict=False):
        self.step_calls.append((noise_pred, t, latents))
        self.last_latents = latents - 0.1 * noise_pred
        return (self.last_latents,)


class FakeTransformerConfig:
    patch_size = PATCH_SIZE
    in_channels = LATENT_CHANNELS


class FakeTransformer:
    def __init__(self):
        self.config = FakeTransformerConfig()
        self.inner_dim = INNER_DIM
        self.num_heads = NUM_HEADS
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.calls = []

    def __call__(
        self,
        latent_model_input,
        t_expand,
        encoder_hidden_states,
        text_embedding_mask,
        encoder_hidden_states_t5,
        text_embedding_mask_t5,
        image_meta_size,
        style,
        image_rotary_emb,
        return_dict=False,
    ):
        self.calls.append(
            dict(
                latent_model_input=latent_model_input,
                t_expand=t_expand,
                encoder_hidden_states=encoder_hidden_states,
                text_embedding_mask=text_embedding_mask,
                encoder_hidden_states_t5=encoder_hidden_states_t5,
                text_embedding_mask_t5=text_embedding_mask_t5,
                image_meta_size=image_meta_size,
                style=style,
                image_rotary_emb=image_rotary_emb,
            )
        )
        batch = latent_model_input.shape[0]
        half = batch // 2
        # First half of the batch is the unconditional pass, second half the
        # conditional pass (matches torch.cat([negative, positive]) ordering).
        # Noise channels (first LATENT_CHANNELS) carry a value distinguishing
        # the two; variance channels (last LATENT_CHANNELS) carry a sentinel
        # that must never reach scheduler.step if the chunk/guidance split
        # is implemented correctly.
        noise = torch.cat(
            [torch.zeros(half, LATENT_CHANNELS, 8, 8), torch.ones(batch - half, LATENT_CHANNELS, 8, 8)],
            dim=0,
        )
        variance = torch.full((batch, LATENT_CHANNELS, 8, 8), -99.0)
        return (torch.cat([noise, variance], dim=1),)


class FakeHunyuanPipe:
    def __init__(self):
        self.image_processor = FakeImageProcessor3()
        self.vae = FakeVae3()
        self.scheduler = FakeDenoiseScheduler()
        self.transformer = FakeTransformer()
        self.encode_prompt_calls = []
        self.safety_checker_calls = []
        self.free_hooks_calls = 0

    def encode_prompt(
        self,
        *,
        prompt,
        device,
        dtype,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        max_sequence_length=None,
        text_encoder_index=0,
    ):
        self.encode_prompt_calls.append(
            dict(
                text_encoder_index=text_encoder_index,
                max_sequence_length=max_sequence_length,
                do_classifier_free_guidance=do_classifier_free_guidance,
                seq_len=prompt_embeds.shape[1],
            )
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask

    def run_safety_checker(self, image, device, dtype):
        self.safety_checker_calls.append(image)
        return image, None

    def maybe_free_model_hooks(self):
        self.free_hooks_calls += 1


def _fake_conditioning():
    return {
        "prompt_embeds": torch.rand(1, 77, HIDDEN_SIZE),
        "negative_prompt_embeds": torch.rand(1, 77, HIDDEN_SIZE),
        "prompt_attention_mask": torch.ones(1, 77),
        "negative_prompt_attention_mask": torch.ones(1, 77),
        "prompt_embeds_2": torch.rand(1, 256, HIDDEN_SIZE),
        "negative_prompt_embeds_2": torch.rand(1, 256, HIDDEN_SIZE),
        "prompt_attention_mask_2": torch.ones(1, 256),
        "negative_prompt_attention_mask_2": torch.ones(1, 256),
    }


def test_run_hunyuan_img2img_contract():
    pipe = FakeHunyuanPipe()
    conditioning = _fake_conditioning()
    guidance_scale = 3.5

    expected_timesteps, expected_effective_steps = select_timesteps(
        FakeDenoiseScheduler(), 4, 0.5, "cpu"
    )

    images = run_hunyuan_img2img(
        pipe,
        "source-image",
        conditioning,
        width=64,
        height=64,
        steps=4,
        strength=0.5,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(0),
        progress=None,
        should_cancel=None,
    )

    # Both encoders ran with the Diffusers 0.39 txt2img sequence-length
    # contract (77 for CLIP/BERT, 256 for mT5) and CFG on.
    assert [c["text_encoder_index"] for c in pipe.encode_prompt_calls] == [0, 1]
    assert [c["max_sequence_length"] for c in pipe.encode_prompt_calls] == [77, 256]
    assert [c["seq_len"] for c in pipe.encode_prompt_calls] == [77, 256]
    assert all(c["do_classifier_free_guidance"] for c in pipe.encode_prompt_calls)

    # Only the strength-selected suffix of timesteps ran (2 of 4 possible).
    assert expected_effective_steps == 2
    assert len(pipe.transformer.calls) == expected_effective_steps
    assert [c[1] for c in pipe.scheduler.step_calls] == list(expected_timesteps)

    call = pipe.transformer.calls[0]
    assert call["encoder_hidden_states"].shape == (2, 77, HIDDEN_SIZE)
    assert call["text_embedding_mask"].shape == (2, 77)
    assert call["encoder_hidden_states_t5"].shape == (2, 256, HIDDEN_SIZE)
    assert call["text_embedding_mask_t5"].shape == (2, 256)
    assert call["image_meta_size"].shape == (2, 6)
    assert call["style"].shape == (2,)
    assert isinstance(call["image_rotary_emb"], tuple)

    # Classifier-free guidance split the variance channels off before
    # scheduler.step: the sentinel -99 variance value must never appear.
    for noise_pred, _t, _latents in pipe.scheduler.step_calls:
        assert noise_pred.shape == (1, LATENT_CHANNELS, 8, 8)
        expected = torch.full_like(noise_pred, guidance_scale)
        assert torch.allclose(noise_pred, expected)

    # The VAE decoded the scaled final latents.
    assert len(pipe.vae.decode_calls) == 1
    expected_decode_arg = pipe.scheduler.last_latents / pipe.vae.config.scaling_factor
    assert torch.allclose(pipe.vae.decode_calls[0], expected_decode_arg)

    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert pipe.free_hooks_calls == 1


def test_run_hunyuan_img2img_cancellation_stops_before_next_step():
    pipe = FakeHunyuanPipe()
    conditioning = _fake_conditioning()

    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] >= 1

    with pytest.raises(concurrent.futures.CancelledError):
        run_hunyuan_img2img(
            pipe,
            "source-image",
            conditioning,
            width=64,
            height=64,
            steps=4,
            strength=0.5,
            guidance_scale=3.5,
            generator=torch.Generator(device="cpu").manual_seed(0),
            progress=None,
            should_cancel=should_cancel,
        )

    assert len(pipe.transformer.calls) == 1
    assert len(pipe.scheduler.step_calls) == 1
