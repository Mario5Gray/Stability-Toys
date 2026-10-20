"""Hunyuan-only img2img execution component.

Shares the already-loaded Hunyuan transformer, VAE, text encoders,
tokenizers, and scheduler. Diffusers 0.39 has no HunyuanDiT img2img
pipeline, so this module owns the denoising path directly rather than
routing through an SD/SDXL pipeline.
"""


def select_timesteps(scheduler, steps, strength, device):
    """Pick the strength-selected suffix of ``scheduler``'s timesteps.

    Returns ``(timesteps, effective_steps)``. ``effective_steps`` is the
    count of timesteps that will actually run; a low ``strength`` keeps
    more of the source image by running fewer of the final steps.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if not 0 < strength <= 1:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    for method in ("set_timesteps", "add_noise"):
        if not callable(getattr(scheduler, method, None)):
            raise ValueError(
                f"{type(scheduler).__name__} does not support required "
                f"scheduler method {method!r}"
            )

    scheduler.set_timesteps(steps, device=device)
    effective_steps = min(steps, max(1, int(steps * strength)))
    offset = (steps - effective_steps) * scheduler.order
    timesteps = scheduler.timesteps[offset:]
    if not timesteps:
        raise ValueError("scheduler produced an empty timestep schedule")
    return timesteps, effective_steps


def prepare_source_latents(pipe, image, *, width, height, timestep, generator, dtype, device):
    """Encode ``image`` with ``pipe``'s loaded VAE and noise it to ``timestep``.

    Returns the noisy source latent, ready as the denoising loop's initial
    latent. Dimensions are validated before the VAE is touched.
    """
    from diffusers.utils.torch_utils import randn_tensor

    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    image_tensor = pipe.image_processor.preprocess(image, height=height, width=width)
    image_tensor = image_tensor.to(device=pipe.vae.device, dtype=pipe.vae.dtype)

    latent_dist = pipe.vae.encode(image_tensor).latent_dist
    source_latent = latent_dist.sample(generator=generator) * pipe.vae.config.scaling_factor
    source_latent = source_latent.to(device=device, dtype=dtype)

    noise = randn_tensor(source_latent.shape, generator=generator, device=source_latent.device, dtype=dtype)
    return pipe.scheduler.add_noise(source_latent, noise, timestep)
