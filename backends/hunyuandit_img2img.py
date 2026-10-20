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
