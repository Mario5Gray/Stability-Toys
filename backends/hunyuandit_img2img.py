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

    if width <= 0 or height <= 0 or width % 8 != 0 or height % 8 != 0:
        raise ValueError(
            f"width and height must be positive and divisible by 8, got {width}x{height}"
        )

    image_tensor = pipe.image_processor.preprocess(image, height=height, width=width)
    image_tensor = image_tensor.to(device=pipe.vae.device, dtype=pipe.vae.dtype)

    latent_dist = pipe.vae.encode(image_tensor).latent_dist
    source_latent = latent_dist.sample(generator=generator) * pipe.vae.config.scaling_factor
    source_latent = source_latent.to(device=device, dtype=dtype)

    noise = randn_tensor(source_latent.shape, generator=generator, device=source_latent.device, dtype=dtype)
    return pipe.scheduler.add_noise(source_latent, noise, timestep)


def run_hunyuan_img2img(
    pipe,
    image,
    conditioning,
    *,
    width,
    height,
    steps,
    strength,
    guidance_scale,
    generator,
    progress=None,
    should_cancel=None,
):
    """Run the Hunyuan-only img2img denoising loop.

    ``conditioning`` is the dict produced by the worker's existing
    conditioning artifact (``prompt_embeds``/``negative_prompt_embeds`` for
    the CLIP/BERT encoder, the ``_2`` pair for mT5), the same shape fed to
    ``HunyuanDiTPipeline.__call__`` today. This mirrors that call's
    denoising loop exactly (diffusers 0.39
    ``pipelines/hunyuandit/pipeline_hunyuandit.py``) since diffusers has no
    HunyuanDiT img2img pipeline to call instead.
    """
    import torch
    from concurrent.futures import CancelledError

    from diffusers.models.embeddings import get_2d_rotary_pos_embed
    from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import get_resize_crop_region_for_grid

    device = pipe.transformer.device
    dtype = pipe.transformer.dtype
    do_cfg = guidance_scale > 1.0
    num_images_per_prompt = 1

    timesteps, effective_steps = select_timesteps(pipe.scheduler, steps, strength, device)
    latents = prepare_source_latents(
        pipe,
        image,
        width=width,
        height=height,
        timestep=timesteps[0],
        generator=generator,
        dtype=dtype,
        device=device,
    )

    prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask = (
        pipe.encode_prompt(
            prompt=None,
            device=device,
            dtype=dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            prompt_embeds=conditioning["prompt_embeds"],
            negative_prompt_embeds=conditioning.get("negative_prompt_embeds"),
            prompt_attention_mask=conditioning["prompt_attention_mask"],
            negative_prompt_attention_mask=conditioning.get("negative_prompt_attention_mask"),
            max_sequence_length=77,
            text_encoder_index=0,
        )
    )
    prompt_embeds_2, negative_prompt_embeds_2, prompt_attention_mask_2, negative_prompt_attention_mask_2 = (
        pipe.encode_prompt(
            prompt=None,
            device=device,
            dtype=dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            prompt_embeds=conditioning["prompt_embeds_2"],
            negative_prompt_embeds=conditioning.get("negative_prompt_embeds_2"),
            prompt_attention_mask=conditioning["prompt_attention_mask_2"],
            negative_prompt_attention_mask=conditioning.get("negative_prompt_attention_mask_2"),
            max_sequence_length=256,
            text_encoder_index=1,
        )
    )

    batch_size = conditioning["prompt_embeds"].shape[0]

    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
        prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
        prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])

    grid_height = height // 8 // pipe.transformer.config.patch_size
    grid_width = width // 8 // pipe.transformer.config.patch_size
    base_size = 512 // 8 // pipe.transformer.config.patch_size
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
    image_rotary_emb = get_2d_rotary_pos_embed(
        pipe.transformer.inner_dim // pipe.transformer.num_heads,
        grid_crops_coords,
        (grid_height, grid_width),
        device=device,
        output_type="pt",
    )

    style = torch.tensor([0], device=device)
    if do_cfg:
        style = torch.cat([style] * 2, dim=0)
    style = style.repeat(batch_size * num_images_per_prompt)

    add_time_ids = torch.tensor([[1024, 1024, height, width, 0, 0]], dtype=prompt_embeds.dtype, device=device)
    if do_cfg:
        add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        t_expand = torch.tensor([t] * latent_model_input.shape[0], device=device, dtype=latent_model_input.dtype)

        noise_pred = pipe.transformer(
            latent_model_input,
            t_expand,
            encoder_hidden_states=prompt_embeds,
            text_embedding_mask=prompt_attention_mask,
            encoder_hidden_states_t5=prompt_embeds_2,
            text_embedding_mask_t5=prompt_attention_mask_2,
            image_meta_size=add_time_ids,
            style=style,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        noise_pred, _ = noise_pred.chunk(2, dim=1)
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if should_cancel is not None and should_cancel():
            raise CancelledError("job cancelled at step boundary")
        if progress is not None:
            progress(i + 1, effective_steps, "denoise")

    image_out = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image_out, has_nsfw_concept = pipe.run_safety_checker(image_out, device, prompt_embeds.dtype)
    if has_nsfw_concept is None:
        do_denormalize = [True] * image_out.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    images = pipe.image_processor.postprocess(image_out, output_type="pil", do_denormalize=do_denormalize)
    pipe.maybe_free_model_hooks()
    return images
