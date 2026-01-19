import argparse
import json
import time

import PIL
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import (
    LCMScheduler
)

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os

import torch  # Only used for `torch.from_tensor` in `pipe.scheduler.step()`
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from typing import Callable, List, Optional, Union, Tuple
from PIL import Image

from rknnlite.api import RKNNLite

import os
import json
import time
from typing import List, Any, Optional, Union

import numpy as np
from rknnlite.api import RKNNLite

class RKNN2Model:
    """Wrapper for running RKNPU2 (RKNNLite) models"""

    def __init__(
        self,
        model_dir: str,
        *,
        core_mask: Optional[Union[str, int]] = None,
        multi_context: bool = True,
        data_format: str = "nhwc",
        verbose_shapes: bool = False,
        runtime_kwargs: Optional[dict] = None,
        force_fp32=True,
        **_ignored: Any,
    ):
        """
        Params are designed to match the pipeline service pattern:
          RKNN2Model(path, **rknn_context_cfg)

        - core_mask: can be None (defaults), string ("NPU_CORE_0"/"NPU_CORE_1"/"NPU_CORE_2"/"NPU_CORE_AUTO"),
                    or RKNNLite constant/int if you pass it directly.
          NOTE: you said multi-core causes kernel crash; leave default to AUTO.
        - multi_context: kept for compatibility with pool patterns. This class is already per-instance.
        - data_format: passed to inference (default "nchw")
        - verbose_shapes: log/print input/output shapes (disable for server)
        - runtime_kwargs: optional extra kwargs to pass into init_runtime(...)
        - **_ignored: allows you to pass context_name/worker_id etc without breaking
        """
        self.data_format = data_format.lower()
        self.force_fp32 = force_fp32        
        self.model_dir = model_dir
        self.data_format = data_format
        self.verbose_shapes = verbose_shapes
        self.multi_context = multi_context
        self.runtime_kwargs = runtime_kwargs or {}
        self.modelname = os.path.basename(model_dir.rstrip("/"))


        # Known-good key orders (fallback)
        self.key_orders = {
            "text_encoder": ("input_ids",),
            "unet": ("sample", "timestep", "encoder_hidden_states", "timestep_cond"),
            "vae_decoder": ("latent_sample",),  # change to match your call
        }        

        logger.info(f"Loading {model_dir}")
        start = time.time()

        cfg_path = os.path.join(model_dir, "config.json")
        rknn_path = os.path.join(model_dir, "model.rknn")

        if not (os.path.exists(model_dir) and os.path.exists(rknn_path)):
            raise FileNotFoundError(f"Missing model dir or model.rknn: {model_dir}")

        self.config = json.load(open(cfg_path, "r"))

        self.rknnlite = RKNNLite()
        self.rknnlite.load_rknn(rknn_path)

        # Resolve core mask
        resolved_core_mask = self._resolve_core_mask(core_mask)

        # IMPORTANT: Use AUTO by default because you noted multi-core can crash.
        # If you later confirm stability, pass core_mask="NPU_CORE_0"/"NPU_CORE_1"/"NPU_CORE_2" per worker.
        self.rknnlite.init_runtime(core_mask=resolved_core_mask, **self.runtime_kwargs)

        load_time = time.time() - start
        logger.info(f"Done loading {model_dir}. Took {load_time:.1f} seconds.")

        self.modelname = os.path.basename(model_dir.rstrip("/"))
        self.inference_time = 0

    def _resolve_core_mask(self, core_mask: Optional[Union[str, int]]) -> int:
        if core_mask is None:
            return RKNNLite.NPU_CORE_AUTO

        # Allow passing RKNNLite constant directly
        if isinstance(core_mask, int):
            return core_mask

        # Allow passing names
        if isinstance(core_mask, str):
            key = core_mask.strip().upper()
            mapping = {
                "NPU_CORE_AUTO": RKNNLite.NPU_CORE_AUTO,
                "NPU_CORE_0": RKNNLite.NPU_CORE_0,
                "NPU_CORE_1": RKNNLite.NPU_CORE_1,
                "NPU_CORE_2": RKNNLite.NPU_CORE_2,
                # Some people write these:
                "AUTO": RKNNLite.NPU_CORE_AUTO,
                "0": RKNNLite.NPU_CORE_0,
                "1": RKNNLite.NPU_CORE_1,
                "2": RKNNLite.NPU_CORE_2,
            }
            if key not in mapping:
                raise ValueError(f"Unknown core_mask string: {core_mask!r}")
            return mapping[key]

        raise TypeError(f"core_mask must be None, int, or str; got {type(core_mask)}")

    def __call__(self, **kwargs):
        import numpy as np

        def prep(x):
            if isinstance(x, np.ndarray):
                # dtype safety
                if x.dtype == np.float64:
                    x = x.astype(np.float32, copy=False)
                elif x.dtype == np.float16:
                    x = x.astype(np.float32, copy=False)

                # layout safety: only transpose 4D tensors at RKNN boundary
                if x.ndim == 4:
                    if self.data_format == "nhwc" and x.shape[1] in (1, 3, 4):      # NCHW -> NHWC
                        x = x.transpose(0, 2, 3, 1)
                    elif self.data_format == "nchw" and x.shape[-1] in (1, 3, 4):   # NHWC -> NCHW
                        x = x.transpose(0, 3, 1, 2)

                x = np.ascontiguousarray(x)
            return x

        # deterministic per-model input ordering
        if self.modelname == "text_encoder":
            order = ("input_ids",)
        elif self.modelname == "unet":
            order = ("sample", "timestep", "encoder_hidden_states", "timestep_cond")
        elif self.modelname == "vae_decoder":
            order = ("latent_sample",)
        else:
            order = tuple(sorted(kwargs.keys()))

        input_list = [prep(kwargs[k]) for k in order]

        if self.modelname == "vae_decoder":
            x = input_list[0]
            logger.info("vae in[0] shape=%s dtype=%s contiguous=%s", x.shape, x.dtype, x.flags['C_CONTIGUOUS'])
        results = self.rknnlite.inference(inputs=input_list, data_format=self.data_format)

        logger.info("%s out[0] shape=%s dtype=%s", self.modelname, results[0].shape, results[0].dtype)
        return results

class RKNN2LatentConsistencyPipeline(DiffusionPipeline):

    def __init__(
            self,
            text_encoder: RKNN2Model,
            unet: RKNN2Model,
            vae_decoder: RKNN2Model,
            scheduler: LCMScheduler,
            tokenizer: CLIPTokenizer,
            force_zeros_for_empty_prompt: Optional[bool] = True,
            feature_extractor: Optional[CLIPFeatureExtractor] = None,
            text_encoder_2: Optional[RKNN2Model] = None,
            tokenizer_2: Optional[CLIPTokenizer] = None
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.force_zeros_for_empty_prompt = force_zeros_for_empty_prompt
        self.safety_checker = None

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        self.unet = unet
        self.vae_decoder = vae_decoder

        VAE_DECODER_UPSAMPLE_FACTOR = 8
        self.vae_scale_factor = VAE_DECODER_UPSAMPLE_FACTOR

    @staticmethod
    def postprocess(
        image: np.ndarray,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
        ):
        def numpy_to_pil(images: np.ndarray):
            """
            Convert a numpy image or a batch of images to a PIL image.
            """
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            if images.shape[-1] == 1:
                # special case for grayscale (single channel) images
                pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
            else:
                pil_images = [Image.fromarray(image) for image in images]

            return pil_images
        
        def denormalize(images: np.ndarray):
            """
            Denormalize an image array to [0,1].
            """
            return np.clip(images / 2 + 0.5, 0, 1)
    
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support np array"
            )
        if output_type not in ["latent", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            logger.warning(deprecation_message)
            output_type = "np"

        if output_type == "latent":
            return image
        
        if do_denormalize is None:
            raise ValueError("do_denormalize is required for postprocessing")

        image = np.stack(
            [denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])], axis=0
        )
        image = image.transpose((0, 2, 3, 1))

        if output_type == "pil":
            image = numpy_to_pil(image)

        return image

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, list]],
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`Union[str, List[str]]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from https://github.com/huggingface/diffusers/blob/v0.17.1/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L217
    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int],
        width: Optional[int],
        callback_steps: int,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Keep latents in NCHW everywhere in Python, and only convert to NHWC right at the RKNN boundary for models that require it.
    #That means:
    #•   Before UNet RKNN call: NCHW -> NHWC
    #•   After UNet RKNN call: NHWC -> NCHW (only if the raw output is NHWC)
    #•   VAE decoder input: if it expects NHWC, convert right before it too.
    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, np.random.RandomState):
                latents = generator.randn(*shape).astype(dtype)
            elif isinstance(generator, torch.Generator):
                latents = torch.randn(*shape, generator=generator).numpy().astype(dtype)
            else:
                raise ValueError(
                    f"Expected `generator` to be of type `np.random.RandomState` or `torch.Generator`, but got"
                    f" {type(generator)}."
                )
        elif latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        return latents

    # Adapted from https://github.com/huggingface/diffusers/blob/v0.22.0/src/diffusers/pipelines/latent_consistency/pipeline_latent_consistency.py#L264
    def __call__(
        self,
        prompt: Union[str, List[str]] = "",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        original_inference_steps: int = None,
        guidance_scale: float = 8.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[np.random.RandomState, torch.Generator]] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`Optional[Union[str, List[str]]]`, defaults to None):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`Optional[int]`, defaults to None):
                The height in pixels of the generated image.
            width (`Optional[int]`, defaults to None):
                The width in pixels of the generated image.
            num_inference_steps (`int`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, defaults to 1):
                The number of images to generate per prompt.
            generator (`Optional[Union[np.random.RandomState, torch.Generator]]`, defaults to `None`):
                A np.random.RandomState to make generation deterministic.
            latents (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (Optional[Callable], defaults to `None`):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            guidance_rescale (`float`, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        height = height or self.unet.config["sample_size"] * self.vae_scale_factor
        width = width or self.unet.config["sample_size"] * self.vae_scale_factor

        # Don't need to get negative prompts due to LCM guided distillation
        negative_prompt = None
        negative_prompt_embeds = None

        # check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random.RandomState()

        start_time = time.time()
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            False,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        encode_prompt_time = time.time() - start_time
        print(f"Prompt encoding time: {encode_prompt_time:.2f}s")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, original_inference_steps=original_inference_steps)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config["in_channels"],
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        bs = batch_size * num_images_per_prompt
        # get Guidance Scale Embedding
        w = np.full(bs, guidance_scale - 1, dtype=prompt_embeds.dtype)
        w_embedding = self.get_guidance_scale_embedding(
            w, embedding_dim=self.unet.config["time_cond_proj_dim"], dtype=prompt_embeds.dtype
        )

        # Adapted from diffusers to extend it for other runtimes than ORT
        #timestep_dtype = np.int64
        timestep_dtype = np.int32
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        inference_start = time.time()
        for i, t in enumerate(self.progress_bar(timesteps)):
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(
                sample=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=w_embedding,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), return_dict=False
            )
            latents, denoised = latents.numpy(), denoised.numpy()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        inference_time = time.time() - inference_start
        print(f"Inference time: {inference_time:.2f}s")

        decode_start = time.time()
        if output_type == "latent":
            image = denoised
            has_nsfw_concept = None
        else:
            t0 = time.time()
            denoised /= self.vae_decoder.config["scaling_factor"]
            t1 = time.time()

            t_inf0 = time.time()
            outs = [self.vae_decoder(latent_sample=denoised[i:i+1])[0] for i in range(denoised.shape[0])]
            t_inf1 = time.time()

            t_cat0 = time.time()
            image = np.concatenate(outs)
            t_cat1 = time.time()

            has_nsfw_concept = None  # skip safety checker

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        t_post0 = time.time()
        image = self.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        t_post1 = time.time()

        print("scale:", t1-t0, "vae_inf:", t_inf1-t_inf0, "concat:", t_cat1-t_cat0, "post:", t_post1-t_post0)

        decode_time = time.time() - decode_start
        print(f"Decode time: {decode_time:.2f}s")

        total_time = encode_prompt_time + inference_time + decode_time
        print(f"Total time: {total_time:.2f}s")

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


    # Adapted from https://github.com/huggingface/diffusers/blob/v0.22.0/src/diffusers/pipelines/latent_consistency/pipeline_latent_consistency.py#L264
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=None):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        w = w * 1000
        half_dim = embedding_dim // 2
        emb = np.log(10000.0) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
        emb = w[:, None] * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)

        if embedding_dim % 2 == 1:  # zero pad
            emb = np.pad(emb, [(0, 0), (0, 1)])

        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

def get_image_path(args, **override_kwargs):
    """ mkdir output folder and encode metadata in the filename
    """
    out_folder = os.path.join(args.o, "_".join(args.prompt.replace("/", "_").rsplit(" ")))
    os.makedirs(out_folder, exist_ok=True)

    out_fname = f"randomSeed_{override_kwargs.get('seed', None) or args.seed}"

    out_fname += f"_LCM_"
    out_fname += f"_numInferenceSteps{override_kwargs.get('num_inference_steps', None) or args.num_inference_steps}"

    return os.path.join(out_folder, out_fname + ".png")


def prepare_controlnet_cond(image_path, height, width):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((height, width), resample=Image.LANCZOS)
    image = np.array(image).transpose(2, 0, 1) / 255.0
    return image

#args.prompt seed=4234924 i=model_path o=output_path size=256x256 num_inference_steps guidance_scale
def generate_png_bytes(args):
    logger.info(f"Setting random seed to {args.seed}")

    scheduler_config_path = os.path.join(args.i, "scheduler/scheduler_config.json")
    with open(scheduler_config_path, "r") as f:
        scheduler_config = json.load(f)

    user_specified_scheduler = LCMScheduler.from_config(scheduler_config)

    pipe = RKNN2LatentConsistencyPipeline(
        text_encoder=RKNN2Model(self.paths.text_encoder, data_format="nchw", **self.rknn_context_cfg),
        unet=RKNN2Model(self.paths.unet, data_format="nhwc", **self.rknn_context_cfg),
        vae_decoder=RKNN2Model(self.paths.vae_decoder, data_format="nchw", **self.rknn_context_cfg),
        scheduler=user_specified_scheduler,
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16"),
    )

    logger.info("Beginning image generation.")

    result = pipe(
        prompt=args.prompt,
        height=int(args.size.split("x")[0]),
        width=int(args.size.split("x")[1]),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=np.random.RandomState(args.seed),
    )

    pil_image = result["images"][0]

    # Convert to PNG bytes
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)

    return buf.getvalue()

def main(args):
    logger.info(f"Setting random seed to {args.seed}")

    # load scheduler from scheduler/scheduler_config.json
    scheduler_config_path = os.path.join(args.i, "scheduler/scheduler_config.json")
    with open(scheduler_config_path, "r") as f:
        scheduler_config = json.load(f)
    user_specified_scheduler = LCMScheduler.from_config(scheduler_config)

    logger.info("Using scheduler: %s", user_specified_scheduler.__class__.__name__)

    # Parse size as WIDTHxHEIGHT (common CLI convention)
    w_str, h_str = args.size.lower().split("x")
    width, height = int(w_str), int(h_str)

    pipe = RKNN2LatentConsistencyPipeline(
        text_encoder=RKNN2Model(os.path.join(args.i, "text_encoder"), data_format="nchw"),
        unet=RKNN2Model(os.path.join(args.i, "unet"), data_format="nhwc"),
        vae_decoder=RKNN2Model(os.path.join(args.i, "vae_decoder"), data_format="nhwc"),
        scheduler=user_specified_scheduler,
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16"),
    )

    logger.info("Beginning image generation.")
    out = pipe(
        prompt=args.prompt,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=np.random.RandomState(args.seed),
    )

    out_path = get_image_path(args)
    logger.info("Saving generated image to %s", out_path)
    out["images"][0].save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        required=True,
        help="The text prompt to be used for text-to-image generation.")
    parser.add_argument(
        "-i",
        required=True,
        help=("Path to model directory"))
    parser.add_argument("-o", required=True)
    parser.add_argument("--seed",
                        default=93,
                        type=int,
                        help="Random seed to be able to reproduce results")
    parser.add_argument(
        "-s",
        "--size",
        default="256x256",
        type=str,
        help="Image size")
    parser.add_argument(
        "--num-inference-steps",
        default=4,
        type=int,
        help="The number of iterations the unet model will be executed throughout the reverse diffusion process")
    parser.add_argument(
        "--guidance-scale",
        default=7.5,
        type=float,
        help="Controls the influence of the text prompt on sampling process (0=random images)")

    args = parser.parse_args()
    main(args)    
