from __future__ import annotations

from .artifacts import ConditioningCompatibility, MaterializedConditioning
from .contracts import (
    ConditioningRequest,
    ConditioningServiceRequirements,
    ModelContext,
)
from .invocation import CompletedInvocation


def _load_compel():
    try:
        from compel import Compel, ReturnedEmbeddingsType
    except ModuleNotFoundError as error:
        if error.name != "compel":
            raise
        raise RuntimeError(
            "Compel is not installed; install requirements-conditioning.txt "
            "or select the native conditioning service"
        ) from error
    return Compel, ReturnedEmbeddingsType


def _dtype_name(dtype: object) -> str:
    name = str(dtype)
    return name.removeprefix("torch.")


def _to_live_dtype(tensor: object, dtype: object) -> object:
    to_dtype = getattr(tensor, "to", None)
    if to_dtype is None:
        return tensor
    return to_dtype(dtype=dtype)


class CompelConditioningService:
    requirements = ConditioningServiceRequirements(local_encoder_bundle=True)

    def invoke(
        self,
        request: ConditioningRequest,
        context: ModelContext,
    ) -> CompletedInvocation:
        try:
            return CompletedInvocation.success(self._materialize(request, context))
        except Exception as error:
            return CompletedInvocation.failure(error)

    def _materialize(
        self,
        request: ConditioningRequest,
        context: ModelContext,
    ) -> MaterializedConditioning:
        bundle = context.local_encoder_bundle
        if bundle is None:
            raise RuntimeError("Compel conditioning requires a local encoder bundle")

        Compel, ReturnedEmbeddingsType = _load_compel()
        descriptor = context.descriptor
        prompt = request.prompt
        negative_prompt = request.negative_prompt or ""

        if descriptor.model_family == "sdxl":
            slots = self._materialize_sdxl(
                Compel,
                ReturnedEmbeddingsType,
                bundle,
                descriptor.device,
                prompt,
                negative_prompt,
            )
        else:
            slots = self._materialize_sd15(
                Compel, bundle, descriptor.device, prompt, negative_prompt
            )

        live_dtype = bundle.live_dtype()
        normalized_slots = {
            name: _to_live_dtype(tensor, live_dtype) for name, tensor in slots.items()
        }
        return MaterializedConditioning(
            slots=normalized_slots,
            compatibility=ConditioningCompatibility(
                model_family=descriptor.model_family,
                encoder_identities=descriptor.encoder_identities,
                hidden_dimensions=descriptor.hidden_dimensions,
                pooled_required=descriptor.pooled_required,
                dtype_name=_dtype_name(live_dtype),
            ),
        )

    def _materialize_sd15(
        self,
        Compel,
        bundle,
        device: str,
        prompt: str,
        negative_prompt: str,
    ) -> dict[str, object]:
        tokenizers = bundle.tokenizers()
        text_encoders = bundle.text_encoders()
        compel = Compel(
            tokenizer=tokenizers[0],
            text_encoder=text_encoders[0],
            device=device,
            truncate_long_prompts=False,
        )
        prompt_embeds = compel(prompt)
        negative_prompt_embeds = compel(negative_prompt)
        prompt_embeds, negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    def _materialize_sdxl(
        self,
        Compel,
        ReturnedEmbeddingsType,
        bundle,
        device: str,
        prompt: str,
        negative_prompt: str,
    ) -> dict[str, object]:
        compel = Compel(
            tokenizer=list(bundle.tokenizers()),
            text_encoder=list(bundle.text_encoders()),
            returned_embeddings_type=(
                ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
            ),
            requires_pooled=[False, True],
            device=device,
            truncate_long_prompts=False,
        )
        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)
        empty_prompt_embeds = negative_prompt_embeds
        if negative_prompt:
            empty_prompt_embeds, _ = compel("")
        prompt_embeds, negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds],
            precomputed_padding=empty_prompt_embeds,
        )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }
