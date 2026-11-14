"""SD1.5 and SDXL mode LoRAs through the diffusers fuse (STABL-fvwsvjvn).

Tiny real pipelines: the fuse writes real weights, so the tests check the
fused math, the text encoder fuse, and the refusals before any load.
"""
from __future__ import annotations

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from diffusers import (  # noqa: E402
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from safetensors.torch import save_file  # noqa: E402
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection  # noqa: E402

from backends.cuda_worker import DiffusersCudaWorker, DiffusersSDXLCudaWorker  # noqa: E402
from backends.mode_loras import LoraSpec  # noqa: E402
from backends.mps_worker import DiffusersMpsWorker  # noqa: E402
from backends.unet_mode_loras import UNetLoraError, UNetModeLoraMixin  # noqa: E402

UNET_TARGET = "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q"
TE_TARGET = "text_model.encoder.layers.0.self_attn.q_proj"


def _te_config() -> CLIPTextConfig:
    return CLIPTextConfig(
        hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4,
        vocab_size=1000, projection_dim=32, bos_token_id=0, eos_token_id=2, layer_norm_eps=1e-5,
    )


def _sd_pipe() -> StableDiffusionPipeline:
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=1, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32,
    )
    return StableDiffusionPipeline(
        vae=None, text_encoder=CLIPTextModel(_te_config()), tokenizer=None, unet=unet, scheduler=None,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )


def _sdxl_pipe() -> StableDiffusionXLPipeline:
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=1, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=64,
        attention_head_dim=(2, 4), use_linear_projection=True, addition_embed_type="text_time",
        addition_time_embed_dim=8, transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80,
    )
    return StableDiffusionXLPipeline(
        vae=None, text_encoder=CLIPTextModel(_te_config()),
        text_encoder_2=CLIPTextModelWithProjection(_te_config()), tokenizer=None, tokenizer_2=None,
        unet=unet, scheduler=None,
    )


def _worker(pipe, cls=DiffusersCudaWorker, **overrides):
    w = cls.__new__(cls)
    w.pipe = pipe
    w.worker_id = 0
    w._quantize = "none"
    w._checkpoint_precision = "unknown"
    w._offload = "none"
    w._style_loaded = {}
    w._img2img_pipe = None
    for k, v in overrides.items():
        setattr(w, k, v)
    return w


def _lora(pairs, rank=4, seed=0) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """pairs: (key prefix, linear). Returns the state dict and (A, B) per prefix."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    ab: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for prefix, lin in pairs:
        a = torch.randn(rank, lin.in_features, generator=g) * 0.1
        b = torch.randn(lin.out_features, rank, generator=g) * 0.1
        sd[f"{prefix}.lora_A.weight"] = a
        sd[f"{prefix}.lora_B.weight"] = b
        ab[prefix] = (a, b)
    return sd, ab


def _save(path, sd) -> str:
    save_file(sd, str(path))
    return str(path)


def _no_tuner_layers(module: torch.nn.Module) -> bool:
    from peft.tuners.tuners_utils import BaseTunerLayer

    return not any(isinstance(m, BaseTunerLayer) for m in module.modules())


def test_sd_unet_and_text_encoder_fuse_math(tmp_path):
    w = _worker(_sd_pipe())
    unet_lin = w.pipe.unet.get_submodule(UNET_TARGET)
    te_lin = w.pipe.text_encoder.get_submodule(TE_TARGET)
    unet_w0 = unet_lin.weight.detach().clone()
    te_w0 = te_lin.weight.detach().clone()
    sd, ab = _lora([(f"unet.{UNET_TARGET}", unet_lin), (f"text_encoder.{TE_TARGET}", te_lin)])
    path = _save(tmp_path / "sd.safetensors", sd)

    w.configure_loras([LoraSpec(path, 0.5)])

    a, b = ab[f"unet.{UNET_TARGET}"]
    torch.testing.assert_close(
        w.pipe.unet.get_submodule(UNET_TARGET).weight, unet_w0 + 0.5 * (b @ a), atol=1e-5, rtol=0
    )
    a, b = ab[f"text_encoder.{TE_TARGET}"]
    torch.testing.assert_close(
        w.pipe.text_encoder.get_submodule(TE_TARGET).weight, te_w0 + 0.5 * (b @ a), atol=1e-5, rtol=0
    )
    assert _no_tuner_layers(w.pipe.unet)
    assert _no_tuner_layers(w.pipe.text_encoder)


def test_sdxl_text_encoder_2_fuses(tmp_path):
    w = _worker(_sdxl_pipe(), cls=DiffusersSDXLCudaWorker)
    lin = w.pipe.text_encoder_2.get_submodule(TE_TARGET)
    w0 = lin.weight.detach().clone()
    sd, ab = _lora([(f"text_encoder_2.{TE_TARGET}", lin)])
    path = _save(tmp_path / "te2.safetensors", sd)

    w.configure_loras([LoraSpec(path, 1.0)])

    a, b = ab[f"text_encoder_2.{TE_TARGET}"]
    torch.testing.assert_close(
        w.pipe.text_encoder_2.get_submodule(TE_TARGET).weight, w0 + 1.0 * (b @ a), atol=1e-5, rtol=0
    )
    assert _no_tuner_layers(w.pipe.text_encoder_2)


@pytest.mark.parametrize(
    "overrides,needle",
    [
        ({"_quantize": "fp8"}, "fp8"),
        ({"_checkpoint_precision": "fp8"}, "fp8"),
    ],
)
def test_fp8_refuses(tmp_path, overrides, needle):
    w = _worker(_sd_pipe(), **overrides)
    missing = str(tmp_path / "missing.safetensors")

    with pytest.raises(UNetLoraError, match=needle):
        w.configure_loras([LoraSpec(missing, 1.0)])


def test_sequential_offload_refuses(tmp_path):
    w = _worker(_sd_pipe(), _offload="sequential")

    with pytest.raises(UNetLoraError, match="sequential"):
        w.configure_loras([LoraSpec(str(tmp_path / "missing.safetensors"), 1.0)])


def test_style_adapters_loaded_refuses(tmp_path):
    w = _worker(_sd_pipe(), _style_loaded={"papercut": True})

    with pytest.raises(UNetLoraError, match="style"):
        w.configure_loras([LoraSpec(str(tmp_path / "missing.safetensors"), 1.0)])


def test_partial_fit_refuses(tmp_path):
    w = _worker(_sd_pipe())
    lin = w.pipe.unet.get_submodule(UNET_TARGET)
    w0 = lin.weight.detach().clone()
    sd, _ = _lora([(f"unet.{UNET_TARGET}", lin)])
    sd["unet.not_a_module.lora_A.weight"] = torch.randn(4, 64)
    sd["unet.not_a_module.lora_B.weight"] = torch.randn(64, 4)
    path = _save(tmp_path / "partial.safetensors", sd)

    with pytest.raises(UNetLoraError, match="partial.safetensors"):
        w.configure_loras([LoraSpec(path, 1.0)])

    torch.testing.assert_close(w.pipe.unet.get_submodule(UNET_TARGET).weight, w0, atol=0, rtol=0)


@pytest.mark.parametrize("cls", [DiffusersCudaWorker, DiffusersSDXLCudaWorker, DiffusersMpsWorker])
def test_workers_have_configure_loras(cls):
    assert issubclass(cls, UNetModeLoraMixin)
    assert cls.configure_loras is UNetModeLoraMixin.configure_loras


def test_img2img_shares_fused_unet(tmp_path):
    w = _worker(_sd_pipe())
    lin = w.pipe.unet.get_submodule(UNET_TARGET)
    w0 = lin.weight.detach().clone()
    sd, _ = _lora([(f"unet.{UNET_TARGET}", lin)])
    w.configure_loras([LoraSpec(_save(tmp_path / "a.safetensors", sd), 1.0)])

    img2img = StableDiffusionImg2ImgPipeline(**w.pipe.components)

    assert img2img.unet is w.pipe.unet
    assert not torch.equal(img2img.unet.get_submodule(UNET_TARGET).weight, w0)
