"""Shared diffusers LoRA fuse helper (STABL-fvwsvjvn).

The Z-Image worker tests (tests/test_zimage_lora_worker.py) cover the fuse
steps through the helper. These tests cover the down-tensor count that the
fit check compares against.
"""
from __future__ import annotations

import torch

from backends.diffusers_lora_fuse import count_lora_down_tensors


def _t() -> torch.Tensor:
    return torch.zeros(1)


def test_counts_old_diffusers_dot_keys():
    sd = {"unet.x.lora.down.weight": _t(), "unet.x.lora.up.weight": _t()}

    assert count_lora_down_tensors(sd, include_text_encoder=False) == 1


def _mixed() -> dict[str, torch.Tensor]:
    return {
        "diffusion_model.x.lora_A.weight": _t(),
        "diffusion_model.x.lora_B.weight": _t(),
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": _t(),
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": _t(),
    }


def test_excludes_text_encoder_keys_when_not_fused():
    assert count_lora_down_tensors(_mixed(), include_text_encoder=False) == 1


def test_includes_text_encoder_keys_when_fused():
    assert count_lora_down_tensors(_mixed(), include_text_encoder=True) == 2


def test_alpha_keys_not_counted():
    sd = _mixed()
    sd["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.alpha"] = _t()

    assert count_lora_down_tensors(sd, include_text_encoder=True) == 2
