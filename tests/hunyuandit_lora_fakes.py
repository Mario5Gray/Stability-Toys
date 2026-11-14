"""Test helpers for the HunyuanDiT Tencent LoRA merge (STABL-ykowysrg).

The fake transformer uses the diffusers HunyuanDiT2DModel module names that the
merge targets. The fake LoRA uses the Tencent original hydit key names.
"""
from __future__ import annotations

import torch
from torch import nn

TENCENT_PREFIX = "base_model.model."


class _Attn(nn.Module):
    def __init__(self, hidden: int, kv_in: int, dtype: torch.dtype):
        super().__init__()
        self.to_q = nn.Linear(hidden, hidden, dtype=dtype)
        self.to_k = nn.Linear(kv_in, hidden, dtype=dtype)
        self.to_v = nn.Linear(kv_in, hidden, dtype=dtype)
        self.to_out = nn.ModuleList([nn.Linear(hidden, hidden, dtype=dtype)])


class _Block(nn.Module):
    def __init__(self, hidden: int, cross: int, dtype: torch.dtype):
        super().__init__()
        self.attn1 = _Attn(hidden, hidden, dtype)
        self.attn2 = _Attn(hidden, cross, dtype)


class _Pooler(nn.Module):
    def __init__(self, pooler: int, dtype: torch.dtype):
        super().__init__()
        self.q_proj = nn.Linear(pooler, pooler, dtype=dtype)


class _TimeExtraEmb(nn.Module):
    def __init__(self, pooler: int, dtype: torch.dtype):
        super().__init__()
        self.pooler = _Pooler(pooler, dtype)


class FakeTransformer(nn.Module):
    def __init__(self, num_blocks: int, hidden: int, cross: int, pooler: int, dtype: torch.dtype):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(hidden, cross, dtype) for _ in range(num_blocks))
        self.time_extra_emb = _TimeExtraEmb(pooler, dtype)


def make_fake_transformer(num_blocks=2, hidden=8, cross=6, pooler=5, dtype=torch.float32, seed=7):
    torch.manual_seed(seed)
    return FakeTransformer(num_blocks, hidden, cross, pooler, dtype)


def make_fake_lora(transformer, rank=2, seed=0, prefix=True):
    """Build a Tencent-layout LoRA dict that fits `transformer`."""
    g = torch.Generator().manual_seed(seed)
    hidden = transformer.blocks[0].attn1.to_q.in_features
    cross = transformer.blocks[0].attn2.to_k.in_features
    pooler = transformer.time_extra_emb.pooler.q_proj.in_features
    # (source key, in_features, out_features)
    per_block = (
        ("attn1.Wqkv", hidden, 3 * hidden),
        ("attn1.out_proj", hidden, hidden),
        ("attn2.q_proj", hidden, hidden),
        ("attn2.kv_proj", cross, 2 * hidden),
        ("attn2.out_proj", hidden, hidden),
    )
    out: dict[str, torch.Tensor] = {}

    def put(src, fan_in, fan_out):
        out[f"{src}.lora_A.weight"] = torch.randn(rank, fan_in, generator=g) * 0.1
        out[f"{src}.lora_B.weight"] = torch.randn(fan_out, rank, generator=g) * 0.1

    for i in range(len(transformer.blocks)):
        for src, fan_in, fan_out in per_block:
            put(f"blocks.{i}.{src}", fan_in, fan_out)
    put("pooler.q_proj", pooler, pooler)
    if prefix:
        out = {TENCENT_PREFIX + k: v for k, v in out.items()}
    return out


def tencent_reference_merge(transformer_state_dict, lora_state_dict, lora_scale, num_layers):
    """Tencent's published load_hunyuan_dit_lora, with num_layers as a parameter."""
    for i in range(num_layers):
        Wqkv = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_A.weight"])
        q, k, v = torch.chunk(Wqkv, 3, dim=0)
        transformer_state_dict[f"blocks.{i}.attn1.to_q.weight"] += lora_scale * q
        transformer_state_dict[f"blocks.{i}.attn1.to_k.weight"] += lora_scale * k
        transformer_state_dict[f"blocks.{i}.attn1.to_v.weight"] += lora_scale * v

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn1.to_out.0.weight"] += lora_scale * out_proj

        q_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn2.to_q.weight"] += lora_scale * q_proj

        kv_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_A.weight"])
        k, v = torch.chunk(kv_proj, 2, dim=0)
        transformer_state_dict[f"blocks.{i}.attn2.to_k.weight"] += lora_scale * k
        transformer_state_dict[f"blocks.{i}.attn2.to_v.weight"] += lora_scale * v

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn2.to_out.0.weight"] += lora_scale * out_proj

    q_proj = torch.matmul(lora_state_dict["pooler.q_proj.lora_B.weight"], lora_state_dict["pooler.q_proj.lora_A.weight"])
    transformer_state_dict["time_extra_emb.pooler.q_proj.weight"] += lora_scale * q_proj

    return transformer_state_dict


def clone_state(transformer):
    return {k: v.detach().clone() for k, v in transformer.state_dict().items()}


def save_lora(path, lora):
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in lora.items()}, str(path))
    return path
