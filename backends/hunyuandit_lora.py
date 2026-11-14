"""Merge Tencent HunyuanDiT PEFT LoRAs into a diffusers HunyuanDiT2DModel.

diffusers 0.39 has no LoRA loader for HunyuanDiT. Tencent publishes LoRAs
(for example HYDiT-LoRA jade and porcelain) with the original hydit key names.
This module maps those keys to the diffusers module names and adds the LoRA
delta into the weights, the same math as Tencent's load_hunyuan_dit_lora.

The merge is permanent for the loaded pipeline. A mode reload restores the
base weights. Alpha is not applied, the same as the Tencent reference code.
See docs/superpowers/specs/2026-11-14-hunyuandit-mode-lora-design.md
(STABL-ykowysrg).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

TENCENT_PREFIX = "base_model.model."
ADAPTER_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"

# Source key (per block, under "blocks.<i>.") -> diffusers target modules.
# A source with several targets is split into equal row chunks.
_BLOCK_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attn1.Wqkv", ("attn1.to_q", "attn1.to_k", "attn1.to_v")),
    ("attn1.out_proj", ("attn1.to_out.0",)),
    ("attn2.q_proj", ("attn2.to_q",)),
    ("attn2.kv_proj", ("attn2.to_k", "attn2.to_v")),
    ("attn2.out_proj", ("attn2.to_out.0",)),
)
_GLOBAL_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pooler.q_proj", ("time_extra_emb.pooler.q_proj",)),
)

_MAX_REPORTED_KEYS = 5


class HunyuanDiTLoraError(ValueError):
    """A HunyuanDiT LoRA cannot be read, validated, or merged."""


def resolve_lora_file(path: str | Path) -> Path:
    """Return the LoRA file. A directory resolves to <dir>/adapter_model.safetensors."""
    p = Path(path)
    if p.is_dir():
        p = p / ADAPTER_FILE
    if not p.is_file():
        raise HunyuanDiTLoraError(f"HunyuanDiT LoRA file not found: {p}")
    return p


def read_tencent_lora(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a LoRA file to CPU tensors and strip the Tencent key prefix."""
    from safetensors.torch import load_file

    raw = load_file(str(path), device="cpu")
    return {
        (k[len(TENCENT_PREFIX):] if k.startswith(TENCENT_PREFIX) else k): v
        for k, v in raw.items()
    }


def read_adapter_config(lora_file: Path) -> dict:
    """Return the sibling adapter_config.json as a dict, or {} when absent."""
    cfg = Path(lora_file).with_name(ADAPTER_CONFIG)
    if not cfg.is_file():
        return {}
    try:
        return json.loads(cfg.read_text())
    except (OSError, ValueError) as e:
        raise HunyuanDiTLoraError(f"cannot read {cfg}: {e}") from e


def _pairs(transformer: torch.nn.Module) -> list[tuple[str, tuple[str, ...]]]:
    """Every (source, target module names) pair this transformer expects."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for i in range(len(transformer.blocks)):
        for src, targets in _BLOCK_MAP:
            out.append((f"blocks.{i}.{src}", tuple(f"blocks.{i}.{t}" for t in targets)))
    out.extend(_GLOBAL_MAP)
    return out


def _ab(src: str) -> tuple[str, str]:
    return f"{src}.lora_A.weight", f"{src}.lora_B.weight"


def _short(keys) -> str:
    keys = sorted(keys)
    more = len(keys) - _MAX_REPORTED_KEYS
    shown = ", ".join(keys[:_MAX_REPORTED_KEYS])
    return shown + (f" (+{more} more)" if more > 0 else "")


def validate_tencent_lora(transformer: torch.nn.Module, lora: dict[str, torch.Tensor]) -> None:
    """Check the full key set and every shape. Raise HunyuanDiTLoraError on a mismatch."""
    pairs = _pairs(transformer)
    expected = {k for src, _ in pairs for k in _ab(src)}
    got = set(lora)
    missing, extra = expected - got, got - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} key(s): {_short(missing)}")
        if extra:
            parts.append(f"unexpected {len(extra)} key(s): {_short(extra)}")
        raise HunyuanDiTLoraError(
            "not a Tencent HunyuanDiT LoRA for this transformer: " + "; ".join(parts)
        )

    for src, targets in pairs:
        a_key, b_key = _ab(src)
        a, b = lora[a_key], lora[b_key]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise HunyuanDiTLoraError(
                f"{src}: bad LoRA shapes A={tuple(a.shape)} B={tuple(b.shape)}"
            )
        mods = [transformer.get_submodule(t) for t in targets]
        in_features = {m.weight.shape[1] for m in mods}
        out_features = {m.weight.shape[0] for m in mods}
        if len(in_features) != 1 or len(out_features) != 1:
            raise HunyuanDiTLoraError(f"{src}: targets {targets} differ in shape")
        (fan_in,), (fan_out,) = in_features, out_features
        if a.shape[1] != fan_in or b.shape[0] != fan_out * len(targets):
            raise HunyuanDiTLoraError(
                f"{src}: LoRA A={tuple(a.shape)} B={tuple(b.shape)} does not fit "
                f"{len(targets)} target(s) of shape ({fan_out}, {fan_in})"
            )


def merge_tencent_lora(
    transformer: torch.nn.Module, lora: dict[str, torch.Tensor], strength: float
) -> int:
    """Validate, then add strength * (B @ A) into the target weights.

    The product runs in fp32 on the weight's device. The sum is cast back to
    the weight's dtype. Return the number of target weights changed.
    """
    validate_tencent_lora(transformer, lora)
    pairs = _pairs(transformer)
    # Sequential CPU offload keeps the real weights in accelerate hooks and
    # leaves meta tensors on the modules. A write there changes nothing.
    meta = [t for _, targets in pairs for t in targets
            if transformer.get_submodule(t).weight.is_meta]
    if meta:
        raise HunyuanDiTLoraError(
            f"cannot merge into meta weights ({_short(meta)}); "
            "sequential CPU offload is not supported with mode LoRAs"
        )
    changed = 0
    with torch.no_grad():
        for src, targets in pairs:
            a_key, b_key = _ab(src)
            weights = [transformer.get_submodule(t).weight for t in targets]
            device = weights[0].device
            delta = lora[b_key].to(device, torch.float32) @ lora[a_key].to(device, torch.float32)
            for w, chunk in zip(weights, torch.chunk(delta, len(targets), dim=0)):
                w.data.copy_((w.data.float() + float(strength) * chunk).to(w.dtype))
                changed += 1
    return changed
