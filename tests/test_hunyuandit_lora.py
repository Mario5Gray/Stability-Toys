"""Tencent HunyuanDiT LoRA validate + merge (STABL-ykowysrg)."""
from __future__ import annotations

import json

import pytest
import torch

from backends.hunyuandit_lora import (
    TENCENT_PREFIX,
    HunyuanDiTLoraError,
    merge_tencent_lora,
    read_adapter_config,
    read_tencent_lora,
    resolve_lora_file,
    validate_tencent_lora,
)
from tests.hunyuandit_lora_fakes import (
    clone_state,
    make_fake_lora,
    make_fake_transformer,
    save_lora,
    tencent_reference_merge,
)


def _assert_state_equal(transformer, expected):
    for k, v in transformer.state_dict().items():
        assert torch.equal(v, expected[k]), k


def test_resolve_lora_file_accepts_dir_and_file(tmp_path):
    f = tmp_path / "adapter_model.safetensors"
    f.write_bytes(b"x")
    assert resolve_lora_file(tmp_path) == f
    assert resolve_lora_file(str(f)) == f


def test_resolve_lora_file_missing_raises(tmp_path):
    with pytest.raises(HunyuanDiTLoraError):
        resolve_lora_file(tmp_path / "nope.safetensors")
    with pytest.raises(HunyuanDiTLoraError):
        resolve_lora_file(tmp_path)  # dir without adapter_model.safetensors


def test_read_strips_tencent_prefix(tmp_path):
    t = make_fake_transformer()
    lora = make_fake_lora(t, prefix=True)
    path = save_lora(tmp_path / "adapter_model.safetensors", lora)
    got = read_tencent_lora(path)
    assert len(got) == len(lora)
    assert all(k.startswith(("blocks.", "pooler.")) for k in got)
    assert TENCENT_PREFIX == "base_model.model."


def test_read_adapter_config_optional(tmp_path):
    f = tmp_path / "adapter_model.safetensors"
    assert read_adapter_config(f) == {}
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 64}))
    assert read_adapter_config(f) == {"r": 64, "lora_alpha": 64}


def test_merge_matches_tencent_reference():
    t = make_fake_transformer()
    lora = make_fake_lora(t, prefix=False)
    expected = tencent_reference_merge(clone_state(t), lora, 1.0, num_layers=2)
    merge_tencent_lora(t, lora, 1.0)
    for k, v in t.state_dict().items():
        assert torch.allclose(v, expected[k], atol=1e-6), k


def test_strength_scales_delta():
    lora = make_fake_lora(make_fake_transformer(), prefix=False)
    t_full = make_fake_transformer()
    t_half = make_fake_transformer()
    before = clone_state(t_full)
    merge_tencent_lora(t_full, lora, 1.0)
    merge_tencent_lora(t_half, lora, 0.5)
    full, half = t_full.state_dict(), t_half.state_dict()
    for k in before:
        assert torch.allclose(half[k] - before[k], 0.5 * (full[k] - before[k]), atol=1e-6), k


def test_merge_returns_target_count():
    t = make_fake_transformer()
    assert merge_tencent_lora(t, make_fake_lora(t, prefix=False), 1.0) == 8 * 2 + 1


def test_fp16_weight_merge_computes_in_fp32():
    t = make_fake_transformer(dtype=torch.float16)
    lora = make_fake_lora(t, prefix=False)
    w = t.blocks[0].attn1.to_out[0].weight.detach().clone()
    a = lora["blocks.0.attn1.out_proj.lora_A.weight"]
    b = lora["blocks.0.attn1.out_proj.lora_B.weight"]
    expected = (w.float() + b.float() @ a.float()).half()
    merge_tencent_lora(t, lora, 1.0)
    assert torch.equal(t.blocks[0].attn1.to_out[0].weight, expected)


def test_validate_rejects_missing_key():
    t = make_fake_transformer()
    lora = make_fake_lora(t, prefix=False)
    del lora["blocks.1.attn2.kv_proj.lora_B.weight"]
    before = clone_state(t)
    with pytest.raises(HunyuanDiTLoraError, match="blocks.1.attn2.kv_proj"):
        merge_tencent_lora(t, lora, 1.0)
    _assert_state_equal(t, before)


def test_validate_rejects_extra_key():
    t = make_fake_transformer()
    lora = make_fake_lora(t, prefix=False)
    lora["blocks.0.mlp.fc1.lora_A.weight"] = torch.zeros(2, 8)
    with pytest.raises(HunyuanDiTLoraError, match="blocks.0.mlp.fc1"):
        validate_tencent_lora(t, lora)


def test_validate_rejects_wrong_shape():
    t = make_fake_transformer()
    lora = make_fake_lora(t, prefix=False)
    lora["blocks.0.attn1.out_proj.lora_A.weight"] = torch.zeros(3, 8)  # rank 3 vs B rank 2
    before = clone_state(t)
    with pytest.raises(HunyuanDiTLoraError, match="blocks.0.attn1.out_proj"):
        merge_tencent_lora(t, lora, 1.0)
    _assert_state_equal(t, before)


def test_validate_rejects_sdxl_style_keys():
    t = make_fake_transformer()
    lora = {"lora_unet_down_blocks_0_attentions_0.lora_down.weight": torch.zeros(2, 8)}
    with pytest.raises(HunyuanDiTLoraError, match=r"blocks\.0\.attn1\.Wqkv"):
        validate_tencent_lora(t, lora)


def test_merge_refuses_meta_weights():
    """Sequential CPU offload leaves module weights on meta. A merge there would
    change nothing and the mode would load without its LoRA (STABL-ykowysrg)."""
    t = make_fake_transformer()
    lora = make_fake_lora(t, prefix=False)
    t.blocks[1].attn2.to_v.to("meta")
    before = t.blocks[0].attn1.to_q.weight.detach().clone()
    with pytest.raises(HunyuanDiTLoraError, match="meta"):
        merge_tencent_lora(t, lora, 1.0)
    assert torch.equal(t.blocks[0].attn1.to_q.weight, before)  # no partial write
