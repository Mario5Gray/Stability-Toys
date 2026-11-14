"""DiffusersHunyuanDiTCudaWorker.configure_loras (STABL-ykowysrg)."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
import torch

from backends.cuda_worker import DiffusersHunyuanDiTCudaWorker
from backends.hunyuan_mps_worker import DiffusersHunyuanDiTMpsWorker
from backends.hunyuandit_lora import HunyuanDiTLoraError, merge_tencent_lora
from backends.mode_loras import LoraSpec
from tests.hunyuandit_lora_fakes import (
    clone_state,
    make_fake_lora,
    make_fake_transformer,
    save_lora,
)


def _worker():
    w = DiffusersHunyuanDiTCudaWorker.__new__(DiffusersHunyuanDiTCudaWorker)
    w.pipe = SimpleNamespace(transformer=make_fake_transformer())
    w._quantize = "none"
    w._checkpoint_precision = "fp16"
    w.worker_id = 0
    return w


def _lora_dir(tmp_path, name, transformer, seed=0, config=None):
    d = tmp_path / name
    d.mkdir()
    save_lora(d / "adapter_model.safetensors", make_fake_lora(transformer, seed=seed))
    if config is not None:
        (d / "adapter_config.json").write_text(json.dumps(config))
    return d


def _assert_state_equal(transformer, expected):
    for k, v in transformer.state_dict().items():
        assert torch.equal(v, expected[k]), k


def test_configure_loras_merges_two_specs_additively(tmp_path):
    w = _worker()
    t = w.pipe.transformer
    a = _lora_dir(tmp_path, "a", t, seed=1)
    b = _lora_dir(tmp_path, "b", t, seed=2)

    ref = make_fake_transformer()  # same seed -> same base weights
    merge_tencent_lora(ref, make_fake_lora(ref, seed=1, prefix=False), 1.0)
    merge_tencent_lora(ref, make_fake_lora(ref, seed=2, prefix=False), 0.5)

    w.configure_loras([
        LoraSpec(str(a / "adapter_model.safetensors"), 1.0),
        LoraSpec(str(b / "adapter_model.safetensors"), 0.5),
    ])
    expected = ref.state_dict()
    for k, v in t.state_dict().items():
        assert torch.allclose(v, expected[k], atol=1e-6), k


def test_configure_loras_accepts_directory_path(tmp_path):
    w = _worker()
    before = clone_state(w.pipe.transformer)
    w.configure_loras([LoraSpec(str(_lora_dir(tmp_path, "a", w.pipe.transformer)), 1.0)])
    after = w.pipe.transformer.state_dict()
    assert not torch.equal(after["blocks.0.attn1.to_q.weight"], before["blocks.0.attn1.to_q.weight"])


def test_quantized_transformer_refuses_before_read(tmp_path):
    w = _worker()
    w._quantize = "fp8"
    with pytest.raises(HunyuanDiTLoraError, match="quantized"):
        w.configure_loras([LoraSpec(str(tmp_path / "missing.safetensors"), 1.0)])


def test_checkpoint_fp8_refuses(tmp_path):
    w = _worker()
    w._checkpoint_precision = "fp8"
    with pytest.raises(HunyuanDiTLoraError, match="quantized"):
        w.configure_loras([LoraSpec(str(_lora_dir(tmp_path, "a", w.pipe.transformer)), 1.0)])


def test_non_finite_strength_refuses(tmp_path):
    w = _worker()
    before = clone_state(w.pipe.transformer)
    d = _lora_dir(tmp_path, "a", w.pipe.transformer)
    with pytest.raises(HunyuanDiTLoraError, match="strength"):
        w.configure_loras([LoraSpec(str(d), float("nan"))])
    _assert_state_equal(w.pipe.transformer, before)


def test_second_bad_lora_changes_no_weight(tmp_path):
    w = _worker()
    t = w.pipe.transformer
    good = _lora_dir(tmp_path, "good", t)
    bad_lora = make_fake_lora(t)
    del bad_lora["base_model.model.pooler.q_proj.lora_A.weight"]
    bad = tmp_path / "bad.safetensors"
    save_lora(bad, bad_lora)
    before = clone_state(t)
    with pytest.raises(HunyuanDiTLoraError, match="pooler.q_proj"):
        w.configure_loras([LoraSpec(str(good), 1.0), LoraSpec(str(bad), 1.0)])
    _assert_state_equal(t, before)


def test_configure_loras_logs_rank_and_alpha(tmp_path, caplog):
    w = _worker()
    d = _lora_dir(tmp_path, "a", w.pipe.transformer, config={"r": 2, "lora_alpha": 2})
    with caplog.at_level(logging.INFO, logger="backends.cuda_worker"):
        w.configure_loras([LoraSpec(str(d), 0.8)])
    assert "strength=0.8 r=2 alpha=2 targets=17" in caplog.text


def test_mps_worker_inherits_configure_loras():
    assert DiffusersHunyuanDiTMpsWorker.configure_loras is DiffusersHunyuanDiTCudaWorker.configure_loras


def test_controlnet_pipe_reuses_base_pipe(monkeypatch):
    seen = {}

    class _FakeCnPipe:
        @classmethod
        def from_pipe(cls, pipe, **kw):
            seen["pipe"] = pipe
            return cls()

    monkeypatch.setattr("backends.cuda_worker._hunyuandit_controlnet_pipeline_cls", lambda: _FakeCnPipe)
    w = _worker()
    w._build_controlnet_pipe(object())
    assert seen["pipe"] is w.pipe
