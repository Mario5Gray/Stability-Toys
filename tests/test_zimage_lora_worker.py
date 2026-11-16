"""Z-Image mode LoRA load and fuse (STABL-gftujmyi).

The tests use a tiny real ZImageTransformer2DModel in a ZImagePipeline with
no scheduler, VAE, or text encoder. The LoRA files use ComfyUI key names.
"""
from __future__ import annotations

import logging
from unittest import mock

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")

from diffusers import ZImageImg2ImgPipeline, ZImagePipeline, ZImageTransformer2DModel  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from backends.mode_loras import LoraSpec  # noqa: E402
from backends.zimage_cuda_worker import (  # noqa: E402
    DiffusersZImageCudaWorker,
    ZImageLoraError,
)
from backends.zimage_mps_worker import DiffusersZImageMpsWorker  # noqa: E402

TARGET = "layers.0.attention.to_q"


def _tiny_transformer() -> ZImageTransformer2DModel:
    torch.manual_seed(0)
    return ZImageTransformer2DModel(
        dim=64,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=32,
        axes_dims=[8, 12, 12],
        axes_lens=[64, 32, 32],
    )


@pytest.fixture(params=[DiffusersZImageCudaWorker, DiffusersZImageMpsWorker])
def worker_type(request):
    return request.param


def _worker(worker_type=DiffusersZImageCudaWorker):
    w = worker_type.__new__(worker_type)
    w.pipe = ZImagePipeline(
        scheduler=None, vae=None, text_encoder=None, tokenizer=None, transformer=_tiny_transformer()
    )
    w.worker_id = 0
    w._img2img_pipe = None
    return w


def _module(worker, name: str) -> torch.nn.Linear:
    return worker.pipe.transformer.get_submodule(name)


def _comfy_lora(worker, module_names, rank=4, seed=0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for name in module_names:
        lin = _module(worker, name)
        out[f"diffusion_model.{name}.lora_A.weight"] = torch.randn(rank, lin.in_features, generator=g) * 0.1
        out[f"diffusion_model.{name}.lora_B.weight"] = torch.randn(lin.out_features, rank, generator=g) * 0.1
    return out


def _save(path, lora) -> str:
    save_file({k: v.contiguous() for k, v in lora.items()}, str(path))
    return str(path)


def _delta(lora, name) -> torch.Tensor:
    a = lora[f"diffusion_model.{name}.lora_A.weight"].float()
    b = lora[f"diffusion_model.{name}.lora_B.weight"].float()
    return b @ a


def _clone(worker) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in worker.pipe.transformer.state_dict().items()}


def test_fused_weight_matches_lora_math(tmp_path, worker_type):
    w = _worker(worker_type)
    base = _module(w, TARGET).weight.detach().clone()
    lora = _comfy_lora(w, [TARGET])
    path = _save(tmp_path / "a.safetensors", lora)

    w.configure_loras([LoraSpec(path, 0.7)])

    expected = base.float() + 0.7 * _delta(lora, TARGET)
    torch.testing.assert_close(_module(w, TARGET).weight.float(), expected, atol=1e-5, rtol=0)


def test_two_loras_add(tmp_path, worker_type):
    w = _worker(worker_type)
    base = _module(w, TARGET).weight.detach().clone()
    l1 = _comfy_lora(w, [TARGET], seed=1)
    l2 = _comfy_lora(w, [TARGET], seed=2)
    p1 = _save(tmp_path / "one.safetensors", l1)
    p2 = _save(tmp_path / "two.safetensors", l2)

    w.configure_loras([LoraSpec(p1, 1.0), LoraSpec(p2, 0.5)])

    expected = base.float() + _delta(l1, TARGET) + 0.5 * _delta(l2, TARGET)
    torch.testing.assert_close(_module(w, TARGET).weight.float(), expected, atol=1e-5, rtol=0)


def test_no_adapter_layers_remain(tmp_path, worker_type):
    from peft.tuners.tuners_utils import BaseTunerLayer

    w = _worker(worker_type)
    path = _save(tmp_path / "a.safetensors", _comfy_lora(w, [TARGET, "layers.1.attention.to_k"]))

    w.configure_loras([LoraSpec(path, 1.0)])

    tuners = [n for n, m in w.pipe.transformer.named_modules() if isinstance(m, BaseTunerLayer)]
    assert tuners == []


def test_zero_strength_leaves_weights(tmp_path, worker_type):
    w = _worker(worker_type)
    before = _clone(w)
    path = _save(tmp_path / "a.safetensors", _comfy_lora(w, [TARGET]))

    w.configure_loras([LoraSpec(path, 0.0)])

    after = w.pipe.transformer.state_dict()
    assert set(after) == set(before)
    for k, v in before.items():
        torch.testing.assert_close(after[k], v, atol=0, rtol=0)


def test_foreign_lora_refuses(tmp_path, worker_type):
    w = _worker(worker_type)
    g = torch.Generator().manual_seed(0)
    sdxl = {
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": torch.randn(4, 320, generator=g),
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": torch.randn(320, 4, generator=g),
    }
    path = _save(tmp_path / "sdxl.safetensors", sdxl)

    with pytest.raises(ZImageLoraError, match="sdxl.safetensors"):
        w.configure_loras([LoraSpec(path, 1.0)])


def test_non_finite_strength_refuses_before_load(tmp_path, worker_type):
    w = _worker(worker_type)
    missing = str(tmp_path / "does-not-exist.safetensors")

    with pytest.raises(ZImageLoraError, match="strength"):
        w.configure_loras([LoraSpec(missing, float("inf"))])


def test_directory_with_one_file_loads(tmp_path, worker_type):
    w = _worker(worker_type)
    base = _module(w, TARGET).weight.detach().clone()
    d = tmp_path / "lora_dir"
    d.mkdir()
    _save(d / "only.safetensors", _comfy_lora(w, [TARGET]))

    w.configure_loras([LoraSpec(str(d), 1.0)])

    assert not torch.equal(_module(w, TARGET).weight, base)


def test_directory_with_two_files_refuses(tmp_path, worker_type):
    w = _worker(worker_type)
    d = tmp_path / "lora_dir"
    d.mkdir()
    _save(d / "a.safetensors", _comfy_lora(w, [TARGET], seed=1))
    _save(d / "b.safetensors", _comfy_lora(w, [TARGET], seed=2))

    with pytest.raises(ZImageLoraError):
        w.configure_loras([LoraSpec(str(d), 1.0)])


def test_logs_targeted_module_count(tmp_path, caplog, worker_type):
    w = _worker(worker_type)
    path = _save(tmp_path / "a.safetensors", _comfy_lora(w, [TARGET]))

    with caplog.at_level(logging.INFO, logger="backends.zimage_cuda_worker"):
        w.configure_loras([LoraSpec(path, 0.7)])

    assert "strength=0.7 modules=1" in caplog.text
    assert f"[{worker_type.log_tag}]" in caplog.text


def test_img2img_pipe_shares_fused_transformer(tmp_path, worker_type):
    w = _worker(worker_type)
    base = _module(w, TARGET).weight.detach().clone()
    path = _save(tmp_path / "a.safetensors", _comfy_lora(w, [TARGET]))

    w.configure_loras([LoraSpec(path, 1.0)])
    with mock.patch(
        "backends.zimage_cuda_worker._zimage_img2img_pipeline_cls", return_value=ZImageImg2ImgPipeline
    ):
        img2img = w._ensure_img2img_pipe()

    assert img2img.transformer is w.pipe.transformer
    assert not torch.equal(img2img.transformer.get_submodule(TARGET).weight, base)


def test_partial_fit_lora_refuses(tmp_path, worker_type):
    w = _worker(worker_type)
    base = _module(w, TARGET).weight.detach().clone()
    lora = _comfy_lora(w, [TARGET])
    lora["diffusion_model.not_a_module.lora_A.weight"] = torch.randn(4, 64)
    lora["diffusion_model.not_a_module.lora_B.weight"] = torch.randn(64, 4)
    path = _save(tmp_path / "partial.safetensors", lora)

    with pytest.raises(ZImageLoraError, match="partial.safetensors"):
        w.configure_loras([LoraSpec(path, 1.0)])

    torch.testing.assert_close(_module(w, TARGET).weight, base, atol=0, rtol=0)
