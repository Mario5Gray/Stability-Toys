"""Phase 0 corrective detector tests (STABL-ichgkgno, Task 1).

The detector must not silently classify a HunyuanDiT Diffusers directory as SDXL.
HunyuanDiT has `transformer/` (no `unet/`) and two text encoders (BERT + T5); the
ungated dual-encoder fallback in VariantClassifier previously read those two
encoders as SDXL. These tests drive the *real* detector on metadata-only
fixtures and assert the corrective architecture facts plus an additive-only delta
for existing SD forms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from utils.model_detector import ModelVariant, detect_model

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "models" / "hunyuandit-v1.1-diffusers"


@pytest.fixture
def hunyuandit_dir() -> Path:
    return FIXTURE_DIR


def _write_diffusers_dir(root: Path, model_index: dict, component_configs: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps(model_index))
    for component, config in component_configs.items():
        comp_dir = root / component
        comp_dir.mkdir(parents=True, exist_ok=True)
        (comp_dir / "config.json").write_text(json.dumps(config))
    return root


def _sdxl_diffusers_dir(tmp_path: Path) -> Path:
    return _write_diffusers_dir(
        tmp_path / "sdxl",
        model_index={
            "_class_name": "StableDiffusionXLPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
            "vae": ["diffusers", "AutoencoderKL"],
        },
        component_configs={
            "unet": {"cross_attention_dim": 2048, "in_channels": 4, "out_channels": 4},
            "text_encoder": {"hidden_size": 768},
            "text_encoder_2": {"hidden_size": 1280},
        },
    )


def _sd15_diffusers_dir(tmp_path: Path) -> Path:
    return _write_diffusers_dir(
        tmp_path / "sd15",
        model_index={
            "_class_name": "StableDiffusionPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        },
        component_configs={
            "unet": {"cross_attention_dim": 768, "in_channels": 4, "out_channels": 4},
            "text_encoder": {"hidden_size": 768},
        },
    )


# --- Corrective Hunyuan classification -------------------------------------

def test_hunyuandit_directory_is_not_classified_as_sdxl(hunyuandit_dir: Path):
    info = detect_model(str(hunyuandit_dir))
    assert info.variant not in {ModelVariant.SDXL_BASE, ModelVariant.SDXL_REFINER}


def test_hunyuandit_directory_is_not_classified_as_sd2(hunyuandit_dir: Path):
    info = detect_model(str(hunyuandit_dir))
    assert info.variant not in {ModelVariant.SD20, ModelVariant.SD21}


def test_hunyuandit_transformer_cad_does_not_populate_unet_cad(hunyuandit_dir: Path):
    info = detect_model(str(hunyuandit_dir))
    assert info.cross_attention_dim is None


def test_hunyuandit_directory_reports_transformer_architecture_facts(hunyuandit_dir: Path):
    info = detect_model(str(hunyuandit_dir))
    assert info.base_arch == "transformer"
    assert info.transformer_kind == "hunyuandit"


# --- Additive-only delta for existing SD forms -----------------------------

def test_sdxl_diffusers_still_sdxl_with_unet_base_arch(tmp_path: Path):
    info = detect_model(str(_sdxl_diffusers_dir(tmp_path)))
    assert info.variant == ModelVariant.SDXL_BASE
    assert info.base_arch == "unet"
    assert info.transformer_kind is None


def test_sd15_diffusers_still_sd15_with_unet_base_arch(tmp_path: Path):
    info = detect_model(str(_sd15_diffusers_dir(tmp_path)))
    assert info.variant == ModelVariant.SD15
    assert info.base_arch == "unet"
    assert info.transformer_kind is None


def test_sd_safetensors_reports_unet_base_arch(tmp_path: Path):
    path = tmp_path / "sd_model.safetensors"
    save_file(
        {"conv_in.weight": np.zeros((320, 4, 3, 3), dtype=np.float32)},
        str(path),
    )
    info = detect_model(str(path))
    assert info.base_arch == "unet"


def test_sd_checkpoint_reports_unet_base_arch(tmp_path: Path):
    path = tmp_path / "sd_model.ckpt"
    torch.save({"conv_in.weight": torch.zeros(320, 4, 3, 3)}, str(path))
    info = detect_model(str(path))
    assert info.base_arch == "unet"
