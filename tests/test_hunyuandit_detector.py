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

# Classification and dispatch fields whose values must not change: proving the
# architecture facts (base_arch / transformer_kind) are the ONLY behavior delta.
_SNAPSHOT_FIELDS = (
    "variant",
    "cross_attention_dim",
    "text_encoder_hidden_size",
    "text_encoder_2_hidden_size",
    "unet_in_channels",
    "is_lora",
    "format",
    "loader_format",
    "compatible_worker",
    "required_cross_attention_dim",
    "base_arch",
    "transformer_kind",
)


def _snapshot(info) -> dict:
    d = info.to_dict()
    return {k: d[k] for k in _SNAPSHOT_FIELDS}


def test_sdxl_diffusers_snapshot_delta_is_only_architecture_facts(tmp_path: Path):
    info = detect_model(str(_sdxl_diffusers_dir(tmp_path)))
    assert _snapshot(info) == {
        "variant": "sdxl-base",
        "cross_attention_dim": 2048,
        "text_encoder_hidden_size": 768,
        "text_encoder_2_hidden_size": 1280,
        "unet_in_channels": 4,
        "is_lora": False,
        "format": "diffusers",
        "loader_format": "diffusers_dir",
        "compatible_worker": "backends.cuda_worker.DiffusersSDXLCudaWorker",
        "required_cross_attention_dim": 2048,
        "base_arch": "unet",          # additive fact
        "transformer_kind": None,     # additive fact
    }


def test_sd15_diffusers_snapshot_delta_is_only_architecture_facts(tmp_path: Path):
    info = detect_model(str(_sd15_diffusers_dir(tmp_path)))
    assert _snapshot(info) == {
        "variant": "sd15",
        "cross_attention_dim": 768,
        "text_encoder_hidden_size": 768,
        "text_encoder_2_hidden_size": None,
        "unet_in_channels": 4,
        "is_lora": False,
        "format": "diffusers",
        "loader_format": "diffusers_dir",
        "compatible_worker": "backends.cuda_worker.DiffusersCudaWorker",
        "required_cross_attention_dim": 768,
        "base_arch": "unet",          # additive fact
        "transformer_kind": None,     # additive fact
    }


def test_sd_safetensors_snapshot_delta_is_only_architecture_facts(tmp_path: Path):
    path = tmp_path / "sd15.safetensors"
    save_file(
        {
            "model.diffusion_model.input_blocks.0.0.weight": np.zeros((320, 4, 3, 3), dtype=np.float32),
            # attn2.to_k is Linear(context_dim -> inner_dim), and torch stores
            # weights (out_features, in_features), so the LDM tensor is
            # (inner_dim, cross_attention_dim) and CAD is shape[1]. Matches the
            # fixture in tests/test_model_detector.py.
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight": np.zeros((1280, 768), dtype=np.float32),
        },
        str(path),
    )
    info = detect_model(str(path))
    assert _snapshot(info) == {
        "variant": "sd15",
        "cross_attention_dim": 768,
        "text_encoder_hidden_size": None,
        "text_encoder_2_hidden_size": None,
        "unet_in_channels": 4,
        "is_lora": False,
        "format": "safetensors",
        "loader_format": "single_file",
        "compatible_worker": "backends.cuda_worker.DiffusersCudaWorker",
        "required_cross_attention_dim": 768,
        "base_arch": "unet",          # additive fact
        "transformer_kind": None,     # additive fact
    }


def test_sd_checkpoint_snapshot_delta_is_only_architecture_facts(tmp_path: Path):
    path = tmp_path / "sd15.ckpt"
    torch.save(
        {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3)},
        str(path),
    )
    info = detect_model(str(path))
    assert _snapshot(info) == {
        "variant": "sd15",
        "cross_attention_dim": 768,
        "text_encoder_hidden_size": None,
        "text_encoder_2_hidden_size": None,
        "unet_in_channels": None,
        "is_lora": False,
        "format": "checkpoint",
        "loader_format": "single_file",
        "compatible_worker": "backends.cuda_worker.DiffusersCudaWorker",
        "required_cross_attention_dim": 768,
        "base_arch": "unet",          # additive fact
        "transformer_kind": None,     # additive fact
    }


# --- base_arch requires evidence; ambiguity stays unknown ------------------

def test_non_unet_safetensors_is_not_marked_unet(tmp_path: Path):
    # Only a text-encoder tensor, no UNet keys: architecture is not provable.
    path = tmp_path / "encoder_only.safetensors"
    save_file(
        {"text_model.encoder.layers.0.self_attn.k_proj.weight": np.zeros((768, 768), dtype=np.float32)},
        str(path),
    )
    info = detect_model(str(path))
    assert info.base_arch == "unknown"


def test_non_unet_checkpoint_is_not_marked_unet(tmp_path: Path):
    # No UNet/diffusion keys: must not be asserted as a UNet family.
    path = tmp_path / "misc.ckpt"
    torch.save({"some_projection.weight": torch.zeros(16, 16)}, str(path))
    info = detect_model(str(path))
    assert info.base_arch == "unknown"


def test_ambiguous_unet_and_transformer_stays_unknown(tmp_path: Path):
    # A directory declaring BOTH a UNet and a transformer is ambiguous; the
    # detector must not guess an architecture.
    root = _write_diffusers_dir(
        tmp_path / "ambiguous",
        model_index={
            "_class_name": "SomePipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "transformer": ["diffusers", "HunyuanDiT2DModel"],
            "text_encoder": ["transformers", "BertModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        },
        component_configs={
            "unet": {"cross_attention_dim": 2048, "in_channels": 4},
            "transformer": {"_class_name": "HunyuanDiT2DModel"},
            "text_encoder": {"hidden_size": 1024},
        },
    )
    info = detect_model(str(root))
    assert info.base_arch == "unknown"
    assert info.transformer_kind is None
    # Ambiguous architecture must not become a dispatchable SDXL via the UNet
    # cross_attention_dim: it stays UNKNOWN with no compatible worker.
    assert info.variant == ModelVariant.UNKNOWN
    assert info.compatible_worker is None


def test_hunyuandit_directory_has_no_compatible_worker(hunyuandit_dir: Path):
    # A transformer family is not dispatchable through the UNet worker path.
    info = detect_model(str(hunyuandit_dir))
    assert info.variant == ModelVariant.UNKNOWN
    assert info.compatible_worker is None
