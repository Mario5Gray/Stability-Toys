import json
import pytest
from utils.model_detector import ModelInfo, detect_model
from backends.family_profiles import (
    FAMILY_REGISTRY, ZIMAGE_PROFILE, family_ids, resolve_family,
)

def _zimage_dir(tmp_path, transformer_class="ZImageTransformer2DModel"):
    root = tmp_path / "Z-Image-Turbo-BF16"
    (root / "transformer").mkdir(parents=True)
    (root / "text_encoder").mkdir()
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "ZImagePipeline",
        "transformer": ["diffusers", transformer_class],
        "text_encoder": ["transformers", "Qwen3Model"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": transformer_class, "dim": 3840}))
    (root / "text_encoder" / "config.json").write_text(json.dumps({"hidden_size": 2560}))
    return root

def test_detector_marks_zimage_folder(tmp_path):
    info = detect_model(str(_zimage_dir(tmp_path)))
    assert info.base_arch == "transformer"
    assert info.transformer_kind == "zimage"
    assert info.loader_format == "diffusers_dir"
    assert info.cross_attention_dim is None

def test_unknown_transformer_class_is_not_zimage(tmp_path):
    info = detect_model(str(_zimage_dir(tmp_path, "SomeOtherTransformer")))
    assert info.transformer_kind is None

def test_profile_values():
    assert ZIMAGE_PROFILE.family_id == "zimage"
    assert ZIMAGE_PROFILE.encoder_roles == ("text_encoder",)
    assert ZIMAGE_PROFILE.pooled_required is False
    assert ZIMAGE_PROFILE.pooled_projection_role is None
    assert ZIMAGE_PROFILE.control_image_kwarg == "control_image"

def test_resolve_family_selects_exactly_zimage(tmp_path):
    info = detect_model(str(_zimage_dir(tmp_path)))
    assert resolve_family(info) is ZIMAGE_PROFILE
    assert "zimage" in family_ids()

def test_zimage_predicate_rejects_other_families():
    detect = next(r.detect for r in FAMILY_REGISTRY if r.profile is ZIMAGE_PROFILE)
    assert not detect(ModelInfo(path="x", base_arch="transformer", transformer_kind="hunyuandit"))
    assert not detect(ModelInfo(path="x", base_arch="unet", cross_attention_dim=2048))
