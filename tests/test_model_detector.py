import torch
from safetensors.torch import save_file

from utils.model_detector import ModelVariant, detect_model


def test_detect_model_uses_input_axis_for_ldm_attn2_cross_attention_dim(tmp_path):
    path = tmp_path / "mxcp_like.safetensors"
    save_file(
        {
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight": torch.zeros((1280, 768), dtype=torch.float16),
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight": torch.zeros((77, 768), dtype=torch.float16),
        },
        str(path),
    )

    info = detect_model(str(path))

    assert info.cross_attention_dim == 768
    assert info.variant == ModelVariant.SD15
