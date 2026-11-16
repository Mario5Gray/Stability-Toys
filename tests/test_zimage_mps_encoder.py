"""Native MPS proof for the int8 Z-Image encoder store (STABL-mbhlhinl).

These tests use real Transformers, Quanto and MPS. They run only with
RUN_ZIMAGE_MPS_NATIVE=1. A skip is not native acceptance.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModel, Qwen3Config

from backends.zimage_text_encoder import load_int8_text_encoder

REPO_ROOT = Path(__file__).resolve().parents[1]
IDS = [[1, 2, 3, 4]]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ZIMAGE_MPS_NATIVE") != "1",
    reason="explicit native MPS test",
)


def _tiny_model(root: Path) -> Path:
    """Write tiny local Qwen3 weights. No remote model identifier is used."""

    torch.manual_seed(0)
    encoder_dir = root / "text_encoder"
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        dtype="bfloat16",
    )
    encoder = AutoModel.from_config(config, dtype=torch.bfloat16)
    encoder.save_pretrained(encoder_dir, safe_serialization=True)
    return root


@pytest.fixture
def tiny_model(tmp_path):
    return _tiny_model(tmp_path / "model")


def _output(encoder) -> torch.Tensor:
    ids = torch.tensor(IDS, device="mps")
    with torch.inference_mode():
        return encoder(ids).last_hidden_state.detach().cpu()


def test_mps_encoder_filesystem_hit(tiny_model, tmp_path):
    assert torch.backends.mps.is_available(), "native MPS unavailable"
    store = tmp_path / "encoder-store"
    fresh = load_int8_text_encoder(
        str(tiny_model), device="mps", dtype=torch.bfloat16, store_root=store,
    )
    expected = _output(fresh)
    del fresh
    torch.mps.empty_cache()
    with patch(
        "backends.zimage_text_encoder._build_entry",
        side_effect=AssertionError("store hit rebuilt encoder"),
    ):
        loaded = load_int8_text_encoder(
            str(tiny_model), device="mps", dtype=torch.bfloat16, store_root=store,
        )
    actual = _output(loaded)
    assert actual.dtype is torch.bfloat16
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


_BUILD_SCRIPT = """
import sys

import torch
from safetensors.torch import save_file

from backends.zimage_text_encoder import load_int8_text_encoder

model_root, store_root, out_path = sys.argv[1:4]
encoder = load_int8_text_encoder(
    model_root, device="mps", dtype=torch.bfloat16, store_root=store_root,
)
with torch.inference_mode():
    ids = torch.tensor([[1, 2, 3, 4]], device="mps")
    out = encoder(ids).last_hidden_state.detach().cpu()
save_file({"out": out.contiguous()}, out_path)
print("build", out.dtype, tuple(out.shape))
"""

_HIT_SCRIPT = """
import sys

import torch
from safetensors.torch import load_file

import backends.zimage_text_encoder as store_module

model_root, store_root, expected_path = sys.argv[1:4]


def refuse(*args, **kwargs):
    raise AssertionError("restart reload rebuilt the encoder")


store_module._build_entry = refuse
encoder = store_module.load_int8_text_encoder(
    model_root, device="mps", dtype=torch.bfloat16, store_root=store_root,
)
with torch.inference_mode():
    ids = torch.tensor([[1, 2, 3, 4]], device="mps")
    out = encoder(ids).last_hidden_state.detach().cpu()
expected = load_file(expected_path)["out"]
torch.testing.assert_close(out, expected, rtol=0, atol=0)
print("hit", out.dtype, tuple(out.shape))
"""


def _run_script(script: str, *args: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script, *map(str, args)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_restart_hit_after_process_rebuild(tiny_model, tmp_path):
    assert torch.backends.mps.is_available(), "native MPS unavailable"
    store = tmp_path / "encoder-store"
    expected = tmp_path / "expected.safetensors"
    build = _run_script(_BUILD_SCRIPT, tiny_model, store, expected)
    (tmp_path / "build.stdout").write_text(build.stdout)
    (tmp_path / "build.stderr").write_text(build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    assert expected.is_file()
    hit = _run_script(_HIT_SCRIPT, tiny_model, store, expected)
    (tmp_path / "hit.stdout").write_text(hit.stdout)
    (tmp_path / "hit.stderr").write_text(hit.stderr)
    assert hit.returncode == 0, hit.stdout + hit.stderr
    assert hit.stdout.startswith("hit")
