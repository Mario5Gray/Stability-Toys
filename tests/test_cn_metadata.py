import json
import sys
import tomllib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cn_metadata import build_map_metadata, save_with_metadata, CHUNK_KEY, SCHEMA_VERSION


def test_build_map_metadata_common_and_params():
    payload = build_map_metadata(
        tool="canny_map",
        control_type="canny",
        source_size=(320, 240),
        params={"low_threshold": 50, "high_threshold": 150},
    )
    assert payload["tool"] == "canny_map"
    assert payload["version"] == SCHEMA_VERSION
    assert payload["control_type"] == "canny"
    assert payload["source_width"] == 320
    assert payload["source_height"] == 240
    assert isinstance(payload["created_at"], str) and payload["created_at"]
    assert payload["low_threshold"] == 50
    assert payload["high_threshold"] == 150


def test_save_with_metadata_roundtrip(tmp_path):
    img = Image.new("L", (16, 12), 200)
    dest = tmp_path / "map.png"
    payload = build_map_metadata(
        tool="depth_map",
        control_type="depth",
        source_size=(16, 12),
        params={"model": "depth-anything"},
    )
    save_with_metadata(img, dest, payload)

    out = Image.open(dest)
    assert out.format == "PNG"
    assert json.loads(out.text[CHUNK_KEY]) == payload


def test_pyproject_lists_cn_metadata_module():
    data = tomllib.loads((ROOT / "scripts" / "pyproject.toml").read_text())
    assert "cn_metadata" in data["tool"]["setuptools"]["py-modules"]
