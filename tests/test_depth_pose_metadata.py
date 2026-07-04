import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import depth_map
import pose_map


def _fixture(path: Path, size=(32, 24)):
    Image.new("RGB", size, "gray").save(path)


def test_depth_map_embeds_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        depth_map, "depth_anything",
        lambda img, size, device: Image.new("L", img.size, 128),
    )
    src = tmp_path / "src.png"
    dest = tmp_path / "depth.png"
    _fixture(src)
    monkeypatch.setattr(
        sys, "argv",
        ["depth_map", str(src), str(dest), "--model", "depth-anything", "--size", "small"],
    )
    depth_map.main()

    meta = json.loads(Image.open(dest).text["controlnet_map"])
    assert meta["tool"] == "depth_map"
    assert meta["control_type"] == "depth"
    assert meta["model"] == "depth-anything"
    assert meta["size"] == "small"
    assert meta["device"] == "cpu"
    assert (meta["source_width"], meta["source_height"]) == (32, 24)


def test_pose_map_embeds_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pose_map, "dwpose",
        lambda img: Image.new("RGB", img.size, "black"),
    )
    src = tmp_path / "src.png"
    dest = tmp_path / "pose.png"
    _fixture(src)
    monkeypatch.setattr(sys, "argv", ["pose_map", str(src), str(dest)])
    pose_map.main()

    meta = json.loads(Image.open(dest).text["controlnet_map"])
    assert meta["tool"] == "pose_map"
    assert meta["control_type"] == "pose"
    assert meta["model"] == "dwpose"
    assert meta["parts"] == ["body", "face", "hands"]
    assert (meta["source_width"], meta["source_height"]) == (32, 24)
