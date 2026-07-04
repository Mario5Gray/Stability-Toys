import io
import json

from PIL import Image, PngImagePlugin

from server.controlnet_metadata import read_control_map_metadata


def _png_with_chunk(value: str) -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("controlnet_map", value)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _png_no_chunk() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_valid_chunk_returns_dict():
    payload = {"tool": "canny_map", "control_type": "canny", "version": 1}
    out = read_control_map_metadata(_png_with_chunk(json.dumps(payload)))
    assert out == payload


def test_absent_chunk_returns_none():
    assert read_control_map_metadata(_png_no_chunk()) is None


def test_malformed_json_returns_none():
    assert read_control_map_metadata(_png_with_chunk("{not valid json")) is None


def test_non_dict_json_returns_none():
    assert read_control_map_metadata(_png_with_chunk("[1, 2, 3]")) is None


def test_non_png_bytes_returns_none():
    assert read_control_map_metadata(b"this is not a png") is None
