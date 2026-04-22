import io

import numpy as np
import pytest
import transformers
import transformers.pipelines
from PIL import Image

from server.controlnet_preprocessors import (
    CannyPreprocessor,
    ControlMapPreprocessor,
    ControlMapResult,
    DepthPreprocessor,
    PreprocessorRegistry,
    pil_to_png_bytes,
    png_bytes_to_pil,
)


class _FakePreprocessor:
    preprocessor_id = "fake"
    control_type = "fake"

    def run(self, image_bytes: bytes, options: dict) -> ControlMapResult:
        return ControlMapResult(
            preprocessor_id="fake",
            control_type="fake",
            image_bytes=b"output",
            width=8,
            height=8,
        )


def _solid_rgb_png_bytes(
    size: tuple[int, int] = (4, 4),
    color: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    image = Image.new("RGB", size, color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_fake_preprocessor_satisfies_protocol():
    assert isinstance(_FakePreprocessor(), ControlMapPreprocessor)


def test_registry_get_returns_none_for_unknown():
    reg = PreprocessorRegistry()
    assert reg.get("no-such") is None


def test_registry_dispatch_registered_preprocessor():
    reg = PreprocessorRegistry()
    reg.register(_FakePreprocessor())
    result = reg.dispatch("fake", b"input", {})
    assert result.preprocessor_id == "fake"
    assert result.image_bytes == b"output"
    assert result.width == 8
    assert result.height == 8
    assert result.media_type == "image/png"


def test_registry_dispatch_unknown_raises():
    reg = PreprocessorRegistry()
    with pytest.raises(ValueError, match="unknown preprocessor"):
        reg.dispatch("missing", b"x", {})


def test_registry_register_rejects_duplicate_preprocessor_id():
    reg = PreprocessorRegistry()
    reg.register(_FakePreprocessor())

    with pytest.raises(ValueError, match="already registered"):
        reg.register(_FakePreprocessor())


def test_registry_register_can_explicitly_replace_existing_preprocessor():
    class _ReplacementPreprocessor:
        preprocessor_id = "fake"
        control_type = "fake"

        def run(self, image_bytes: bytes, options: dict) -> ControlMapResult:
            return ControlMapResult(
                preprocessor_id="fake",
                control_type="fake",
                image_bytes=b"replacement",
                width=9,
                height=9,
            )

    reg = PreprocessorRegistry()
    reg.register(_FakePreprocessor())
    reg.register(_ReplacementPreprocessor(), replace=True)

    result = reg.dispatch("fake", b"input", {})
    assert result.image_bytes == b"replacement"
    assert result.width == 9
    assert result.height == 9


def test_control_map_result_defaults():
    result = ControlMapResult(
        preprocessor_id="canny",
        control_type="canny",
        image_bytes=b"data",
        width=64,
        height=64,
    )
    assert result.media_type == "image/png"


def test_pil_to_png_bytes_round_trips_through_png_bytes_to_pil():
    source = Image.new("RGB", (3, 2), color=(12, 34, 56))

    encoded = pil_to_png_bytes(source)
    decoded = png_bytes_to_pil(encoded)

    assert decoded.size == (3, 2)
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (12, 34, 56)


def test_png_bytes_to_pil_raises_value_error_for_invalid_bytes():
    with pytest.raises(ValueError, match="failed to decode image bytes"):
        png_bytes_to_pil(b"not-an-image")


def test_canny_preprocessor_returns_png_control_map():
    preprocessor = CannyPreprocessor()

    result = preprocessor.run(
        _solid_rgb_png_bytes(),
        {"low_threshold": "10", "high_threshold": "20"},
    )

    decoded = png_bytes_to_pil(result.image_bytes)
    assert result.preprocessor_id == "canny"
    assert result.control_type == "canny"
    assert result.width == 4
    assert result.height == 4
    assert result.media_type == "image/png"
    assert decoded.size == (4, 4)


def test_canny_preprocessor_raises_for_invalid_bytes():
    preprocessor = CannyPreprocessor()

    with pytest.raises(ValueError, match="could not decode source image"):
        preprocessor.run(b"not-an-image", {})


def test_depth_preprocessor_uses_stubbed_pipeline_and_normalizes_output():
    class _StubDepthPreprocessor(DepthPreprocessor):
        def __init__(self) -> None:
            super().__init__(model_id="stub-model")
            self.pipe_calls = 0

        def _get_pipe(self):
            def _pipe(pil_img):
                self.pipe_calls += 1
                return {
                    "depth": Image.fromarray(
                        np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
                    )
                }

            return _pipe

    preprocessor = _StubDepthPreprocessor()

    result = preprocessor.run(_solid_rgb_png_bytes(size=(2, 2)), {})

    decoded = png_bytes_to_pil(result.image_bytes)
    assert preprocessor.pipe_calls == 1
    assert result.preprocessor_id == "depth"
    assert result.control_type == "depth"
    assert result.width == 2
    assert result.height == 2
    assert result.media_type == "image/png"
    assert decoded.size == (2, 2)
    assert decoded.getpixel((0, 0)) == (0, 0, 0)
    assert decoded.getpixel((1, 1)) == (255, 255, 255)


def test_depth_preprocessor_emits_rgb_png_control_map():
    class _StubDepthPreprocessor(DepthPreprocessor):
        def _get_pipe(self):
            def _pipe(pil_img):
                return {
                    "depth": Image.fromarray(
                        np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
                    )
                }

            return _pipe

    preprocessor = _StubDepthPreprocessor(model_id="stub-model")

    result = preprocessor.run(_solid_rgb_png_bytes(size=(2, 2)), {})

    decoded = Image.open(io.BytesIO(result.image_bytes))
    assert decoded.size == (2, 2)
    assert decoded.mode == "RGB"


def test_depth_preprocessor_get_pipe_caches_pipeline_instance(monkeypatch):
    created = []

    def _fake_pipeline(task, model):
        assert task == "depth-estimation"
        assert model == "stub-model"
        pipe = object()
        created.append(pipe)
        return pipe

    monkeypatch.setattr(transformers, "pipeline", _fake_pipeline)
    monkeypatch.setattr(transformers.pipelines, "pipeline", _fake_pipeline)

    preprocessor = DepthPreprocessor(model_id="stub-model")

    first = preprocessor._get_pipe()
    second = preprocessor._get_pipe()

    assert first is second
    assert created == [first]


def test_depth_preprocessor_rejects_invalid_source_bytes_before_pipe_call():
    class _StubDepthPreprocessor(DepthPreprocessor):
        def __init__(self) -> None:
            super().__init__(model_id="stub-model")
            self.pipe_calls = 0

        def _get_pipe(self):
            def _pipe(pil_img):
                self.pipe_calls += 1
                return {"depth": Image.new("L", pil_img.size, color=1)}

            return _pipe

    preprocessor = _StubDepthPreprocessor()

    with pytest.raises(ValueError, match="failed to decode image bytes"):
        preprocessor.run(b"not-an-image", {})

    assert preprocessor.pipe_calls == 0


def test_depth_preprocessor_normalizes_flat_depth_map_to_black():
    class _FlatDepthPreprocessor(DepthPreprocessor):
        def _get_pipe(self):
            def _pipe(pil_img):
                return {
                    "depth": Image.fromarray(
                        np.full((pil_img.height, pil_img.width), 7.0, dtype=np.float32)
                    )
                }

            return _pipe

    preprocessor = _FlatDepthPreprocessor(model_id="stub-model")

    result = preprocessor.run(_solid_rgb_png_bytes(size=(2, 2)), {})

    decoded = png_bytes_to_pil(result.image_bytes)
    assert decoded.getpixel((0, 0)) == (0, 0, 0)
    assert decoded.getpixel((1, 1)) == (0, 0, 0)

