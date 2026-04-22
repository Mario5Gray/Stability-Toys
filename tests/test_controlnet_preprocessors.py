import pytest

from server.controlnet_preprocessors import (
    ControlMapPreprocessor,
    ControlMapResult,
    PreprocessorRegistry,
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


def test_control_map_result_defaults():
    result = ControlMapResult(
        preprocessor_id="canny",
        control_type="canny",
        image_bytes=b"data",
        width=64,
        height=64,
    )
    assert result.media_type == "image/png"
