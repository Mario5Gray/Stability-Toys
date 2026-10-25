import io

import numpy as np
import pytest
import transformers
import transformers.pipelines
from PIL import Image

import server.controlnet_preprocessors as preprocessors
from server.controlnet_preprocessors import (
    CannyPreprocessor,
    ControlMapPreprocessor,
    ControlMapResult,
    DEFAULT_REGISTRY,
    DepthPreprocessor,
    PosePreprocessor,
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


def test_canny_threshold_controls_edge_detection_sensitivity():
    # Sharp half-black / half-white edge: Sobel gradient ≈ 1020.
    # Loose thresholds (0, 10)  → gradient >> high  → edge detected.
    # Tight thresholds (2000, 4000) → gradient << high → no edge detected.
    # If thresholds were silently replaced by defaults (100, 200) the tight
    # assertion would fail because 1020 > 200.
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:, 8:] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    source = buf.getvalue()

    preprocessor = CannyPreprocessor()

    loose = preprocessor.run(source, {"low_threshold": 0, "high_threshold": 10})
    tight = preprocessor.run(source, {"low_threshold": 2000, "high_threshold": 4000})

    loose_arr = np.array(png_bytes_to_pil(loose.image_bytes))
    tight_arr = np.array(png_bytes_to_pil(tight.image_bytes))

    assert loose_arr.max() > 0, "loose thresholds must detect the sharp edge"
    assert tight_arr.max() == 0, "tight thresholds must suppress all edges"


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


def _stub_pose_preprocessor(calls: list[dict] | None = None) -> PosePreprocessor:
    recorded: list[dict] = [] if calls is None else calls

    class _StubPosePreprocessor(PosePreprocessor):
        def _get_detector(self):
            def _detector(pil_img, **kwargs):
                recorded.append({"size": pil_img.size, **kwargs})
                return Image.new("RGB", pil_img.size, color=(0, 0, 0))

            return _detector

    return _StubPosePreprocessor()


def test_pose_preprocessor_satisfies_protocol():
    assert isinstance(PosePreprocessor(), ControlMapPreprocessor)


def test_pose_preprocessor_is_registered_in_default_registry():
    registered = DEFAULT_REGISTRY.get("pose")
    assert isinstance(registered, PosePreprocessor)


def test_pose_preprocessor_generates_body_and_face_map_on_cpu():
    calls: list[dict] = []
    preprocessor = _stub_pose_preprocessor(calls)

    result = preprocessor.run(_solid_rgb_png_bytes(size=(64, 48)), {})

    decoded = png_bytes_to_pil(result.image_bytes)
    assert result.preprocessor_id == "pose"
    assert result.control_type == "pose"
    assert result.media_type == "image/png"
    assert result.width == 64
    assert result.height == 48
    assert decoded.size == (64, 48)
    assert result.metadata == {
        "model": "openpose",
        "parts": ["body", "face"],
        "device": "cpu",
        "max_res": 1024,
    }

    assert len(calls) == 1
    assert calls[0]["include_body"] is True
    assert calls[0]["include_face"] is True
    assert calls[0]["include_hand"] is False


def test_pose_preprocessor_resizes_oversized_source_to_max_res():
    calls: list[dict] = []
    preprocessor = _stub_pose_preprocessor(calls)

    result = preprocessor.run(
        _solid_rgb_png_bytes(size=(400, 200)),
        {"max_res": 100},
    )

    assert calls[0]["size"] == (100, 50)
    assert (result.width, result.height) == (100, 50)
    assert result.metadata["max_res"] == 100


def test_pose_preprocessor_honours_parts_option():
    calls: list[dict] = []
    preprocessor = _stub_pose_preprocessor(calls)

    result = preprocessor.run(_solid_rgb_png_bytes(size=(16, 16)), {"parts": "body"})

    assert calls[0]["include_body"] is True
    assert calls[0]["include_face"] is False
    assert result.metadata["parts"] == ["body"]


def test_pose_preprocessor_rejects_hands_part():
    preprocessor = PosePreprocessor()

    with pytest.raises(ValueError, match="unsupported pose part"):
        preprocessor.run(_solid_rgb_png_bytes(), {"parts": "body,hands"})


def test_pose_preprocessor_rejects_unsupported_model():
    preprocessor = PosePreprocessor()

    with pytest.raises(ValueError, match="unsupported pose model"):
        preprocessor.run(_solid_rgb_png_bytes(), {"model": "dwpose"})


def test_pose_preprocessor_rejects_non_cpu_device():
    calls: list[dict] = []
    preprocessor = _stub_pose_preprocessor(calls)

    with pytest.raises(ValueError, match="pose preprocessing runs on cpu"):
        preprocessor.run(_solid_rgb_png_bytes(), {"device": "mps"})

    assert calls == []


def test_pose_preprocessor_rejects_out_of_range_max_res():
    preprocessor = PosePreprocessor()

    with pytest.raises(ValueError, match="max_res"):
        preprocessor.run(_solid_rgb_png_bytes(), {"max_res": 8})


def test_pose_preprocessor_rejects_non_integer_max_res():
    preprocessor = PosePreprocessor()

    with pytest.raises(ValueError, match="max_res must be an integer"):
        preprocessor.run(_solid_rgb_png_bytes(), {"max_res": "wide"})


def test_pose_preprocessor_rejects_empty_parts():
    preprocessor = PosePreprocessor()

    with pytest.raises(ValueError, match="parts must not be empty"):
        preprocessor.run(_solid_rgb_png_bytes(), {"parts": " , "})


def test_pose_preprocessor_rejects_non_string_parts_type():
    preprocessor = PosePreprocessor()

    with pytest.raises(ValueError, match="parts must be a string or a list"):
        preprocessor.run(_solid_rgb_png_bytes(), {"parts": 5})


def test_pose_preprocessor_treats_null_parts_as_default():
    calls: list[dict] = []
    preprocessor = _stub_pose_preprocessor(calls)

    result = preprocessor.run(_solid_rgb_png_bytes(size=(16, 16)), {"parts": None})

    assert calls[0]["include_body"] is True
    assert calls[0]["include_face"] is True
    assert result.metadata["parts"] == ["body", "face"]


def test_pose_preprocessor_resizes_detector_output_to_source_size():
    class _OffSizePosePreprocessor(PosePreprocessor):
        def _get_detector(self):
            def _detector(pil_img, **kwargs):
                return Image.new("RGB", (128, 128), color=(0, 0, 0))

            return _detector

    result = _OffSizePosePreprocessor().run(_solid_rgb_png_bytes(size=(40, 30)), {})

    assert (result.width, result.height) == (40, 30)
    assert png_bytes_to_pil(result.image_bytes).size == (40, 30)


def test_pose_preprocessor_rejects_invalid_source_bytes_before_detector_call():
    calls: list[dict] = []
    preprocessor = _stub_pose_preprocessor(calls)

    with pytest.raises(ValueError, match="failed to decode image bytes"):
        preprocessor.run(b"not-an-image", {})

    assert calls == []


def test_pose_preprocessor_reports_missing_runtime(monkeypatch):
    def _missing():
        raise ModuleNotFoundError("No module named 'controlnet_aux'", name="controlnet_aux")

    monkeypatch.setattr(preprocessors, "_import_controlnet_aux", _missing)

    with pytest.raises(RuntimeError, match="controlnet-aux"):
        PosePreprocessor().run(_solid_rgb_png_bytes(), {})


def test_pose_preprocessor_loads_detector_once_and_caches_it(monkeypatch):
    created: list[tuple[str, object]] = []

    class _FakeOpenposeDetector:
        @classmethod
        def from_pretrained(cls, repo):
            detector = object()
            created.append((repo, detector))
            return detector

    class _FakeModule:
        OpenposeDetector = _FakeOpenposeDetector

    monkeypatch.setattr(preprocessors, "_import_controlnet_aux", lambda: _FakeModule)

    preprocessor = PosePreprocessor()

    first = preprocessor._get_detector()
    second = preprocessor._get_detector()

    assert first is second
    assert created == [("lllyasviel/Annotators", first)]

