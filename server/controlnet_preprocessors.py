"""
ControlNet preprocessor seam: protocol, result type, registry, shared image helpers,
and v1 concrete implementations (CannyPreprocessor, DepthPreprocessor).
"""

import io
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ControlMapResult:
    preprocessor_id: str
    control_type: str
    image_bytes: bytes
    width: int
    height: int
    media_type: str = "image/png"


@runtime_checkable
class ControlMapPreprocessor(Protocol):
    preprocessor_id: str
    control_type: str

    def run(self, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult: ...


class PreprocessorRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, ControlMapPreprocessor] = {}

    def register(self, preprocessor: ControlMapPreprocessor, *, replace: bool = False) -> None:
        existing = self._registry.get(preprocessor.preprocessor_id)
        if existing is not None and not replace:
            raise ValueError(
                f"preprocessor {preprocessor.preprocessor_id!r} is already registered"
            )
        self._registry[preprocessor.preprocessor_id] = preprocessor

    def get(self, preprocessor_id: str) -> ControlMapPreprocessor | None:
        return self._registry.get(preprocessor_id)

    def dispatch(self, preprocessor_id: str, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult:
        preprocessor = self.get(preprocessor_id)
        if preprocessor is None:
            raise ValueError(f"unknown preprocessor {preprocessor_id!r}")
        return preprocessor.run(image_bytes, options)


def pil_to_png_bytes(pil_image) -> bytes:
    """Encode a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_pil(image_bytes: bytes):
    """Decode PNG (or any PIL-supported format) bytes to a PIL Image."""
    from PIL import Image

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"failed to decode image bytes: {exc}") from exc


class CannyPreprocessor:
    preprocessor_id = "canny"
    control_type = "canny"

    def run(self, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult:
        import cv2
        import numpy as np
        from PIL import Image

        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("CannyPreprocessor: could not decode source image")

        low = int(options.get("low_threshold", 100))
        high = int(options.get("high_threshold", 200))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        height, width = edges.shape

        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result_pil = Image.fromarray(edges_rgb)
        return ControlMapResult(
            preprocessor_id=self.preprocessor_id,
            control_type=self.control_type,
            image_bytes=pil_to_png_bytes(result_pil),
            width=width,
            height=height,
        )


class DepthPreprocessor:
    preprocessor_id = "depth"
    control_type = "depth"
    _DEFAULT_MODEL = "LiheYoung/depth-anything-small-hf"

    def __init__(self, model_id: str = _DEFAULT_MODEL) -> None:
        self._model_id = model_id
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is None:
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline("depth-estimation", model=self._model_id)
        return self._pipe

    def run(self, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult:
        import numpy as np
        from PIL import Image

        pil_img = png_bytes_to_pil(image_bytes)
        width, height = pil_img.size

        result = self._get_pipe()(pil_img)
        depth_pil = result["depth"]

        depth_arr = np.array(depth_pil, dtype=np.float32)
        d_min, d_max = depth_arr.min(), depth_arr.max()
        if d_max > d_min:
            normalized = ((depth_arr - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_arr, dtype=np.uint8)

        depth_rgb = Image.fromarray(normalized).convert("RGB")
        return ControlMapResult(
            preprocessor_id=self.preprocessor_id,
            control_type=self.control_type,
            image_bytes=pil_to_png_bytes(depth_rgb),
            width=width,
            height=height,
        )


DEFAULT_REGISTRY = PreprocessorRegistry()
DEFAULT_REGISTRY.register(CannyPreprocessor())
DEFAULT_REGISTRY.register(DepthPreprocessor())
