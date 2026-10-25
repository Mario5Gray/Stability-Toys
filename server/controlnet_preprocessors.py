"""
ControlNet preprocessor seam: protocol, result type, registry, shared image helpers,
and v1 concrete implementations (CannyPreprocessor, DepthPreprocessor,
PosePreprocessor).

Runtime requirement: PosePreprocessor needs the ``controlnet-aux`` package. That
package is a server requirement (``requirements.txt``). The import stays lazy so
the server and other preprocessors work when the package is absent.
"""

import io
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ControlMapResult:
    preprocessor_id: str
    control_type: str
    image_bytes: bytes
    width: int
    height: int
    media_type: str = "image/png"
    metadata: dict[str, Any] = field(default_factory=dict)


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


def _import_controlnet_aux():
    """Import controlnet_aux. Separate function so tests can simulate absence."""
    import controlnet_aux

    return controlnet_aux


class PosePreprocessor:
    """Pose-map preprocessor for the HunyuanDiT Pose ControlNet.

    Output format: OpenPose-style skeleton. Black background, colored 18-keypoint
    body skeleton and joint dots, white face keypoint dots, no hands. The
    checkpoint ships this format in its own ``pose.jpg``.

    Execution device: cpu only. The detector loads on cpu and this class passes
    no device to it. A non-cpu ``device`` option raises a stable ValueError.
    """

    preprocessor_id = "pose"
    control_type = "pose"
    _DEFAULT_MODEL = "openpose"
    _SUPPORTED_MODELS = ("openpose",)
    _SUPPORTED_PARTS = ("body", "face")
    _DEFAULT_PARTS = ("body", "face")
    _DEFAULT_MAX_RES = 1024
    _MIN_MAX_RES = 64
    _MAX_MAX_RES = 4096
    _ANNOTATORS_REPO = "lllyasviel/Annotators"

    def __init__(self, annotators_repo: str = _ANNOTATORS_REPO) -> None:
        self._annotators_repo = annotators_repo
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            try:
                module = _import_controlnet_aux()
            except ImportError as exc:
                raise RuntimeError(
                    "pose preprocessing needs controlnet-aux. "
                    "Install the pinned version with: "
                    "pip install 'controlnet-aux==0.0.10'"
                ) from exc
            self._detector = module.OpenposeDetector.from_pretrained(self._annotators_repo)
        return self._detector

    def _parse_model(self, options: dict[str, Any]) -> str:
        model = str(options.get("model", self._DEFAULT_MODEL)).strip().lower()
        if model not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"unsupported pose model {model!r}. Supported: "
                f"{', '.join(self._SUPPORTED_MODELS)}"
            )
        return model

    def _parse_device(self, options: dict[str, Any]) -> str:
        device = str(options.get("device", "cpu")).strip().lower()
        if device != "cpu":
            raise ValueError(
                f"pose preprocessing runs on cpu only. Requested device {device!r}."
            )
        return device

    def _parse_parts(self, options: dict[str, Any]) -> list[str]:
        raw = options.get("parts", self._DEFAULT_PARTS)
        if raw is None:
            raw = self._DEFAULT_PARTS
        if isinstance(raw, str):
            parts = [part.strip().lower() for part in raw.split(",")]
        elif isinstance(raw, (list, tuple, set, frozenset)):
            parts = [str(part).strip().lower() for part in raw]
        else:
            raise ValueError(
                f"parts must be a string or a list of strings, got {type(raw).__name__}"
            )
        parts = [part for part in parts if part]
        if not parts:
            raise ValueError("pose parts must not be empty")
        for part in parts:
            if part not in self._SUPPORTED_PARTS:
                raise ValueError(
                    f"unsupported pose part {part!r}. Supported: "
                    f"{', '.join(self._SUPPORTED_PARTS)}"
                )
        return sorted(set(parts))

    def _parse_max_res(self, options: dict[str, Any]) -> int:
        raw = options.get("max_res", self._DEFAULT_MAX_RES)
        try:
            max_res = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_res must be an integer, got {raw!r}") from exc
        if max_res < self._MIN_MAX_RES or max_res > self._MAX_MAX_RES:
            raise ValueError(
                f"max_res must be between {self._MIN_MAX_RES} and "
                f"{self._MAX_MAX_RES}, got {max_res}"
            )
        return max_res

    def run(self, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult:
        from PIL import Image

        model = self._parse_model(options)
        device = self._parse_device(options)
        parts = self._parse_parts(options)
        max_res = self._parse_max_res(options)

        pil_img = png_bytes_to_pil(image_bytes)
        pil_img = self._cap_long_edge(pil_img, max_res)
        width, height = pil_img.size

        detector = self._get_detector()
        pose_img = detector(
            pil_img,
            detect_resolution=min(width, height),
            image_resolution=min(width, height),
            include_body="body" in parts,
            include_face="face" in parts,
            include_hand=False,
            output_type="pil",
        )
        if pose_img.size != (width, height):
            pose_img = pose_img.resize((width, height), Image.Resampling.LANCZOS)

        return ControlMapResult(
            preprocessor_id=self.preprocessor_id,
            control_type=self.control_type,
            image_bytes=pil_to_png_bytes(pose_img),
            width=width,
            height=height,
            metadata={
                "model": model,
                "parts": parts,
                "device": device,
                "max_res": max_res,
            },
        )

    @staticmethod
    def _cap_long_edge(pil_img, max_res: int):
        from PIL import Image

        width, height = pil_img.size
        longest = max(width, height)
        if longest <= max_res:
            return pil_img
        scale = max_res / longest
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return pil_img.resize(size, Image.Resampling.LANCZOS)


DEFAULT_REGISTRY = PreprocessorRegistry()
DEFAULT_REGISTRY.register(CannyPreprocessor())
DEFAULT_REGISTRY.register(DepthPreprocessor())
DEFAULT_REGISTRY.register(PosePreprocessor())
