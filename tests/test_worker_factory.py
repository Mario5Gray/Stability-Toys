"""
Functional tests for worker_factory.

Tests automatic worker type detection and capability-aware worker creation.
"""

import sys
import types
from unittest.mock import Mock, MagicMock, patch

import pytest

from utils.model_detector import ModelInfo, ModelVariant

# Mock heavyweight dependencies just long enough to import the module under test,
# then restore sys.modules immediately so other test files aren't poisoned.
_MOCKED_MODULES = ["torch", "torch.cuda", "safetensors", "safetensors.torch", "diffusers"]
_saved_modules = {k: sys.modules.get(k) for k in _MOCKED_MODULES}

for _mod in _MOCKED_MODULES:
    sys.modules[_mod] = MagicMock()

from backends.worker_factory import detect_worker_type, create_cuda_worker

for _mod, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _orig


class TestDetectWorkerType:
    """Test automatic worker type detection."""

    @patch("backends.worker_factory.detect_model")
    @patch("os.path.exists", return_value=True)
    def test_detect_sdxl_base_2048(self, mock_exists, mock_detect):
        """SDXL Base uses the SDXL worker."""
        mock_detect.return_value = ModelInfo(
            path="/models/sdxl-base.safetensors",
            variant=ModelVariant.SDXL_BASE,
            cross_attention_dim=2048,
            confidence=0.95,
        )

        worker_type = detect_worker_type("/models/sdxl-base.safetensors")

        assert worker_type == "sdxl"
        mock_detect.assert_called_once_with("/models/sdxl-base.safetensors")

    @patch("backends.worker_factory.detect_model")
    @patch("os.path.exists", return_value=True)
    def test_detect_sd21_1024(self, mock_exists, mock_detect):
        """SD2.x continues to use the SD1.5 worker implementation."""
        mock_detect.return_value = ModelInfo(
            path="/models/sd21.safetensors",
            variant=ModelVariant.SD21,
            cross_attention_dim=1024,
            confidence=0.95,
        )

        worker_type = detect_worker_type("/models/sd21.safetensors")

        assert worker_type == "sd15"

    @patch("os.path.exists", return_value=False)
    def test_detect_model_not_found(self, mock_exists):
        """Missing model path fails before detection."""
        with pytest.raises(RuntimeError, match="Model not found"):
            detect_worker_type("/models/missing.safetensors")

    @patch("backends.worker_factory.detect_model")
    @patch("os.path.exists", return_value=True)
    def test_detect_unsupported_dim(self, mock_exists, mock_detect):
        """Unsupported cross-attention dims remain a hard error."""
        mock_detect.return_value = ModelInfo(
            path="/models/unknown.safetensors",
            variant=ModelVariant.UNKNOWN,
            cross_attention_dim=512,
            confidence=0.95,
        )

        with pytest.raises(RuntimeError, match="Unsupported cross_attention_dim: 512"):
            detect_worker_type("/models/unknown.safetensors")


class TestCreateCudaWorker:
    """Test CUDA worker creation."""

    @patch("backends.worker_factory.detect_model")
    def test_create_sdxl_worker_passes_detected_capabilities(self, mock_detect):
        """Detected model capabilities must be forwarded into the SDXL worker."""
        model_info = ModelInfo(
            path="/models/checkpoints/sdxl-base.safetensors",
            variant=ModelVariant.SDXL_BASE,
            cross_attention_dim=2048,
            confidence=0.95,
            loader_format="single_file",
            checkpoint_precision="fp8",
            checkpoint_variant="sdxl-base",
        )
        model_info.scheduler_profile = "native"
        mock_detect.return_value = model_info
        mock_worker = Mock()
        fake_cuda_worker = types.SimpleNamespace(
            DiffusersSDXLCudaWorker=Mock(return_value=mock_worker)
        )

        with patch.dict(sys.modules, {"backends.cuda_worker": fake_cuda_worker}):
            worker = create_cuda_worker(
                worker_id=3,
                model_path="/models/checkpoints/sdxl-base.safetensors",
            )

        assert worker == mock_worker
        kwargs = fake_cuda_worker.DiffusersSDXLCudaWorker.call_args.kwargs
        assert kwargs["worker_id"] == 3
        assert kwargs["model_path"] == "/models/checkpoints/sdxl-base.safetensors"
        assert kwargs["model_info"] is model_info

    @patch("backends.worker_factory.detect_model")
    def test_create_worker_uses_supplied_model_info(self, mock_detect):
        """WorkerPool can pass authoritative merged capabilities without re-detecting."""
        model_info = ModelInfo(
            path="/models/checkpoints/sdxl-base.safetensors",
            variant=ModelVariant.SDXL_BASE,
            cross_attention_dim=2048,
            confidence=0.95,
            loader_format="single_file",
            checkpoint_precision="fp8",
            checkpoint_variant="sdxl-base",
        )
        model_info.scheduler_profile = "native"
        fake_cuda_worker = types.SimpleNamespace(
            DiffusersSDXLCudaWorker=Mock(return_value=Mock())
        )

        with patch.dict(sys.modules, {"backends.cuda_worker": fake_cuda_worker}):
            create_cuda_worker(
                worker_id=5,
                model_path="/models/checkpoints/sdxl-base.safetensors",
                model_info=model_info,
            )

        mock_detect.assert_not_called()
        assert fake_cuda_worker.DiffusersSDXLCudaWorker.call_args.kwargs["model_info"] is model_info

    @patch("backends.worker_factory.detect_model")
    def test_create_sd15_worker_passes_model_info(self, mock_detect):
        """The SD1.5 worker also receives the resolved ModelInfo."""
        model_info = ModelInfo(
            path="/models/checkpoints/sd15.safetensors",
            variant=ModelVariant.SD15,
            cross_attention_dim=768,
            confidence=0.95,
            loader_format="single_file",
            checkpoint_precision="unknown",
            checkpoint_variant="sd15",
        )
        model_info.scheduler_profile = "lcm"
        mock_detect.return_value = model_info
        fake_cuda_worker = types.SimpleNamespace(
            DiffusersCudaWorker=Mock(return_value=Mock())
        )

        with patch.dict(sys.modules, {"backends.cuda_worker": fake_cuda_worker}):
            create_cuda_worker(
                worker_id=2,
                model_path="/models/checkpoints/sd15.safetensors",
            )

        assert fake_cuda_worker.DiffusersCudaWorker.call_args.kwargs["model_info"] is model_info

    @patch("backends.worker_factory.detect_model")
    def test_create_worker_detection_fails(self, mock_detect):
        """Detection failures still surface as runtime errors."""
        mock_detect.side_effect = RuntimeError("Detection failed")

        with pytest.raises(RuntimeError, match="Detection failed"):
            create_cuda_worker(worker_id=1, model_path="/models/broken.safetensors")


def test_model_info_to_dict_includes_recommended_size():
    """Recommended size should serialize with the rest of the top-level capabilities."""
    model_info = ModelInfo(
        path="/models/checkpoints/sdxl-base.safetensors",
        variant=ModelVariant.SDXL_BASE,
        recommended_size="896x1152",
    )

    assert model_info.to_dict()["recommended_size"] == "896x1152"
