"""
Functional tests for worker_factory.

Tests automatic worker type detection based on model inspection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Mock dependencies just long enough to import the module under test,
# then restore sys.modules immediately so other test files aren't poisoned.
_MOCKED_MODULES = ['torch', 'torch.cuda', 'safetensors', 'safetensors.torch', 'diffusers']
_saved_modules = {k: sys.modules.get(k) for k in _MOCKED_MODULES}

for _mod in _MOCKED_MODULES:
    sys.modules[_mod] = MagicMock()

from backends.worker_factory import detect_worker_type, create_cuda_worker

# Restore immediately — the imported module keeps its internal mock references
for _mod, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _orig


class TestDetectWorkerType:
    """Test automatic worker type detection."""

    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "sdxl-base.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_detect_sdxl_base_2048(self, mock_exists, mock_detect):
        """Test SDXL Base detection (cross_attention_dim=2048)."""
        mock_info = Mock()
        mock_info.cross_attention_dim = 2048
        mock_info.variant = Mock(value="sdxl")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        worker_type = detect_worker_type()
        assert worker_type == "sdxl"

    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "sdxl-refiner.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_detect_sdxl_refiner_1280(self, mock_exists, mock_detect):
        """Test SDXL Refiner detection (cross_attention_dim=1280)."""
        mock_info = Mock()
        mock_info.cross_attention_dim = 1280
        mock_info.variant = Mock(value="sdxl-refiner")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        worker_type = detect_worker_type()
        assert worker_type == "sdxl"

    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "sd15.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_detect_sd15_768(self, mock_exists, mock_detect):
        """Test SD1.5 detection (cross_attention_dim=768)."""
        mock_info = Mock()
        mock_info.cross_attention_dim = 768
        mock_info.variant = Mock(value="sd15")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        worker_type = detect_worker_type()
        assert worker_type == "sd15"

    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "sd21.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_detect_sd21_1024(self, mock_exists, mock_detect):
        """Test SD2.x detection (cross_attention_dim=1024)."""
        mock_info = Mock()
        mock_info.cross_attention_dim = 1024
        mock_info.variant = Mock(value="sd21")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        worker_type = detect_worker_type()
        assert worker_type == "sd15"  # SD2.x uses sd15 worker

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_missing_model_root(self):
        """Test detection fails without MODEL_ROOT."""
        with pytest.raises(RuntimeError) as exc_info:
            detect_worker_type()
        assert "MODEL_ROOT" in str(exc_info.value)

    @patch.dict(os.environ, {"MODEL_ROOT": "/models"}, clear=True)
    def test_detect_missing_model(self):
        """Test detection fails without MODEL."""
        with pytest.raises(RuntimeError) as exc_info:
            detect_worker_type()
        assert "MODEL environment variable" in str(exc_info.value)

    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "nonexistent.safetensors"})
    @patch('os.path.exists', return_value=False)
    def test_detect_model_not_found(self, mock_exists):
        """Test detection fails when model file doesn't exist."""
        with pytest.raises(RuntimeError) as exc_info:
            detect_worker_type()
        assert "Model not found" in str(exc_info.value)

    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "unknown.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_detect_unsupported_dim(self, mock_exists, mock_detect):
        """Test detection fails with unsupported cross_attention_dim."""
        mock_info = Mock()
        mock_info.cross_attention_dim = 512  # Unsupported
        mock_info.variant = Mock(value="unknown")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        with pytest.raises(RuntimeError) as exc_info:
            detect_worker_type()
        assert "Unsupported cross_attention_dim: 512" in str(exc_info.value)

    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "broken.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_detect_inspection_fails(self, mock_exists, mock_detect):
        """Test detection fails when model inspection raises error."""
        mock_detect.side_effect = Exception("Failed to read model")

        with pytest.raises(RuntimeError) as exc_info:
            detect_worker_type()
        assert "Model detection failed" in str(exc_info.value)


class TestCreateCudaWorker:
    """Test CUDA worker creation."""

    @patch('backends.worker_factory.detect_worker_type')
    @patch('backends.cuda_worker.DiffusersSDXLCudaWorker')
    def test_create_sdxl_worker(self, mock_sdxl_class, mock_detect):
        """Test creating SDXL worker."""
        mock_detect.return_value = "sdxl"
        mock_worker = Mock()
        mock_sdxl_class.return_value = mock_worker

        worker = create_cuda_worker(worker_id=1)

        mock_detect.assert_called_once()
        mock_sdxl_class.assert_called_once_with(worker_id=1)
        assert worker == mock_worker

    @patch('backends.worker_factory.detect_worker_type')
    @patch('backends.cuda_worker.DiffusersCudaWorker')
    def test_create_sd15_worker(self, mock_sd15_class, mock_detect):
        """Test creating SD1.5 worker."""
        mock_detect.return_value = "sd15"
        mock_worker = Mock()
        mock_sd15_class.return_value = mock_worker

        worker = create_cuda_worker(worker_id=2)

        mock_detect.assert_called_once()
        mock_sd15_class.assert_called_once_with(worker_id=2)
        assert worker == mock_worker

    @patch('backends.worker_factory.detect_worker_type')
    def test_create_worker_detection_fails(self, mock_detect):
        """Test worker creation fails when detection fails."""
        mock_detect.side_effect = RuntimeError("Detection failed")

        with pytest.raises(RuntimeError) as exc_info:
            create_cuda_worker(worker_id=1)
        assert "Detection failed" in str(exc_info.value)


class TestCrossAttentionDimValues:
    """Test all supported cross_attention_dim values."""

    @pytest.mark.parametrize("dim,expected_type", [
        (768, "sd15"),      # SD1.5
        (1024, "sd15"),     # SD2.x (uses sd15 worker)
        (1280, "sdxl"),     # SDXL Refiner
        (2048, "sdxl"),     # SDXL Base
    ])
    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "test.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_all_supported_dimensions(self, mock_exists, mock_detect, dim, expected_type):
        """Test all supported cross_attention_dim values."""
        mock_info = Mock()
        mock_info.cross_attention_dim = dim
        mock_info.variant = Mock(value="test")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        worker_type = detect_worker_type()
        assert worker_type == expected_type

    @pytest.mark.parametrize("unsupported_dim", [
        256, 512, 640, 3072, 4096
    ])
    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/models", "MODEL": "test.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_unsupported_dimensions(self, mock_exists, mock_detect, unsupported_dim):
        """Test that unsupported dimensions raise errors."""
        mock_info = Mock()
        mock_info.cross_attention_dim = unsupported_dim
        mock_info.variant = Mock(value="unknown")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        with pytest.raises(RuntimeError) as exc_info:
            detect_worker_type()
        assert f"Unsupported cross_attention_dim: {unsupported_dim}" in str(exc_info.value)


class TestEnvironmentVariables:
    """Test environment variable handling."""

    @patch('utils.model_detector.detect_model')
    @patch.dict(os.environ, {"MODEL_ROOT": "/path/to/models", "MODEL": "model.safetensors"})
    @patch('os.path.exists', return_value=True)
    def test_uses_model_root_and_model(self, mock_exists, mock_detect):
        """Test that MODEL_ROOT and MODEL are used correctly."""
        mock_info = Mock()
        mock_info.cross_attention_dim = 768
        mock_info.variant = Mock(value="sd15")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        detect_worker_type()

        # Should call detect_model with joined path
        mock_detect.assert_called_once_with("/path/to/models/model.safetensors")

    @patch.dict(os.environ, {"MODEL_ROOT": "  /models  ", "MODEL": "  test.safetensors  "})
    @patch('utils.model_detector.detect_model')
    @patch('os.path.exists', return_value=True)
    def test_strips_whitespace(self, mock_exists, mock_detect):
        """Test that environment variables are stripped of whitespace."""
        mock_info = Mock()
        mock_info.cross_attention_dim = 768
        mock_info.variant = Mock(value="sd15")
        mock_info.confidence = 0.95
        mock_detect.return_value = mock_info

        detect_worker_type()

        # Should strip whitespace
        mock_detect.assert_called_once_with("/models/test.safetensors")
