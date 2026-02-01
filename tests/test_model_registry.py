"""
Functional tests for ModelRegistry.

Tests VRAM tracking, model registration, and capacity checking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock torch just long enough to import model_registry,
# then restore sys.modules immediately so other test files aren't poisoned.
_MOCKED_MODULES = ['torch', 'torch.cuda']
_saved_modules = {k: sys.modules.get(k) for k in _MOCKED_MODULES}

torch_mock = MagicMock(name="torch")
cuda_mock = MagicMock(name="torch.cuda")
torch_mock.cuda = cuda_mock

sys.modules['torch'] = torch_mock
sys.modules['torch.cuda'] = cuda_mock

from backends.model_registry import ModelRegistry

# Restore immediately
for _mod, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_mod, None)
    else:
        sys.modules[_mod] = _orig


@pytest.fixture
def mock_cuda():
    """Mock CUDA functions for testing."""    
    with patch('backends.model_registry.torch.cuda.is_available', return_value=True), \
         patch('backends.model_registry.torch.cuda.get_device_properties') as mock_props, \
         patch('backends.model_registry.torch.cuda.memory_allocated', return_value=0), \
         patch("backends.model_registry.torch.cuda.get_available_vram", return_value=100 * 1024**2),\
         patch("backends.model_registry.torch.cuda.device_memory_used", return_value=100 * 1024**2) :
        
        # Mock GPU with 24GB VRAM
        mock_device_props = Mock()
        mock_device_props.total_memory = 24 * 1024**3  # 24GB in bytes
        mock_device_props.name = "NVIDIA GeForce RTX 3090"
        mock_props.return_value = mock_device_props

        yield

@pytest.fixture
def registry(mock_cuda):
    """Create a fresh ModelRegistry for each test."""
    return ModelRegistry()


class TestModelRegistryInit:
    """Test registry initialization."""

    def test_init_with_cuda(self, mock_cuda):
        """Test initialization with CUDA available."""
        registry = ModelRegistry()

        assert registry._device_index == 0
        assert registry._total_vram == 24 * 1024**3
        assert len(registry._loaded) == 0

    def test_get_total_vram(self, registry):
        """Test total VRAM retrieval."""
        assert registry.get_total_vram() == 24 * 1024**3


class TestModelRegistration:
    """Test model registration and unregistration."""

    def test_register_single_model(self, registry):
        """Test registering a single model."""
        registry.register_model(
            name="sdxl-base",
            model_path="/models/sdxl-base.safetensors",
            vram_bytes=12 * 1024**3,
            loras=[]
        )

        assert "sdxl-base" in registry._loaded
        model_info = registry._loaded["sdxl-base"]
        assert model_info.name == "sdxl-base"
        assert model_info.model_path == "/models/sdxl-base.safetensors"
        assert model_info.vram_bytes == 12 * 1024**3
        assert model_info.loras == []

    def test_register_model_with_loras(self, registry):
        """Test registering model with LoRAs."""
        loras = ["/loras/portrait.safetensors", "/loras/detail.safetensors"]

        registry.register_model(
            name="sdxl-portrait",
            model_path="/models/sdxl-base.safetensors",
            vram_bytes=14 * 1024**3,
            loras=loras
        )

        model_info = registry._loaded["sdxl-portrait"]
        assert model_info.loras == loras

    def test_register_multiple_models(self, registry):
        """Test registering multiple models."""
        registry.register_model("model1", "/path/1", 5 * 1024**3, loras=[])
        registry.register_model("model2", "/path/2", 7 * 1024**3, loras=[])

        assert len(registry._loaded) == 2
        assert "model1" in registry._loaded
        assert "model2" in registry._loaded

    def test_register_overwrites_existing(self, registry):
        """Test that re-registering overwrites existing entry."""
        registry.register_model("test", "/path/1", 5 * 1024**3, loras=[])
        registry.register_model("test", "/path/2", 10 * 1024**3, loras=[])

        assert len(registry._loaded) == 1
        assert registry._loaded["test"].model_path == "/path/2"
        assert registry._loaded["test"].vram_bytes == 10 * 1024**3

    def test_unregister_model(self, registry):
        """Test unregistering a model."""
        registry.register_model("test", "/path", 5 * 1024**3, loras=[])
        assert "test" in registry._loaded

        registry.unregister_model("test")
        assert "test" not in registry._loaded

    def test_unregister_nonexistent_model(self, registry):
        """Test unregistering non-existent model (should not raise)."""
        registry.unregister_model("nonexistent")  # Should not raise
        assert len(registry._loaded) == 0

    def test_clear_all_models(self, registry):
        """Test clearing all models."""
        registry.register_model("model1", "/path/1", 5 * 1024**3, loras=[])
        registry.register_model("model2", "/path/2", 7 * 1024**3, loras=[])

        registry.clear()
        assert len(registry._loaded) == 0


class TestVRAMTracking:
    """Test VRAM tracking and availability calculations."""

    @patch('backends.model_registry.torch.cuda.memory_allocated')
    def test_get_used_vram_empty(self, mock_allocated, registry):
        """Test VRAM usage with no models loaded."""
        mock_allocated.return_value = 100 * 1024**2  # 100MB baseline

        used = registry.get_used_vram()        
        assert used == 100 * 1024**2

    @patch('backends.model_registry.torch.cuda.device_memory_used')
    def test_get_used_vram_with_models(self, mock_allocated, registry):
        """Test VRAM usage with models loaded."""
        mock_allocated.return_value = 15 * 1024**3  # 15GB allocated

        registry.register_model("test", "/path", 12 * 1024**3, loras=[])
        used = registry.get_used_vram()
        assert used == 15 * 1024**3

    @patch('backends.model_registry.torch.cuda.device_memory_used')
    def test_get_available_vram(self, mock_allocated, registry):
        """Test available VRAM calculation."""
        mock_allocated.return_value = 10 * 1024**3  # 10GB used

        available = registry.get_available_vram()
        # 24GB total - 10GB used = 14GB available
        assert available == 14 * 1024**3

    @patch('backends.model_registry.torch.cuda.memory_allocated')
    def test_can_fit_when_space_available(self, mock_allocated, registry):
        """Test can_fit returns True when space available."""
        mock_allocated.return_value = 10 * 1024**3  # 10GB used, 14GB available

        # Should fit: 14GB available > 12GB requested
        assert registry.can_fit(12 * 1024**3) is True

    @patch('backends.model_registry.torch.cuda.device_memory_used')
    def test_can_fit_when_no_space(self, mock_allocated, registry):
        """Test can_fit returns False when no space."""
        mock_allocated.return_value = 20 * 1024**3  # 20GB used, 4GB available
                
        # Should not fit: 4GB available < 12GB requested
        assert registry.can_fit(12 * 1024**3) is False

    @patch('backends.model_registry.torch.cuda.memory_allocated')
    def test_can_fit_exact_fit(self, mock_allocated, registry):
        """Test can_fit with exact available space."""
        mock_allocated.return_value = 20 * 1024**3  # 20GB used, 4GB available

        # Exact fit should work
        assert registry.can_fit(4 * 1024**3) is True


class TestVRAMStats:
    """Test VRAM statistics output."""

    @patch('backends.model_registry.torch.cuda.device_memory_used')
    def test_get_vram_stats_empty(self, mock_allocated, registry):
        """Test VRAM stats with no models loaded."""
        mock_allocated.return_value = 100 * 1024**2  # 100MB

        stats = registry.get_vram_stats()

        assert stats["device"] == "NVIDIA GeForce RTX 3090"
        assert stats["total_gb"] == pytest.approx(24.0, rel=0.1)
        assert stats["used_gb"] == pytest.approx(0.1, rel=0.1)
        assert stats["available_gb"] == pytest.approx(23.9, rel=0.1)
        assert stats["models_loaded"] == 0
        assert stats["models"] == []

    @patch('backends.model_registry.torch.cuda.device_memory_used')
    def test_get_vram_stats_with_models(self, mock_allocated, registry):
        """Test VRAM stats with models loaded."""
        mock_allocated.return_value = 15 * 1024**3  # 15GB

        registry.register_model(
            name="sdxl-base",
            model_path="/models/sdxl-base.safetensors",
            vram_bytes=12 * 1024**3,
            loras=[]
        )
        registry.register_model(
            name="sd15-fast",
            model_path="/models/sd15-fast.safetensors",
            vram_bytes=3 * 1024**3,
            loras=["/loras/test.safetensors"]
        )

        stats = registry.get_vram_stats()

        assert stats["total_gb"] == pytest.approx(24.0, rel=0.1)
        assert stats["used_gb"] == pytest.approx(15.0, rel=0.1)
        assert stats["available_gb"] == pytest.approx(9.0, rel=0.1)
        assert stats["usage_percent"] == pytest.approx(62.5, rel=0.1)
        assert stats["models_loaded"] == 2

        # Check model details
        assert len(stats["models"]) == 2

        sdxl_stats = next(m for m in stats["models"] if m["name"] == "sdxl-base")
        assert sdxl_stats["model_path"] == "/models/sdxl-base.safetensors"
        assert sdxl_stats["vram_gb"] == pytest.approx(12.0, rel=0.1)
        assert sdxl_stats["loras"] == []

        sd15_stats = next(m for m in stats["models"] if m["name"] == "sd15-fast")
        assert sd15_stats["vram_gb"] == pytest.approx(3.0, rel=0.1)
        assert sd15_stats["loras"] == ["/loras/test.safetensors"]


class TestHelperMethods:
    """Test helper methods."""

    def test_get_loaded_models_empty(self, registry):
        """Test getting loaded models when empty."""
        models = registry.get_loaded_models()
        assert models == {}

    def test_get_loaded_models_with_models(self, registry):
        """Test getting all loaded models."""
        registry.register_model("model1", "/path/1", 5 * 1024**3, loras=[])
        registry.register_model("model2", "/path/2", 7 * 1024**3, loras=[])

        models = registry.get_loaded_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models

    def test_is_loaded_true(self, registry):
        """Test checking if model is loaded (true case)."""
        registry.register_model("test", "/path", 5 * 1024**3, loras=[])
        assert registry.is_loaded("test") is True

    def test_is_loaded_false(self, registry):
        """Test checking if model is loaded (false case)."""
        assert registry.is_loaded("nonexistent") is False

    def test_get_model_exists(self, registry):
        """Test getting model when it exists."""
        registry.register_model("test", "/path", 5 * 1024**3, loras=["/lora1"])

        model = registry.get_model("test")
        assert model is not None
        assert model.name == "test"
        assert model.model_path == "/path"
        assert model.vram_bytes == 5 * 1024**3
        assert model.loras == ["/lora1"]

    def test_get_model_not_exists(self, registry):
        """Test getting model when it doesn't exist."""
        model = registry.get_model("nonexistent")
        assert model is None


class TestEstimateVRAM:
    """Test VRAM estimation utility."""

    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_estimate_vram_from_file_size(self, mock_getsize, mock_exists, registry):
        """Test VRAM estimation (file_size * 1.2)."""
        mock_exists.return_value = True
        mock_getsize.return_value = 10 * 1024**3  # 10GB file

        estimated = registry.estimate_model_vram("/models/test.safetensors")

        # Should be file_size * 1.2
        assert estimated == pytest.approx(12 * 1024**3, rel=0.01)

    @patch('os.path.exists')
    def test_estimate_vram_file_not_found(self, mock_exists, registry):
        """Test VRAM estimation when file doesn't exist."""
        mock_exists.return_value = False

        estimated = registry.estimate_model_vram("/models/nonexistent.safetensors")
        assert estimated == 0


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_register_unregister_thread_safe(self, registry):
        """Test that registration is thread-safe (uses lock)."""
        import threading

        def register():
            for i in range(10):
                registry.register_model(f"model-{i}", f"/path/{i}", 1024**3, loras=[])

        def unregister():
            for i in range(10):
                registry.unregister_model(f"model-{i}")

        t1 = threading.Thread(target=register)
        t2 = threading.Thread(target=unregister)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should complete without errors
        assert True
