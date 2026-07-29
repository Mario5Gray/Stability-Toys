"""
Functional tests for ModelRegistry as a pure DeviceMemory view.

The registry holds no torch and no pynvml: tests stub the DeviceMemory
interface (the mrrrbmjp vaccine — no torch/pynvml call-site mocks).
Registration/lifecycle/estimate tests don't touch VRAM sourcing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from backends.device_memory import (
    ConsumerMemory,
    DeviceMemorySnapshot,
    MemoryTopology,
)
from backends.model_registry import ModelRegistry


def _snap(consumers=(), total=24 * 1024**3, free=10 * 1024**3):
    return DeviceMemorySnapshot(
        device_uuid="GPU-test",
        topology=MemoryTopology.DISCRETE,
        total_bytes=total,
        free_bytes=free,
        consumers=consumers,
    )


@pytest.fixture
def device_memory():
    dm = Mock()
    dm.device_name = "NVIDIA GeForce RTX 3090"
    dm.cached_snapshot.return_value = _snap()
    dm.available_for_load.return_value = 10 * 1024**3
    return dm


@pytest.fixture
def registry(device_memory):
    """Create a fresh ModelRegistry for each test, viewing the stub device."""
    return ModelRegistry(device_memory=device_memory)


class TestModelRegistryInit:
    """Test registry initialization."""

    def test_placeholder_registry_reports_backend_without_vram_fields(self):
        from backends.model_registry import PlaceholderModelRegistry

        registry = PlaceholderModelRegistry("cpu")
        stats = registry.get_vram_stats()

        assert stats["backend"] == "cpu"
        assert stats["device"] == "CPU placeholder"
        assert stats["models_loaded"] == 0

    def test_init_loads_empty(self, registry):
        """Initialization over the device view; no models loaded."""
        assert len(registry._loaded) == 0


class TestDeviceMemoryView:
    """The registry is a pure view: cached_snapshot/available_for_load only."""

    def test_total_from_cached_snapshot(self, registry, device_memory):
        assert registry.get_total_vram() == 24 * 1024**3
        device_memory.cached_snapshot.assert_called()

    def test_available_from_available_for_load(self, registry):
        assert registry.get_available_vram() == 10 * 1024**3

    def test_allocated_reserved_from_worker_consumer_entry(self, registry, device_memory):
        worker = ConsumerMemory(label="worker", pid=1,
                                allocated_bytes=3 * 1024**3,
                                reserved_bytes=5 * 1024**3)
        device_memory.cached_snapshot.return_value = _snap(consumers=(worker,))
        assert registry.get_allocated_vram() == 3 * 1024**3
        assert registry.get_reserved_vram() == 5 * 1024**3
        assert registry.get_used_vram() == 5 * 1024**3  # reserved alias

    def test_no_worker_consumer_reads_zero(self, registry):
        assert registry.get_allocated_vram() == 0
        assert registry.get_reserved_vram() == 0

    def test_registry_never_calls_fresh_snapshot(self, registry, device_memory):
        registry.get_total_vram()
        registry.get_allocated_vram()
        registry.get_reserved_vram()
        device_memory.snapshot.assert_not_called()  # no fan-out from the view

    def test_registry_has_no_torch(self):
        import backends.model_registry as mr
        assert not hasattr(mr, "torch"), "registry must not import torch"


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


class TestCanFit:
    """can_fit reads driver-truth free via available_for_load (5% slack gate)."""

    def test_can_fit_rejects_when_driver_free_is_short(self, registry, device_memory):
        """Rejects even when total - torch_reserved would wrongly say it fits."""
        device_memory.available_for_load.return_value = 10 * 1024**3
        assert registry.can_fit(12 * 1024**3) is False

    def test_can_fit_when_space_available(self, registry, device_memory):
        device_memory.available_for_load.return_value = 14 * 1024**3
        assert registry.can_fit(12 * 1024**3) is True

    def test_can_fit_when_no_space(self, registry, device_memory):
        device_memory.available_for_load.return_value = 4 * 1024**3
        assert registry.can_fit(12 * 1024**3) is False

    def test_can_fit_exact_fit_within_slack(self, registry, device_memory):
        device_memory.available_for_load.return_value = 4 * 1024**3
        assert registry.can_fit(4 * 1024**3) is True  # 4 < 4 + 5% slack

    def test_can_fit_false_when_no_device(self, registry, device_memory):
        """Null-provider parity with the old no-CUDA guard: 0 free -> reject."""
        device_memory.available_for_load.return_value = 0
        assert registry.can_fit(4 * 1024**3) is False


class TestVRAMStats:
    """Test VRAM statistics output via the view."""

    def test_get_vram_stats_empty(self, registry, device_memory):
        """used_gb is device-used (total - driver_free); reserved_gb is the
        worker consumer's torch pool. The two must not be conflated."""
        worker = ConsumerMemory(label="worker", pid=1,
                                allocated_bytes=50 * 1024**2,
                                reserved_bytes=100 * 1024**2)
        device_memory.cached_snapshot.return_value = _snap(
            consumers=(worker,), total=24 * 1024**3, free=23 * 1024**3)
        device_memory.available_for_load.return_value = 23 * 1024**3

        stats = registry.get_vram_stats()

        assert stats["device"] == "NVIDIA GeForce RTX 3090"
        assert stats["total_gb"] == pytest.approx(24.0, rel=0.1)
        assert stats["available_gb"] == pytest.approx(23.0, rel=0.1)   # driver free
        assert stats["used_gb"] == pytest.approx(1.0, rel=0.1)         # total - free
        assert stats["usage_percent"] == pytest.approx(4.17, rel=0.1)  # used / total
        assert stats["reserved_gb"] == pytest.approx(0.1, rel=0.1)     # worker pool
        assert stats["models_loaded"] == 0
        assert stats["models"] == []

    def test_get_vram_stats_with_models(self, registry, device_memory):
        """Stats with models: device-used reflects the driver, not torch reserved."""
        worker = ConsumerMemory(label="worker", pid=1,
                                allocated_bytes=14 * 1024**3,
                                reserved_bytes=15 * 1024**3)
        device_memory.cached_snapshot.return_value = _snap(
            consumers=(worker,), total=24 * 1024**3, free=4 * 1024**3)
        device_memory.available_for_load.return_value = 4 * 1024**3

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
        assert stats["available_gb"] == pytest.approx(4.0, rel=0.1)   # driver free
        assert stats["used_gb"] == pytest.approx(20.0, rel=0.1)       # total - free
        assert stats["usage_percent"] == pytest.approx(83.3, rel=0.1)
        assert stats["reserved_gb"] == pytest.approx(15.0, rel=0.1)   # worker pool
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

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=MagicMock)
    def test_estimate_vram_fp8_safetensors(self, mock_open, mock_getsize, mock_exists, registry):
        """fp8 safetensors uses multiplier 1.1, not 1.2."""
        import json
        import struct

        mock_exists.return_value = True
        mock_getsize.return_value = 4 * 1024**3  # 4 GB fp8 file

        # Build a minimal safetensors header with fp8 dtype
        header = {"model.weight": {"dtype": "F8_E4M3", "shape": [1024, 1024], "data_offsets": [0, 2097152]}}
        header_bytes = json.dumps(header).encode()
        header_size_bytes = struct.pack("<Q", len(header_bytes))

        mock_file = MagicMock()
        mock_file.__enter__ = lambda s: s
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.side_effect = [header_size_bytes, header_bytes]
        mock_open.return_value = mock_file

        estimated = registry.estimate_model_vram("/models/sdxl_fp8.safetensors")

        # fp8 multiplier is 1.1
        assert estimated == pytest.approx(4 * 1.1 * 1024**3, rel=0.01)

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('builtins.open', side_effect=OSError("unreadable"))
    def test_estimate_vram_header_read_error_falls_back(self, mock_open, mock_getsize, mock_exists, registry):
        """Header read failure falls back to 1.2 multiplier."""
        mock_exists.return_value = True
        mock_getsize.return_value = 6 * 1024**3

        estimated = registry.estimate_model_vram("/models/broken.safetensors")

        assert estimated == pytest.approx(6 * 1.2 * 1024**3, rel=0.01)


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
