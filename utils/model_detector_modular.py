"""
Modular model detection system with improved separation of concerns.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
import os
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Standardized model information structure."""
    path: str
    model_type: str = "unknown"
    architecture: str = "unknown"
    variant: str = "unknown"
    format: str = "unknown"  # safetensors, diffusers, checkpoint
    compatibility: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


class Detector(Protocol):
    """Protocol for detection interceptors."""
    
    @abstractmethod
    def can_handle(self, path: str) -> bool:
        """Check if this detector can handle the given path."""
        pass
    
    @abstractmethod
    def detect(self, path: str, info: ModelInfo) -> float:
        """
        Detect model information and update ModelInfo.
        
        Returns:
            Confidence level (0.0 - 1.0)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this detector."""
        pass


class BaseDetector(ABC):
    """Base class for detectors with common functionality."""
    
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @abstractmethod
    def can_handle(self, path: str) -> bool:
        """Check if this detector can handle the given path."""
        pass
    
    @abstractmethod
    def detect(self, path: str, info: ModelInfo) -> float:
        """
        Detect model information and update ModelInfo.
        
        Returns:
            Confidence level (0.0 - 1.0)
        """
        pass


class SafetensorsDetector(BaseDetector):
    """Detects model type from .safetensors files."""
    
    def __init__(self):
        super().__init__("safetensors")
    
    def can_handle(self, path: str) -> bool:
        return path.endswith(".safetensors")
    
    def detect(self, path: str, info: ModelInfo) -> float:
        # Implementation would go here
        # This is a simplified example
        try:
            # In a real implementation, you'd read the safetensors file
            # and extract metadata
            info.format = "safetensors"
            info.metadata["file_size"] = os.path.getsize(path)
            return 0.9
        except Exception as e:
            logger.warning(f"Failed to detect safetensors model: {e}")
            return 0.0


class DiffusersDetector(BaseDetector):
    """Detects model type from diffusers directories."""
    
    def __init__(self):
        super().__init__("diffusers")
    
    def can_handle(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return os.path.exists(os.path.join(path, "model_index.json"))
    
    def detect(self, path: str, info: ModelInfo) -> float:
        try:
            model_index_path = os.path.join(path, "model_index.json")
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
            
            info.format = "diffusers"
            info.architecture = model_index.get("_class_name", "unknown")
            info.metadata["model_index"] = model_index
            return 0.95
        except Exception as e:
            logger.warning(f"Failed to detect diffusers model: {e}")
            return 0.0


class CheckpointDetector(BaseDetector):
    """Detects model type from .ckpt/.pt/.pth files."""
    
    def __init__(self):
        super().__init__("checkpoint")
    
    def can_handle(self, path: str) -> bool:
        return any(path.endswith(ext) for ext in [".ckpt", ".pt", ".pth"])
    
    def detect(self, path: str, info: ModelInfo) -> float:
        try:
            # In a real implementation, you'd load the checkpoint
            # and analyze its structure
            info.format = "checkpoint"
            info.metadata["file_size"] = os.path.getsize(path)
            return 0.8
        except Exception as e:
            logger.warning(f"Failed to detect checkpoint model: {e}")
            return 0.0


class VariantClassifier(BaseDetector):
    """Classifies model variant based on collected information."""
    
    def __init__(self):
        super().__init__("variant_classifier")
    
    def can_handle(self, path: str) -> bool:
        # Can handle any path, runs after other detectors
        return True
    
    def detect(self, path: str, info: ModelInfo) -> float:
        # Classify based on architecture and other metadata
        arch = info.architecture.lower()
        if "lcm" in arch or "latentconsistency" in arch:
            info.variant = "lcm"
        elif "turbo" in arch:
            info.variant = "turbo"
        elif "refiner" in arch:
            info.variant = "refiner"
        else:
            info.variant = "standard"
        
        return 0.9


class CompatibilityResolver(BaseDetector):
    """Resolves worker compatibility based on variant."""
    
    def __init__(self):
        super().__init__("compatibility_resolver")
    
    def can_handle(self, path: str) -> bool:
        return True  # Runs after other detectors
    
    def detect(self, path: str, info: ModelInfo) -> float:
        # Set compatibility information based on variant
        compatibility = {}
        
        if info.variant == "lcm":
            compatibility["worker"] = "cuda_lcm"
            compatibility["steps_range"] = [1, 8]
        elif info.variant == "turbo":
            compatibility["worker"] = "cuda_turbo"
            compatibility["steps_range"] = [1, 1]
        elif info.variant == "refiner":
            compatibility["worker"] = "sdxl"
            compatibility["steps_range"] = [20, 50]
        else:
            compatibility["worker"] = "cuda"
            compatibility["steps_range"] = [20, 50]
        
        info.compatibility = compatibility
        return 0.95


class ResolutionDetector(BaseDetector):
    """Detects sizing/resolution policy for a model."""
    
    def __init__(self):
        super().__init__("resolution_detector")
    
    def can_handle(self, path: str) -> bool:
        return True  # Can run on any model
    
    def detect(self, path: str, info: ModelInfo) -> float:
        size_policy = {
            "downsample_factor": 8,
            "divisible_by_px": 8,
            "native_resolution_px": 512,
            "latent_sample_size": 64,
            "recommended_sizes": ["512x512", "768x512", "768x768"]
        }
        
        # Adjust based on model type
        if "xl" in info.architecture.lower() or "sdxl" in info.architecture.lower():
            size_policy["native_resolution_px"] = 1024
            size_policy["latent_sample_size"] = 128
            size_policy["recommended_sizes"] = ["1024x1024", "1280x768", "1536x640"]
        
        info.metadata["size_policy"] = size_policy
        return 0.9


class ModelDetector:
    """
    Main detector class that chains multiple detection interceptors.
    
    Features:
    - Pluggable detector architecture
    - Confidence-based result aggregation
    - Ordered execution of detectors
    - Extensible design
    """
    
    def __init__(self):
        self.detectors: List[Detector] = []
        self._default_detectors_registered = False
    
    def add_detector(self, detector: Detector) -> None:
        """Add a detector to the chain."""
        self.detectors.append(detector)
        self._default_detectors_registered = False
    
    def _register_default_detectors(self) -> None:
        """Register default detectors if none are registered."""
        if not self.detectors:
            self.add_detector(SafetensorsDetector())
            self.add_detector(DiffusersDetector())
            self.add_detector(CheckpointDetector())
            self.add_detector(VariantClassifier())
            self.add_detector(CompatibilityResolver())
            self.add_detector(ResolutionDetector())
            self._default_detectors_registered = True
    
    def detect(self, path: str) -> ModelInfo:
        """
        Detect model information by running all applicable detectors.
        
        Returns:
            ModelInfo with aggregated information
        """
        if not self._default_detectors_registered:
            self._register_default_detectors()
        
        info = ModelInfo(path=path)
        
        # Run detectors in order
        for detector in self.detectors:
            if detector.can_handle(path):
                try:
                    confidence = detector.detect(path, info)
                    info.confidence = max(info.confidence, confidence)
                except Exception as e:
                    logger.warning(f"Detector {detector.name} failed on {path}: {e}")
                    continue
        
        return info


def detect_model(path: str) -> ModelInfo:
    """
    Convenience function to detect a single model.
    
    Args:
        path: Path to model file or directory
        
    Returns:
        ModelInfo with detection results
    """
    detector = ModelDetector()
    return detector.detect(path)
