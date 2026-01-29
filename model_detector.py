#!/usr/bin/env python3
"""
Extensible Model Type Detection System

Architecture: Stack of detection interceptors (plugins)
- Each detector analyzes specific aspects
- Results are merged into final ModelInfo
- Easy to add new detectors without modifying existing code

Usage:
    from model_detector import ModelDetector, detect_model

    # Quick detection
    info = detect_model("/path/to/model.safetensors")
    print(info.to_json())

    # Custom detector stack
    detector = ModelDetector()
    detector.add_detector(CustomDetector())
    info = detector.detect("/path/to/model.safetensors")

Supports:
    - .safetensors and .ckpt checkpoints
    - Diffusers directories
    - LoRA files (both formats)
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, List, Any, Protocol
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum


# ============================================================================
# Data Types
# ============================================================================

class ModelVariant(Enum):
    """Supported model variants."""
    SD15 = "sd15"
    SD20 = "sd20"
    SD21 = "sd21"
    SDXL_BASE = "sdxl-base"
    SDXL_REFINER = "sdxl-refiner"
    LORA_SD15 = "lora-sd15"
    LORA_SDXL = "lora-sdxl"
    UNKNOWN = "unknown"

    @property
    def is_sdxl(self) -> bool:
        return self in (ModelVariant.SDXL_BASE, ModelVariant.SDXL_REFINER, ModelVariant.LORA_SDXL)

    @property
    def is_sd15(self) -> bool:
        return self in (ModelVariant.SD15, ModelVariant.SD20, ModelVariant.SD21, ModelVariant.LORA_SD15)

    @property
    def is_lora(self) -> bool:
        return self in (ModelVariant.LORA_SD15, ModelVariant.LORA_SDXL)


@dataclass
class ModelInfo:
    """Complete model detection result."""
    path: str
    variant: ModelVariant = ModelVariant.UNKNOWN

    # Architecture
    cross_attention_dim: Optional[int] = None
    text_encoder_hidden_size: Optional[int] = None
    text_encoder_2_hidden_size: Optional[int] = None
    unet_in_channels: Optional[int] = None
    unet_out_channels: Optional[int] = None
    vae_latent_channels: Optional[int] = None

    # Format
    format: str = "unknown"  # safetensors, checkpoint, diffusers
    is_lora: bool = False

    # Metadata
    confidence: float = 0.0  # 0.0 to 1.0
    detected_by: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Compatibility
    compatible_worker: Optional[str] = None
    required_cross_attention_dim: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "path": self.path,
            "variant": self.variant.value,
            "cross_attention_dim": self.cross_attention_dim,
            "text_encoder_hidden_size": self.text_encoder_hidden_size,
            "text_encoder_2_hidden_size": self.text_encoder_2_hidden_size,
            "unet_in_channels": self.unet_in_channels,
            "format": self.format,
            "is_lora": self.is_lora,
            "confidence": self.confidence,
            "detected_by": self.detected_by,
            "compatible_worker": self.compatible_worker,
            "required_cross_attention_dim": self.required_cross_attention_dim,
            "metadata": self.metadata,
        }
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Detector Protocol (Interface)
# ============================================================================

class Detector(Protocol):
    """
    Protocol for detection interceptors.

    Each detector:
    1. Checks if it can handle the file/directory
    2. Extracts information it knows about
    3. Updates ModelInfo with findings
    4. Returns confidence level
    """

    name: str

    def can_handle(self, path: str) -> bool:
        """Check if this detector can handle the given path."""
        ...

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        """
        Detect model information and update info object.

        Args:
            path: Path to model file or directory
            info: ModelInfo to update (may contain info from previous detectors)

        Returns:
            Updated ModelInfo with new findings
        """
        ...


# ============================================================================
# Base Detector Class
# ============================================================================

class BaseDetector(ABC):
    """Base class for detectors with common functionality."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def can_handle(self, path: str) -> bool:
        """Check if this detector can handle the given path."""
        pass

    @abstractmethod
    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        """Detect and update model information."""
        pass

    def _mark_detection(self, info: ModelInfo):
        """Mark that this detector contributed to the result."""
        if self.name not in info.detected_by:
            info.detected_by.append(self.name)


# ============================================================================
# Concrete Detectors
# ============================================================================

class SafetensorsDetector(BaseDetector):
    """Detects model type from .safetensors files."""

    def __init__(self):
        super().__init__("SafetensorsDetector")

    def can_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".safetensors"

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        try:
            from safetensors import safe_open
        except ImportError:
            return info

        info.format = "safetensors"

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())

            # Check if LoRA
            if any("lora" in k.lower() for k in keys):
                info.is_lora = True
                self._detect_lora(keys, info)
                self._mark_detection(info)
                return info

            # Extract architecture info
            self._extract_cross_attention(keys, f, info)
            self._extract_text_encoder_info(keys, f, info)
            self._extract_unet_info(keys, f, info)

        self._mark_detection(info)
        return info

    def _detect_lora(self, keys: List[str], info: ModelInfo):
        """Detect LoRA type."""
        has_te2 = any("text_encoder_2" in k or "lora_te2" in k for k in keys)

        if has_te2:
            info.variant = ModelVariant.LORA_SDXL
            info.cross_attention_dim = 2048
            info.confidence = 0.9
        else:
            info.variant = ModelVariant.LORA_SD15
            info.cross_attention_dim = 768
            info.confidence = 0.9

    def _extract_cross_attention(self, keys: List[str], f, info: ModelInfo):
        """Extract cross-attention dimension from UNet."""
        # Try middle block first
        for key in keys:
            if "middle_block" in key and "transformer_blocks" in key and "attn2.to_k.weight" in key:
                try:
                    tensor = f.get_tensor(key)
                    info.cross_attention_dim = tensor.shape[0]
                    return
                except Exception:
                    pass

        # Try input blocks
        for key in keys:
            if "input_blocks" in key and "transformer_blocks" in key and "attn2.to_k.weight" in key:
                try:
                    tensor = f.get_tensor(key)
                    info.cross_attention_dim = tensor.shape[0]
                    return
                except Exception:
                    pass

    def _extract_text_encoder_info(self, keys: List[str], f, info: ModelInfo):
        """Extract text encoder configuration."""
        # Check for dual text encoders
        has_te1 = any("text_encoder." in k or "conditioner.embedders.0" in k for k in keys)
        has_te2 = any("text_encoder_2." in k or "conditioner.embedders.1" in k for k in keys)

        if has_te2:
            info.metadata["has_dual_text_encoders"] = True

        # Get hidden sizes
        for key in keys:
            if "text_model.encoder.layers.0.self_attn.k_proj.weight" in key:
                try:
                    tensor = f.get_tensor(key)
                    if "text_encoder." in key:
                        info.text_encoder_hidden_size = tensor.shape[1]
                    elif "text_encoder_2." in key:
                        info.text_encoder_2_hidden_size = tensor.shape[1]
                except Exception:
                    pass

    def _extract_unet_info(self, keys: List[str], f, info: ModelInfo):
        """Extract UNet configuration."""
        for key in keys:
            if "input_blocks.0.0.weight" in key or "conv_in.weight" in key:
                try:
                    tensor = f.get_tensor(key)
                    info.unet_in_channels = tensor.shape[1]
                    return
                except Exception:
                    pass


class DiffusersDetector(BaseDetector):
    """Detects model type from diffusers directories."""

    def __init__(self):
        super().__init__("DiffusersDetector")

    def can_handle(self, path: str) -> bool:
        path_obj = Path(path)
        return path_obj.is_dir() and (path_obj / "model_index.json").exists()

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        info.format = "diffusers"
        path_obj = Path(path)

        # Read model_index.json
        with open(path_obj / "model_index.json", "r") as f:
            model_index = json.load(f)

        # Check for dual text encoders
        if "text_encoder_2" in model_index:
            info.metadata["has_dual_text_encoders"] = True

        # Read UNet config
        unet_config_path = path_obj / "unet" / "config.json"
        if unet_config_path.exists():
            with open(unet_config_path, "r") as f:
                unet_config = json.load(f)
                info.cross_attention_dim = unet_config.get("cross_attention_dim")
                info.unet_in_channels = unet_config.get("in_channels")
                info.unet_out_channels = unet_config.get("out_channels")

        # Read text encoder config
        te_config_path = path_obj / "text_encoder" / "config.json"
        if te_config_path.exists():
            with open(te_config_path, "r") as f:
                te_config = json.load(f)
                info.text_encoder_hidden_size = te_config.get("hidden_size")

        # Read text encoder 2 config (SDXL)
        te2_config_path = path_obj / "text_encoder_2" / "config.json"
        if te2_config_path.exists():
            with open(te2_config_path, "r") as f:
                te2_config = json.load(f)
                info.text_encoder_2_hidden_size = te2_config.get("hidden_size")

        self._mark_detection(info)
        return info


class CheckpointDetector(BaseDetector):
    """Detects model type from .ckpt/.pt/.pth files."""

    def __init__(self):
        super().__init__("CheckpointDetector")

    def can_handle(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in [".ckpt", ".pt", ".pth"]

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        try:
            import torch
        except ImportError:
            return info

        info.format = "checkpoint"

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                keys = checkpoint.get("state_dict", checkpoint).keys()
            else:
                keys = []

            keys_list = list(keys)

            # Check for LoRA
            if any("lora" in k.lower() for k in keys_list):
                info.is_lora = True
                has_te2 = any("text_encoder_2" in k for k in keys_list)
                info.variant = ModelVariant.LORA_SDXL if has_te2 else ModelVariant.LORA_SD15
                info.cross_attention_dim = 2048 if has_te2 else 768
                info.confidence = 0.8
            else:
                # Check for dual text encoders
                has_te2 = any("text_encoder_2" in k or "conditioner.embedders.1" in k for k in keys_list)
                info.metadata["has_dual_text_encoders"] = has_te2

                # Infer cross-attention from presence of te2
                if has_te2:
                    info.cross_attention_dim = 2048
                else:
                    # Assume SD1.5 for single text encoder
                    info.cross_attention_dim = 768

            self._mark_detection(info)

        except Exception as e:
            info.metadata["checkpoint_error"] = str(e)

        return info


class VariantClassifier(BaseDetector):
    """
    Classifies model variant based on collected information.

    This should run AFTER other detectors have gathered architectural info.
    """

    def __init__(self):
        super().__init__("VariantClassifier")

    def can_handle(self, path: str) -> bool:
        # Always can handle (runs on all paths)
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        # Skip if already classified (e.g., by LoRA detector)
        if info.variant != ModelVariant.UNKNOWN:
            self._mark_detection(info)
            return info

        # Classify based on cross-attention dimension
        if info.cross_attention_dim == 2048:
            # SDXL
            has_dual = info.metadata.get("has_dual_text_encoders", False)
            if has_dual:
                # TODO: Detect refiner vs base (refiner uses different TE setup)
                info.variant = ModelVariant.SDXL_BASE
                info.confidence = max(info.confidence, 0.95)
            else:
                info.variant = ModelVariant.SDXL_BASE
                info.confidence = max(info.confidence, 0.85)

        elif info.cross_attention_dim == 768:
            # SD 1.x
            # Differentiate by text encoder hidden size
            if info.text_encoder_hidden_size == 1024:
                info.variant = ModelVariant.SD21
                info.confidence = max(info.confidence, 0.95)
            else:
                info.variant = ModelVariant.SD15
                info.confidence = max(info.confidence, 0.9)

        elif info.cross_attention_dim == 1024:
            # SD 2.0
            info.variant = ModelVariant.SD20
            info.confidence = max(info.confidence, 0.95)

        else:
            # Unknown - try to infer from other features
            if info.text_encoder_2_hidden_size or info.metadata.get("has_dual_text_encoders"):
                info.variant = ModelVariant.SDXL_BASE
                info.confidence = max(info.confidence, 0.6)
            elif info.text_encoder_hidden_size == 768:
                info.variant = ModelVariant.SD15
                info.confidence = max(info.confidence, 0.5)
            elif info.text_encoder_hidden_size == 1024:
                info.variant = ModelVariant.SD21
                info.confidence = max(info.confidence, 0.5)

        self._mark_detection(info)
        return info


class CompatibilityResolver(BaseDetector):
    """
    Resolves worker compatibility based on variant.

    This should run LAST after variant is determined.
    """

    def __init__(self):
        super().__init__("CompatibilityResolver")

    def can_handle(self, path: str) -> bool:
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        if info.variant.is_sdxl:
            info.compatible_worker = "backends.cuda_worker.DiffusersSDXLCudaWorker"
            info.required_cross_attention_dim = 2048
        elif info.variant.is_sd15:
            info.compatible_worker = "backends.cuda_worker.DiffusersCudaWorker"
            info.required_cross_attention_dim = 768
        else:
            info.compatible_worker = None
            info.required_cross_attention_dim = None

        self._mark_detection(info)
        return info


# ============================================================================
# Main Detector Class
# ============================================================================

class ModelDetector:
    """
    Main detector class that chains multiple detection interceptors.

    Usage:
        detector = ModelDetector()
        detector.add_detector(CustomDetector())  # Add custom detector
        info = detector.detect("/path/to/model")
    """

    def __init__(self):
        self.detectors: List[Detector] = []
        self._setup_default_detectors()

    def _setup_default_detectors(self):
        """Setup default detector stack."""
        # Format-specific detectors (extract raw info)
        self.add_detector(SafetensorsDetector())
        self.add_detector(DiffusersDetector())
        self.add_detector(CheckpointDetector())

        # Analysis detectors (interpret info)
        self.add_detector(VariantClassifier())
        self.add_detector(CompatibilityResolver())

    def add_detector(self, detector: Detector):
        """Add a detector to the stack."""
        self.detectors.append(detector)

    def detect(self, path: str) -> ModelInfo:
        """
        Run detection pipeline on a model.

        Args:
            path: Path to model file or directory

        Returns:
            ModelInfo with detection results
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        # Initialize info
        info = ModelInfo(path=path)

        # Run each detector in sequence
        for detector in self.detectors:
            if detector.can_handle(path):
                info = detector.detect(path, info)

        return info


# ============================================================================
# Convenience Function
# ============================================================================

def detect_model(path: str) -> ModelInfo:
    """
    Quick detection using default detector stack.

    Args:
        path: Path to model file or directory

    Returns:
        ModelInfo with detection results
    """
    detector = ModelDetector()
    return detector.detect(path)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect Stable Diffusion model variant (SD1.5, SD2.x, SDXL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("path", help="Path to model file or directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")

    args = parser.parse_args()

    try:
        info = detect_model(args.path)

        if args.json or args.pretty:
            indent = 2 if args.pretty else None
            print(info.to_json(indent=indent))
        else:
            # Simple text output
            print(f"Variant: {info.variant.value}")
            print(f"Cross-Attention Dim: {info.cross_attention_dim}")
            print(f"Format: {info.format}")
            print(f"Is LoRA: {info.is_lora}")
            print(f"Confidence: {info.confidence:.2f}")
            print(f"Compatible Worker: {info.compatible_worker}")
            print(f"Detected By: {', '.join(info.detected_by)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
