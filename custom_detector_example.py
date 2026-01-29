#!/usr/bin/env python3
"""
Example: How to add custom detection logic to the model detector.

This demonstrates the extensible interceptor pattern.
"""

from model_detector import (
    ModelDetector,
    BaseDetector,
    ModelInfo,
    ModelVariant,
    detect_model,
)
from pathlib import Path
from typing import Dict, Any
import json


# ============================================================================
# Example 1: LCM Detector
# ============================================================================

class LCMDetector(BaseDetector):
    """
    Detects if a model is an LCM (Latent Consistency Model) variant.

    LCM models can run with fewer steps (1-8 steps).
    """

    def __init__(self):
        super().__init__("LCMDetector")

    def can_handle(self, path: str) -> bool:
        # Can handle any format
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        path_lower = path.lower()

        # Check filename for LCM indicators
        if "lcm" in path_lower or "latent-consistency" in path_lower:
            info.metadata["is_lcm"] = True
            info.metadata["recommended_steps"] = "4-8"
            info.metadata["recommended_guidance"] = "1.0"
            self._mark_detection(info)

        return info


# ============================================================================
# Example 2: Turbo Detector
# ============================================================================

class TurboDetector(BaseDetector):
    """
    Detects SDXL-Turbo or SD-Turbo variants.

    Turbo models are optimized for 1-step generation.
    """

    def __init__(self):
        super().__init__("TurboDetector")

    def can_handle(self, path: str) -> bool:
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        path_lower = path.lower()

        if "turbo" in path_lower:
            info.metadata["is_turbo"] = True
            info.metadata["recommended_steps"] = "1-2"
            info.metadata["recommended_guidance"] = "0.0"

            # Refine variant if we know it's SDXL
            if info.variant == ModelVariant.SDXL_BASE:
                info.metadata["turbo_type"] = "sdxl-turbo"
            elif info.variant in [ModelVariant.SD15, ModelVariant.SD21]:
                info.metadata["turbo_type"] = "sd-turbo"

            self._mark_detection(info)

        return info


# ============================================================================
# Example 3: Refiner Detector
# ============================================================================

class RefinerDetector(BaseDetector):
    """
    Detects SDXL Refiner models.

    Refiners are used for upscaling/refining SDXL base outputs.
    """

    def __init__(self):
        super().__init__("RefinerDetector")

    def can_handle(self, path: str) -> bool:
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        path_lower = path.lower()

        # Check for refiner indicators
        if "refiner" in path_lower or "refinement" in path_lower:
            if info.cross_attention_dim == 2048:
                info.variant = ModelVariant.SDXL_REFINER
                info.metadata["is_refiner"] = True
                info.metadata["use_with"] = "sdxl-base"
                self._mark_detection(info)

        return info


# ============================================================================
# Example 4: License Detector
# ============================================================================

class LicenseDetector(BaseDetector):
    """
    Detects model license from metadata or filename.

    Useful for compliance tracking.
    """

    def __init__(self):
        super().__init__("LicenseDetector")

    def can_handle(self, path: str) -> bool:
        # Only handle diffusers (has metadata)
        return Path(path).is_dir() and (Path(path) / "model_index.json").exists()

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        model_card_path = Path(path) / "README.md"

        if model_card_path.exists():
            content = model_card_path.read_text().lower()

            # Check for license mentions
            licenses = {
                "creativeml-openrail-m": ["openrail", "creativeml"],
                "apache-2.0": ["apache", "apache-2.0"],
                "mit": ["mit license"],
                "cc-by-nc": ["cc-by-nc", "non-commercial"],
            }

            for license_name, indicators in licenses.items():
                if any(ind in content for ind in indicators):
                    info.metadata["license"] = license_name
                    self._mark_detection(info)
                    break

        return info


# ============================================================================
# Example 5: LoRA Strength Analyzer
# ============================================================================

class LoRAStrengthAnalyzer(BaseDetector):
    """
    Analyzes LoRA tensor magnitudes to suggest optimal strength.

    This is a more advanced example showing data analysis.
    """

    def __init__(self):
        super().__init__("LoRAStrengthAnalyzer")

    def can_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".safetensors"

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        if not info.is_lora:
            return info

        try:
            from safetensors import safe_open
            import numpy as np

            magnitudes = []

            with safe_open(path, framework="pt", device="cpu") as f:
                keys = list(f.keys())

                # Sample a few LoRA tensors
                lora_keys = [k for k in keys if "lora" in k.lower()][:10]

                for key in lora_keys:
                    try:
                        tensor = f.get_tensor(key)
                        # Convert to numpy and get magnitude
                        magnitude = np.abs(tensor.numpy()).mean()
                        magnitudes.append(float(magnitude))
                    except Exception:
                        pass

            if magnitudes:
                avg_magnitude = np.mean(magnitudes)

                # Suggest strength based on magnitude
                if avg_magnitude < 0.01:
                    suggested_strength = [0.8, 1.0, 1.2, 1.5]
                elif avg_magnitude < 0.05:
                    suggested_strength = [0.6, 0.8, 1.0, 1.2]
                else:
                    suggested_strength = [0.4, 0.6, 0.8, 1.0]

                info.metadata["lora_avg_magnitude"] = avg_magnitude
                info.metadata["suggested_strength_levels"] = suggested_strength
                self._mark_detection(info)

        except Exception as e:
            info.metadata["lora_analysis_error"] = str(e)

        return info


# ============================================================================
# Usage Examples
# ============================================================================

def example_basic_detection():
    """Example 1: Basic detection with default detectors."""
    print("Example 1: Basic Detection")
    print("-" * 60)

    info = detect_model("/path/to/model.safetensors")
    print(info.to_json())
    print()


def example_custom_detector():
    """Example 2: Adding custom detectors."""
    print("Example 2: Custom Detectors")
    print("-" * 60)

    # Create detector with custom logic
    detector = ModelDetector()

    # Add custom detectors to the stack
    detector.add_detector(LCMDetector())
    detector.add_detector(TurboDetector())
    detector.add_detector(RefinerDetector())
    detector.add_detector(LicenseDetector())
    detector.add_detector(LoRAStrengthAnalyzer())

    # Detect
    info = detector.detect("/path/to/sdxl-turbo.safetensors")

    # Now info will have additional metadata
    print(info.to_json())
    print()


def example_integration():
    """Example 3: Integration with server loading logic."""
    print("Example 3: Server Integration")
    print("-" * 60)

    def load_model_with_detection(model_path: str):
        """Load model using detected configuration."""
        # Create detector with all custom logic
        detector = ModelDetector()
        detector.add_detector(LCMDetector())
        detector.add_detector(TurboDetector())

        # Detect model
        info = detector.detect(model_path)

        # Choose worker based on detection
        if info.variant.is_sdxl:
            from backends.cuda_worker import DiffusersSDXLCudaWorker
            worker = DiffusersSDXLCudaWorker(worker_id=0)
        elif info.variant.is_sd15:
            from backends.cuda_worker import DiffusersCudaWorker
            worker = DiffusersCudaWorker(worker_id=0)
        else:
            raise ValueError(f"Unknown variant: {info.variant}")

        # Configure based on metadata
        config = {
            "worker": worker,
            "default_steps": 4,  # Default
            "default_guidance": 7.5,  # Default
        }

        # Override with detected settings
        if info.metadata.get("is_lcm"):
            config["default_steps"] = 4
            config["default_guidance"] = 1.0
        elif info.metadata.get("is_turbo"):
            config["default_steps"] = 1
            config["default_guidance"] = 0.0

        return config

    # Example usage
    config = load_model_with_detection("sdxl-lcm.safetensors")
    print(f"Worker: {type(config['worker']).__name__}")
    print(f"Steps: {config['default_steps']}")
    print(f"Guidance: {config['default_guidance']}")
    print()


def example_lora_registration():
    """Example 4: Auto-register LoRAs with detected settings."""
    print("Example 4: LoRA Auto-Registration")
    print("-" * 60)

    def register_lora(lora_path: str) -> Dict[str, Any]:
        """Automatically register a LoRA with detected settings."""
        # Detect with strength analyzer
        detector = ModelDetector()
        detector.add_detector(LoRAStrengthAnalyzer())

        info = detector.detect(lora_path)

        if not info.is_lora:
            raise ValueError(f"Not a LoRA: {lora_path}")

        # Generate StyleDef config
        lora_name = Path(lora_path).stem.replace("-", "_").replace(" ", "_").lower()

        style_def = {
            "id": lora_name,
            "title": lora_name.replace("_", " ").title(),
            "lora_path": lora_path,
            "adapter_name": f"style_{lora_name}",
            "levels": info.metadata.get("suggested_strength_levels", [0.5, 0.75, 1.0, 1.25]),
            "required_cross_attention_dim": info.required_cross_attention_dim,
        }

        return style_def

    # Example
    style_def = register_lora("/models/loras/anime_style.safetensors")
    print(json.dumps(style_def, indent=2))
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Custom Detector Examples")
    print("=" * 60)
    print()

    # Run examples
    # example_basic_detection()
    # example_custom_detector()
    # example_integration()
    example_lora_registration()

    print("\nTo use these in your code:")
    print("1. Import: from custom_detector_example import LCMDetector")
    print("2. Create: detector = ModelDetector()")
    print("3. Add: detector.add_detector(LCMDetector())")
    print("4. Detect: info = detector.detect(path)")
