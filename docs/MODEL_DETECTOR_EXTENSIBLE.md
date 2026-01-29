## Extensible Model Detection System

### Architecture Overview

The model detector uses a **stack of detection interceptors** (plugins) that analyze models in sequence. Each detector adds information to a shared `ModelInfo` object.

```
┌─────────────────────────────────────────────────────────┐
│                    ModelDetector                         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 1. SafetensorsDetector                             │ │
│  │    → Extracts architecture from .safetensors       │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 2. DiffusersDetector                               │ │
│  │    → Reads config.json files                       │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 3. CheckpointDetector                              │ │
│  │    → Analyzes .ckpt files                          │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 4. VariantClassifier                               │ │
│  │    → Determines SD1.5/SD2.x/SDXL                  │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 5. CompatibilityResolver                           │ │
│  │    → Maps to worker class                          │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Custom Detectors (optional)                        │ │
│  │    → LCMDetector, TurboDetector, etc.             │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                         ↓
                   ModelInfo (JSON)
```

## Quick Start

### Basic Usage

```python
from utils.model_detector import detect_model

# Detect model
info = detect_model("/path/to/model.safetensors")

# JSON output for automation
print(info.to_json())
```

### Command Line

```bash
# JSON output (for automation)
python -m utils.model_detector model.safetensors --json

# Pretty JSON
python -m utils.model_detector model.safetensors --pretty

# Simple text
python -m utils.model_detector model.safetensors
```

## ModelInfo Output

```json
{
  "path": "/models/sdxl-base.safetensors",
  "variant": "sdxl-base",
  "cross_attention_dim": 2048,
  "text_encoder_hidden_size": null,
  "text_encoder_2_hidden_size": null,
  "unet_in_channels": 4,
  "format": "safetensors",
  "is_lora": false,
  "confidence": 0.95,
  "detected_by": [
    "SafetensorsDetector",
    "VariantClassifier",
    "CompatibilityResolver"
  ],
  "compatible_worker": "backends.cuda_worker.DiffusersSDXLCudaWorker",
  "required_cross_attention_dim": 2048,
  "metadata": {
    "has_dual_text_encoders": true
  }
}
```

## Supported Variants

| Variant | Cross-Attention Dim | Worker |
|---------|---------------------|--------|
| `sd15` | 768 | `DiffusersCudaWorker` |
| `sd20` | 1024 | `DiffusersCudaWorker` |
| `sd21` | 768 | `DiffusersCudaWorker` |
| `sdxl-base` | 2048 | `DiffusersSDXLCudaWorker` |
| `sdxl-refiner` | 2048 | `DiffusersSDXLCudaWorker` |
| `lora-sd15` | 768 | Compatible with SD1.5 |
| `lora-sdxl` | 2048 | Compatible with SDXL |

## Adding Custom Detectors

### Step 1: Create Detector Class

```python
from utils.model_detector import BaseDetector, ModelInfo

class MyCustomDetector(BaseDetector):
    def __init__(self):
        super().__init__("MyCustomDetector")

    def can_handle(self, path: str) -> bool:
        # Return True if this detector can analyze this path
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        # Add your detection logic
        info.metadata["my_custom_field"] = "value"

        # Mark that this detector contributed
        self._mark_detection(info)

        return info
```

### Step 2: Use Custom Detector

```python
from utils.model_detector import ModelDetector

# Create detector
detector = ModelDetector()

# Add custom detector to stack
detector.add_detector(MyCustomDetector())

# Detect
info = detector.detect("/path/to/model")
```

## Example Custom Detectors

### LCM Detector

Detects Latent Consistency Models:

```python
class LCMDetector(BaseDetector):
    def __init__(self):
        super().__init__("LCMDetector")

    def can_handle(self, path: str) -> bool:
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        if "lcm" in path.lower():
            info.metadata["is_lcm"] = True
            info.metadata["recommended_steps"] = "4-8"
            info.metadata["recommended_guidance"] = "1.0"
            self._mark_detection(info)
        return info
```

### Turbo Detector

Detects SDXL-Turbo models:

```python
class TurboDetector(BaseDetector):
    def __init__(self):
        super().__init__("TurboDetector")

    def can_handle(self, path: str) -> bool:
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        if "turbo" in path.lower():
            info.metadata["is_turbo"] = True
            info.metadata["recommended_steps"] = "1-2"
            info.metadata["recommended_guidance"] = "0.0"
            self._mark_detection(info)
        return info
```

See `utils/custom_detector_example.py` for more examples.

## Integration with Server

### Auto-Configure Worker

```python
from utils.model_detector import detect_model

def load_model_auto(model_path: str):
    """Automatically select worker based on detection."""
    info = detect_model(model_path)

    if info.variant.is_sdxl:
        from backends.cuda_worker import DiffusersSDXLCudaWorker
        return DiffusersSDXLCudaWorker(worker_id=0)
    elif info.variant.is_sd15:
        from backends.cuda_worker import DiffusersCudaWorker
        return DiffusersCudaWorker(worker_id=0)
    else:
        raise ValueError(f"Unsupported variant: {info.variant}")
```

### Auto-Register LoRAs

```python
def register_lora_auto(lora_path: str):
    """Automatically register LoRA with detected settings."""
    from utils.model_detector import detect_model

    info = detect_model(lora_path)

    if not info.is_lora:
        raise ValueError("Not a LoRA")

    # Generate StyleDef
    lora_name = Path(lora_path).stem.lower()

    return {
        "id": lora_name,
        "title": lora_name.replace("_", " ").title(),
        "lora_path": lora_path,
        "adapter_name": f"style_{lora_name}",
        "levels": [0.5, 0.75, 1.0, 1.25],
        "required_cross_attention_dim": info.required_cross_attention_dim,
    }
```

## Detector Protocol

All detectors must implement this protocol:

```python
class Detector(Protocol):
    name: str

    def can_handle(self, path: str) -> bool:
        """Return True if this detector can handle the path."""
        ...

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        """Analyze model and update info object."""
        ...
```

## Best Practices

### 1. Check Before Modifying

Only update fields if you have new information:

```python
# Good: Only set if not already set
if info.cross_attention_dim is None:
    info.cross_attention_dim = 2048

# Avoid: Overwriting existing detection
info.cross_attention_dim = 2048  # May override previous detector
```

### 2. Use Metadata for Custom Fields

Store custom fields in `metadata` dict:

```python
# Good: Custom fields in metadata
info.metadata["is_lcm"] = True
info.metadata["recommended_steps"] = "4-8"

# Avoid: Adding fields directly (won't serialize)
info.is_lcm = True  # Won't appear in to_dict()
```

### 3. Always Mark Detection

Call `_mark_detection()` when contributing:

```python
def detect(self, path: str, info: ModelInfo) -> ModelInfo:
    # ... your logic ...
    self._mark_detection(info)  # Important!
    return info
```

### 4. Handle Errors Gracefully

Store errors in metadata:

```python
try:
    # ... analysis ...
except Exception as e:
    info.metadata["my_detector_error"] = str(e)
```

## Detector Ordering

Detectors run in order added. Typical stack:

1. **Format Detectors** (SafetensorsDetector, DiffusersDetector)
   - Extract raw architectural information
   - Don't interpret, just collect data

2. **Classifier Detectors** (VariantClassifier)
   - Interpret collected data
   - Determine variant

3. **Resolver Detectors** (CompatibilityResolver)
   - Map variant to worker
   - Determine compatibility

4. **Custom Detectors** (LCMDetector, TurboDetector)
   - Add specialized metadata
   - Provide recommendations

## Testing Custom Detectors

```python
import pytest
from utils.model_detector import ModelDetector, ModelInfo

def test_my_detector():
    detector = ModelDetector()
    detector.add_detector(MyCustomDetector())

    info = detector.detect("/path/to/test/model.safetensors")

    assert "my_custom_field" in info.metadata
    assert "MyCustomDetector" in info.detected_by
```

## Performance

| Format | Size | Detection Time |
|--------|------|----------------|
| .safetensors | 6.9 GB | ~2 seconds |
| .ckpt | 6.9 GB | ~30 seconds |
| Diffusers | Multiple files | ~1 second |

## Advanced: Conditional Detectors

Run detectors only when conditions are met:

```python
class ConditionalDetector(BaseDetector):
    def can_handle(self, path: str) -> bool:
        # Only run for SDXL models
        # Note: Can't access info here, check in detect()
        return True

    def detect(self, path: str, info: ModelInfo) -> ModelInfo:
        # Check condition first
        if info.variant != ModelVariant.SDXL_BASE:
            return info  # Skip

        # ... SDXL-specific logic ...
        self._mark_detection(info)
        return info
```

## Summary

**Architecture**: Stack of interceptor plugins
**Extensibility**: Add detectors without modifying core
**Output**: JSON for automation
**Use Case**: Check model before loading

**Main API**:
```python
from utils.model_detector import detect_model
info = detect_model(path)
print(info.to_json())
```

**Custom Detectors**:
```python
from utils.model_detector import ModelDetector
detector = ModelDetector()
detector.add_detector(MyDetector())
info = detector.detect(path)
```
