# Model Type Detection Guide

## Overview

The `detect_model_type.py` tool automatically detects if a Stable Diffusion model or LoRA is compatible with SD1.5 or SDXL without loading the entire model into memory.

## Why You Need This

**Problem**: You have unlabeled models and don't know which worker to use:
- `DiffusersCudaWorker` for SD1.5 models (cross_attention_dim=768)
- `DiffusersSDXLCudaWorker` for SDXL models (cross_attention_dim=2048)

**Solution**: This tool analyzes the model architecture and tells you exactly which worker to use.

## Quick Start

### Single Model Detection

```bash
python detect_model_type.py /path/to/model.safetensors
```

### Scan Directory

```bash
# Scan all models in directory
python detect_model_type.py --scan /models/

# Scan recursively
python detect_model_type.py --scan /models/ --recursive
```

### JSON Output

```bash
python detect_model_type.py model.safetensors --json > model_info.json
```

## Example Output

```
======================================================================
Model Detection Results
======================================================================

File: /models/sdxl-1.0-base.safetensors
Format: Checkpoint
Model Type: SDXL Base
Confidence: high

Architecture Details:
  Cross-Attention Dim: 2048
  Text Encoder Hidden Size: Unknown
  Dual Text Encoders: Yes
  UNet In Channels: 4

Compatibility:
  ✗ NOT compatible with DiffusersCudaWorker (SD1.5)
  ✓ Compatible with DiffusersSDXLCudaWorker
  Worker: backends.cuda_worker.DiffusersSDXLCudaWorker

======================================================================
```

## Supported Formats

### Model Formats
- ✅ `.safetensors` (recommended, fast)
- ✅ `.ckpt` (legacy PyTorch checkpoint)
- ✅ `.pt` / `.pth` (PyTorch checkpoint)
- ✅ Diffusers directories (with `model_index.json`)

### Model Types Detected
- **SD 1.5** - cross_attention_dim=768, single text encoder
- **SD 2.0/2.1** - cross_attention_dim=1024, single text encoder
- **SDXL Base** - cross_attention_dim=2048, dual text encoders
- **SDXL Refiner** - cross_attention_dim=2048, specialized text encoder
- **LoRA (SD 1.5)** - Compatible with SD1.5 models
- **LoRA (SDXL)** - Compatible with SDXL models

## Detection Method

The tool analyzes model architecture by examining:

1. **Cross-Attention Dimension** - Primary indicator
   - 768 = SD 1.5
   - 1024 = SD 2.0/2.1
   - 2048 = SDXL

2. **Text Encoder Count**
   - Single = SD 1.5/2.x
   - Dual = SDXL

3. **Text Encoder Hidden Size**
   - 768 = SD 1.5
   - 1024 = SD 2.0/2.1
   - Various = SDXL

4. **LoRA Keys** - Checks for LoRA-specific patterns
   - `text_encoder_2` = SDXL LoRA
   - Single encoder = SD1.5 LoRA

## Use Cases

### 1. Before Loading Model

```bash
# Check before adding to server config
python detect_model_type.py new_model.safetensors

# Based on output, set correct worker:
# SD1.5 -> export MODEL=new_model.safetensors (DiffusersCudaWorker)
# SDXL  -> export SDXL_MODEL=new_model.safetensors (DiffusersSDXLCudaWorker)
```

### 2. Organize Model Library

```bash
# Scan and categorize all models
python detect_model_type.py --scan /models/ --recursive --json > inventory.json

# Parse JSON and organize:
# - SD1.5 models -> /models/sd15/
# - SDXL models -> /models/sdxl/
# - SD1.5 LoRAs -> /models/loras/sd15/
# - SDXL LoRAs -> /models/loras/sdxl/
```

### 3. Verify LoRA Compatibility

```bash
# Check LoRA before adding to STYLE_REGISTRY
python detect_model_type.py /models/loras/anime_style.safetensors

# Output shows:
# - LoRA (SD 1.5) -> Add with required_cross_attention_dim=768
# - LoRA (SDXL)   -> Add with required_cross_attention_dim=2048
```

### 4. Automated Model Loading

```python
# Integration example
import json
import subprocess

result = subprocess.run(
    ["python", "detect_model_type.py", model_path, "--json"],
    capture_output=True,
    text=True
)

info = json.loads(result.stdout)

if info["cross_attention_dim"] == 2048:
    # Use SDXL worker
    worker = DiffusersSDXLCudaWorker(0)
elif info["cross_attention_dim"] == 768:
    # Use SD1.5 worker
    worker = DiffusersCudaWorker(0)
```

## Confidence Levels

- **high**: Definitive detection based on multiple indicators
- **medium**: Detected based on partial information
- **low**: Could not determine with confidence

## Adding LoRAs to Your Server

Once you detect a LoRA type, add it to `backends/styles.py`:

### SD1.5 LoRA

```python
STYLE_REGISTRY = {
    "anime": StyleDef(
        id="anime",
        title="Anime Style",
        lora_path="/models/loras/sd15/anime.safetensors",
        adapter_name="style_anime",
        levels=[0.5, 0.75, 1.0, 1.25],
        required_cross_attention_dim=768,  # SD1.5
    ),
}
```

### SDXL LoRA

```python
STYLE_REGISTRY = {
    "papercut": StyleDef(
        id="papercut",
        title="Papercut Style",
        lora_path="/models/loras/sdxl/papercut.safetensors",
        adapter_name="style_papercut",
        levels=[0.8, 0.9, 1.0, 1.15],
        required_cross_attention_dim=2048,  # SDXL
    ),
}
```

The workers automatically filter LoRAs:
- `DiffusersCudaWorker` loads only 768 LoRAs
- `DiffusersSDXLCudaWorker` loads only 2048 LoRAs

## Command Reference

### Basic Usage

```bash
# Detect single model
python detect_model_type.py model.safetensors

# Diffusers directory
python detect_model_type.py /models/stable-diffusion-xl-base-1.0/

# LoRA file
python detect_model_type.py lora.safetensors
```

### Scan Options

```bash
# Scan directory (non-recursive)
python detect_model_type.py --scan /models/

# Scan recursively
python detect_model_type.py --scan /models/ -r

# JSON output
python detect_model_type.py --scan /models/ --json

# No color (for scripts)
python detect_model_type.py model.safetensors --no-color
```

## Troubleshooting

### ImportError: safetensors

```bash
pip install safetensors
```

### ImportError: torch

```bash
pip install torch
```

### "Could not determine model type with confidence"

This means the model has an unusual architecture. Check the notes in output:
- Partial information may still be useful
- Cross-attention dim is the most reliable indicator
- Try opening an issue with model details

### Detection is Slow

- `.safetensors` files are fastest (recommended)
- `.ckpt` files require loading entire checkpoint
- Use `--no-color` for faster output in scripts

## Performance

| Format | Size | Detection Time |
|--------|------|----------------|
| .safetensors | 6.9 GB | ~2 seconds |
| .ckpt | 6.9 GB | ~30 seconds |
| Diffusers dir | Multiple files | ~1 second |
| LoRA .safetensors | 100 MB | <1 second |

## Integration Ideas

### 1. Pre-commit Hook

```bash
# .git/hooks/pre-commit
for model in $(git diff --cached --name-only | grep '\.safetensors$'); do
  python detect_model_type.py "$model" || exit 1
done
```

### 2. Model Validation Script

```bash
#!/bin/bash
# validate_models.sh
for model in /models/*.safetensors; do
  result=$(python detect_model_type.py "$model" --json)
  type=$(echo "$result" | jq -r '.model_type')

  if [ "$type" = "Unknown" ]; then
    echo "WARNING: Could not detect type for $model"
  fi
done
```

### 3. Auto-organize Script

```python
import os
import json
import shutil
from pathlib import Path

def organize_models(source_dir):
    """Organize models into SD15/SDXL directories."""
    models = Path(source_dir).glob("*.safetensors")

    for model in models:
        info = detect_model(str(model))

        if info.cross_attention_dim == 768:
            dest = Path(source_dir) / "sd15" / model.name
        elif info.cross_attention_dim == 2048:
            dest = Path(source_dir) / "sdxl" / model.name
        else:
            dest = Path(source_dir) / "unknown" / model.name

        dest.parent.mkdir(exist_ok=True)
        shutil.move(str(model), str(dest))
        print(f"Moved {model.name} -> {dest}")
```

## Summary

**Key Points:**
- ✅ Fast detection without loading full model
- ✅ Supports .safetensors, .ckpt, diffusers, LoRAs
- ✅ Tells you exactly which worker to use
- ✅ JSON output for automation
- ✅ Scan entire directories

**Main Command:**
```bash
python detect_model_type.py model.safetensors
```

**Result:**
- Know if it's SD1.5 or SDXL
- Know which worker to use
- Know if LoRA is compatible
- Organize your model library
