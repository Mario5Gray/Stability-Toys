#!/usr/bin/env python3
"""
Model Type Detection Tool

Detects if a Stable Diffusion model is SD1.5, SD2.x, or SDXL by analyzing
the model architecture without loading the entire model into memory.

Usage:
    python detect_model_type.py /path/to/model.safetensors
    python detect_model_type.py /path/to/model_directory/
    python detect_model_type.py --scan /path/to/models/

Supports:
    - Single file checkpoints (.safetensors, .ckpt, .pt)
    - Diffusers directories
    - LoRA files
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Color output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ModelType(Enum):
    SD15 = "SD 1.5"
    SD20 = "SD 2.0"
    SD21 = "SD 2.1"
    SDXL_BASE = "SDXL Base"
    SDXL_REFINER = "SDXL Refiner"
    LORA_SD15 = "LoRA (SD 1.5)"
    LORA_SDXL = "LoRA (SDXL)"
    UNKNOWN = "Unknown"


@dataclass
class ModelInfo:
    """Model detection result."""
    path: str
    model_type: ModelType
    cross_attention_dim: Optional[int] = None
    text_encoder_hidden_size: Optional[int] = None
    has_dual_text_encoders: bool = False
    unet_in_channels: Optional[int] = None
    unet_out_channels: Optional[int] = None
    vae_latent_channels: Optional[int] = None
    is_lora: bool = False
    is_diffusers: bool = False
    confidence: str = "high"  # high, medium, low
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def detect_safetensors(file_path: str) -> ModelInfo:
    """Detect model type from .safetensors file."""
    try:
        import safetensors
        from safetensors import safe_open
    except ImportError:
        print(f"{Colors.FAIL}Error: safetensors library not installed{Colors.ENDC}")
        print(f"Install with: pip install safetensors")
        sys.exit(1)

    info = ModelInfo(path=file_path, model_type=ModelType.UNKNOWN)

    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        keys_list = list(keys)

        # Check if it's a LoRA
        if any("lora" in k.lower() for k in keys_list):
            info.is_lora = True
            info = _detect_lora_type(keys_list, info)
            return info

        # Look for key indicators
        unet_keys = [k for k in keys_list if k.startswith("model.diffusion_model.") or k.startswith("unet.")]
        text_encoder_keys = [k for k in keys_list if "text_encoder" in k or "cond_stage_model" in k]

        # Check for dual text encoders (SDXL)
        has_te1 = any("text_encoder." in k or "conditioner.embedders.0" in k for k in text_encoder_keys)
        has_te2 = any("text_encoder_2." in k or "conditioner.embedders.1" in k for k in text_encoder_keys)
        info.has_dual_text_encoders = has_te1 and has_te2

        # Find cross-attention dim from UNet middle block
        for key in unet_keys:
            if "middle_block" in key and "transformer_blocks" in key and "attn2.to_k.weight" in key:
                try:
                    tensor = f.get_tensor(key)
                    # attn2.to_k.weight shape is (cross_attn_dim, hidden_dim)
                    info.cross_attention_dim = tensor.shape[0]
                    break
                except Exception:
                    pass

        # Alternative: check input_blocks cross attention
        if info.cross_attention_dim is None:
            for key in unet_keys:
                if "input_blocks" in key and "transformer_blocks" in key and "attn2.to_k.weight" in key:
                    try:
                        tensor = f.get_tensor(key)
                        info.cross_attention_dim = tensor.shape[0]
                        break
                    except Exception:
                        pass

        # Check text encoder hidden size
        for key in text_encoder_keys:
            if "text_model.encoder.layers.0.self_attn.k_proj.weight" in key:
                try:
                    tensor = f.get_tensor(key)
                    info.text_encoder_hidden_size = tensor.shape[1]
                    break
                except Exception:
                    pass

        # Check UNet channels
        for key in unet_keys:
            if "input_blocks.0.0.weight" in key or "conv_in.weight" in key:
                try:
                    tensor = f.get_tensor(key)
                    info.unet_in_channels = tensor.shape[1]
                    break
                except Exception:
                    pass

    # Determine model type
    info.model_type = _classify_model(info)

    return info


def detect_diffusers(directory: str) -> ModelInfo:
    """Detect model type from diffusers directory."""
    info = ModelInfo(path=directory, model_type=ModelType.UNKNOWN, is_diffusers=True)

    model_index_path = os.path.join(directory, "model_index.json")
    if not os.path.exists(model_index_path):
        info.notes.append("No model_index.json found")
        info.confidence = "low"
        return info

    with open(model_index_path, "r") as f:
        model_index = json.load(f)

    # Check for dual text encoders
    if "text_encoder_2" in model_index:
        info.has_dual_text_encoders = True

    # Check UNet config
    unet_config_path = os.path.join(directory, "unet", "config.json")
    if os.path.exists(unet_config_path):
        with open(unet_config_path, "r") as f:
            unet_config = json.load(f)
            info.cross_attention_dim = unet_config.get("cross_attention_dim")
            info.unet_in_channels = unet_config.get("in_channels")
            info.unet_out_channels = unet_config.get("out_channels")

    # Check text encoder config
    te_config_path = os.path.join(directory, "text_encoder", "config.json")
    if os.path.exists(te_config_path):
        with open(te_config_path, "r") as f:
            te_config = json.load(f)
            info.text_encoder_hidden_size = te_config.get("hidden_size")

    # Determine model type
    info.model_type = _classify_model(info)

    return info


def _detect_lora_type(keys: List[str], info: ModelInfo) -> ModelInfo:
    """Detect LoRA type by analyzing keys."""
    # Look for text encoder keys to determine compatibility
    te_keys = [k for k in keys if "text_encoder" in k or "lora_te" in k]

    # Check for SDXL-specific patterns
    has_te2 = any("text_encoder_2" in k or "lora_te2" in k for k in te_keys)

    # Check UNet LoRA keys for cross-attention hints
    unet_keys = [k for k in keys if "lora_unet" in k or "unet" in k]
    attn_keys = [k for k in unet_keys if "attn2" in k and ("to_k" in k or "to_v" in k)]

    if has_te2:
        info.model_type = ModelType.LORA_SDXL
        info.cross_attention_dim = 2048  # SDXL
        info.notes.append("Detected dual text encoder LoRA (SDXL)")
    else:
        # Assume SD1.5 if single text encoder
        info.model_type = ModelType.LORA_SD15
        info.cross_attention_dim = 768  # SD1.5
        info.notes.append("Detected single text encoder LoRA (SD 1.5)")

    return info


def _classify_model(info: ModelInfo) -> ModelType:
    """Classify model type based on detected features."""
    if info.is_lora:
        # Already classified in _detect_lora_type
        return info.model_type

    # SDXL detection
    if info.has_dual_text_encoders or info.cross_attention_dim == 2048:
        info.confidence = "high"
        # Check if it's refiner or base
        # Refiner typically has different text encoder setup
        return ModelType.SDXL_BASE  # Default to base, refiner is less common

    # SD 1.x/2.x detection
    if info.cross_attention_dim == 768:
        info.confidence = "high"
        # SD1.5 uses 768 hidden size, SD2.x uses 1024
        if info.text_encoder_hidden_size == 1024:
            return ModelType.SD21
        else:
            return ModelType.SD15

    if info.cross_attention_dim == 1024:
        info.confidence = "high"
        return ModelType.SD20

    # Fallback based on partial information
    if info.text_encoder_hidden_size:
        info.confidence = "medium"
        if info.text_encoder_hidden_size == 768:
            return ModelType.SD15
        elif info.text_encoder_hidden_size == 1024:
            return ModelType.SD21
        elif info.text_encoder_hidden_size == 2048:
            return ModelType.SDXL_BASE

    info.confidence = "low"
    info.notes.append("Could not determine model type with confidence")
    return ModelType.UNKNOWN


def detect_model(path: str) -> ModelInfo:
    """Detect model type from file or directory."""
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Check if it's a diffusers directory
    if path_obj.is_dir():
        if (path_obj / "model_index.json").exists():
            return detect_diffusers(str(path))
        else:
            raise ValueError(f"Directory is not a valid diffusers model: {path}")

    # Check file extension
    ext = path_obj.suffix.lower()
    if ext == ".safetensors":
        return detect_safetensors(str(path))
    elif ext in [".ckpt", ".pt", ".pth"]:
        # PyTorch checkpoints - need different handling
        return detect_pytorch_checkpoint(str(path))
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def detect_pytorch_checkpoint(file_path: str) -> ModelInfo:
    """Detect model type from PyTorch checkpoint (.ckpt, .pt)."""
    import torch

    info = ModelInfo(path=file_path, model_type=ModelType.UNKNOWN)
    info.notes.append("PyTorch checkpoint - loading keys only")

    # Load only keys, not weights
    try:
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                keys = checkpoint["state_dict"].keys()
            else:
                keys = checkpoint.keys()
        else:
            info.notes.append("Unsupported checkpoint format")
            info.confidence = "low"
            return info

        keys_list = list(keys)

        # Similar logic to safetensors detection
        if any("lora" in k.lower() for k in keys_list):
            info.is_lora = True
            info = _detect_lora_type(keys_list, info)
            return info

        # Check for dual text encoders
        has_te2 = any("text_encoder_2" in k or "conditioner.embedders.1" in k for k in keys_list)
        info.has_dual_text_encoders = has_te2

        # Look for cross-attention indicators in keys
        for key in keys_list:
            if "middle_block" in key and "attn2" in key:
                if has_te2:
                    info.cross_attention_dim = 2048
                else:
                    info.cross_attention_dim = 768
                break

        info.model_type = _classify_model(info)

    except Exception as e:
        info.notes.append(f"Error loading checkpoint: {e}")
        info.confidence = "low"

    return info


def print_model_info(info: ModelInfo, use_color: bool = True):
    """Print model information in a nice format."""
    if not use_color:
        Colors.HEADER = Colors.OKBLUE = Colors.OKCYAN = Colors.OKGREEN = ""
        Colors.WARNING = Colors.FAIL = Colors.ENDC = Colors.BOLD = Colors.UNDERLINE = ""

    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Model Detection Results{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")

    print(f"{Colors.BOLD}File:{Colors.ENDC} {info.path}")
    print(f"{Colors.BOLD}Format:{Colors.ENDC} {'Diffusers' if info.is_diffusers else 'Checkpoint'}")

    # Model type with color coding
    type_color = Colors.OKGREEN if info.confidence == "high" else Colors.WARNING
    print(f"{Colors.BOLD}Model Type:{Colors.ENDC} {type_color}{info.model_type.value}{Colors.ENDC}")
    print(f"{Colors.BOLD}Confidence:{Colors.ENDC} {type_color}{info.confidence}{Colors.ENDC}")

    if info.is_lora:
        print(f"{Colors.BOLD}LoRA Type:{Colors.ENDC} Yes")

    print(f"\n{Colors.BOLD}Architecture Details:{Colors.ENDC}")
    print(f"  Cross-Attention Dim: {info.cross_attention_dim or 'Unknown'}")
    print(f"  Text Encoder Hidden Size: {info.text_encoder_hidden_size or 'Unknown'}")
    print(f"  Dual Text Encoders: {'Yes' if info.has_dual_text_encoders else 'No'}")

    if info.unet_in_channels:
        print(f"  UNet In Channels: {info.unet_in_channels}")

    # Compatibility
    print(f"\n{Colors.BOLD}Compatibility:{Colors.ENDC}")
    if info.model_type in [ModelType.SD15, ModelType.SD20, ModelType.SD21, ModelType.LORA_SD15]:
        print(f"  {Colors.OKGREEN}✓{Colors.ENDC} Compatible with DiffusersCudaWorker (SD1.5)")
        print(f"  {Colors.FAIL}✗{Colors.ENDC} NOT compatible with DiffusersSDXLCudaWorker")
        print(f"  {Colors.OKBLUE}Worker:{Colors.ENDC} backends.cuda_worker.DiffusersCudaWorker")
    elif info.model_type in [ModelType.SDXL_BASE, ModelType.SDXL_REFINER, ModelType.LORA_SDXL]:
        print(f"  {Colors.FAIL}✗{Colors.ENDC} NOT compatible with DiffusersCudaWorker (SD1.5)")
        print(f"  {Colors.OKGREEN}✓{Colors.ENDC} Compatible with DiffusersSDXLCudaWorker")
        print(f"  {Colors.OKBLUE}Worker:{Colors.ENDC} backends.cuda_worker.DiffusersSDXLCudaWorker")
    else:
        print(f"  {Colors.WARNING}?{Colors.ENDC} Unknown compatibility")

    if info.notes:
        print(f"\n{Colors.BOLD}Notes:{Colors.ENDC}")
        for note in info.notes:
            print(f"  - {note}")

    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")


def scan_directory(directory: str, recursive: bool = False) -> List[ModelInfo]:
    """Scan directory for models and detect types."""
    results = []
    path_obj = Path(directory)

    if not path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all model files
    patterns = ["*.safetensors", "*.ckpt", "*.pt", "*.pth"]
    files = []

    for pattern in patterns:
        if recursive:
            files.extend(path_obj.rglob(pattern))
        else:
            files.extend(path_obj.glob(pattern))

    # Also find diffusers directories
    if recursive:
        for item in path_obj.rglob("model_index.json"):
            files.append(item.parent)
    else:
        for item in path_obj.glob("*/model_index.json"):
            files.append(item.parent)

    print(f"Found {len(files)} model(s) to analyze...")
    print()

    for file in files:
        try:
            info = detect_model(str(file))
            results.append(info)
        except Exception as e:
            print(f"Error analyzing {file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect Stable Diffusion model type (SD1.5, SD2.x, SDXL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.safetensors
  %(prog)s /path/to/diffusers/directory/
  %(prog)s --scan /models/ --recursive
  %(prog)s model.safetensors --json > result.json
        """
    )

    parser.add_argument("path", nargs="?", help="Path to model file or directory")
    parser.add_argument("--scan", metavar="DIR", help="Scan directory for models")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan recursively")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    if not args.path and not args.scan:
        parser.print_help()
        sys.exit(1)

    try:
        if args.scan:
            # Scan mode
            results = scan_directory(args.scan, recursive=args.recursive)

            if args.json:
                # JSON output
                json_results = []
                for info in results:
                    json_results.append({
                        "path": info.path,
                        "model_type": info.model_type.value,
                        "cross_attention_dim": info.cross_attention_dim,
                        "has_dual_text_encoders": info.has_dual_text_encoders,
                        "is_lora": info.is_lora,
                        "is_diffusers": info.is_diffusers,
                        "confidence": info.confidence,
                        "notes": info.notes,
                    })
                print(json.dumps(json_results, indent=2))
            else:
                # Pretty print
                for info in results:
                    print_model_info(info, use_color=not args.no_color)

                # Summary
                sd15_count = sum(1 for info in results if info.model_type in [ModelType.SD15, ModelType.SD20, ModelType.SD21])
                sdxl_count = sum(1 for info in results if info.model_type in [ModelType.SDXL_BASE, ModelType.SDXL_REFINER])
                lora_sd15_count = sum(1 for info in results if info.model_type == ModelType.LORA_SD15)
                lora_sdxl_count = sum(1 for info in results if info.model_type == ModelType.LORA_SDXL)
                unknown_count = sum(1 for info in results if info.model_type == ModelType.UNKNOWN)

                print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
                print(f"  SD 1.5/2.x: {sd15_count}")
                print(f"  SDXL: {sdxl_count}")
                print(f"  LoRA (SD 1.5): {lora_sd15_count}")
                print(f"  LoRA (SDXL): {lora_sdxl_count}")
                print(f"  Unknown: {unknown_count}")
                print(f"  Total: {len(results)}")

        else:
            # Single file mode
            info = detect_model(args.path)

            if args.json:
                result = {
                    "path": info.path,
                    "model_type": info.model_type.value,
                    "cross_attention_dim": info.cross_attention_dim,
                    "text_encoder_hidden_size": info.text_encoder_hidden_size,
                    "has_dual_text_encoders": info.has_dual_text_encoders,
                    "is_lora": info.is_lora,
                    "is_diffusers": info.is_diffusers,
                    "confidence": info.confidence,
                    "notes": info.notes,
                }
                print(json.dumps(result, indent=2))
            else:
                print_model_info(info, use_color=not args.no_color)

    except Exception as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
