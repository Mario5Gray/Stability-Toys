"""
Mode configuration management.

Loads and validates modes.yml configuration file containing:
- Global paths (model_root, lora_root)
- Default mode
- Mode definitions (model, loras, defaults)
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import yaml

MODE_CONFIG_PATH = os.environ.get("MODE_CONFIG_PATH", "conf")

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """LoRA configuration within a mode."""
    path: str
    strength: float = 1.0
    adapter_name: Optional[str] = None

    def __post_init__(self):
        if self.adapter_name is None:
            # Generate adapter name from filename
            self.adapter_name = f"lora_{Path(self.path).stem}"


@dataclass
class ModeConfig:
    """Configuration for a single mode."""
    name: str
    model: str  # Path relative to model_root
    loras: List[LoRAConfig] = field(default_factory=list)
    default_size: str = "512x512"
    default_steps: int = 4
    default_guidance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Resolved absolute paths (set after loading)
    model_path: Optional[str] = None
    lora_paths: List[str] = field(default_factory=list)


@dataclass
class ModesYAML:
    """Root configuration from modes.yml."""
    model_root: str
    lora_root: str
    default_mode: str
    modes: Dict[str, ModeConfig]


class ModeConfigManager:
    """
    Manages mode configurations from modes.yml.

    Responsibilities:
    - Load and validate modes.yml
    - Resolve paths relative to model_root/lora_root
    - Provide access to mode definitions
    - Validate mode consistency
    """

    def __init__(self, config_path: str):
        """
        Initialize mode configuration manager.

        Args:
            config_path: Path to modes.yml (relative to project root)
        """
        if not config_path:
            raise ValueError(f"Couldnt find a modes.yml to configure models with! current CONFIG_PATH={self.config_path}")
        
        self.config_path = Path(os.path.join(Path(config_path), "modes.yml"))

        self.config: Optional[ModesYAML] = None
        self._load_config()

    def _load_config(self):
        """Load and validate modes.yml."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"modes.yml not found at {self.config_path}. "
                f"Create this file to define model loading modes."
            )

        logger.info(f"[ModeConfig] Loading configuration from {self.config_path}")

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        # Validate required fields
        if not data:
            raise ValueError("modes.yml is empty")

        if "model_root" not in data:
            raise ValueError("modes.yml missing required field: model_root")
        if "default_mode" not in data:
            raise ValueError("modes.yml missing required field: default_mode")
        if "modes" not in data or not data["modes"]:
            raise ValueError("modes.yml missing or empty: modes")

        # Parse configuration
        model_root = Path(data["model_root"]).expanduser()
        lora_root = Path(data.get("lora_root", data["model_root"])).expanduser()
        default_mode = data["default_mode"]

        # Parse mode definitions
        modes = {}
        for mode_name, mode_data in data["modes"].items():
            if "model" not in mode_data:
                raise ValueError(f"Mode '{mode_name}' missing required field: model")

            # Parse LoRAs
            loras = []
            for lora_def in mode_data.get("loras", []):
                if isinstance(lora_def, str):
                    # Simple format: just path
                    loras.append(LoRAConfig(path=lora_def))
                elif isinstance(lora_def, dict):
                    # Full format: {path, strength, adapter_name}
                    loras.append(LoRAConfig(
                        path=lora_def["path"],
                        strength=lora_def.get("strength", 1.0),
                        adapter_name=lora_def.get("adapter_name"),
                    ))

            mode = ModeConfig(
                name=mode_name,
                model=mode_data["model"],
                loras=loras,
                default_size=mode_data.get("default_size", "512x512"),
                default_steps=mode_data.get("default_steps", 4),
                default_guidance=mode_data.get("default_guidance", 1.0),
                metadata=mode_data.get("metadata", {}),
            )

            # Resolve absolute paths
            mode.model_path = str(model_root / mode.model)
            mode.lora_paths = [str(lora_root / lora.path) for lora in mode.loras]

            modes[mode_name] = mode

        # Validate default mode exists
        if default_mode not in modes:
            raise ValueError(
                f"default_mode '{default_mode}' not found in modes. "
                f"Available modes: {list(modes.keys())}"
            )

        self.config = ModesYAML(
            model_root=str(model_root),
            lora_root=str(lora_root),
            default_mode=default_mode,
            modes=modes,
        )

        logger.info(f"[ModeConfig] Loaded {len(modes)} modes")
        logger.info(f"[ModeConfig] Default mode: {default_mode}")
        logger.info(f"[ModeConfig] Model root: {model_root}")
        logger.info(f"[ModeConfig] LoRA root: {lora_root}")

        # Validate paths exist
        self._validate_paths()

    def _validate_paths(self):
        """Validate that model and LoRA paths exist."""
        errors = []

        # Check model_root exists
        if not Path(self.config.model_root).exists():
            errors.append(f"model_root does not exist: {self.config.model_root}")

        # Check lora_root exists
        if not Path(self.config.lora_root).exists():
            errors.append(f"lora_root does not exist: {self.config.lora_root}")

        # Check each mode's model and LoRAs
        for mode_name, mode in self.config.modes.items():
            if not Path(mode.model_path).exists():
                errors.append(f"Mode '{mode_name}' model not found: {mode.model_path}")

            for i, lora_path in enumerate(mode.lora_paths):
                if not Path(lora_path).exists():
                    errors.append(
                        f"Mode '{mode_name}' LoRA {i} not found: {lora_path}"
                    )

        if errors:
            logger.warning("[ModeConfig] Path validation warnings:")
            for error in errors:
                logger.warning(f"  - {error}")
            # Don't raise - allow starting with missing models for development

    def save_config(self, data: Dict[str, Any]):
        """
        Save configuration data to modes.yml and reload.

        Args:
            data: Dict with model_root, lora_root, default_mode, modes
        """
        # Build YAML-friendly structure
        yaml_data = {
            "model_root": data["model_root"],
            "lora_root": data["lora_root"],
            "default_mode": data["default_mode"],
            "modes": {},
        }

        for mode_name, mode_data in data["modes"].items():
            mode_entry = {
                "model": mode_data["model"],
                "default_size": mode_data.get("default_size", "512x512"),
                "default_steps": mode_data.get("default_steps", 4),
                "default_guidance": mode_data.get("default_guidance", 1.0),
            }
            loras = mode_data.get("loras", [])
            if loras:
                mode_entry["loras"] = [
                    {"path": lora["path"], "strength": lora.get("strength", 1.0)}
                    if lora.get("strength", 1.0) != 1.0
                    else lora["path"]
                    for lora in loras
                ]
            yaml_data["modes"][mode_name] = mode_entry

        # Write atomically-ish: write to temp then rename
        tmp_path = self.config_path.with_suffix(".yml.tmp")
        with open(tmp_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        tmp_path.rename(self.config_path)

        logger.info(f"[ModeConfig] Saved configuration to {self.config_path}")
        self._load_config()

    def reload(self):
        """Reload configuration from disk."""
        logger.info("[ModeConfig] Reloading configuration")
        self._load_config()

    def get_mode(self, name: str) -> ModeConfig:
        """
        Get mode configuration by name.

        Args:
            name: Mode name

        Returns:
            ModeConfig

        Raises:
            KeyError if mode not found
        """
        if name not in self.config.modes:
            raise KeyError(
                f"Mode '{name}' not found. Available: {list(self.config.modes.keys())}"
            )
        return self.config.modes[name]

    def list_modes(self) -> List[str]:
        """Get list of all mode names."""
        return list(self.config.modes.keys())

    def get_default_mode(self) -> str:
        """Get the default mode name."""
        return self.config.default_mode

    def get_default_mode_config(self) -> ModeConfig:
        """Get the default mode configuration."""
        return self.get_mode(self.config.default_mode)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "model_root": self.config.model_root,
            "lora_root": self.config.lora_root,
            "default_mode": self.config.default_mode,
            "modes": {
                name: {
                    "model": mode.model,
                    "model_path": mode.model_path,
                    "loras": [
                        {
                            "path": lora.path,
                            "strength": lora.strength,
                            "adapter_name": lora.adapter_name,
                        }
                        for lora in mode.loras
                    ],
                    "default_size": mode.default_size,
                    "default_steps": mode.default_steps,
                    "default_guidance": mode.default_guidance,
                    "metadata": mode.metadata,
                }
                for name, mode in self.config.modes.items()
            },
        }


# Global instance (initialized on first import)
_config_manager: Optional[ModeConfigManager] = None


def get_mode_config(confPath: Optional[str] = None) -> ModeConfigManager:
    """Get global mode configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        if confPath is None:
            _config_manager = ModeConfigManager(MODE_CONFIG_PATH)
        else:
            _config_manager = ModeConfigManager(confPath) 

    return _config_manager


def reload_mode_config():
    """Reload global mode configuration from disk."""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload()
