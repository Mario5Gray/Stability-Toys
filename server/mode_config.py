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
class ChatBackendConfig:
    """OpenAI-compatible chat backend settings for a mode."""
    endpoint: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None


@dataclass
class ChatConnectionConfig:
    """Reusable transport settings for chat backends."""
    endpoint: str
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class ChatDelegateConfig:
    """Named bundle: connection + model + inference params bound to one logical chat persona."""
    name: str
    connection: str          # key into chat_connections
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None


@dataclass
class ControlNetControlTypePolicy:
    """Per-control-type policy within a mode's controlnet_policy."""
    default_model_id: Optional[str] = None
    allowed_model_ids: List[str] = field(default_factory=list)
    allow_preprocess: bool = True
    default_strength: float = 1.0
    min_strength: float = 0.0
    max_strength: float = 2.0


@dataclass
class ControlNetPolicy:
    """Mode-owned ControlNet policy.

    When `enabled` is False, any request carrying `controlnets` is rejected.
    `allowed_control_types` maps canonical control-type names (e.g. 'canny')
    to their per-type policy. Absent control types are forbidden.
    """
    enabled: bool = False
    max_attachments: int = 0
    allow_reuse_emitted_maps: bool = False
    allowed_control_types: Dict[str, ControlNetControlTypePolicy] = field(default_factory=dict)


@dataclass
class ModeConfig:
    """Configuration for a single mode."""
    name: str
    model: str  # Path relative to model_root
    loras: List[LoRAConfig] = field(default_factory=list)
    resolution_set: Optional[str] = None
    resolution_options: List[Dict[str, str]] = field(default_factory=list)
    default_size: str = "512x512"
    default_steps: int = 4
    default_guidance: float = 1.0
    maximum_len: Optional[int] = None
    loader_format: Optional[str] = None
    checkpoint_precision: Optional[str] = None
    checkpoint_variant: Optional[str] = None
    scheduler_profile: Optional[str] = None
    recommended_size: Optional[str] = None
    runtime_quantize: Optional[str] = None
    runtime_offload: Optional[str] = None
    runtime_attention_slicing: Optional[bool] = None
    runtime_enable_xformers: Optional[bool] = None
    negative_prompt_templates: Dict[str, str] = field(default_factory=dict)
    default_negative_prompt_template: Optional[str] = None
    allow_custom_negative_prompt: bool = False
    allowed_scheduler_ids: Optional[List[str]] = None
    default_scheduler_id: Optional[str] = None
    chat_delegate: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    controlnet_policy: ControlNetPolicy = field(default_factory=ControlNetPolicy)

    # Resolved absolute paths (set after loading)
    model_path: Optional[str] = None
    lora_paths: List[str] = field(default_factory=list)


@dataclass
class ModesYAML:
    """Root configuration from modes.yml."""
    model_root: str
    lora_root: str
    default_mode: str
    resolution_sets: Dict[str, List[Dict[str, str]]]
    chat_connections: Dict[str, ChatConnectionConfig]
    chat_delegates: Dict[str, ChatDelegateConfig]
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
            raise ValueError(
                "Couldnt find a modes.yml to configure models with! current CONFIG_PATH is empty"
            )
        
        self.config_path = Path(os.path.join(Path(config_path), "modes.yml"))

        self.config: ModesYAML = None  # type: ignore[assignment]
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
        if (
            "resolution_sets" not in data
            or not isinstance(data["resolution_sets"], dict)
            or not data["resolution_sets"]
            or "default" not in data["resolution_sets"]
        ):
            raise ValueError("modes.yml missing required field: resolution_sets.default")

        # Parse configuration
        model_root = Path(data["model_root"]).expanduser()
        lora_root = Path(data.get("lora_root", data["model_root"])).expanduser()
        default_mode = data["default_mode"]

        resolution_sets: Dict[str, List[Dict[str, str]]] = {}
        for set_name, entries in data["resolution_sets"].items():
            if entries is None:
                entries = []
            if not isinstance(entries, list):
                raise ValueError(f"resolution_sets.{set_name} must be a list")
            resolved_entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(f"resolution_sets.{set_name} entries must be mappings")
                if "size" not in entry:
                    raise ValueError(f"resolution_sets.{set_name} entries missing required field: size")
                if "aspect_ratio" not in entry:
                    raise ValueError(f"resolution_sets.{set_name} entries missing required field: aspect_ratio")
                resolved_entries.append({
                    "size": str(entry["size"]),
                    "aspect_ratio": str(entry["aspect_ratio"]),
                })
            resolution_sets[set_name] = resolved_entries

        if "chat" in data:
            raise ValueError(
                "modes.yml contains legacy top-level 'chat'; migrate to 'chat_connections' plus mode chat_* fields"
            )

        raw_chat_connections = data.get("chat_connections")
        if raw_chat_connections is None:
            raw_chat_connections = {}
        if not isinstance(raw_chat_connections, dict):
            raise ValueError("modes.yml field 'chat_connections' must be a mapping")
        chat_connections = {
            connection_name: self._parse_chat_connection_config(connection_name, chat_data)
            for connection_name, chat_data in raw_chat_connections.items()
        }

        raw_chat_delegates = data.get("chat_delegates")
        if raw_chat_delegates is None:
            raw_chat_delegates = {}
        if not isinstance(raw_chat_delegates, dict):
            raise ValueError("modes.yml field 'chat_delegates' must be a mapping")
        chat_delegates = {
            delegate_name: self._parse_chat_delegate_config(delegate_name, delegate_data, chat_connections)
            for delegate_name, delegate_data in raw_chat_delegates.items()
        }

        # Parse mode definitions
        modes = {}
        for mode_name, mode_data in data["modes"].items():
            if "model" not in mode_data:
                raise ValueError(f"Mode '{mode_name}' missing required field: model")
            if "chat" in mode_data and mode_data.get("chat") is not None:
                raise ValueError(
                    f"Mode '{mode_name}' contains legacy mode-scoped chat config; use chat_delegates"
                )
            for old_field in ("chat_connection", "chat_model", "chat_max_tokens", "chat_temperature", "chat_system_prompt"):
                if mode_data.get(old_field) is not None:
                    raise ValueError(
                        f"Mode '{mode_name}' uses removed field '{old_field}'; "
                        f"define a chat_delegate in chat_delegates: and reference it via chat_delegate:"
                    )

            resolution_set = mode_data.get("resolution_set") or "default"
            if resolution_set not in resolution_sets:
                raise ValueError(
                    f"Mode '{mode_name}' references unknown resolution_set '{resolution_set}'"
                )
            resolution_options = resolution_sets[resolution_set]
            default_size = mode_data.get("default_size", "512x512")
            if default_size not in {entry["size"] for entry in resolution_options}:
                raise ValueError(
                    f"Mode '{mode_name}' default_size '{default_size}' is not present in resolution_set '{resolution_set}'"
                )

            chat_delegate = self._normalize_optional_string(mode_data.get("chat_delegate"))
            if chat_delegate and chat_delegate not in chat_delegates:
                raise ValueError(f"Mode '{mode_name}' references unknown chat_delegate '{chat_delegate}'")

            # Parse LoRAs
            loras = []
            for lora_def in mode_data.get("loras", []):
                if isinstance(lora_def, str):
                    loras.append(LoRAConfig(path=lora_def))
                elif isinstance(lora_def, dict):
                    loras.append(LoRAConfig(
                        path=lora_def["path"],
                        strength=lora_def.get("strength", 1.0),
                        adapter_name=lora_def.get("adapter_name"),
                    ))

            mode = ModeConfig(
                name=mode_name,
                model=mode_data["model"],
                loras=loras,
                resolution_set=resolution_set,
                resolution_options=resolution_options,
                default_size=default_size,
                default_steps=mode_data.get("default_steps", 4),
                default_guidance=mode_data.get("default_guidance", 1.0),
                maximum_len=mode_data.get("maximum_len"),
                loader_format=mode_data.get("loader_format"),
                checkpoint_precision=mode_data.get("checkpoint_precision"),
                checkpoint_variant=mode_data.get("checkpoint_variant"),
                scheduler_profile=mode_data.get("scheduler_profile"),
                recommended_size=mode_data.get("recommended_size"),
                runtime_quantize=mode_data.get("runtime_quantize"),
                runtime_offload=mode_data.get("runtime_offload"),
                runtime_attention_slicing=mode_data.get("runtime_attention_slicing"),
                runtime_enable_xformers=mode_data.get("runtime_enable_xformers"),
                negative_prompt_templates=mode_data.get("negative_prompt_templates", {}) or {},
                default_negative_prompt_template=mode_data.get("default_negative_prompt_template"),
                allow_custom_negative_prompt=bool(mode_data.get("allow_custom_negative_prompt", False)),
                allowed_scheduler_ids=mode_data.get("allowed_scheduler_ids"),
                default_scheduler_id=mode_data.get("default_scheduler_id"),
                chat_delegate=chat_delegate,
                metadata=mode_data.get("metadata", {}),
                controlnet_policy=self._parse_controlnet_policy(mode_name, mode_data.get("controlnet_policy")),
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
            resolution_sets=resolution_sets,
            chat_connections=chat_connections,
            chat_delegates=chat_delegates,
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
        assert self.config is not None
        errors = []

        # Check model_root exists
        if not Path(self.config.model_root).exists():
            errors.append(f"model_root does not exist: {self.config.model_root}")

        # Check lora_root exists
        if not Path(self.config.lora_root).exists():
            errors.append(f"lora_root does not exist: {self.config.lora_root}")

        # Check each mode's model and LoRAs
        for mode_name, mode in self.config.modes.items():
            if not Path(mode.model_path or "").exists():
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

    _ALLOWED_CONTROLNET_POLICY_KEYS = {
        "enabled", "max_attachments", "allow_reuse_emitted_maps", "allowed_control_types",
    }

    _ALLOWED_CONTROLNET_TYPE_KEYS = {
        "default_model_id", "allowed_model_ids", "allow_preprocess",
        "default_strength", "min_strength", "max_strength",
    }

    def _parse_controlnet_policy(self, mode_name: str, raw: Any) -> ControlNetPolicy:
        if raw is None:
            return ControlNetPolicy()
        if not isinstance(raw, dict):
            raise ValueError(f"Mode '{mode_name}' controlnet_policy must be a mapping")
        unknown = set(raw.keys()) - self._ALLOWED_CONTROLNET_POLICY_KEYS
        if unknown:
            raise ValueError(
                f"Mode '{mode_name}' controlnet_policy has unknown keys: {sorted(unknown)}"
            )
        allowed_types_raw = raw.get("allowed_control_types") or {}
        if not isinstance(allowed_types_raw, dict):
            raise ValueError(
                f"Mode '{mode_name}' controlnet_policy.allowed_control_types must be a mapping"
            )
        allowed_types: Dict[str, ControlNetControlTypePolicy] = {}
        for type_name, type_raw in allowed_types_raw.items():
            if type_raw is None:
                type_raw = {}
            if not isinstance(type_raw, dict):
                raise ValueError(
                    f"Mode '{mode_name}' controlnet_policy.allowed_control_types.{type_name} must be a mapping"
                )
            unknown_type = set(type_raw.keys()) - self._ALLOWED_CONTROLNET_TYPE_KEYS
            if unknown_type:
                raise ValueError(
                    f"Mode '{mode_name}' controlnet_policy.allowed_control_types.{type_name} has unknown keys: {sorted(unknown_type)}"
                )
            allowed_ids = type_raw.get("allowed_model_ids") or []
            if not isinstance(allowed_ids, list) or not all(isinstance(x, str) for x in allowed_ids):
                raise ValueError(
                    f"Mode '{mode_name}' controlnet_policy.allowed_control_types.{type_name}.allowed_model_ids must be a list of strings"
                )
            allowed_types[type_name] = ControlNetControlTypePolicy(
                default_model_id=type_raw.get("default_model_id"),
                allowed_model_ids=list(allowed_ids),
                allow_preprocess=bool(type_raw.get("allow_preprocess", True)),
                default_strength=float(type_raw.get("default_strength", 1.0)),
                min_strength=float(type_raw.get("min_strength", 0.0)),
                max_strength=float(type_raw.get("max_strength", 2.0)),
            )
        return ControlNetPolicy(
            enabled=bool(raw.get("enabled", False)),
            max_attachments=int(raw.get("max_attachments", 0)),
            allow_reuse_emitted_maps=bool(raw.get("allow_reuse_emitted_maps", False)),
            allowed_control_types=allowed_types,
        )

    def _parse_chat_connection_config(self, connection_name: str, chat_data: Dict[str, Any]) -> ChatConnectionConfig:
        """Parse reusable chat connection settings."""
        if not isinstance(chat_data, dict):
            raise ValueError(f"Chat connection '{connection_name}' must be a mapping")
        endpoint = (chat_data.get("endpoint") or "").strip()
        if not endpoint:
            raise ValueError(f"Chat connection '{connection_name}' missing required field: endpoint")
        return ChatConnectionConfig(
            endpoint=endpoint,
            api_key_env=(chat_data.get("api_key_env") or "OPENAI_API_KEY").strip(),
        )

    def _parse_chat_delegate_config(
        self,
        delegate_name: str,
        delegate_data: Dict[str, Any],
        chat_connections: Dict[str, ChatConnectionConfig],
    ) -> ChatDelegateConfig:
        """Parse a chat_delegates entry, validating the referenced connection exists."""
        if not isinstance(delegate_data, dict):
            raise ValueError(f"Chat delegate '{delegate_name}' must be a mapping")
        connection = (delegate_data.get("connection") or "").strip()
        if not connection:
            raise ValueError(f"Chat delegate '{delegate_name}' missing required field: connection")
        if connection not in chat_connections:
            raise ValueError(f"Chat delegate '{delegate_name}' references unknown connection '{connection}'")
        model = (delegate_data.get("model") or "").strip()
        if not model:
            raise ValueError(f"Chat delegate '{delegate_name}' missing required field: model")
        max_tokens = self._parse_optional_int(delegate_data.get("max_tokens"), f"chat_delegate '{delegate_name}'", "max_tokens")
        temperature = self._parse_optional_float(delegate_data.get("temperature"), f"chat_delegate '{delegate_name}'", "temperature")
        system_prompt = self._normalize_optional_string(delegate_data.get("system_prompt"))
        return ChatDelegateConfig(
            name=delegate_name,
            connection=connection,
            model=model,
            max_tokens=max_tokens if max_tokens is not None else 1024,
            temperature=temperature if temperature is not None else 0.7,
            system_prompt=system_prompt,
        )

    def _parse_optional_int(self, value: Any, owner: str, field_name: str) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{owner} has invalid {field_name}") from e

    def _parse_optional_float(self, value: Any, owner: str, field_name: str) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{owner} has invalid {field_name}") from e

    def _normalize_optional_string(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        if normalized == "":
            return None
        return normalized

    def _merge_system_prompts(self, mode_prompt: Optional[str], override_prompt: Optional[str]) -> Optional[str]:
        prompts = [prompt for prompt in (self._normalize_optional_string(mode_prompt), self._normalize_optional_string(override_prompt)) if prompt]
        if not prompts:
            return None
        return "\n\n".join(prompts)

    def save_config(self, data: Dict[str, Any]):
        """
        Save configuration data to modes.yml and reload.

        Args:
            data: Dict with model_root, lora_root, default_mode, resolution_sets, modes
        """
        if "resolution_sets" not in data or not isinstance(data["resolution_sets"], dict):
            raise ValueError("save_config requires resolution_sets")

        # Build YAML-friendly structure
        yaml_data = {
            "model_root": data["model_root"],
            "lora_root": data["lora_root"],
            "default_mode": data["default_mode"],
            "resolution_sets": data["resolution_sets"],
            "modes": {},
        }

        for mode_name, mode_data in data["modes"].items():
            mode_entry = {
                "model": mode_data["model"],
                "default_size": mode_data.get("default_size", "512x512"),
                "default_steps": mode_data.get("default_steps", 4),
                "default_guidance": mode_data.get("default_guidance", 1.0),
            }
            if mode_data.get("maximum_len") is not None:
                mode_entry["maximum_len"] = mode_data.get("maximum_len")
            if mode_data.get("chat") is not None:
                raise ValueError(
                    f"Mode '{mode_name}' contains legacy mode-scoped chat config; use chat_delegates"
                )
            for old_field in ("chat_connection", "chat_model", "chat_max_tokens", "chat_temperature", "chat_system_prompt"):
                if mode_data.get(old_field) is not None:
                    raise ValueError(
                        f"Mode '{mode_name}' uses removed field '{old_field}'; "
                        f"define a chat_delegate in chat_delegates: and reference it via chat_delegate:"
                    )
            chat_delegate = self._normalize_optional_string(mode_data.get("chat_delegate"))
            if chat_delegate and chat_delegate not in (data.get("chat_delegates") or {}):
                raise ValueError(f"Mode '{mode_name}' references unknown chat_delegate '{chat_delegate}'")
            if mode_data.get("resolution_set") is not None:
                mode_entry["resolution_set"] = mode_data.get("resolution_set")
            for cap_field in (
                "loader_format",
                "checkpoint_precision",
                "checkpoint_variant",
                "scheduler_profile",
                "recommended_size",
                "runtime_quantize",
                "runtime_offload",
                "runtime_attention_slicing",
                "runtime_enable_xformers",
            ):
                val = mode_data.get(cap_field)
                if val is not None:
                    mode_entry[cap_field] = val
            mode_entry["negative_prompt_templates"] = mode_data.get("negative_prompt_templates", {}) or {}
            if mode_data.get("default_negative_prompt_template") is not None:
                mode_entry["default_negative_prompt_template"] = mode_data.get("default_negative_prompt_template")
            mode_entry["allow_custom_negative_prompt"] = bool(mode_data.get("allow_custom_negative_prompt", False))
            if "allowed_scheduler_ids" in mode_data:
                mode_entry["allowed_scheduler_ids"] = mode_data.get("allowed_scheduler_ids")
            if mode_data.get("default_scheduler_id") is not None:
                mode_entry["default_scheduler_id"] = mode_data.get("default_scheduler_id")
            if chat_delegate is not None:
                mode_entry["chat_delegate"] = chat_delegate
            loras = mode_data.get("loras", [])
            if loras:
                mode_entry["loras"] = [
                    {"path": lora["path"], "strength": lora.get("strength", 1.0)}
                    if lora.get("strength", 1.0) != 1.0
                    else lora["path"]
                    for lora in loras
                ]
            yaml_data["modes"][mode_name] = mode_entry

        if data.get("chat") not in (None, {}):
            raise ValueError("save_config does not support legacy top-level 'chat'; use 'chat_delegates'")
        raw_chat_connections = data.get("chat_connections") or {}
        if not isinstance(raw_chat_connections, dict):
            raise ValueError("save_config requires chat_connections to be a mapping when provided")
        if raw_chat_connections:
            yaml_data["chat_connections"] = {}
            for connection_name, connection_data in raw_chat_connections.items():
                conn = self._parse_chat_connection_config(connection_name, connection_data)
                yaml_data["chat_connections"][connection_name] = {
                    "endpoint": conn.endpoint,
                    "api_key_env": conn.api_key_env,
                }

        raw_chat_delegates = data.get("chat_delegates") or {}
        if not isinstance(raw_chat_delegates, dict):
            raise ValueError("save_config requires chat_delegates to be a mapping when provided")
        if raw_chat_delegates:
            parsed_connections = {
                k: self._parse_chat_connection_config(k, v)
                for k, v in raw_chat_connections.items()
            }
            yaml_data["chat_delegates"] = {}
            for delegate_name, delegate_data in raw_chat_delegates.items():
                d = self._parse_chat_delegate_config(delegate_name, delegate_data, parsed_connections)
                entry: Dict[str, Any] = {"connection": d.connection, "model": d.model}
                if d.max_tokens != 1024:
                    entry["max_tokens"] = d.max_tokens
                if d.temperature != 0.7:
                    entry["temperature"] = d.temperature
                if d.system_prompt is not None:
                    entry["system_prompt"] = d.system_prompt
                yaml_data["chat_delegates"][delegate_name] = entry

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

    def resolve_chat_config(self, mode_name: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[ChatBackendConfig]:
        """Resolve a mode's chat config via its chat_delegate."""
        overrides = overrides or {}
        mode = self.get_mode(mode_name)
        if not mode.chat_delegate:
            return None

        delegate = self.config.chat_delegates[mode.chat_delegate]
        connection = self.config.chat_connections[delegate.connection]

        model = self._normalize_optional_string(overrides.get("model")) or delegate.model
        max_tokens = self._parse_optional_int(overrides.get("max_tokens"), f"mode '{mode_name}'", "max_tokens")
        if max_tokens is None:
            max_tokens = delegate.max_tokens
        temperature = self._parse_optional_float(overrides.get("temperature"), f"mode '{mode_name}'", "temperature")
        if temperature is None:
            temperature = delegate.temperature
        system_prompt = self._merge_system_prompts(delegate.system_prompt, overrides.get("system_prompt"))

        return ChatBackendConfig(
            endpoint=connection.endpoint,
            model=model,
            api_key_env=connection.api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "model_root": self.config.model_root,
            "lora_root": self.config.lora_root,
            "default_mode": self.config.default_mode,
            "resolution_sets": self.config.resolution_sets,
            "chat_connections": {
                connection_name: {
                    "endpoint": chat_connection.endpoint,
                    "api_key_env": chat_connection.api_key_env,
                }
                for connection_name, chat_connection in self.config.chat_connections.items()
            },
            "chat_delegates": {
                delegate_name: {
                    "connection": delegate.connection,
                    "model": delegate.model,
                    "max_tokens": delegate.max_tokens,
                    "temperature": delegate.temperature,
                    "system_prompt": delegate.system_prompt,
                }
                for delegate_name, delegate in self.config.chat_delegates.items()
            },
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
                    "resolution_set": mode.resolution_set,
                    "resolution_options": [
                        {
                            "size": option["size"],
                            "aspect_ratio": option["aspect_ratio"],
                        }
                        for option in mode.resolution_options
                    ],
                    "default_size": mode.default_size,
                    "default_steps": mode.default_steps,
                    "default_guidance": mode.default_guidance,
                    "maximum_len": mode.maximum_len,
                    "loader_format": mode.loader_format,
                    "checkpoint_precision": mode.checkpoint_precision,
                    "checkpoint_variant": mode.checkpoint_variant,
                    "scheduler_profile": mode.scheduler_profile,
                    "recommended_size": mode.recommended_size,
                    "runtime_quantize": mode.runtime_quantize,
                    "runtime_offload": mode.runtime_offload,
                    "runtime_attention_slicing": mode.runtime_attention_slicing,
                    "runtime_enable_xformers": mode.runtime_enable_xformers,
                    "negative_prompt_templates": mode.negative_prompt_templates,
                    "default_negative_prompt_template": mode.default_negative_prompt_template,
                    "allow_custom_negative_prompt": mode.allow_custom_negative_prompt,
                    "allowed_scheduler_ids": mode.allowed_scheduler_ids,
                    "default_scheduler_id": mode.default_scheduler_id,
                    "chat_delegate": mode.chat_delegate,
                    "metadata": mode.metadata,
                    "controlnet_policy": {
                        "enabled": mode.controlnet_policy.enabled,
                        "max_attachments": mode.controlnet_policy.max_attachments,
                        "allow_reuse_emitted_maps": mode.controlnet_policy.allow_reuse_emitted_maps,
                        "allowed_control_types": {
                            type_name: {
                                "default_model_id": type_policy.default_model_id,
                                "allowed_model_ids": list(type_policy.allowed_model_ids),
                                "allow_preprocess": type_policy.allow_preprocess,
                                "default_strength": type_policy.default_strength,
                                "min_strength": type_policy.min_strength,
                                "max_strength": type_policy.max_strength,
                            }
                            for type_name, type_policy in mode.controlnet_policy.allowed_control_types.items()
                        },
                    },
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
