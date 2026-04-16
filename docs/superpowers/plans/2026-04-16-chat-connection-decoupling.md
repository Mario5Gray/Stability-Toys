# Chat Connection Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mode-name-keyed chat config with reusable `chat_connections`, flat mode-owned `chat_*` defaults, and neutral per-request behavioral overrides shared by advisor and WebSocket chat paths.

**Architecture:** Keep transport settings in `ModeConfigManager` as named connection records, keep model/default behavior on each mode, and centralize resolution in one mode-level chat resolver. Leave connection selection config-owned, allow request overrides only for behavior (`model`, `max_tokens`, `temperature`, `system_prompt`), and keep client lifecycle fresh-per-request in both advisor and WebSocket callers.

**Tech Stack:** Python 3 in Miniforge base, FastAPI, Pydantic, existing `ChatCompletionsClient`, pytest, drift, Fiberplane CLI

---

## File Structure

### Config And Resolution

- Modify: `server/mode_config.py`
  Replace top-level `chat` parsing with `chat_connections`, extend `ModeConfig` with flat `chat_*` fields, add `ChatConnectionConfig`, add a shared resolver, and update `save_config()` / `to_dict()` to round-trip the new shape.
- Modify: `conf/modes.yml`
  Convert the checked-in config from legacy top-level `chat` keyed by mode name to `chat_connections` plus flat mode `chat_*` fields.

### Runtime Consumers

- Modify: `server/advisor_service.py`
  Switch from `get_chat_config(mode_name)` to the new resolver and preserve existing maximum-length clamp behavior while supporting neutral overrides the advisor surface exposes.
- Modify: `server/ws_routes.py`
  Switch chat resolution to the new resolver, support neutral overrides from `params`, and preserve streaming / completion behavior.

### Tests

- Modify: `tests/test_mode_config.py`
  Replace legacy `chat`-shape assertions with `chat_connections` + flat mode assertions and add resolver/validation coverage.
- Modify: `tests/test_advisor_service.py`
  Verify resolver usage, additive `system_prompt` stacking, empty-string override behavior, and continued clamp behavior.
- Modify: `tests/test_ws_routes.py`
  Verify resolver usage, neutral request override behavior, additive `system_prompt` stacking, and backend-error passthrough expectations.

### Docs And Tracking

- Modify: `docs/superpowers/specs/2026-04-07-chat-completions-backend-design.md`
  Update stale prose that still describes a nested per-mode `chat` block.
- Modify: `drift.lock`
  Refresh provenance after doc updates.

---

### Task 1: Refactor Mode Config To `chat_connections` And Flat Mode Fields

**Files:**
- Modify: `server/mode_config.py`
- Modify: `tests/test_mode_config.py`

- [ ] **Step 1: Write the failing config-shape and validation tests**

```python
# tests/test_mode_config.py
def test_mode_config_parses_chat_connections_and_flat_mode_chat_fields(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-chat
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
chat_connections:
  local_default:
    endpoint: http://localhost:11434/v1
    api_key_env: OPENAI_API_KEY
modes:
  sdxl-chat:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
    chat_connection: local_default
    chat_model: llama3.2
    chat_max_tokens: 768
    chat_temperature: 0.4
    chat_system_prompt: You are concise.
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sdxl-chat")
    chat_cfg = manager.resolve_chat_config("sdxl-chat")

    assert mode.chat_connection == "local_default"
    assert mode.chat_model == "llama3.2"
    assert chat_cfg is not None
    assert chat_cfg.endpoint == "http://localhost:11434/v1"
    assert chat_cfg.model == "llama3.2"
    assert chat_cfg.max_tokens == 768
    assert chat_cfg.temperature == 0.4
    assert chat_cfg.system_prompt == "You are concise."
    assert manager.to_dict()["chat_connections"]["local_default"]["endpoint"] == "http://localhost:11434/v1"


def test_mode_config_rejects_legacy_top_level_chat_block(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-chat
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
chat:
  sdxl-chat:
    endpoint: http://localhost:11434/v1
    model: llama3.2
modes:
  sdxl-chat:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    with pytest.raises(ValueError, match="legacy top-level 'chat'"):
        ModeConfigManager(str(tmp_path))


def test_mode_config_rejects_unknown_chat_connection(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-chat
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
chat_connections:
  local_default:
    endpoint: http://localhost:11434/v1
modes:
  sdxl-chat:
    model: checkpoints/sdxl/sdxl-base.safetensors
    default_size: 512x512
    chat_connection: missing_connection
    chat_model: llama3.2
""".strip()
    )

    from server.mode_config import ModeConfigManager

    with pytest.raises(ValueError, match="unknown chat_connection 'missing_connection'"):
        ModeConfigManager(str(tmp_path))
```

- [ ] **Step 2: Run the mode-config tests to verify they fail on the current implementation**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_mode_config.py -k "chat_connection or legacy_top_level_chat or flat_mode_chat_fields" -q
```

Expected:

- parse failures because `chat_connections`, flat `chat_*` fields, and `resolve_chat_config()` do not exist yet

- [ ] **Step 3: Implement the new config model and resolver**

```python
# server/mode_config.py
@dataclass
class ChatConnectionConfig:
    endpoint: str
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class ChatBackendConfig:
    endpoint: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None


@dataclass
class ModeConfig:
    name: str
    model: str
    loras: List[LoRAConfig] = field(default_factory=list)
    resolution_set: Optional[str] = None
    resolution_options: List[Dict[str, str]] = field(default_factory=list)
    default_size: str = "512x512"
    default_steps: int = 4
    default_guidance: float = 1.0
    maximum_len: Optional[int] = None
    chat_connection: Optional[str] = None
    chat_model: Optional[str] = None
    chat_max_tokens: Optional[int] = None
    chat_temperature: Optional[float] = None
    chat_system_prompt: Optional[str] = None
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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModesYAML:
    model_root: str
    lora_root: str
    default_mode: str
    resolution_sets: Dict[str, List[Dict[str, str]]]
    chat_connections: Dict[str, ChatConnectionConfig]
    modes: Dict[str, ModeConfig]
```

```python
# server/mode_config.py
def _parse_chat_connection_config(self, connection_name: str, chat_data: Dict[str, Any]) -> ChatConnectionConfig:
    if not isinstance(chat_data, dict):
        raise ValueError(f"Chat connection '{connection_name}' must be a mapping")
    endpoint = (chat_data.get("endpoint") or "").strip()
    if not endpoint:
        raise ValueError(f"Chat connection '{connection_name}' missing required field: endpoint")
    return ChatConnectionConfig(
        endpoint=endpoint,
        api_key_env=(chat_data.get("api_key_env") or "OPENAI_API_KEY").strip(),
    )


def resolve_chat_config(self, mode_name: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[ChatBackendConfig]:
    mode = self.get_mode(mode_name)
    if not mode.chat_connection:
        return None

    connection = self.config.chat_connections[mode.chat_connection]
    merged_model = self._normalize_optional_string((overrides or {}).get("model")) or mode.chat_model
    merged_system_prompt = self._merge_system_prompts(
        mode.chat_system_prompt,
        self._normalize_optional_string((overrides or {}).get("system_prompt")),
    )
    merged_max_tokens = (overrides or {}).get("max_tokens", mode.chat_max_tokens)
    merged_temperature = (overrides or {}).get("temperature", mode.chat_temperature)

    return ChatBackendConfig(
        endpoint=connection.endpoint,
        model=merged_model or "",
        api_key_env=connection.api_key_env,
        max_tokens=int(merged_max_tokens if merged_max_tokens is not None else 1024),
        temperature=float(merged_temperature if merged_temperature is not None else 0.7),
        system_prompt=merged_system_prompt,
    )
```

```python
# server/mode_config.py
if "chat" in data:
    raise ValueError("modes.yml contains legacy top-level 'chat'; migrate to 'chat_connections' plus mode chat_* fields")

raw_chat_connections = data.get("chat_connections") or {}
if not isinstance(raw_chat_connections, dict):
    raise ValueError("modes.yml field 'chat_connections' must be a mapping")

chat_connections = {
    name: self._parse_chat_connection_config(name, chat_data)
    for name, chat_data in raw_chat_connections.items()
}

chat_connection = mode_data.get("chat_connection")
chat_model = mode_data.get("chat_model")
if chat_connection and chat_connection not in chat_connections:
    raise ValueError(f"Mode '{mode_name}' references unknown chat_connection '{chat_connection}'")
if chat_connection and not chat_model:
    raise ValueError(f"Mode '{mode_name}' must set chat_model when chat_connection is configured")
if chat_model and not chat_connection:
    raise ValueError(f"Mode '{mode_name}' must set chat_connection when chat_model is configured")
```

- [ ] **Step 4: Run the mode-config tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_mode_config.py -k "chat_connection or legacy_top_level_chat or flat_mode_chat_fields" -q
```

Expected:

- all new config-shape and validation tests pass

- [ ] **Step 5: Commit**

```bash
git add server/mode_config.py tests/test_mode_config.py
git commit -m "refactor: decouple chat connections from mode names"
```

### Task 2: Move Advisor Digest To The Shared Resolver

**Files:**
- Modify: `server/advisor_service.py`
- Modify: `tests/test_advisor_service.py`

- [ ] **Step 1: Write the failing advisor tests for resolver use and prompt stacking**

```python
# tests/test_advisor_service.py
@pytest.mark.asyncio
async def test_generate_digest_uses_resolved_chat_config_and_appends_request_system_prompt():
    from server.advisor_service import AdvisorDigestRequest, generate_digest

    chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="llama3.2",
        api_key_env="OPENAI_API_KEY",
        max_tokens=256,
        temperature=0.6,
        system_prompt="Mode prompt",
    )
    mode = SimpleNamespace(maximum_len=120)
    config = SimpleNamespace(
        get_default_mode=lambda: "sdxl-general",
        get_mode=lambda name: mode,
        resolve_chat_config=lambda name, overrides=None: chat_cfg,
    )
    client_inst = SimpleNamespace(complete=AsyncMock(return_value="digest text"))

    req = AdvisorDigestRequest(
        gallery_id="gal_1",
        evidence={"version": 1, "gallery_id": "gal_1", "items": []},
        system_prompt="Request prompt",
        temperature=0.2,
        length_limit=400,
    )

    with patch("server.advisor_service.get_mode_config", return_value=config), \
            patch("server.advisor_service.ChatCompletionsClient", return_value=client_inst):
        await generate_digest(req)

    messages, kwargs = client_inst.complete.await_args.args[0], client_inst.complete.await_args.kwargs
    assert messages[0] == {"role": "system", "content": "Mode prompt"}
    assert messages[1] == {"role": "system", "content": "Request prompt"}
    assert kwargs["max_tokens"] == 120
    assert kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_generate_digest_treats_empty_string_system_prompt_override_as_missing():
    from server.advisor_service import AdvisorDigestRequest, generate_digest

    chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="llama3.2",
        api_key_env="OPENAI_API_KEY",
        max_tokens=256,
        temperature=0.6,
        system_prompt="Mode prompt",
    )
    mode = SimpleNamespace(maximum_len=None)
    config = SimpleNamespace(
        get_default_mode=lambda: "sdxl-general",
        get_mode=lambda name: mode,
        resolve_chat_config=lambda name, overrides=None: chat_cfg,
    )
    client_inst = SimpleNamespace(complete=AsyncMock(return_value="digest text"))

    req = AdvisorDigestRequest(
        gallery_id="gal_1",
        evidence={"version": 1, "gallery_id": "gal_1", "items": []},
        system_prompt="",
    )

    with patch("server.advisor_service.get_mode_config", return_value=config), \
            patch("server.advisor_service.ChatCompletionsClient", return_value=client_inst):
        await generate_digest(req)

    messages = client_inst.complete.await_args.args[0]
    assert messages[0] == {"role": "system", "content": "Mode prompt"}
    assert [m for m in messages if m["role"] == "system"] == [{"role": "system", "content": "Mode prompt"}]
```

- [ ] **Step 2: Run the advisor tests to verify they fail on the current resolver contract**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_advisor_service.py -q
```

Expected:

- failures because `AdvisorDigestRequest` does not yet accept the extra neutral override fields and `generate_digest()` still expects `get_chat_config()`

- [ ] **Step 3: Implement advisor resolver usage and neutral override plumbing**

```python
# server/advisor_service.py
class AdvisorDigestRequest(BaseModel):
    gallery_id: str
    evidence: Dict[str, Any]
    mode: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    length_limit: Optional[int] = Field(default=None, ge=1, le=4096)
```

```python
# server/advisor_service.py
mode_name = req.mode or mode_config.get_default_mode()
mode = mode_config.get_mode(mode_name)
chat_cfg = mode_config.resolve_chat_config(
    mode_name,
    overrides={
        "model": req.model,
        "temperature": req.temperature,
        "system_prompt": req.system_prompt,
    },
)
if chat_cfg is None:
    raise ValueError("advisor digest requires chat configuration for the active mode")

messages = _build_messages(req, fingerprint, effective_limit)
if chat_cfg.system_prompt:
    for prompt in chat_cfg.system_prompt.split("\n\n"):
        messages.insert(0, {"role": "system", "content": prompt})
```

```python
# server/advisor_service.py
effective_limit = req.length_limit
mode_limit = getattr(mode, "maximum_len", None)
if mode_limit is not None:
    effective_limit = min(effective_limit if effective_limit is not None else int(mode_limit), int(mode_limit))
```

- [ ] **Step 4: Run the advisor tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_advisor_service.py -q
```

Expected:

- existing digest tests still pass
- new stacking / empty-string tests pass

- [ ] **Step 5: Commit**

```bash
git add server/advisor_service.py tests/test_advisor_service.py
git commit -m "refactor: resolve advisor chat config from mode connections"
```

### Task 3: Move WebSocket Chat To The Shared Resolver And Neutral Overrides

**Files:**
- Modify: `server/ws_routes.py`
- Modify: `tests/test_ws_routes.py`

- [ ] **Step 1: Write the failing WebSocket tests for overrides and stack semantics**

```python
# tests/test_ws_routes.py
def test_chat_job_appends_request_system_prompt_after_mode_prompt(self):
    app.state.use_mode_system = True
    pool = MagicMock()
    pool.get_current_mode.return_value = "sdxl-general"
    app.state.worker_pool = pool

    fake_chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="llama3.2",
        api_key_env="OPENAI_API_KEY",
        max_tokens=128,
        temperature=0.4,
        system_prompt="Mode prompt",
    )
    complete_mock = AsyncMock(return_value="assistant reply")

    try:
        with patch("server.ws_routes.get_mode_config") as get_mode_config, \
                patch("backends.chat_client.ChatCompletionsClient.complete", new=complete_mock):
            get_mode_config.return_value = SimpleNamespace(
                get_mode=lambda name: SimpleNamespace(name=name, maximum_len=None),
                resolve_chat_config=lambda name, overrides=None: fake_chat_cfg,
                get_default_mode=lambda: "sdxl-general",
            )

            with client.websocket_connect("/v1/ws") as ws:
                ws.receive_json()
                ws.send_json({
                    "type": "job:submit",
                    "id": "chat-override",
                    "jobType": "chat",
                    "params": {
                        "prompt": "hello",
                        "stream": False,
                        "system_prompt": "Request prompt",
                    },
                })
                ws.receive_json()
                ws.receive_json()

        messages = complete_mock.await_args.args[0]
        assert messages[0] == {"role": "system", "content": "Mode prompt"}
        assert messages[1] == {"role": "system", "content": "Request prompt"}
    finally:
        app.state.use_mode_system = False
        app.state.worker_pool = None


def test_chat_job_passes_model_override_without_connection_cross_validation(self):
    app.state.use_mode_system = True
    pool = MagicMock()
    pool.get_current_mode.return_value = "sdxl-general"
    app.state.worker_pool = pool

    failing_complete = AsyncMock(side_effect=RuntimeError("backend rejected model"))
    fake_chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="override-model",
        api_key_env="OPENAI_API_KEY",
        max_tokens=128,
        temperature=0.4,
        system_prompt=None,
    )

    try:
        with patch("server.ws_routes.get_mode_config") as get_mode_config, \
                patch("backends.chat_client.ChatCompletionsClient.complete", new=failing_complete):
            get_mode_config.return_value = SimpleNamespace(
                get_mode=lambda name: SimpleNamespace(name=name, maximum_len=None),
                resolve_chat_config=lambda name, overrides=None: fake_chat_cfg,
                get_default_mode=lambda: "sdxl-general",
            )

            with client.websocket_connect("/v1/ws") as ws:
                ws.receive_json()
                ws.send_json({
                    "type": "job:submit",
                    "id": "chat-model-override",
                    "jobType": "chat",
                    "params": {
                        "prompt": "hello",
                        "stream": False,
                        "model": "override-model",
                    },
                })
                ws.receive_json()
                err = ws.receive_json()
                assert err["type"] == "job:error"
                assert "backend rejected model" in err["error"]
    finally:
        app.state.use_mode_system = False
        app.state.worker_pool = None
```

- [ ] **Step 2: Run the WebSocket chat tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_ws_routes.py -k "chat_job" -q
```

Expected:

- failures because `_resolve_chat_config()` still calls `get_chat_config()` and does not pass neutral overrides

- [ ] **Step 3: Implement resolver-driven WebSocket chat config**

```python
# server/ws_routes.py
def _resolve_chat_config(state, params: dict) -> Optional[tuple[ChatConfig, Optional[int]]]:
    mode_name = params.get("mode")
    mode_config = get_mode_config()
    if not mode_name:
        if getattr(state, "use_mode_system", False):
            pool = getattr(state, "worker_pool", None)
            if pool is not None:
                mode_name = pool.get_current_mode()
        if not mode_name:
            mode_name = mode_config.get_default_mode()
    if not mode_name:
        return None
    mode = mode_config.get_mode(mode_name)
    maximum_len = getattr(mode, "maximum_len", None)
    chat_cfg = mode_config.resolve_chat_config(
        mode_name,
        overrides={
            "model": params.get("model"),
            "max_tokens": params.get("max_tokens"),
            "temperature": params.get("temperature"),
            "system_prompt": params.get("system_prompt"),
        },
    )
    if chat_cfg is None:
        return None

    return (
        ChatConfig(
            endpoint=chat_cfg.endpoint,
            model=chat_cfg.model,
            api_key_env=chat_cfg.api_key_env,
            max_tokens=chat_cfg.max_tokens,
            temperature=chat_cfg.temperature,
            system_prompt=chat_cfg.system_prompt,
        ),
        maximum_len,
    )
```

```python
# server/ws_routes.py
def _build_chat_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        for item in system_prompt.split("\n\n"):
            if item:
                messages.append({"role": "system", "content": item})
    messages.append({"role": "user", "content": prompt})
    return messages
```

- [ ] **Step 4: Run the WebSocket chat tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_ws_routes.py -k "chat_job" -q
```

Expected:

- existing chat job tests continue to pass
- new override and error-passthrough tests pass

- [ ] **Step 5: Commit**

```bash
git add server/ws_routes.py tests/test_ws_routes.py
git commit -m "refactor: use mode chat resolver in websocket chat jobs"
```

### Task 4: Migrate Checked-In Config, Refresh Stale Docs, And Verify

**Files:**
- Modify: `conf/modes.yml`
- Modify: `server/mode_config.py`
- Modify: `tests/test_mode_config.py`
- Modify: `docs/superpowers/specs/2026-04-07-chat-completions-backend-design.md`
- Modify: `drift.lock`

- [ ] **Step 1: Write the failing round-trip and export tests for the new YAML shape**

```python
# tests/test_mode_config.py
def test_mode_config_save_config_round_trips_chat_connections(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
chat_connections:
  local_default:
    endpoint: http://localhost:11434/v1
    api_key_env: OPENAI_API_KEY
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    default_size: 512x512
    chat_connection: local_default
    chat_model: llama3.2
    chat_max_tokens: 512
    chat_temperature: 0.5
    chat_system_prompt: You are concise.
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    manager.save_config(manager.to_dict())

    saved = yaml.safe_load(cfg.read_text())
    assert "chat" not in saved
    assert saved["chat_connections"]["local_default"]["endpoint"] == "http://localhost:11434/v1"
    assert saved["modes"]["sdxl"]["chat_connection"] == "local_default"
    assert saved["modes"]["sdxl"]["chat_model"] == "llama3.2"
```

- [ ] **Step 2: Run the focused migration tests to verify they fail before the export path is updated**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_mode_config.py -k "round_trips_chat_connections" -q
```

Expected:

- failure because `save_config()` and `to_dict()` still emit or expect the legacy top-level `chat` block

- [ ] **Step 3: Update checked-in config, export paths, and stale chat-backend docs**

```yaml
# conf/modes.yml
chat_connections:
  local_default:
    endpoint: "http://node2.lan:8080"
    api_key_env: "OPENAI_API_KEY"

modes:
  SDXL:
    model: checkpoints/sdxl4GB2GBImprovedFP8_fp8FullCheckpoint.safetensors
    loader_format: single_file
    checkpoint_precision: fp8
    checkpoint_variant: sdxl-base
    scheduler_profile: native
    runtime_quantize: none
    runtime_offload: model
    runtime_attention_slicing: true
    runtime_enable_xformers: true
    resolution_set: sdxl
    default_size: 1024x1024
    recommended_size: 1024x1024
    default_steps: 11
    default_guidance: 3.0
    chat_connection: local_default
    chat_model: "gemma3-1b"
    chat_max_tokens: 750
    chat_temperature: 0.4
    chat_system_prompt: "You are a concise SDXL prompt advisor."
```

```python
# server/mode_config.py
def save_config(self, data: Dict[str, Any]):
    yaml_data = {
        "model_root": data["model_root"],
        "lora_root": data["lora_root"],
        "default_mode": data["default_mode"],
        "resolution_sets": data["resolution_sets"],
        "modes": {},
    }

    raw_chat_connections = data.get("chat_connections") or {}
    if raw_chat_connections:
        yaml_data["chat_connections"] = {
            name: {
                "endpoint": cfg["endpoint"],
                "api_key_env": cfg.get("api_key_env", "OPENAI_API_KEY"),
            }
            for name, cfg in raw_chat_connections.items()
        }

    for mode_name, mode_data in data["modes"].items():
        mode_entry = {
            "model": mode_data["model"],
            "default_size": mode_data.get("default_size", "512x512"),
            "default_steps": mode_data.get("default_steps", 4),
            "default_guidance": mode_data.get("default_guidance", 1.0),
        }
        if mode_data.get("maximum_len") is not None:
            mode_entry["maximum_len"] = mode_data["maximum_len"]
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
            if mode_data.get(cap_field) is not None:
                mode_entry[cap_field] = mode_data[cap_field]
        for key in ("chat_connection", "chat_model", "chat_max_tokens", "chat_temperature", "chat_system_prompt"):
            if mode_data.get(key) is not None:
                mode_entry[key] = mode_data[key]
```

```markdown
# docs/superpowers/specs/2026-04-07-chat-completions-backend-design.md
- replace the old nested `chat:` mode example with `chat_connections` plus flat mode `chat_*` fields
- update prose that says "Modes gain an optional `chat` block"
- keep the doc historically accurate about backend goals, but not stale about the config schema
```

- [ ] **Step 4: Run the full targeted verification and refresh drift**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_mode_config.py tests/test_advisor_service.py -q
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_ws_routes.py -k "chat_job" -q
drift refs server/mode_config.py
drift link docs/superpowers/specs/2026-04-07-chat-completions-backend-design.md server/mode_config.py
drift link docs/superpowers/specs/2026-04-07-chat-completions-backend-design.md server/ws_routes.py
drift check --changed server --changed conf --changed tests --changed docs/superpowers/specs
```

Expected:

- all targeted tests pass
- drift check reports `ok` for changed drift-managed docs

- [ ] **Step 5: Commit**

```bash
git add conf/modes.yml server/mode_config.py tests/test_mode_config.py docs/superpowers/specs/2026-04-07-chat-completions-backend-design.md drift.lock
git commit -m "docs: migrate chat config schema and refresh drift"
```

---

## Self-Review

- Spec coverage:
  - `chat_connections` + flat `chat_*` fields covered by Task 1 and Task 4.
  - shared resolver semantics covered by Task 1 implementation and Task 2 / Task 3 consumer updates.
  - additive `system_prompt` stacking and empty-string handling covered by Task 2 and Task 3 tests.
  - no connection override and no model/connection cross-validation covered by Task 3 tests and resolver design.
  - manual migration and `save_config()` / `to_dict()` updates covered by Task 4.
  - fresh-per-request client lifecycle covered by Task 2 and Task 3 architecture and no caching work is introduced.
- Placeholder scan:
  - no placeholder markers remain in task steps or code snippets.
- Type consistency:
  - the plan consistently uses `ChatConnectionConfig`, `ChatBackendConfig`, `ModeConfig.chat_*`, and `ModeConfigManager.resolve_chat_config()`.
