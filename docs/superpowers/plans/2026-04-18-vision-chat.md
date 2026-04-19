# Vision Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eye icon to the chat composer that sends the currently selected image (longest-edge resized, as base64) alongside any draft text to a vision-capable LLM delegate, streaming the response into a CHAT bubble while generation continues in parallel.

**Architecture:** Vision delegates are configured in `modes.yml` via four new optional fields on `chat_delegates` entries (`vision`, `vision_system_prompt`, `vision_default_prompt`, `vision_resize`). The frontend resizes the active image on an offscreen canvas, encodes it as base64, and forwards it through the existing `useChatJob` WS hook (which already handles streaming). The backend detects `image_b64` in job params and builds an OpenAI multimodal message.

**Tech Stack:** Python dataclasses + PyYAML (config), FastAPI WebSocket (backend), React + Vite (frontend), canvas API (image resize), Vitest (frontend tests), pytest (backend tests)

---

## Files

| File | Action | Responsibility |
| --- | --- | --- |
| `server/mode_config.py` | Modify | Add 4 vision fields to `ChatDelegateConfig`; parse them in `_parse_chat_delegate_config`; emit them from `to_dict` |
| `server/ws_routes.py` | Modify | Widen `_build_chat_messages` types; add vision branch to `_run_chat` |
| `backends/chat_client.py` | Modify | Widen `messages` type annotations to `List[Dict[str, Any]]` |
| `server/model_routes.py` | Modify | Expose `vision_enabled`, `vision_resize`, `vision_default_prompt` per mode in `/api/modes` |
| `lcm-sr-ui/src/hooks/useChatJob.js` | Modify | Forward optional `image_b64` param in `job:submit` message |
| `lcm-sr-ui/src/utils/resizeImageToLongestEdge.js` | Create | Pure async utility: load image URL → canvas → base64, longest-edge constrained |
| `lcm-sr-ui/src/App.jsx` | Modify | Add `visionImageUrl`, `visionEnabled`, `visionResize`, `visionDefaultPrompt` to `slashCtx` |
| `lcm-sr-ui/src/components/chat/MessageComposer.jsx` | Modify | Add Eye icon button + `handleVisionSend` |
| `conf/modes.yml` | Modify | Add vision fields to `sdxl_advisor` delegate |
| `tests/test_mode_config.py` | Modify | Tests for vision field parsing and `to_dict` output |
| `tests/test_ws_routes.py` | Modify | Tests for vision gating and multimodal message building |
| `lcm-sr-ui/src/utils/resizeImageToLongestEdge.test.js` | Create | Unit tests for resize utility |
| `lcm-sr-ui/src/hooks/useChatJob.test.jsx` | Modify | Test that `image_b64` is forwarded in `job:submit` |

---

## Task 1: Config — ChatDelegateConfig vision fields

**Files:**
- Modify: `server/mode_config.py`
- Modify: `tests/test_mode_config.py`

- [ ] **Step 1: Write failing test for vision field parsing**

Add to `tests/test_mode_config.py`:

```python
def test_mode_config_parses_chat_delegate_vision_fields(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text("""
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
chat_connections:
  local:
    endpoint: http://localhost:8080/v1
chat_delegates:
  phi4_vision:
    connection: local
    model: phi4-mm-Q4_K_M
    vision: true
    vision_system_prompt: "You are a visual analyst."
    vision_default_prompt: "What do you see?"
    vision_resize: 768
modes:
  sdxl:
    model: checkpoints/sdxl.safetensors
    default_size: 512x512
    chat_delegate: phi4_vision
""".strip())

    from server.mode_config import ModeConfigManager
    manager = ModeConfigManager(str(tmp_path))
    d = manager.config.chat_delegates["phi4_vision"]
    assert d.vision is True
    assert d.vision_system_prompt == "You are a visual analyst."
    assert d.vision_default_prompt == "What do you see?"
    assert d.vision_resize == 768


def test_mode_config_chat_delegate_vision_defaults(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text("""
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
chat_connections:
  local:
    endpoint: http://localhost:8080/v1
chat_delegates:
  text_only:
    connection: local
    model: gemma3-1b
modes:
  sdxl:
    model: checkpoints/sdxl.safetensors
    default_size: 512x512
    chat_delegate: text_only
""".strip())

    from server.mode_config import ModeConfigManager
    manager = ModeConfigManager(str(tmp_path))
    d = manager.config.chat_delegates["text_only"]
    assert d.vision is False
    assert d.vision_system_prompt is None
    assert d.vision_default_prompt == "Describe this image."
    assert d.vision_resize == 512
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_mode_config.py::test_mode_config_parses_chat_delegate_vision_fields tests/test_mode_config.py::test_mode_config_chat_delegate_vision_defaults -v
```

Expected: `FAILED` — `ChatDelegateConfig` has no `vision` field.

- [ ] **Step 3: Add vision fields to `ChatDelegateConfig` dataclass**

In `server/mode_config.py`, update the `ChatDelegateConfig` dataclass (currently ends at `system_prompt`):

```python
@dataclass
class ChatDelegateConfig:
    """Named bundle: connection + model + inference params bound to one logical chat persona."""
    name: str
    connection: str          # key into chat_connections
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    vision: bool = False
    vision_system_prompt: Optional[str] = None
    vision_default_prompt: str = "Describe this image."
    vision_resize: int = 512
```

- [ ] **Step 4: Parse vision fields in `_parse_chat_delegate_config`**

In `server/mode_config.py`, at the end of `_parse_chat_delegate_config` (currently line ~383), extend the `return ChatDelegateConfig(...)` call:

```python
        vision = bool(delegate_data.get("vision", False))
        vision_system_prompt = self._normalize_optional_string(delegate_data.get("vision_system_prompt"))
        vision_default_prompt = str(delegate_data.get("vision_default_prompt", "Describe this image.")).strip() or "Describe this image."
        vision_resize_raw = self._parse_optional_int(delegate_data.get("vision_resize"), f"chat_delegate '{delegate_name}'", "vision_resize")
        vision_resize = vision_resize_raw if vision_resize_raw is not None else 512

        return ChatDelegateConfig(
            name=delegate_name,
            connection=connection,
            model=model,
            max_tokens=max_tokens if max_tokens is not None else 1024,
            temperature=temperature if temperature is not None else 0.7,
            system_prompt=system_prompt,
            vision=vision,
            vision_system_prompt=vision_system_prompt,
            vision_default_prompt=vision_default_prompt,
            vision_resize=vision_resize,
        )
```

- [ ] **Step 5: Emit vision fields from `to_dict`**

In `server/mode_config.py`, inside `to_dict` update the `chat_delegates` comprehension (currently lines ~621-628):

```python
            "chat_delegates": {
                delegate_name: {
                    "connection": delegate.connection,
                    "model": delegate.model,
                    "max_tokens": delegate.max_tokens,
                    "temperature": delegate.temperature,
                    "system_prompt": delegate.system_prompt,
                    "vision": delegate.vision,
                    "vision_system_prompt": delegate.vision_system_prompt,
                    "vision_default_prompt": delegate.vision_default_prompt,
                    "vision_resize": delegate.vision_resize,
                }
                for delegate_name, delegate in self.config.chat_delegates.items()
            },
```

Also add derived vision fields inside the per-mode dict in the `modes` comprehension (after `"chat_delegate": mode.chat_delegate`):

```python
                    "chat_delegate": mode.chat_delegate,
                    # Vision fields derived from the referenced delegate
                    **({
                        "vision_enabled": bool(_d.vision),
                        "vision_resize": _d.vision_resize,
                        "vision_default_prompt": _d.vision_default_prompt,
                    } if (_d := self.config.chat_delegates.get(mode.chat_delegate)) else {
                        "vision_enabled": False,
                        "vision_resize": 512,
                        "vision_default_prompt": "Describe this image.",
                    }),
```

- [ ] **Step 6: Write failing test for `to_dict` vision fields**

Add to `tests/test_mode_config.py`:

```python
def test_mode_config_to_dict_includes_vision_fields(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text("""
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
chat_connections:
  local:
    endpoint: http://localhost:8080/v1
chat_delegates:
  phi4:
    connection: local
    model: phi4-mm
    vision: true
    vision_resize: 256
    vision_default_prompt: "Describe."
modes:
  sdxl:
    model: checkpoints/sdxl.safetensors
    default_size: 512x512
    chat_delegate: phi4
""".strip())

    from server.mode_config import ModeConfigManager
    manager = ModeConfigManager(str(tmp_path))
    d = manager.to_dict()
    assert d["chat_delegates"]["phi4"]["vision"] is True
    assert d["chat_delegates"]["phi4"]["vision_resize"] == 256
    assert d["modes"]["sdxl"]["vision_enabled"] is True
    assert d["modes"]["sdxl"]["vision_resize"] == 256
    assert d["modes"]["sdxl"]["vision_default_prompt"] == "Describe."


def test_mode_config_to_dict_vision_fields_default_when_no_delegate(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text("""
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
modes:
  sdxl:
    model: checkpoints/sdxl.safetensors
    default_size: 512x512
""".strip())

    from server.mode_config import ModeConfigManager
    manager = ModeConfigManager(str(tmp_path))
    d = manager.to_dict()
    assert d["modes"]["sdxl"]["vision_enabled"] is False
    assert d["modes"]["sdxl"]["vision_resize"] == 512
    assert d["modes"]["sdxl"]["vision_default_prompt"] == "Describe this image."
```

- [ ] **Step 7: Run all config tests**

```bash
python -m pytest tests/test_mode_config.py -v
```

Expected: all 30 tests `PASSED`.

- [ ] **Step 8: Commit**

```bash
git add server/mode_config.py tests/test_mode_config.py
git commit -m "feat(config): add vision fields to ChatDelegateConfig"
```

---

## Task 2: useChatJob — forward image_b64

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useChatJob.js`
- Modify: `lcm-sr-ui/src/hooks/useChatJob.test.jsx`

- [ ] **Step 1: Write failing test**

Add to `lcm-sr-ui/src/hooks/useChatJob.test.jsx` (after existing tests):

```jsx
it('forwards image_b64 in job:submit params', () => {
  const { result } = renderHook(() => useChatJob());
  const onAck = vi.fn();

  act(() => {
    result.current.start({
      prompt: 'describe this',
      image_b64: 'abc123base64',
      onAck,
      onDelta: vi.fn(),
      onComplete: vi.fn(),
      onError: vi.fn(),
    });
  });

  expect(wsMock.client.send).toHaveBeenCalledWith(
    expect.objectContaining({
      type: 'job:submit',
      jobType: 'chat',
      params: expect.objectContaining({
        prompt: 'describe this',
        image_b64: 'abc123base64',
        stream: true,
      }),
    })
  );
});

it('omits image_b64 from params when not provided', () => {
  const { result } = renderHook(() => useChatJob());

  act(() => {
    result.current.start({
      prompt: 'hello',
      onAck: vi.fn(),
      onDelta: vi.fn(),
      onComplete: vi.fn(),
      onError: vi.fn(),
    });
  });

  const sent = wsMock.client.send.mock.calls[0][0];
  expect(sent.params).not.toHaveProperty('image_b64');
});
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd lcm-sr-ui && npx vitest run src/hooks/useChatJob.test.jsx 2>&1 | tail -20
```

Expected: `FAILED` — `image_b64` absent from sent params.

- [ ] **Step 3: Update `useChatJob.js` to accept and forward `image_b64`**

In `lcm-sr-ui/src/hooks/useChatJob.js`, update the `start` callback destructure and `wsClient.send` call:

```js
  const start = useCallback(({ prompt, image_b64, onAck, onDelta, onComplete, onError }) => {
```

And the `wsClient.send` call (currently `params: { prompt, stream: true }`):

```js
    wsClient.send({
      type: 'job:submit',
      id: corrId,
      jobType: 'chat',
      params: {
        prompt,
        ...(image_b64 !== undefined ? { image_b64 } : {}),
        stream: true,
      },
    });
```

- [ ] **Step 4: Run all useChatJob tests**

```bash
cd lcm-sr-ui && npx vitest run src/hooks/useChatJob.test.jsx 2>&1 | tail -20
```

Expected: all tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useChatJob.js lcm-sr-ui/src/hooks/useChatJob.test.jsx
git commit -m "feat(ws): forward image_b64 in chat job:submit params"
```

---

## Task 3: Backend — multimodal message building + vision gating

**Files:**
- Modify: `backends/chat_client.py`
- Modify: `server/ws_routes.py`
- Modify: `tests/test_ws_routes.py`

- [ ] **Step 1: Write failing tests for `_build_chat_messages` with list content**

Add to the `TestWsRoutes` class in `tests/test_ws_routes.py`:

```python
def test_build_chat_messages_text_only(self):
    from server.ws_routes import _build_chat_messages
    msgs = _build_chat_messages("hello", "You are concise.")
    assert msgs == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "hello"},
    ]

def test_build_chat_messages_list_content_passes_through(self):
    from server.ws_routes import _build_chat_messages
    user_content = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    msgs = _build_chat_messages(user_content, None)
    assert msgs == [{"role": "user", "content": user_content}]

def test_build_chat_messages_list_content_with_system(self):
    from server.ws_routes import _build_chat_messages
    user_content = [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]
    msgs = _build_chat_messages(user_content, "You are a visual analyst.")
    assert msgs[0] == {"role": "system", "content": "You are a visual analyst."}
    assert msgs[1] == {"role": "user", "content": user_content}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_ws_routes.py::TestWsRoutes::test_build_chat_messages_text_only tests/test_ws_routes.py::TestWsRoutes::test_build_chat_messages_list_content_passes_through tests/test_ws_routes.py::TestWsRoutes::test_build_chat_messages_list_content_with_system -v
```

Expected: `FAILED` — `_build_chat_messages` rejects list input.

- [ ] **Step 3: Widen `_build_chat_messages` in `server/ws_routes.py`**

Replace the current `_build_chat_messages` function (lines ~343–351):

```python
def _build_chat_messages(
    prompt: Union[str, List[Any]],
    system_prompt: Optional[str],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        for item in system_prompt.split("\n\n"):
            content = item.strip()
            if content:
                messages.append({"role": "system", "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages
```

Add `Union` and `Any` to the existing imports at the top of the file — `from typing import Any, Dict, Optional, List, Union` (currently `Union` is absent).

- [ ] **Step 4: Widen type annotations in `backends/chat_client.py`**

In `backends/chat_client.py`, change `List[Dict[str, str]]` → `List[Dict[str, Any]]` in four places:
- `_request_payload` parameter `messages`
- `_message_summary` parameter `messages`
- `complete` parameter `messages`
- `stream` parameter `messages`

Also add `Any` to the import line: `from typing import AsyncIterator, Dict, List, Optional, Any`

The actual change for each signature — example for `complete`:

```python
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
```

Apply identically to `stream`, `_request_payload`, and `_message_summary`.

- [ ] **Step 5: Write failing tests for vision gating in `_run_chat`**

Add to `TestWsRoutes` in `tests/test_ws_routes.py`:

```python
def test_chat_job_vision_rejected_when_delegate_not_vision_enabled(self):
    app.state.use_mode_system = True
    pool = MagicMock()
    pool.get_current_mode.return_value = "sdxl-general"
    app.state.worker_pool = pool

    fake_chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="gemma3-1b",
        api_key_env="OPENAI_API_KEY",
        max_tokens=128,
        temperature=0.4,
        system_prompt=None,
    )
    fake_delegate = SimpleNamespace(vision=False)

    try:
        with patch("server.ws_routes.get_mode_config") as gmc:
            gmc.return_value = SimpleNamespace(
                get_mode=lambda name: SimpleNamespace(
                    name=name, maximum_len=None, chat_delegate="text_advisor"
                ),
                resolve_chat_config=lambda name, overrides=None: fake_chat_cfg,
                get_default_mode=lambda: "sdxl-general",
                config=SimpleNamespace(
                    chat_delegates={"text_advisor": fake_delegate}
                ),
            )
            with client.websocket_connect("/v1/ws") as ws:
                ws.receive_json()
                ws.send_json({
                    "type": "job:submit",
                    "id": "vision-1",
                    "jobType": "chat",
                    "params": {
                        "prompt": "what do you see?",
                        "image_b64": "abc123",
                        "stream": False,
                    },
                })
                ack = ws.receive_json()
                assert ack["type"] == "job:ack"
                err = ws.receive_json()
                assert err["type"] == "job:error"
                assert "vision not enabled" in err["error"].lower()
    finally:
        app.state.use_mode_system = False
        app.state.worker_pool = None


def test_chat_job_vision_builds_multimodal_message(self):
    app.state.use_mode_system = True
    pool = MagicMock()
    pool.get_current_mode.return_value = "sdxl-general"
    app.state.worker_pool = pool

    fake_chat_cfg = SimpleNamespace(
        endpoint="http://localhost:11434/v1",
        model="phi4-mm",
        api_key_env="OPENAI_API_KEY",
        max_tokens=128,
        temperature=0.4,
        system_prompt="You are concise.",
    )
    fake_delegate = SimpleNamespace(
        vision=True,
        vision_system_prompt="You are a visual analyst.",
    )
    captured_messages = []

    async def fake_complete(messages, **kwargs):
        captured_messages.extend(messages)
        return "I see a cat."

    try:
        with patch("server.ws_routes.get_mode_config") as gmc, \
                patch("backends.chat_client.ChatCompletionsClient.complete", new=AsyncMock(side_effect=fake_complete)):
            gmc.return_value = SimpleNamespace(
                get_mode=lambda name: SimpleNamespace(
                    name=name, maximum_len=None, chat_delegate="phi4_vision"
                ),
                resolve_chat_config=lambda name, overrides=None: fake_chat_cfg,
                get_default_mode=lambda: "sdxl-general",
                config=SimpleNamespace(
                    chat_delegates={"phi4_vision": fake_delegate}
                ),
            )
            with client.websocket_connect("/v1/ws") as ws:
                ws.receive_json()
                ws.send_json({
                    "type": "job:submit",
                    "id": "vision-2",
                    "jobType": "chat",
                    "params": {
                        "prompt": "what do you see?",
                        "image_b64": "abc123",
                        "stream": False,
                    },
                })
                ack = ws.receive_json()
                assert ack["type"] == "job:ack"
                done = ws.receive_json()
                assert done["type"] == "job:complete"
                assert done["outputs"] == [{"text": "I see a cat."}]

        # Check multimodal message was built correctly
        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0] == {"type": "text", "text": "what do you see?"}
        assert user_msg["content"][1]["type"] == "image_url"
        assert user_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

        # System prompt should use vision_system_prompt
        sys_msg = next(m for m in captured_messages if m["role"] == "system")
        assert sys_msg["content"] == "You are a visual analyst."
    finally:
        app.state.use_mode_system = False
        app.state.worker_pool = None
```

- [ ] **Step 6: Run tests to confirm they fail**

```bash
python -m pytest tests/test_ws_routes.py::TestWsRoutes::test_chat_job_vision_rejected_when_delegate_not_vision_enabled tests/test_ws_routes.py::TestWsRoutes::test_chat_job_vision_builds_multimodal_message -v
```

Expected: `FAILED`.

- [ ] **Step 7: Add vision branch to `_run_chat` in `server/ws_routes.py`**

Replace the body of `_run_chat` starting after `chat_cfg, maximum_len = chat_context` (keep all existing logic, insert the vision block before `messages = _build_chat_messages(...)`):

```python
async def _run_chat(ws: WebSocket, client_id: str, job_id: str, params: dict) -> None:
    """Run a chat completions job via an OpenAI-compatible backend."""
    try:
        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Missing prompt"})
            return

        state = _get_app_state(ws)
        chat_context = _resolve_chat_config(state, params)
        if chat_context is None:
            await hub.send(
                client_id,
                {"type": "job:error", "jobId": job_id, "error": "chat not configured for this mode"},
            )
            return
        chat_cfg, maximum_len = chat_context

        image_b64 = params.get("image_b64")

        if image_b64:
            # Resolve delegate directly to access vision metadata
            # (_resolve_chat_config returns ChatConfig which drops delegate fields)
            _mode_config = get_mode_config()
            _mode_name = params.get("mode") or (
                state.worker_pool.get_current_mode()
                if getattr(state, "use_mode_system", False) and getattr(state, "worker_pool", None)
                else None
            ) or _mode_config.get_default_mode()
            _mode = _mode_config.get_mode(_mode_name)
            _delegate = (
                _mode_config.config.chat_delegates.get(_mode.chat_delegate)
                if _mode.chat_delegate else None
            )
            if not (_delegate and _delegate.vision):
                await hub.send(
                    client_id,
                    {"type": "job:error", "jobId": job_id, "error": "Vision not enabled for this delegate"},
                )
                return
            effective_system = (
                _delegate.vision_system_prompt if _delegate.vision_system_prompt
                else chat_cfg.system_prompt
            )
            user_content: Any = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        else:
            effective_system = chat_cfg.system_prompt
            user_content = prompt

        client = ChatCompletionsClient(chat_cfg)
        stream = bool(params.get("stream", True))
        max_tokens = chat_cfg.max_tokens
        if maximum_len is not None:
            max_tokens = min(max_tokens, int(maximum_len))
        temperature = chat_cfg.temperature
        messages = _build_chat_messages(user_content, effective_system)

        if stream:
            chunks: List[str] = []
            async for delta in client.stream(messages, max_tokens=max_tokens, temperature=temperature):
                chunks.append(delta)
                await hub.send(
                    client_id,
                    {"type": "job:progress", "jobId": job_id, "delta": delta},
                )
            full_text = "".join(chunks)
        else:
            full_text = await client.complete(messages, max_tokens=max_tokens, temperature=temperature)

        await hub.send(
            client_id,
            {
                "type": "job:complete",
                "jobId": job_id,
                "outputs": [{"text": full_text}],
                "meta": {
                    "model": chat_cfg.model,
                    "endpoint_base": chat_cfg.endpoint.rstrip("/"),
                },
            },
        )
    except Exception as e:
        logger.error("Chat job %s failed: %s", job_id, e, exc_info=True)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": str(e)})
```

- [ ] **Step 8: Run all ws_routes tests**

```bash
python -m pytest tests/test_ws_routes.py -v 2>&1 | tail -30
```

Expected: all tests `PASSED`.

- [ ] **Step 9: Commit**

```bash
git add server/ws_routes.py backends/chat_client.py tests/test_ws_routes.py
git commit -m "feat(backend): add vision branch to _run_chat with multimodal message building"
```

---

## Task 4: model_routes — expose vision fields per mode

**Files:**
- Modify: `server/model_routes.py`

- [ ] **Step 1: Add vision fields to the `/api/modes` mode dict**

In `server/model_routes.py`, inside the `list_modes` route, the per-mode dict is built starting around line 155. Add three fields after `"chat_enabled"`:

```python
                "chat_enabled": bool(mode_data.get("chat_delegate")),
                "vision_enabled": bool(mode_data.get("vision_enabled")),
                "vision_resize": mode_data.get("vision_resize", 512),
                "vision_default_prompt": mode_data.get("vision_default_prompt", "Describe this image."),
```

- [ ] **Step 2: Run config + ws_routes tests to confirm nothing broken**

```bash
python -m pytest tests/test_mode_config.py tests/test_ws_routes.py -v 2>&1 | tail -10
```

Expected: all `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add server/model_routes.py
git commit -m "feat(api): expose vision_enabled, vision_resize, vision_default_prompt in /api/modes"
```

---

## Task 5: Frontend — `resizeImageToLongestEdge` utility

**Files:**
- Create: `lcm-sr-ui/src/utils/resizeImageToLongestEdge.js`
- Create: `lcm-sr-ui/src/utils/resizeImageToLongestEdge.test.js`

- [ ] **Step 1: Write failing tests**

Create `lcm-sr-ui/src/utils/resizeImageToLongestEdge.test.js`:

```js
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { resizeImageToLongestEdge } from './resizeImageToLongestEdge';

// jsdom doesn't implement canvas, so we stub it
function makeCanvas(drawWidth, drawHeight) {
  let capturedDraw = null;
  const canvas = {
    width: 0,
    height: 0,
    getContext: () => ({
      drawImage: (img, x, y, w, h) => { capturedDraw = { w, h }; },
    }),
    toDataURL: (type) => `data:${type};base64,fakebase64`,
    _capturedDraw: () => capturedDraw,
  };
  return canvas;
}

describe('resizeImageToLongestEdge', () => {
  let canvasInstances = [];

  beforeEach(() => {
    canvasInstances = [];
    vi.stubGlobal('document', {
      createElement: (tag) => {
        if (tag !== 'canvas') return {};
        const c = makeCanvas();
        canvasInstances.push(c);
        return c;
      },
    });
    vi.stubGlobal('Image', class {
      set src(url) {
        this.width = this._w ?? 1024;
        this.height = this._h ?? 768;
        this.onload?.();
      }
    });
  });

  it('constrains landscape image longest edge to maxPx', async () => {
    // 1024x768 → longest edge 1024, scale to 512x384
    const result = await resizeImageToLongestEdge('fake://landscape', 512);
    const canvas = canvasInstances[0];
    expect(canvas.width).toBe(512);
    expect(canvas.height).toBe(384);
    expect(result).toBe('fakebase64');
  });

  it('constrains portrait image longest edge to maxPx', async () => {
    vi.stubGlobal('Image', class {
      set src(url) {
        this.width = 768;
        this.height = 1024;
        this.onload?.();
      }
    });
    const result = await resizeImageToLongestEdge('fake://portrait', 512);
    const canvas = canvasInstances[0];
    expect(canvas.width).toBe(384);
    expect(canvas.height).toBe(512);
  });

  it('does not upscale images smaller than maxPx', async () => {
    vi.stubGlobal('Image', class {
      set src(url) {
        this.width = 256;
        this.height = 128;
        this.onload?.();
      }
    });
    const result = await resizeImageToLongestEdge('fake://small', 512);
    const canvas = canvasInstances[0];
    expect(canvas.width).toBe(256);
    expect(canvas.height).toBe(128);
  });

  it('handles square image', async () => {
    vi.stubGlobal('Image', class {
      set src(url) {
        this.width = 1024;
        this.height = 1024;
        this.onload?.();
      }
    });
    const result = await resizeImageToLongestEdge('fake://square', 512);
    const canvas = canvasInstances[0];
    expect(canvas.width).toBe(512);
    expect(canvas.height).toBe(512);
  });

  it('strips data: prefix from toDataURL result', async () => {
    const result = await resizeImageToLongestEdge('fake://any', 512);
    expect(result).not.toContain('data:');
    expect(result).not.toContain('image/png;base64,');
  });
});
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd lcm-sr-ui && npx vitest run src/utils/resizeImageToLongestEdge.test.js 2>&1 | tail -15
```

Expected: `FAILED` — module not found.

- [ ] **Step 3: Implement `resizeImageToLongestEdge.js`**

Create `lcm-sr-ui/src/utils/resizeImageToLongestEdge.js`:

```js
/**
 * Load an image from a URL onto an offscreen canvas, scale so the longest
 * edge equals maxPx (aspect ratio preserved, never upscaled), and return
 * the raw base64 PNG string (no "data:image/png;base64," prefix).
 *
 * @param {string} url
 * @param {number} maxPx  longest-edge pixel limit (default 512)
 * @returns {Promise<string>} bare base64
 */
export function resizeImageToLongestEdge(url, maxPx = 512) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const { naturalWidth: w, naturalHeight: h } = img;
      const longest = Math.max(w, h);
      const scale = longest > maxPx ? maxPx / longest : 1;
      const dw = Math.round(w * scale);
      const dh = Math.round(h * scale);

      const canvas = document.createElement('canvas');
      canvas.width = dw;
      canvas.height = dh;
      canvas.getContext('2d').drawImage(img, 0, 0, dw, dh);

      const dataUrl = canvas.toDataURL('image/png');
      // strip "data:image/png;base64," prefix
      resolve(dataUrl.replace(/^data:[^;]+;base64,/, ''));
    };
    img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    img.src = url;
  });
}
```

Note: in tests, `img.naturalWidth` / `img.naturalHeight` are set via the stub. When the Image class stub sets `this.width` / `this.height`, update the test to set `naturalWidth` / `naturalHeight` OR update the implementation to fall back to `img.width` / `img.height`. Use `img.naturalWidth || img.width` for browser compatibility:

```js
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
```

- [ ] **Step 4: Run resize tests**

```bash
cd lcm-sr-ui && npx vitest run src/utils/resizeImageToLongestEdge.test.js 2>&1 | tail -15
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/utils/resizeImageToLongestEdge.js lcm-sr-ui/src/utils/resizeImageToLongestEdge.test.js
git commit -m "feat(ui): add resizeImageToLongestEdge canvas utility"
```

---

## Task 6: slashCtx vision fields in App.jsx

**Files:**
- Modify: `lcm-sr-ui/src/App.jsx`

- [ ] **Step 1: Add vision fields to `slashCtx` useMemo**

In `lcm-sr-ui/src/App.jsx`, find the `slashCtx` useMemo (currently around line 807). Add four fields after `wsConnected`:

```js
  const slashCtx = useMemo(() => ({
    addMessage,
    updateMessage,
    createErrorMessage,
    activeMode: modeState.activeModeName ?? null,
    chatEnabled: Boolean(modeState.activeMode?.chat_enabled),
    inputMode,
    setInputMode,
    chatJob,
    runGenerate,
    onSendPrompt: defaultComposer.onSendPrompt,
    wsConnected: ws.connected,
    visionImageUrl: selectedMsg?.serverImageUrl || selectedMsg?.imageUrl || null,
    visionEnabled: Boolean(modeState.activeMode?.vision_enabled),
    visionResize: modeState.activeMode?.vision_resize ?? 512,
    visionDefaultPrompt: modeState.activeMode?.vision_default_prompt ?? 'Describe this image.',
  }), [
    addMessage,
    updateMessage,
    createErrorMessage,
    modeState.activeModeName,
    modeState.activeMode,
    inputMode,
    chatJob,
    runGenerate,
    defaultComposer.onSendPrompt,
    ws.connected,
    selectedMsg,
  ]);
```

- [ ] **Step 2: Run the frontend test suite to confirm nothing regressed**

```bash
cd lcm-sr-ui && npx vitest run 2>&1 | tail -15
```

Expected: all tests `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add lcm-sr-ui/src/App.jsx
git commit -m "feat(ui): add vision fields to slashCtx"
```

---

## Task 7: MessageComposer — eye icon and handleVisionSend

**Files:**
- Modify: `lcm-sr-ui/src/components/chat/MessageComposer.jsx`

- [ ] **Step 1: Add Eye import and handleVisionSend**

In `lcm-sr-ui/src/components/chat/MessageComposer.jsx`:

1. Add `Eye` to the lucide-react import (currently `import { MessageSquare, Image } from 'lucide-react'`):

```js
import { MessageSquare, Image, Eye } from 'lucide-react';
```

2. Add the resize utility import after the existing imports:

```js
import { resizeImageToLongestEdge } from '../../utils/resizeImageToLongestEdge';
import { nowId } from '../../utils/helpers';
import { MESSAGE_KINDS, MESSAGE_ROLES } from '../../utils/constants';
```

3. Add `handleVisionSend` inside the component body, after the `send` callback:

```js
  const handleVisionSend = React.useCallback(async () => {
    if (!slashCtx?.visionImageUrl) return;

    const prompt = draft.trim() || (slashCtx.visionDefaultPrompt ?? 'Describe this image.');
    const maxPx = slashCtx.visionResize ?? 512;

    let image_b64;
    try {
      image_b64 = await resizeImageToLongestEdge(slashCtx.visionImageUrl, maxPx);
    } catch (err) {
      slashCtx.addMessage(slashCtx.createErrorMessage(`Vision resize failed: ${err.message}`));
      return;
    }

    const userId = nowId();
    const assistantId = nowId();

    slashCtx.addMessage([
      { id: userId, role: MESSAGE_ROLES.USER, kind: MESSAGE_KINDS.CHAT, text: prompt, ts: Date.now() },
      { id: assistantId, role: MESSAGE_ROLES.ASSISTANT, kind: MESSAGE_KINDS.CHAT, text: '', streaming: true, jobId: null, ts: Date.now() },
    ]);

    let rafBuffer = '', rafPending = false, terminated = false;
    const flushDelta = () => {
      const chunk = rafBuffer; rafBuffer = ''; rafPending = false;
      if (chunk && !terminated) {
        slashCtx.updateMessage(assistantId, (prev) => ({ ...prev, text: prev.text + chunk }));
      }
    };

    const handle = slashCtx.chatJob.start({
      prompt,
      image_b64,
      onAck: ({ jobId }) => slashCtx.updateMessage(assistantId, { jobId }),
      onDelta: (text) => {
        if (terminated) return;
        rafBuffer += text;
        if (!rafPending) { rafPending = true; requestAnimationFrame(flushDelta); }
      },
      onComplete: ({ text }) => {
        terminated = true;
        slashCtx.updateMessage(assistantId, { text, streaming: false, jobId: null });
      },
      onError: (errMsg) => {
        terminated = true;
        slashCtx.updateMessage(assistantId, (prev) => ({
          ...prev,
          kind: MESSAGE_KINDS.ERROR,
          prevText: prev.text,
          text: errMsg,
          streaming: false,
          jobId: null,
        }));
      },
    });

    slashCtx.updateMessage(assistantId, { cancelHandle: handle });
    setDraft('');
  }, [draft, slashCtx]);
```

- [ ] **Step 2: Add Eye button to the JSX**

In the return JSX, find the button column (currently contains Send and Cancel). Add the Eye button between them:

```jsx
        <div className="flex flex-col mt-2 gap-2">
          <Button
            onClick={send}
            disabled={!draft.trim()}
            className="relative overflow-hidden"
          >
            Send
          </Button>

          {slashCtx?.visionEnabled && (
            <Button
              variant="ghost"
              size="icon"
              disabled={!slashCtx.visionImageUrl}
              onClick={handleVisionSend}
              title="Send image to vision model"
              type="button"
            >
              <Eye className="h-4 w-4" />
            </Button>
          )}

          {onCancelAll ? (
            <Button variant="secondary" onClick={onCancelAll} type="button">
              Cancel
            </Button>
          ) : null}
        </div>
```

- [ ] **Step 3: Run full frontend test suite**

```bash
cd lcm-sr-ui && npx vitest run 2>&1 | tail -15
```

Expected: all tests `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add lcm-sr-ui/src/components/chat/MessageComposer.jsx
git commit -m "feat(ui): add vision eye icon and handleVisionSend to MessageComposer"
```

---

## Task 8: modes.yml — enable vision on sdxl_advisor

**Files:**
- Modify: `conf/modes.yml`

- [ ] **Step 1: Add vision fields to `sdxl_advisor` delegate**

In `conf/modes.yml`, update the `sdxl_advisor` entry under `chat_delegates`:

```yaml
chat_delegates:
  sdxl_advisor:
    connection: local_default
    model: "gemma3-1b"
    max_tokens: 750
    temperature: 0.4
    system_prompt: "You are a concise SDXL prompt advisor."
    vision: true
    vision_system_prompt: "You are a visual analyst. Describe what you see concisely, then suggest how the SDXL prompt could be improved."
    vision_default_prompt: "Describe this image."
    vision_resize: 512
```

- [ ] **Step 2: Run all Python tests**

```bash
python -m pytest tests/test_mode_config.py tests/test_ws_routes.py -v 2>&1 | tail -10
```

Expected: all `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add conf/modes.yml
git commit -m "feat(config): enable vision on sdxl_advisor delegate (phi4-mm)"
```

---

## Self-Review

**Spec coverage:**
- ✅ `ChatDelegateConfig` 4 new fields — Task 1
- ✅ `_parse_chat_delegate_config` parses them — Task 1
- ✅ `to_dict` emits vision fields on delegates and derived fields on modes — Task 1
- ✅ `/api/modes` exposes `vision_enabled`, `vision_resize`, `vision_default_prompt` — Task 4
- ✅ `_build_chat_messages` type widened — Task 3
- ✅ `chat_client.py` type annotations widened — Task 3
- ✅ `_run_chat` vision gating + multimodal message — Task 3
- ✅ `useChatJob` forwards `image_b64` — Task 2 (gap caught during review, not in original spec)
- ✅ `resizeImageToLongestEdge` longest-edge resize, no upscale — Task 5
- ✅ `slashCtx` gains vision fields — Task 6
- ✅ Eye icon only visible when `visionEnabled` — Task 7
- ✅ Eye icon disabled when no active image — Task 7
- ✅ Draft used as prompt, `visionDefaultPrompt` when empty — Task 7
- ✅ `modes.yml` updated — Task 8
- ✅ Concurrent generation unaffected (vision is independent async WS task)

**Placeholder scan:** None found.

**Type consistency:**
- `image_b64` used consistently as the key name across `useChatJob.js`, `_run_chat`, and `handleVisionSend`
- `visionImageUrl`, `visionEnabled`, `visionResize`, `visionDefaultPrompt` consistent between `slashCtx` and `MessageComposer`
- `resizeImageToLongestEdge` function name consistent between utility file, test, and import in composer
