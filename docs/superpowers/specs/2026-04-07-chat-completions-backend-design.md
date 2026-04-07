# Chat Completions Backend Client Design

## Summary

This design adds a backend client that speaks the OpenAI chat completions API, so the existing WebSocket job protocol can route text messages to an LLM. The immediate goal is backend plumbing only: a new job type, a config-driven client, and a response shape that fits the existing `job:complete` contract. Frontend chat UI changes are deferred to a follow-up.

The insertion point is the WebSocket job dispatcher in `ws_routes.py`. Today it handles `generate`, `comfy`, and `sr` job types. This design adds a `chat` job type that submits a prompt to a configured OpenAI-compatible endpoint and streams or returns the assistant response through the same `job:complete` envelope the frontend already consumes.

## Goals

- Add a `chat` job type to the WebSocket job protocol
- Call any OpenAI-compatible chat completions endpoint (OpenAI, Ollama, vLLM, local llama-server)
- Configure the endpoint URL, model name, and API key through mode config or environment
- Return the assistant response through the existing `job:complete` WebSocket envelope
- Support streaming via `job:progress` messages for incremental token delivery
- Keep the existing image generation pipeline completely untouched

## Non-Goals

- Frontend chat UI changes (follow-up work)
- Multi-turn conversation context management (future)
- Tool use, function calling, or agent loops (future)
- Remote image generation backends
- Modifying the inbound compat endpoints in `compat_endpoints.py`
- Supporting non-OpenAI-compatible protocols (llama.cpp native, TGI native, etc.)
- Adding a new HTTP REST endpoint for chat (WebSocket is the primary path)

## Current State

- The WebSocket dispatcher (`ws_routes.py:120`) routes on `jobType`: `generate`, `comfy`, `sr`
- Each job type follows the same lifecycle: `job:ack` -> optional `job:progress` -> `job:complete` or `job:error`
- Mode config (`modes.yaml`) defines model paths and generation parameters but has no concept of a chat backend
- The frontend sends `job:submit` with `jobType` and `params` and handles the response envelope generically
- No OpenAI client library is currently in the dependency tree

## Proposed Approach

Add the chat completions client as a thin service layer, similar to how super-resolution is structured: a standalone service with its own config that plugs into the WebSocket job dispatcher without touching the generation pipeline.

The service:
- reads config from mode-level fields or environment variables
- makes HTTP calls to an OpenAI-compatible `/v1/chat/completions` endpoint
- normalizes responses into the existing job envelope shape

Modes gain an optional `chat` block. When a mode has a `chat` block, the frontend can submit `jobType: "chat"` jobs for that mode. Image generation and chat are mutually exclusive per job, not per mode.

## Design

### 1. Mode config extension for chat backends

Files in scope:

- `server/mode_config.py`
- `conf/modes.yaml.example`
- tests for mode config parsing

Design:

- Add an optional `chat` block to mode config:

```yaml
modes:
  sdxl-general:
    model: checkpoints/sdxl-base-1.0.safetensors
    default_size: "1024x1024"
    default_steps: 30
    default_guidance: 7.5
    chat:
      endpoint: "http://localhost:11434/v1"
      model: "llama3.2"
```

- The `chat` block contains:
  - `endpoint` (required): base URL for the OpenAI-compatible API (e.g. `http://localhost:11434/v1`, `https://api.openai.com/v1`)
  - `model` (required): model name to pass in the completions request
  - `api_key_env` (optional): name of an environment variable holding the API key, defaults to `OPENAI_API_KEY`. Omit or leave unset for local endpoints that need no auth.
  - `max_tokens` (optional): default max completion tokens, defaults to 1024
  - `temperature` (optional): default temperature, defaults to 0.7
  - `system_prompt` (optional): system message prepended to every request

- Add a `ChatConfig` dataclass in `mode_config.py`:

```python
@dataclass
class ChatConfig:
    endpoint: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None
```

- `ModeConfig` gains `chat: Optional[ChatConfig] = None`.

- Validation at config load:
  - If `chat` is present, `endpoint` and `model` are required
  - `endpoint` must be a valid URL prefix

- Expose `chat` presence (but not secrets) through `/api/modes` so the frontend knows which modes support chat.

Expected outcome:

- chat backend config lives alongside existing mode config
- no separate config file or global chat settings needed
- modes without a `chat` block behave exactly as before

### 2. Chat completions client service

Files in scope:

- `backends/chat_client.py` (new)
- `requirements.txt`
- tests for the client

Design:

- Add `httpx` to dependencies (async HTTP client, no OpenAI SDK dependency needed).

- Create `ChatCompletionsClient`:

```python
class ChatCompletionsClient:
    """Async client for OpenAI-compatible chat completions."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self._api_key = os.environ.get(config.api_key_env, "")

    async def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a chat completions request. Returns the assistant message content."""
        ...

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completions request. Yields content deltas."""
        ...
```

- The client constructs standard OpenAI chat completions requests:
  - `POST {endpoint}/chat/completions`
  - Body: `{ model, messages, max_tokens, temperature, stream }`
  - Auth: `Authorization: Bearer {api_key}` when key is non-empty

- `complete()` sends `stream: false`, returns the full `choices[0].message.content`.

- `stream()` sends `stream: true`, yields each `choices[0].delta.content` from SSE chunks, following the standard `data: {...}` / `data: [DONE]` protocol.

- Error handling:
  - HTTP 401/403: raise with "authentication failed" context
  - HTTP 429: raise with "rate limited" context
  - HTTP 5xx / connection errors: raise with endpoint context
  - All errors surface as `job:error` through the existing envelope

- The client is stateless per request. Multi-turn context is the caller's responsibility (deferred to future work).

Expected outcome:

- clean async client with no OpenAI SDK dependency
- works with any endpoint that speaks the OpenAI chat completions contract
- streaming support ready for incremental token delivery

### 3. WebSocket `chat` job type

Files in scope:

- `server/ws_routes.py`
- tests for the chat job handler

Design:

- Add a `"chat"` case to the `job:submit` dispatcher alongside `generate`, `comfy`, and `sr`:

```python
elif job_type == "chat":
    t = asyncio.create_task(_run_chat(ws, client_id, job_id, params))
    _track_task(job_id, t)
```

- `_run_chat` handler:
  1. Resolve the active mode's `ChatConfig` from mode config
  2. If no chat config, send `job:error` with "chat not configured for this mode"
  3. Build the messages list: `[system_prompt (if configured), {"role": "user", "content": params["prompt"]}]`
  4. Create a `ChatCompletionsClient` (or reuse a cached one per endpoint)
  5. Stream the response:
     - Each chunk sends `job:progress` with `{ type: "job:progress", jobId, delta: "token text" }`
     - On completion, send `job:complete` with `{ type: "job:complete", jobId, outputs: [{ text: "full response" }], meta: { model, endpoint_base } }`
  6. On error, send `job:error` as usual

- The `params` for a chat job:
  - `prompt` (required): the user message text
  - `stream` (optional, default `true`): whether to stream tokens

- The `job:complete` shape for chat uses `outputs[].text` instead of `outputs[].url`, distinguishing text responses from image responses. The frontend already handles `outputs` as an array; adding a `text` field is additive.

Expected outcome:

- chat jobs follow the same lifecycle as generation jobs
- streaming tokens arrive as `job:progress` deltas
- the frontend can distinguish chat results from image results by checking for `text` vs `url` in outputs

### 4. Expose chat capability in `/api/modes`

Files in scope:

- `server/model_routes.py`

Design:

- Add `chat_enabled: bool` to the serialized mode in `/api/modes`.
- Derive it from `mode.chat is not None`.
- Do not expose `endpoint`, `model`, or key config to the frontend — the frontend only needs to know whether chat is available for a given mode.

Expected outcome:

- frontend can conditionally enable a chat input for modes that support it
- no secrets leak through the API

## Testing Strategy

### Unit tests

- `ChatCompletionsClient`: mock HTTP responses for both streaming and non-streaming, assert correct request shape, auth header presence/absence, error mapping
- Mode config: parse `chat` blocks, validate required fields, verify omitted `chat` results in `None`
- WS handler: mock the client, assert `job:ack` / `job:progress` / `job:complete` sequence for a chat job; assert `job:error` when chat is not configured

### Integration smoke test

- Start with a local Ollama or llama-server instance
- Configure a mode with `chat.endpoint` pointing to it
- Submit a `job:submit` with `jobType: "chat"` over WebSocket
- Assert response arrives through `job:complete`

### Manual validation

1. Configure an SDXL mode with a `chat` block pointing to Ollama
2. Submit a chat job via WebSocket (curl or wscat)
3. Verify streaming tokens arrive as `job:progress`
4. Verify `job:complete` contains the full response text
5. Verify image generation still works normally for the same mode
6. Verify a mode without `chat` returns `job:error` for chat jobs

## Risks And Tradeoffs

### httpx vs openai SDK

Using `httpx` directly instead of the `openai` Python package keeps the dependency lightweight and avoids version coupling. The OpenAI chat completions wire format is stable and simple enough that a direct HTTP client is preferable. If the protocol surface grows (tool use, structured outputs), revisit.

### No multi-turn context

This design is stateless per request — each chat job sends one user message. Multi-turn conversation history must be managed by the caller (the frontend, in a future revision). This is intentional: the backend should not hold conversation state for WebSocket clients that may disconnect.

### Chat and generation coexist per mode

A mode can have both an image model and a chat endpoint. This is by design — a mode represents a workspace configuration, not a single capability. The frontend decides which job type to submit. If this creates confusion, a future revision can add mode-level `capabilities` flags.

### Streaming adds WebSocket message volume

Each token chunk is a `job:progress` message. For a typical LLM response this might be 200-500 messages. The existing WebSocket hub handles this fine — generation jobs already send progress updates. If volume becomes a problem, a future revision can batch deltas.

## Rollout

Implement in this order:

1. Add `ChatConfig` to mode config with parsing and validation
2. Implement `ChatCompletionsClient` with streaming support
3. Wire the `chat` job type into the WebSocket dispatcher
4. Expose `chat_enabled` in `/api/modes`
5. Add tests and manual validation

## Acceptance

This work is complete when:

- a mode with a `chat` block can receive `jobType: "chat"` over WebSocket
- the backend streams tokens from an OpenAI-compatible endpoint as `job:progress` messages
- `job:complete` delivers the full assistant response as `outputs[].text`
- modes without `chat` reject chat jobs with a clear error
- image generation is completely unaffected
- `/api/modes` reports which modes support chat without leaking endpoint details
