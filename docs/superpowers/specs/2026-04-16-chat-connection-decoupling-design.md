# Chat Connection Decoupling Design

## Summary

This design removes the current runtime coupling between chat selection and mode names. Today, top-level chat config is keyed by mode name, so the active mode doubles as both the image-generation mode selector and the chat backend selector. That shape makes transport reuse awkward and pushes future routing changes toward more coupling instead of less.

The replacement keeps configuration flat and snake_case while separating transport from behavioral defaults:

- top-level `chat_connections` stores reusable connection settings such as `endpoint` and `api_key_env`
- each mode stores flat chat defaults such as `chat_connection`, `chat_model`, `chat_max_tokens`, `chat_temperature`, and `chat_system_prompt`
- request payloads may override behavioral fields with neutral names such as `model`, `max_tokens`, `temperature`, and `system_prompt`

This keeps connection ownership in mode config, avoids introducing a profile abstraction, and allows request-time generative feedback without letting callers reroute infrastructure.

## Goals

- Replace mode-name-keyed chat config with reusable named connections
- Keep config flat and snake_case
- Avoid introducing a second abstraction layer such as chat profiles
- Let modes own their default chat model and prompt behavior
- Support request-time overrides for behavioral completion parameters
- Reuse one resolution path for advisor and WebSocket chat callers
- Make migration and validation explicit instead of silently supporting multiple config shapes

## Non-Goals

- Adding a `chat_models` or profile system
- Allowing request payloads to select or override connection ids
- Building a frontend control surface for all chat parameters in this change
- Expanding chat transport beyond the existing OpenAI-compatible backend client
- Generalizing mode config into a larger provider registry beyond chat

## Current State

- [`server/mode_config.py`](/Users/darkbit1001/workspace/Stability-Toys/server/mode_config.py:75) defines `ModesYAML.chat` as a mapping of mode name to `ChatBackendConfig`.
- [`server/mode_config.py`](/Users/darkbit1001/workspace/Stability-Toys/server/mode_config.py:476) exposes `get_chat_config(mode_name)`, which resolves chat by mode name rather than by reusable transport id.
- [`server/advisor_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/advisor_service.py:45) and [`server/ws_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/server/ws_routes.py:318) both depend on that mode-name lookup shape.
- [`conf/modes.yml`](/Users/darkbit1001/workspace/Stability-Toys/conf/modes.yml:1) already keeps most mode fields flat and snake_case, so nested `chat` reintroduction would move against the current YAML style.

The current shape works for one-off mode-specific chat config, but it encodes the wrong boundary. A mode should choose a connection. It should not implicitly be the connection namespace.

## Proposed Approach

Introduce named chat connections at the top level and move mode-specific chat behavior into flat mode fields.

The canonical resolution path becomes:

`mode -> chat_connection id -> connection settings + mode defaults -> request overrides -> final chat config`

This is intentionally simpler than a profile-based design:

- `chat_connections` handles transport and auth lookup
- mode fields handle model and behavioral defaults
- request payloads handle one-off completion tuning

No separate profile entity is needed because the desired reuse boundary is connection data, not reusable behavior bundles.

## Config Design

### Top-Level YAML

Replace top-level `chat` with top-level `chat_connections`.

Proposed shape:

```yaml
chat_connections:
  local_default:
    endpoint: "http://node2.lan:8080"
    api_key_env: "OPENAI_API_KEY"
```

Each connection entry may contain:

- `endpoint` required
- `api_key_env` optional, default `OPENAI_API_KEY`
- future transport-only settings such as timeouts or auth variants if later needed

Connections do not contain:

- `model`
- `max_tokens`
- `temperature`
- `system_prompt`

Those remain mode-owned.

### Mode YAML

Modes gain flat optional chat fields:

```yaml
modes:
  SDXL:
    model: checkpoints/sdxl4GB2GBImprovedFP8_fp8FullCheckpoint.safetensors
    default_size: 1024x1024
    chat_connection: local_default
    chat_model: gemma3-1b
    chat_max_tokens: 750
    chat_temperature: 0.4
    chat_system_prompt: "You are a concise SDXL prompt advisor."
```

Rules:

- `chat_connection` enables chat for the mode
- `chat_model` is required when `chat_connection` is set
- `chat_max_tokens`, `chat_temperature`, and `chat_system_prompt` are optional
- modes with no `chat_connection` are non-chat modes

This preserves a flat mode shape and keeps chat defaults visually aligned with the rest of the mode configuration.

## Runtime Resolution

The current `get_chat_config(mode_name)` abstraction is no longer the right boundary because it exposes the old coupling directly.

Recommended replacement in [`server/mode_config.py`](/Users/darkbit1001/workspace/Stability-Toys/server/mode_config.py:1):

- add a `ChatConnectionConfig` dataclass for transport settings
- extend `ModeConfig` with flat optional `chat_*` fields
- replace `ModesYAML.chat` with `ModesYAML.chat_connections`
- add a resolver that takes a mode name and returns the final chat config for that mode

The resolver should:

1. load the mode
2. return `None` if `chat_connection` is unset
3. validate that the referenced connection exists
4. combine connection transport fields with mode behavioral defaults
5. optionally accept caller-provided behavioral overrides

The resolved object can still be emitted as `ChatConfig` for consumers such as [`server/advisor_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/advisor_service.py:11) and [`server/ws_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/server/ws_routes.py:318), but its derivation should no longer depend on a mode-name-keyed YAML lookup.

## Request Override Design

Request payloads should remain provider-neutral and snake_case.

Allowed override names:

- `model`
- `max_tokens`
- `temperature`
- `system_prompt`

Not allowed:

- `chat_connection`
- connection id override by any other name

Rationale:

- callers can tune one completion request for generative feedback
- config continues to own routing, auth lookup, and endpoint selection
- request payloads describe invocation behavior rather than infrastructure

Precedence:

`request override -> mode chat_* field -> hardcoded fallback`

This rule should be applied consistently in both advisor and WebSocket chat execution paths.

## Validation Rules

Config load should fail fast on the following:

- top-level `chat_connections` is present but not a mapping
- connection entry missing `endpoint`
- mode references unknown `chat_connection`
- mode sets `chat_connection` but omits `chat_model`
- mode sets `chat_model` without `chat_connection`
- legacy top-level `chat` is still present
- legacy nested `modes.<name>.chat` is still present

Numeric parsing should remain explicit for:

- `chat_max_tokens`
- `chat_temperature`

Validation errors should continue to name the offending mode or connection id directly so migration failures are easy to fix.

## Migration Plan

The migration should be direct rather than compatibility-heavy.

### Legacy Shape

```yaml
chat:
  SDXL:
    endpoint: "http://node2.lan:8080"
    model: "gemma3-1b"
    api_key_env: "OPENAI_API_KEY"
    max_tokens: 750
    temperature: 0.4
    system_prompt: "You are a concise SDXL prompt advisor."
```

### New Shape

```yaml
chat_connections:
  local_default:
    endpoint: "http://node2.lan:8080"
    api_key_env: "OPENAI_API_KEY"

modes:
  SDXL:
    chat_connection: local_default
    chat_model: "gemma3-1b"
    chat_max_tokens: 750
    chat_temperature: 0.4
    chat_system_prompt: "You are a concise SDXL prompt advisor."
```

Implementation guidance:

- update the checked-in config to the new shape as part of the refactor
- remove support for top-level `chat`
- keep the existing rejection of nested `modes.<name>.chat`
- update save/export paths so the new shape round-trips cleanly

Failing fast is preferable here because dual-schema support would preserve the same ambiguity this design is trying to remove.

## Consumer Changes

### Advisor Path

[`server/advisor_service.py`](/Users/darkbit1001/workspace/Stability-Toys/server/advisor_service.py:41) should resolve the active mode's chat config through the new resolver instead of looking up chat config by mode name.

It should also accept neutral request overrides where appropriate for advisor digest generation:

- `model`
- `max_tokens`
- `temperature`
- `system_prompt`

If the advisor request surface does not yet expose all of these, the server-side resolution path should still be designed to accept them so future API expansion does not require another resolver redesign.

### WebSocket Chat Path

[`server/ws_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/server/ws_routes.py:299) should use the same resolver for `jobType: "chat"` handling.

The WebSocket `params` object may carry the neutral override names above. The handler should pass only those behavioral overrides into the resolver and must not allow endpoint or connection selection to come from the request.

## Testing Strategy

Add or update tests covering:

- parsing of top-level `chat_connections`
- round-trip save/export of `chat_connections` and flat mode `chat_*` fields
- rejection of unknown `chat_connection`
- rejection of incomplete mode chat config
- rejection of legacy top-level `chat`
- advisor resolution using mode -> connection mapping
- WebSocket chat resolution using the same resolver
- precedence of request overrides over mode defaults
- continued clean behavior for modes without chat enabled

This work should extend the existing tests in:

- [`tests/test_mode_config.py`](/Users/darkbit1001/workspace/Stability-Toys/tests/test_mode_config.py:84)
- [`tests/test_advisor_service.py`](/Users/darkbit1001/workspace/Stability-Toys/tests/test_advisor_service.py:20)
- [`tests/test_ws_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/tests/test_ws_routes.py:712)

## Risks And Tradeoffs

### Why not `chat_models`

`chat_models` would add a reusable behavior profile layer, but the stated requirement is narrower: keyed parameters into reusable connections. Adding a profile concept now would increase indirection without a concrete control surface that needs it.

### Why keep mode fields flat

Nested mode chat blocks would recreate the same shape that was recently moved away from and would make the YAML less consistent with other mode defaults. Flat mode fields are easier to diff, export, and reason about in the current config style.

### Why block connection override in requests

Allowing requests to select connections would blur the separation between invocation behavior and infrastructure routing. That would make policy drift, surprise auth failures, and endpoint inconsistency more likely.

## Tracking Plan

Once this spec is approved, implementation tracking should be created in Fiberplane as one parent issue with child issues for focused delivery:

1. Mode config schema refactor
2. Runtime resolver refactor for advisor and WebSocket consumers
3. Neutral request override support in chat execution paths
4. Config migration, tests, and docs refresh

If analysis work needs to be tracked separately from implementation, those child issues can each carry an initial analysis or validation checklist in their descriptions rather than creating a second parallel hierarchy.

## Success Criteria

- mode config no longer keys chat transport by mode name
- one named connection can be reused by multiple modes
- modes keep flat snake_case chat defaults
- request payloads can override behavioral fields with neutral names
- advisor and WebSocket chat use the same resolution semantics
- legacy chat config shape fails with a direct migration error
