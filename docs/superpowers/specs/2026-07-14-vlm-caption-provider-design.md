# VLM Caption Provider (`openai_vlm`) — Design

**FP issue:** STABL-rylcqort
**Predecessors:**
- `2026-07-11-describe-analysis-interface-design.md` (v1 contract, STABL-tlklfaxz)
- `2026-07-13-describe-transport-cli-design.md` (transport + CLI, STABL-ucomsfel)

**Status:** Authority artifact for the first real-provider track.

## Goal

Ship the first real `DescribeProvider`: caption tasks executed by a visual
LLM behind an OpenAI-compatible `chat/completions` endpoint, plugged into the
existing `build_providers` seam. Everything else — endpoint, orchestrator,
run model, CLI — is already live against `StubProvider` and does not change.

Success: with a delegate configured `provider: openai_vlm`,
`st describe ./photo.png --caption` returns a real model-generated caption;
with the field omitted, every existing config and test behaves exactly as
today.

## Ownership Model (delta only)

| Layer | Owns |
| --- | --- |
| `backends/analysis/vlm_caption.py` | provider implementation: message assembly, image-part building, response parsing, capability declaration |
| `backends/analysis/vlm_client.py` | multimodal `chat/completions` HTTP call: URL join, auth header, timeout, full-response return |
| `server/mode_config.py` | `provider` field + `options` mapping parsing and load-time validation |
| `server/analysis_routes.py` (`build_providers`) | provider selection per delegate; the seam this track was designed for |

The orchestrator, wire contracts, transport, and CLI are **unchanged**.

## Provider

`OpenAIVLMCaptionProvider` implements the existing `DescribeProvider`
protocol (`backends/analysis/providers.py`):

- `supports(task)` returns `task.kind == TaskKind.CAPTION` and nothing else.
  Routing a non-caption task to it therefore yields a `skipped` run with
  `analysis_no_supported_delegate` via the existing pre-dispatch gate.
- `run(provider_run)`:
  1. Build the image content part from `provider_run.target` (see Image
     Delivery).
  2. Assemble messages (see Prompting).
  3. POST to `<connection.endpoint>/chat/completions` via the VLM client.
  4. Return one `ProviderResult` containing **exactly one** `text`
     observation — `task_id`/`target_id` from the plan, `content` =
     `choices[0].message.content` — and `raw_output` set to the **full
     completion response JSON** (the contract's raw-output retention rule;
     never restructured).

### Failure semantics

The provider raises on every failure: HTTP non-2xx, transport error,
timeout, missing/empty `choices[0].message.content`, unresolvable
`asset_ref`. It performs **no retries** (a failed run is a failed run in
v1) and catches nothing it cannot handle. The orchestrator's existing
per-run isolation converts the raise into a `failed` run with
`analysis_run_failed` (provider detail in `error.message`), sibling runs
unaffected. **No new error codes and no orchestrator changes.**

## VLM Client

A dedicated minimal async client in `backends/analysis/vlm_client.py`.
`ChatCompletionsClient` is deliberately **not** reused or modified: it is
typed for text-only messages (`ChatMessage = Dict[str, str]`), returns only
the content string (insufficient for `raw_output`), and is a live dependency
of the chat delegates — reshaping it risks the chat surface for no
analysis-side gain.

The client mirrors `backends/chat_client.py` conventions:

- URL: `f"{endpoint.rstrip('/')}/chat/completions"`.
- Auth: `Authorization: Bearer <env>` when `connection.api_key_env` resolves
  to a non-empty environment value; header omitted otherwise.
- Timeout from the effective options (below).
- Accepts messages whose `content` may be a list of typed parts
  (`{"type": "text", ...}` / `{"type": "image_url", ...}`).
- Returns the **parsed full response dict** — the provider, not the client,
  extracts the caption text.
- Constructor accepts an optional `httpx` transport (or client factory) so
  tests inject `httpx.MockTransport`; production uses a fresh
  `httpx.AsyncClient` per call, matching `chat_client.py`. Connection reuse
  stays deferred (transport spec: caching keyed on config generation is a
  later concern behind the same seam).

## Image Delivery

Exactly-one-of on `DescribeTarget` is already enforced upstream.

- **`asset_ref` target:** `server.asset_store.get_store().resolve(ref)`
  yields bytes + metadata. The provider builds a base64 data-URI
  `image_url` part: `data:<media_type>;base64,<payload>`, `media_type`
  from store metadata with fallback `image/png`. A resolve failure raises →
  `analysis_run_failed` for that run only.
- **`url` target:** the URL string passes through verbatim as the
  `image_url` part. The VLM host fetches it. The server **never fetches
  remote URLs itself** — no SSRF surface, no new fetch path. If the VLM
  host cannot reach the URL, that surfaces as the provider call failing →
  `analysis_run_failed` with the endpoint's error detail.

## Prompting

- System message: a fixed default caption instruction
  (`"You are an image captioning assistant. Describe the image concisely
  and factually."`), overridable per delegate via `options.system_prompt`.
- User message content: the image part, plus — only when
  `CaptionParams.prompt` is set — a text part carrying the caller's
  guidance verbatim.
- `model` in the payload comes from the delegate's existing `model` field.

## Config

### `provider` field (delegate-level, opt-in)

```yaml
analysis_delegates:
  vlm_caption:
    connection: local_vlm
    kind: caption
    model: qwen2.5-vl
    provider: openai_vlm   # NEW; omitted -> stub
```

- Closed set: `stub` (default) | `openai_vlm`.
- **Omitted means `stub`:** every existing config, test, and deployment
  keeps today's behavior. Enabling the real provider is a deliberate
  deployment-config act — the same opt-in discipline as
  `conditioning.service: compel` (native default).
- Load-time validation, fail-fast at parse time in `mode_config.py`,
  matching the existing analysis config discipline:
  - unknown provider value fails load naming the delegate and the valid set;
  - `provider: openai_vlm` on a delegate whose `kind` is not `caption`
    fails load (provider/kind compatibility, same style as the
    route-key == delegate-kind invariant).

### `options` mapping (delegate-level, optional)

```yaml
    options:            # all keys optional
      max_tokens: 256
      temperature: 0.0
      timeout_s: 90
      system_prompt: "Describe the image for an art catalog."
```

| Key | Type | Default |
| --- | --- | --- |
| `max_tokens` | int > 0 | 512 |
| `temperature` | float ≥ 0 | 0.2 |
| `timeout_s` | float > 0 | 60 |
| `system_prompt` | non-empty str | fixed caption instruction |

- Unknown option keys fail config load naming the delegate and key.
- Type/range violations fail config load.
- `options` is accepted on any delegate regardless of `provider` (it is a
  delegate-tuning surface, not an `openai_vlm` exclusive), but only
  `openai_vlm` consumes it in this track; `StubProvider` ignores it.

### Round-trip persistence (required)

`provider` and `options` MUST survive every config write path, or a later
save silently erases the provider policy — the exact defect class the
contract track's review caught for the `analysis_*` sections themselves.
Current code serializes `analysis_delegates` as only
`connection`/`kind`/`model` in both `to_dict()` and `save_config()`'s
yaml rebuild (`server/mode_config.py`), so this track must extend all of:

- `ModeConfigManager.to_dict()` — emit `provider` and `options` on each
  delegate. To keep exports clean, defaults may be omitted (`provider`
  omitted when `stub`, `options` omitted when empty), since omission
  parses back to the same effective config.
- `save_config()`'s `yaml_data` rebuild — same emission rules.
- The bulk `PUT /api/modes` path (`ModesBulkSaveRequest` /
  `save_all_modes` in `server/model_routes.py`) — the analysis sections it
  round-trips must carry the new fields through both the
  omitted-backfill and explicit-payload branches.

Regression tests are part of this track's definition of done: a config
with `provider: openai_vlm` + populated `options` must survive
(1) `to_dict()` → `save_config()` → reload, and (2) a bulk
`PUT /api/modes` that omits the analysis sections, with the effective
provider selection identical afterward.

## Provider Selection (`build_providers`)

`build_providers` in `server/analysis_routes.py` gains access to
`analysis_connections` (signature change is server-internal — it is called
only by the describe endpoint) and switches on `delegate.provider`:

- `stub` → `StubProvider(kind=delegate.kind)` (unchanged).
- `openai_vlm` → `OpenAIVLMCaptionProvider(connection=connections[delegate.connection], model=delegate.model, options=effective_options)`.

Construction stays per-request from live config (transport spec's
lifecycle rule); the provider is cheap to construct — the HTTP work happens
in `run()`.

## Testing

- **Provider unit tests** (`tests/test_analysis_vlm_caption.py`), all via
  injected `httpx.MockTransport`:
  - payload shape pin: `model`, messages array (system + user), data-URI
    image part for an asset_ref target, URL part for a url target, bearer
    header present when the env var is set and absent when not;
  - `CaptionParams.prompt` included as a text part when set, absent
    otherwise;
  - observation content = `choices[0].message.content`; `raw_output` = the
    full response dict;
  - failure modes raise: non-2xx, timeout, missing/empty content,
    unresolvable asset_ref;
  - `supports()` accepts caption, rejects the other four kinds.
- **Config tests** (extend `tests/test_analysis_mode_config.py`):
  provider field parsing + default, unknown-provider failure,
  provider/kind-compatibility failure, options parsing (all keys),
  unknown-option-key failure, type/range failures.
- **Endpoint integration test** (extend `tests/test_analysis_routes.py`):
  a mode configured with `provider: openai_vlm` and a MockTransport-backed
  provider returns 200 with the model caption in observations and full
  raw_output in the run; one failed VLM call among two runs yields
  `partial` (proves per-run isolation through the real provider). Existing
  stub-based tests run unmodified — that is the back-compat proof.
  **At least one of these integration cases must exercise the real
  `build_providers` switch unpatched** — the test injects the mock at the
  httpx-transport level (not by patching the provider factory), so the
  delegate-config → provider-class selection path is what's under test.
- **Config round-trip tests** (per Round-trip persistence above):
  `to_dict()`/`save_config()`/reload and bulk `PUT /api/modes`
  omitted-backfill both preserve `provider` + `options`.
- **Live verification** against the real `node2.lan` VLM endpoint:
  deferred to the human, mirroring the compel track's docs/live
  verification split. Not a test-suite concern.

## Non-Goals

- No detect / ocr / pose / embed providers (YOLO detect is its own track).
- No server-side fetching of `url` targets.
- No connection pooling, client caching, or keep-alive tuning.
- No streaming responses.
- No retry/backoff policy.
- No `Summary` synthesis.
- No new wire-contract fields (Go and Python contracts unchanged).
- No CLI changes.

## Implementation Order (input to the plan)

1. Config: `provider` + `options` parsing, load-time validation, **and
   round-trip persistence** (`to_dict()`, `save_config()`, bulk
   `PUT /api/modes`) with regression tests.
2. VLM client with MockTransport-tested payload/auth/response handling.
3. Provider: image-part building, prompting, response mapping, failures.
4. `build_providers` switch + endpoint integration tests (including the
   unpatched-factory case).
