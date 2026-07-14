# Analysis Configuration

The `describe` capability (analysis family) runs image-analysis tasks — captioning,
object detection, and future OCR/pose/embedding — behind one typed server contract.
This document is the operator reference for configuring it in `conf/modes.yml`.

The user-facing verb is `st describe`; every configuration key and validation error
uses the `analysis_*` vocabulary. For the request/response contract and CLI, see the
design specs under `docs/superpowers/specs/` (`2026-07-11-describe-analysis-interface-design.md`,
`2026-07-13-describe-transport-cli-design.md`, `2026-07-14-vlm-caption-provider-design.md`).
The authoritative schema is the parser in `server/mode_config.py`; this doc tracks it.

## Overview

Analysis config has three top-level sections plus a per-mode reference:

| Section | Purpose |
| --- | --- |
| `analysis_connections` | reusable transport/auth settings for an analyzer backend host |
| `analysis_delegates` | a named analyzer: which connection, which task kind, which model, which provider |
| `analysis_profiles` | maps task kinds to delegates (`task_routes`); a mode selects one profile |
| `modes.<name>.analysis_profile` | opts a mode into a profile |

A `describe` request resolves the **effective mode** (`request.mode`, else the server's
active mode), looks up that mode's `analysis_profile`, and routes each task kind to the
delegate named in the profile's `task_routes`. A mode with no `analysis_profile` cannot
serve describe (the request fails with `analysis_profile_not_found`).

All sections are optional. Omitting them entirely leaves describe unconfigured, which is
the default state of the shipped `conf/modes.yml`.

## Worked example

A mode that captions with a VLM and detects with a (stub) detector:

```yaml
analysis_connections:
  local_vlm:
    endpoint: "http://node2.lan:8080/v1"
    api_key_env: "OPENAI_API_KEY"
  local_detector:
    endpoint: "http://node2.lan:8090"

analysis_delegates:
  vlm_caption:
    connection: local_vlm
    kind: caption
    model: qwen2.5-vl
    provider: openai_vlm       # real VLM; omit for the stub
    options:                   # all optional
      max_tokens: 256
      temperature: 0.0
      timeout_s: 90
      system_prompt: "Describe the image for an art catalog."
  yolo_detect:
    connection: local_detector
    kind: detect
    model: yolo11x

analysis_profiles:
  default:
    task_routes:
      caption: vlm_caption
      detect: yolo_detect

modes:
  SDXL:
    model: sdxl/model.safetensors
    analysis_profile: default
```

With this loaded, `st describe ./photo.png --caption` against mode `SDXL` returns a real
model-generated caption. `--detect` returns stub output until a real detect provider
exists (see Providers).

## `analysis_connections`

Reusable transport settings, referenced by delegates.

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `endpoint` | yes | — | base URL of the analyzer host; for OpenAI-compatible hosts include the `/v1` suffix |
| `api_key_env` | no | `OPENAI_API_KEY` | name of the environment variable holding the bearer token; the token is read at request time, and the `Authorization` header is omitted when the variable is unset or empty |

The API key is never stored in config — only the name of the environment variable that
holds it.

## `analysis_delegates`

A named analyzer backend.

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `connection` | yes | — | must reference a declared `analysis_connections` entry |
| `kind` | yes | — | the task kind this delegate serves: `caption`, `detect`, `ocr`, `pose`, or `embed` |
| `model` | yes | — | model identifier sent to the backend |
| `provider` | no | `stub` | implementation: `stub` or `openai_vlm` |
| `options` | no | `{}` | provider tuning; see Options |

`kind` is a capability declaration — it states what the delegate *can* do; the profile
declares how it is *used*. The two must agree (see the kind invariant under Validation).

### Providers

| `provider` | Behavior |
| --- | --- |
| `stub` (default) | deterministic in-process provider; emits placeholder observations. Use for wiring, tests, and dev without a live backend. |
| `openai_vlm` | real caption provider over an OpenAI-compatible `chat/completions` endpoint. **Requires `kind: caption`.** Local file targets are resolved from the asset store and sent as base64 data-URIs; `http(s)` URL targets pass through verbatim for the VLM host to fetch — the server never fetches remote URLs itself. |

`openai_vlm` is the only real provider today. `detect`/`ocr`/`pose`/`embed` kinds run
against `stub` until their providers land. Leaving `provider` unset keeps every delegate
on the stub, so an existing config's behavior never changes by upgrade alone.

### Options

Optional per-delegate tuning. Unknown keys fail config load. Only `openai_vlm` consumes
these today; the stub ignores them.

| Key | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `max_tokens` | int > 0 | 512 | `openai_vlm` |
| `temperature` | number ≥ 0 | 0.2 | `openai_vlm` |
| `timeout_s` | number > 0 | 60 | `openai_vlm` |
| `system_prompt` | non-empty string | built-in caption instruction | `openai_vlm` |

For `openai_vlm`, the caller's `st describe --prompt "..."` guidance is appended to the
request as a user text part; `system_prompt` sets the system instruction.

## `analysis_profiles`

Maps task kinds to delegate names. A profile's `task_routes` is required and must be
non-empty.

```yaml
analysis_profiles:
  default:
    task_routes:
      caption: vlm_caption      # route key (task kind) : delegate name
      detect: yolo_detect
```

A mode selects exactly one profile. Requests never choose a delegate directly.

## Validation

All analysis config is validated fail-fast at mode load; a bad config raises and the
server does not start (or a reload/bulk save is rejected). The rules and their errors:

| Rule | Error fragment |
| --- | --- |
| `analysis_connections.<n>.endpoint` present | `missing required field: endpoint` |
| delegate `connection` present and declared | `missing required field: connection` / `references unknown connection` |
| delegate `kind` in the closed set | `has invalid kind '<k>' (expected one of ...)` |
| delegate `model` present | `missing required field: model` |
| delegate `provider` in `{stub, openai_vlm}` | `has invalid provider '<p>' (expected one of ...)` |
| `provider: openai_vlm` only on `kind: caption` | `sets provider 'openai_vlm' but kind '<k>' — openai_vlm supports kind 'caption' only` |
| option keys known; values well-typed and in range | `has unknown option '<k>'` / `option <k> must be ...` |
| profile `task_routes` present and non-empty | `missing required mapping: task_routes` |
| every route target is a declared delegate | `route '<k>' references unknown delegate '<d>'` |
| **route key equals the delegate's `kind`** | `analysis_delegate_kind_mismatch: profile '<p>' routes kind '<rk>' to delegate '<d>' of kind '<dk>'` |
| `modes.<n>.analysis_profile` references a declared profile | `Mode '<n>' references unknown analysis_profile '<p>'` |

The kind invariant is deliberate: the delegate declares its capability, the profile
declares its use, and load-time checking catches wiring mistakes (e.g. routing `caption`
to a `detect` delegate) before any request runs.

## Operational notes

- **Enable real providers in deployment config, not repo defaults.** Like Compel
  conditioning, `provider: openai_vlm` belongs in the CUDA/deployment `conf/modes.yml`,
  not the shared repo default. The shipped `conf/modes.yml` intentionally has no analysis
  sections.
- **Live reload.** Analysis policy follows the same reload path as the rest of the mode
  system — SIGHUP, the config file watcher, and `POST /api/modes/reload`. A changed
  profile or delegate takes effect on the next describe request without a process restart.
- **Round-trip safe.** `provider` and `options` survive config export, save, and bulk
  `PUT /api/modes`; defaults are omitted on export (`provider` when `stub`, `options`
  when empty), so a save never silently strips analysis policy.
- **Capability probe.** `GET /api/models/status` reports `capabilities.supports_describe`,
  true when the active mode has an `analysis_profile`.

## Error codes at request time

Configuration errors surface at load. At request time, `POST /v1/describe` (and
`st describe`) can return these operator-facing codes:

| Code | Meaning |
| --- | --- |
| `analysis_mode_not_found` | `request.mode` names an unknown mode |
| `analysis_profile_not_found` | the effective mode has no `analysis_profile` |
| `analysis_no_supported_delegate` | a task kind has no route in the active profile (its runs are `skipped`) |
| `analysis_run_failed` | a delegate was invoked and raised (that run is `failed`; siblings continue) |
| `analysis_invalid_request` / `analysis_target_binding_invalid` | malformed request or a task binding to zero targets |

See the transport spec for the full request/response and exit-code contract.
