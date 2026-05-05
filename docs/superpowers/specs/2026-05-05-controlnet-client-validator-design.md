# ControlNet Client Validator Design

## Goal

Add a pure client-side utility that can validate the Track 3 CUDA ControlNet
backend through the same HTTP and WebSocket APIs used by real callers.

The tool must not start, stop, reconfigure, or otherwise control server-side
process state. It should operate strictly as a black-box client against an
already running Stability Toys server.

## Scope

The first version will provide:

- a repo-local Python CLI at `scripts/controlnet_client.py`
- reusable subcommands for upload and generation flows
- a `validate-track3` workflow that executes the manual checklist currently
  documented in `docs/TESTING_CONTROLNET_TRACK3.md`
- a JSON config file format with CLI overrides
- bundled placeholder PROP assets and an example config
- automatic run artifact capture under a local output directory

The first version will not provide:

- server startup or shutdown
- server mode mutation outside normal request payloads
- remote orchestration
- dashboard/UI output
- packaging as an installable Python distribution

## User Interface

The tool will be invoked directly with Python:

```bash
python scripts/controlnet_client.py <subcommand> [options]
```

Supported subcommands:

- `upload`
- `generate`
- `ws-generate`
- `validate-track3`

All subcommands accept `--config <json-file>`. CLI args override config values.

## Config Model

Config is JSON-backed and supplies defaults for API endpoints, request payload
values, and placeholder assets.

Example shape:

```json
{
  "base_url": "http://127.0.0.1:4200",
  "ws_url": "ws://127.0.0.1:4200/v1/ws",
  "mode": "PROP-mode",
  "prompt": "PROP prompt",
  "negative_prompt": "PROP negative prompt",
  "images": {
    "source": "images/source.png",
    "depth_source": "images/depth-source.png"
  },
  "controlnet": {
    "canny_model_id": "PROP-canny-model-id",
    "depth_model_id": "PROP-depth-model-id",
    "bad_model_id": "PROP-incompatible-model-id"
  }
}
```

Path resolution rules:

- absolute paths are used as-is
- relative paths are resolved from the directory containing the config file

## Bundled Assets

The repo will include placeholder assets under:

- `props/controlnet/track3-example.json`
- `props/controlnet/images/source.png`
- `props/controlnet/images/depth-source.png`

These are examples only. They are expected to be replaced or overridden in
real runs through CLI flags or a custom config file.

## Run Output

By default, each execution writes artifacts to:

`stability-out/controlnet-client/<timestamp>/`

Saved outputs include:

- uploaded asset refs
- request payload snapshots
- HTTP response metadata
- WS completion payloads
- generated images when returned
- validation summary JSON

An explicit `--out-dir` flag may override the base output directory.

## Subcommand Behavior

### `upload`

Uploads a single file to the running server and prints/saves the returned asset
reference and metadata.

### `generate`

Executes an HTTP generation request. It supports preprocess-driven ControlNet
and direct `map_asset_ref` reuse. It captures:

- status code
- response headers
- `X-ControlNet-Artifacts`
- output image bytes

### `ws-generate`

Executes the WebSocket generation path and captures:

- sent request payload
- server frames
- final `job:complete` payload
- `controlnet_artifacts`

### `validate-track3`

Runs the Track 3 CUDA validation checklist:

1. successful `canny` request from `source_asset_ref`
2. successful `depth` request from `source_asset_ref`
3. emitted artifact reuse via `map_asset_ref`
4. two-attachment ordered request
5. incompatible `model_id` rejection before generation
6. repeated requests to observe client-visible cache reuse stability
7. HTTP `X-ControlNet-Artifacts` verification on success
8. WS `job:complete.controlnet_artifacts` verification on success

The validator reports step-by-step pass/fail and writes a machine-readable
summary.

## Architecture

The utility should stay as a single repo-local Python entrypoint with small
internal helpers rather than a package-heavy structure.

Recommended internal responsibilities:

- config loading and CLI override merge
- path resolution helpers
- HTTP upload/generate client helpers
- WebSocket generate client helper
- run directory creation and artifact writing
- validator orchestration for Track 3

## Error Handling

The CLI should:

- fail clearly when required config or file inputs are missing
- preserve server error payloads in saved run artifacts
- distinguish transport failures from validation assertion failures
- continue writing partial evidence even when a validation step fails

`validate-track3` should exit non-zero if any checklist step fails.

## Testing Strategy

Implementation should use TDD.

Tests should cover:

- config loading and CLI override precedence
- relative asset path resolution from config location
- run directory creation and artifact writing
- HTTP header extraction for `X-ControlNet-Artifacts`
- WS completion parsing for `controlnet_artifacts`
- validator step orchestration with stubbed HTTP/WS clients

The first version should prefer deterministic mocked tests over requiring a
live server during unit verification.

## Success Criteria

The first version is successful when a user can:

- point the tool at an already running Stability Toys server
- use bundled PROP examples or override them with real assets
- validate the Track 3 CUDA checklist end-to-end from the client side
- keep the resulting run directory as operational evidence

