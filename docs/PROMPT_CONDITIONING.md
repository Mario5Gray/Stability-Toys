# Prompt Conditioning

Prompt conditioning is selected per mode. The default is native Diffusers prompt
delegation, which preserves the historical `prompt` and `negative_prompt` string
path. CUDA modes may opt in to the local Compel service to materialize prompt
embeddings before generation.

This feature only changes CUDA prompt consumption. It does not add HTTP,
WebSocket, CLI, PNG metadata, RKNN, MLX, CPU, remote, proxy, Redis, or Qdrant
conditioning behavior.

## Configuration

Omitting `conditioning` is the same as native behavior:

```yaml
modes:
  sdxl-general:
    model: sdxl/base
```

An explicit native configuration is also valid:

```yaml
modes:
  sdxl-general:
    model: sdxl/base
    conditioning:
      service: native
      filters: []
      fallback:
        native_when_unconfigured: true
        native_on_failure: false
```

Enable Compel only on CUDA modes whose deployment image includes
`requirements-conditioning.txt`:

```yaml
modes:
  sdxl-general:
    model: sdxl/base
    conditioning:
      service: compel
      filters: []
      fallback:
        native_when_unconfigured: true
        native_on_failure: false
```

Do not enable Compel in shared `conf/modes.yml` as a repo default. For live CUDA
verification, make a temporary operator edit in the CUDA-only deployment
configuration, reload or switch the mode, run the check, then revert the local
edit.

Unknown conditioning keys are rejected during mode load. A mode with
`native_when_unconfigured: false` and no service is also rejected.

## Fallback

`fallback.native_on_failure` is disabled by default. If set to `true`, a Compel
invocation failure can fall back to native prompt delegation. This is an
availability-over-fidelity setting: long prompts can truncate again and may no
longer match Compel output.

Compatibility failures never fall back. CUDA consumers validate materialized
artifacts against the live target pipeline after ControlNet `from_pipe`
construction and img2img normalization. Slot, shape, family, encoder, pooled, or
dtype mismatches fail closed before Diffusers is called, even when
`native_on_failure` is enabled. This guarantee is structural: acceptance runs
outside the native-fallback invocation wrapper.

## Compel Behavior

Compel is not true long-context attention. It chunk-encodes CLIP windows and
concatenates the resulting embeddings.

Operational caveats:

- Chunks are encoded independently; grammar or meaning that crosses a chunk
  boundary can lose coherence.
- Later chunks may have weaker influence than earlier text.
- Tag-style prompts tend to work better than long grammatical prose.
- SDXL pooled conditioning follows Compel's pooled-output behavior, effectively
  representing the first chunk for the pooled slots.
- `negative_prompt: null` is encoded as an empty string so materialized artifacts
  still carry the required negative slots.
- Prompt weighting is strategy-specific. Do not treat this as a general A1111
  prompt-syntax compatibility guarantee.

## Dependency Boundary

Compel is pinned separately in `requirements-conditioning.txt` and installed with
`--no-deps` in CUDA-capable images. The ordinary runtime requirements remain the
authority for Torch, Diffusers, Transformers, and pyparsing. Image package
inspection should show `compel==2.3.1` without Notebook or Jupyter packages pulled
in by the Compel installation.

If a mode explicitly selects `service: compel` and the package is unavailable,
mode load fails before the worker starts accepting generation jobs.
