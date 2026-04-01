# CUDA/RKNN Super-Resolution CLI Design

## Goal

Add a command-line entry point for super-resolution that reuses the existing backend-agnostic service layer instead of going through HTTP.

The CLI is for real operator use, not just tests. It must exercise the same backend selection and model-loading behavior as the server so CUDA SR and RKNN SR can be validated directly from the shell.

## Scope

### In scope

- Add `python -m server.superres_cli`
- Single-image invocation only
- Reuse existing service-layer code:
  - `resolve_superres_backend(...)`
  - `initialize_superres_service(...)`
  - `submit_superres(...)`
- Honor the same env-based backend selection as the server
- Read one input image from disk and write one output image to disk
- Print a short operator summary to stdout

### Out of scope

- Batch mode
- Directory walking or globbing
- Any HTTP dependency
- New SR backend behavior
- Frontend changes

## Interface

The CLI entry point is:

```bash
python -m server.superres_cli --input INPUT --output OUTPUT [options]
```

Arguments:

- `--input`: required input image path
- `--output`: required output image path
- `--magnitude`: optional, default `2`, valid `1..3`
- `--format`: optional, default `png`, valid `png|jpeg`
- `--quality`: optional, default `92`, valid `1..100`

The CLI uses the existing environment variables for backend selection and model resolution, including:

- `BACKEND`
- `CUDA_SR_MODEL`
- `CUDA_SR_TILE`
- `CUDA_SR_FP16`
- RKNN SR env vars already used by the server

## Behavior

The CLI should:

1. Parse and validate arguments
2. Resolve backend selection using the same backend rules as the server
3. Initialize the shared SR service with the same config path used by the server
4. Load the input file bytes
5. Submit one SR job through `submit_superres(...)`
6. Write the resulting bytes to the requested output path
7. Print a short summary including:
   - backend used
   - model filename
   - magnitude
   - output format
   - output path
   - elapsed time
8. Shut down the service cleanly before exit

## Errors

The CLI should fail fast with a non-zero exit code on:

- missing input file
- invalid argument values
- missing required backend/model configuration
- SR runtime failure

Error messages should stay operator-oriented and concise. The CLI does not need HTTP-style status formatting.

## Testing

Unit coverage should focus on:

- argument validation
- backend/service initialization routing
- output file writing
- summary output
- service shutdown on both success and failure

Tests should stub the shared SR service rather than requiring RKNN or CUDA hardware.
