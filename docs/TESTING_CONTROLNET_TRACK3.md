# Track 3 ControlNet CUDA Validation

1. Run one successful `canny` request from `source_asset_ref`.
2. Run one successful `depth` request from `source_asset_ref`.
3. Reuse the emitted artifact with `map_asset_ref`.
4. Run a two-attachment request and confirm both bindings are applied in order.
5. Submit an incompatible `model_id` and confirm fail-fast rejection before generation.
6. Repeat requests and observe cache reuse without OOM.
7. Confirm HTTP `X-ControlNet-Artifacts` header on success.
8. Confirm WS `job:complete.controlnet_artifacts` on success.
