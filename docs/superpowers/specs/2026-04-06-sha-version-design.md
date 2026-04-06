# Shared SHA Version Design

## Summary

This design adds lightweight backend and frontend version reporting using one shared git-derived short SHA. The backend and frontend should report the same value because they are built from the same checkout, and the value should be available in normal runtime status without adding a separate version endpoint.

The source of truth is a single build-time `GIT_SHA` value. Docker injects it once, the backend exposes it through the existing `/api/models/status` payload, and the frontend embeds it at build time and displays it alongside the backend value in the main app header.

## Goals

- Expose a backend version derived from the repo short SHA
- Expose a frontend version derived from the same short SHA
- Keep both values identical for a given build
- Surface version information in an existing operational UI location
- Preserve local development ergonomics when no SHA is provided

## Non-Goals

- Introduce manual semver management
- Add a dedicated version API route
- Execute `git` commands at runtime
- Support distinct frontend and backend SHAs in the same build
- Add more release metadata than the short SHA

## Current State

- The backend already exposes runtime metadata through `/api/models/status`
- The frontend already polls that status through `useModeConfig`
- The main chat header already displays lightweight runtime badges
- The Docker build already has separate UI and server stages, but it does not currently thread a shared version identifier into either side

## Proposed Approach

Use a single build arg, `GIT_SHA`, as the only version input. Map it into both build stages:

- UI build stage receives `VITE_APP_VERSION`
- server stage receives `BACKEND_VERSION`

Both values default to `dev` when not provided. This keeps local non-Docker workflows and ad hoc builds functional without requiring git metadata inside the runtime container.

## Design

### 1. Docker-owned shared version injection

Files in scope:

- `Dockerfile`
- `Dockerfile.live-test` if it should expose the same fallback behavior for local container workflows
- compose files or build commands only if they need documentation updates later

Design:

- Add `ARG GIT_SHA=dev` near the top-level Docker build inputs.
- In the UI build stage, map `GIT_SHA` to `ENV VITE_APP_VERSION=${GIT_SHA}` before `yarn build`.
- In the server stage, map `GIT_SHA` to `ENV BACKEND_VERSION=${GIT_SHA}`.
- Keep the default value `dev` so builds still succeed when no SHA is passed.

Expected outcome:

- one short SHA drives both UI and backend version reporting
- runtime images do not need `.git`
- non-versioned local builds still behave predictably

### 2. Backend version on the existing status path

Files in scope:

- `server/model_routes.py`
- optional small helper module if version lookup logic should not live inline
- backend tests covering `/api/models/status`

Design:

- Read `BACKEND_VERSION` from environment with fallback `dev`.
- Add `backend_version` to the existing `/api/models/status` response.
- Do not create a separate endpoint because the existing status route is already the frontend’s runtime metadata source.

Expected outcome:

- the frontend gets backend version information through its current polling path
- operational tooling can retrieve the backend SHA from the same status response it already uses

### 3. Frontend version embedded at build time

Files in scope:

- `lcm-sr-ui/src/App.jsx`
- `lcm-sr-ui/src/components/chat/ChatHeader.jsx`
- optional tiny helper module in `lcm-sr-ui/src/utils/`
- frontend tests for header rendering and mode status handling

Design:

- Read the frontend version from `import.meta.env.VITE_APP_VERSION` with fallback `dev`.
- Thread that value through the existing app shell into `ChatHeader`.
- Display two subdued badges in the chat header:
  - `UI <sha>`
  - `API <sha>`
- If runtime status has not loaded yet, display `API dev` only if that is the backend fallback actually returned, otherwise use a neutral placeholder such as `API ...` until status arrives.

Expected outcome:

- users can see both build identities at a glance
- support/debugging becomes faster because frontend/backend drift is visible immediately
- the display remains lightweight and does not require a new settings surface

## Testing Strategy

### Backend tests

- Extend status-route tests to assert `backend_version` is included.
- Verify fallback behavior when `BACKEND_VERSION` is unset.

### Frontend tests

- Update any status-hook tests that mock `/api/models/status` so they tolerate and propagate `backend_version`.
- Add a focused render test for `ChatHeader` that asserts both version badges appear with the expected values.

### Manual validation

Validate the following:

1. A Docker build with `--build-arg GIT_SHA=$(git rev-parse --short HEAD)` causes both UI and backend to show the same short SHA.
2. A build without `GIT_SHA` still succeeds and reports `dev`.
3. The chat header shows both version badges without disrupting existing controls.

## Risks And Tradeoffs

### Build commands must provide the SHA deliberately

That is acceptable. The fallback value keeps development friction low, and explicit injection is more reliable than trying to derive git state inside the image.

### Live-test and non-Docker flows may not have a real SHA

That is expected. Those flows should report `dev` unless they are updated to pass the same environment variable explicitly.

### Status payload grows slightly

The added field is small and belongs with the existing runtime metadata, so this is a reasonable tradeoff compared with introducing another endpoint.

## Rollout

Implement in this order:

1. Add shared `GIT_SHA` injection in Docker
2. Expose `backend_version` on `/api/models/status`
3. Embed and display the frontend version in the header
4. Update backend and frontend tests

## Acceptance

This work is complete when:

- the backend reports `backend_version` from `/api/models/status`
- the frontend displays `UI <sha>` and `API <sha>` in the header
- both values match for a Docker build from one checkout
- both sides fall back to `dev` when no SHA is provided
