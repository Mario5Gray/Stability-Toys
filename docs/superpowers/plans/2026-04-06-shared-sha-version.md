# Shared SHA Version Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose one shared git-derived short SHA as both backend and frontend version metadata, and display both values in the main chat UI.

**Architecture:** The build owns version injection through one `GIT_SHA` build arg. Docker maps that single value into the UI build as `VITE_APP_VERSION` and into the server runtime as `BACKEND_VERSION`; the backend adds `backend_version` to `/api/models/status`, and the frontend reads its own build-time version plus the backend runtime version and renders both in the visible chat header. Because the current `ChatHeader` component exists but is not wired into the rendered chat view, the frontend task includes integrating it through `ChatContainer`.

**Tech Stack:** Docker multi-stage builds, FastAPI route serialization, React 19, Vite env injection, Vitest, pytest

---

## File Structure

- `Dockerfile`
  - Owns the production multi-stage build. Needs explicit `ARG GIT_SHA=dev` redeclared inside both `ui-build` and `server` stages before those stages can consume it.
- `Dockerfile.live-test`
  - Dev container path. Should set `BACKEND_VERSION=dev` explicitly so the backend status route behaves consistently in local container workflows.
- `server/model_routes.py`
  - Existing runtime metadata endpoint. Add `backend_version` here instead of creating a new route.
- `tests/test_model_routes.py`
  - Existing route serialization tests. Extend with direct assertions for `backend_version`.
- `lcm-sr-ui/src/utils/version.js`
  - New small helper for frontend build-version lookup with `dev` fallback.
- `lcm-sr-ui/src/hooks/useModeConfig.js`
  - Existing runtime status source. Keep backend version flowing through `runtimeStatus`.
- `lcm-sr-ui/src/components/chat/ChatHeader.jsx`
  - Existing badge component. Extend it with `frontendVersion` and `backendVersion` props.
- `lcm-sr-ui/src/components/chat/ChatContainer.jsx`
  - Actual rendered chat shell. Wire `ChatHeader` into this component and forward version props.
- `lcm-sr-ui/src/App.jsx`
  - Existing composition root. Read frontend version once and pass both version values into `ChatContainer`.
- `lcm-sr-ui/src/hooks/useModeConfig.test.jsx`
  - Existing status hook tests. Update mocked `/api/models/status` payloads to include version metadata where relevant.
- `lcm-sr-ui/src/components/chat/ChatHeader.test.jsx`
  - New focused render test for the version badges.

### Task 1: Add shared SHA injection to Docker builds

**Files:**
- Modify: `Dockerfile`
- Modify: `Dockerfile.live-test`

- [ ] **Step 1: Write the failing packaging contract test**

Add a test to `tests/test_cuda_packaging_contract.py` that asserts the Dockerfile now declares and re-declares `ARG GIT_SHA=dev` in the stages that consume it.

```python
def test_dockerfile_redeclares_git_sha_for_ui_and_server_stages():
    text = Path("Dockerfile").read_text()

    assert "ARG GIT_SHA=dev" in text
    assert text.count("ARG GIT_SHA=dev") >= 3
    assert "ENV VITE_APP_VERSION=${GIT_SHA}" in text
    assert "ENV BACKEND_VERSION=${GIT_SHA}" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cuda_packaging_contract.py::test_dockerfile_redeclares_git_sha_for_ui_and_server_stages -q`
Expected: FAIL because `Dockerfile` does not yet mention `GIT_SHA`.

- [ ] **Step 3: Implement minimal Docker changes**

Update `Dockerfile` with an explicit top-level arg plus stage-local re-declarations and env mappings.

```dockerfile
ARG TARGETPLATFORM
ARG BACKEND
ARG CERTFILE
ARG GIT_SHA=dev

FROM node:20-trixie-slim AS ui-build
ARG GIT_SHA=dev
ENV VITE_APP_VERSION=${GIT_SHA}

FROM python:3.12-slim AS server
ARG GIT_SHA=dev
ENV BACKEND_VERSION=${GIT_SHA}
```

Update `Dockerfile.live-test` so backend version is explicit in that dev container path.

```dockerfile
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_VERSION=dev
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cuda_packaging_contract.py::test_dockerfile_redeclares_git_sha_for_ui_and_server_stages -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Dockerfile Dockerfile.live-test tests/test_cuda_packaging_contract.py
git commit -m "build: inject shared git sha into ui and server"
```

### Task 2: Expose backend version on `/api/models/status`

**Files:**
- Modify: `server/model_routes.py`
- Modify: `tests/test_model_routes.py`

- [ ] **Step 1: Write the failing backend route test**

Add a direct route test to `tests/test_model_routes.py`.

```python
async def test_models_status_includes_backend_version():
    pool = Mock()
    pool.get_current_mode.return_value = "SDXL"
    pool.is_model_loaded.return_value = True
    pool.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {"allocated_gb": 1.5, "reserved_gb": 2.0}

    with patch("server.model_routes.get_worker_pool", return_value=pool), \
         patch("server.model_routes.get_model_registry", return_value=registry), \
         patch.dict(os.environ, {"BACKEND_VERSION": "abc1234"}, clear=False):
        data = await model_routes.get_models_status()

    assert data["backend_version"] == "abc1234"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_routes.py::test_models_status_includes_backend_version -q`
Expected: FAIL with `KeyError: 'backend_version'`.

- [ ] **Step 3: Implement the minimal route change**

In `server/model_routes.py`, read the env with a fallback and add the field to the existing payload.

```python
import os

@router.get("/models/status")
async def get_models_status():
    pool = get_worker_pool()
    registry = get_model_registry()

    current_mode = pool.get_current_mode()
    vram_stats = registry.get_vram_stats()
    queue_size = pool.get_queue_size()
    backend_version = os.environ.get("BACKEND_VERSION", "dev").strip() or "dev"

    return {
        "current_mode": current_mode,
        "is_loaded": pool.is_model_loaded(),
        "queue_size": queue_size,
        "vram": vram_stats,
        "backend_version": backend_version,
    }
```

- [ ] **Step 4: Add fallback coverage and run tests**

Add a second test for the unset/blank fallback.

```python
async def test_models_status_defaults_backend_version_to_dev():
    pool = Mock()
    pool.get_current_mode.return_value = None
    pool.is_model_loaded.return_value = False
    pool.get_queue_size.return_value = 0

    registry = Mock()
    registry.get_vram_stats.return_value = {}

    with patch("server.model_routes.get_worker_pool", return_value=pool), \
         patch("server.model_routes.get_model_registry", return_value=registry), \
         patch.dict(os.environ, {"BACKEND_VERSION": ""}, clear=False):
        data = await model_routes.get_models_status()

    assert data["backend_version"] == "dev"
```

Run: `pytest tests/test_model_routes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/model_routes.py tests/test_model_routes.py
git commit -m "feat: expose backend version in model status"
```

### Task 3: Add frontend version helper and render version badges in the chat shell

**Files:**
- Create: `lcm-sr-ui/src/utils/version.js`
- Modify: `lcm-sr-ui/src/components/chat/ChatHeader.jsx`
- Modify: `lcm-sr-ui/src/components/chat/ChatContainer.jsx`
- Modify: `lcm-sr-ui/src/App.jsx`
- Create: `lcm-sr-ui/src/components/chat/ChatHeader.test.jsx`

- [ ] **Step 1: Write the failing header render test**

Create `lcm-sr-ui/src/components/chat/ChatHeader.test.jsx`.

```jsx
// @vitest-environment jsdom

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ChatHeader } from './ChatHeader';

describe('ChatHeader', () => {
  it('renders frontend and backend version badges', () => {
    render(
      <ChatHeader
        inflightCount={0}
        isDreaming={false}
        srLevel={0}
        frontendVersion="abc1234"
        backendVersion="abc1234"
      />
    );

    expect(screen.getByText('UI abc1234')).toBeInTheDocument();
    expect(screen.getByText('API abc1234')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd lcm-sr-ui && yarn vitest run src/components/chat/ChatHeader.test.jsx`
Expected: FAIL because `ChatHeader` does not yet accept or render version props.

- [ ] **Step 3: Implement the minimal frontend version plumbing**

Create a helper in `lcm-sr-ui/src/utils/version.js`.

```js
export function getFrontendVersion() {
  const value = import.meta.env.VITE_APP_VERSION;
  return typeof value === 'string' && value.trim() ? value.trim() : 'dev';
}
```

Update `ChatHeader.jsx` to accept version props and render subdued badges.

```jsx
export function ChatHeader({
  inflightCount,
  isDreaming,
  srLevel,
  frontendVersion,
  backendVersion,
}) {
  const apiVersion = backendVersion || '...';

  return (
    <CardHeader className="border-b">
      <div className="flex items-center justify-between gap-1">
        <CardTitle className="text-xl">LCM + SR Chat</CardTitle>
        <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
          <Badge variant="secondary">UI {frontendVersion}</Badge>
          <Badge variant="secondary">API {apiVersion}</Badge>
          {srLevel > 0 ? <Badge>SR {srLevel}</Badge> : <Badge variant="outline">{BADGE_LABELS.SR_OFF}</Badge>}
          {isDreaming && <Badge className="gap-1 animate-pulse bg-gradient-to-r from-purple-600 to-pink-600"><Sparkles className="h-1 w-1" />Dreaming</Badge>}
        </div>
      </div>
      <div className="text-sm text-muted-foreground">{UI_MESSAGES.KEYBOARD_TIP}</div>
    </CardHeader>
  );
}
```

Wire `ChatHeader` into `ChatContainer.jsx`.

```jsx
import { ChatHeader } from "./ChatHeader";

export function ChatContainer({
  messages,
  selectedMsgId,
  blurredSelectedMsgId,
  onToggleSelect,
  onCancelRequest,
  setMsgRef,
  composer,
  inflightCount,
  isDreaming,
  dreamMessageId,
  onDreamSave,
  onDreamHistoryPrev,
  onDreamHistoryNext,
  onDreamHistoryLive,
  onRetry,
  serverLabel,
  activeGalleryId,
  onAddToGallery,
  srLevel,
  frontendVersion,
  backendVersion,
}) {
  return (
    <Card className="option-panel-area overflow-hidden rounded-xl shadow-sm h-full flex flex-col">
      <ChatHeader
        inflightCount={inflightCount}
        isDreaming={isDreaming}
        srLevel={srLevel}
        frontendVersion={frontendVersion}
        backendVersion={backendVersion}
      />
      <CardContent className="flex flex-1 flex-col p-0 min-h-0">
```

In `App.jsx`, import the helper, compute the frontend version once, and pass both versions into `ChatContainer`.

```jsx
import { getFrontendVersion } from './utils/version';

export default function App() {
  const frontendVersion = getFrontendVersion();
  const modeState = useModeConfig();
  const backendVersion = modeState.runtimeStatus?.backend_version ?? null;

  return (
    <ChatContainer
      messages={messages}
      selectedMsgId={selectedMsgId}
      blurredSelectedMsgId={blurredSelection?.msgId ?? null}
      onToggleSelect={handleToggleSelectMsg}
      onCancelRequest={(id) => { cleanupMessage(id); cancelRequest(id); deleteMessage(id); }}
      setMsgRef={setMsgRef}
      composer={defaultComposer}
      inflightCount={inflightCount}
      isDreaming={isDreaming}
      dreamMessageId={dreamMessageId}
      onDreamSave={saveDreamAndContinue}
      onDreamHistoryPrev={dreamHistoryPrev}
      onDreamHistoryNext={dreamHistoryNext}
      onDreamHistoryLive={dreamHistoryLive}
      onRetry={(msg) => { if (msg.retryParams) runGenerate(msg.retryParams); }}
      srLevel={params.effective.superresLevel}
      frontendVersion={frontendVersion}
      backendVersion={backendVersion}
      serverLabel={serverLabel}
      onImageDisplayed={onImageDisplayed}
      onImageError={onImageError}
      activeGalleryId={galleryState.activeGalleryId}
      onAddToGallery={onAddToGallery}
    />
  );
}
```

- [ ] **Step 4: Run frontend test to verify it passes**

Run: `cd lcm-sr-ui && yarn vitest run src/components/chat/ChatHeader.test.jsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/utils/version.js lcm-sr-ui/src/components/chat/ChatHeader.jsx lcm-sr-ui/src/components/chat/ChatContainer.jsx lcm-sr-ui/src/components/chat/ChatHeader.test.jsx lcm-sr-ui/src/App.jsx
git commit -m "feat: show shared sha versions in chat header"
```

### Task 4: Keep runtime status tests aligned with version-aware payloads

**Files:**
- Modify: `lcm-sr-ui/src/hooks/useModeConfig.test.jsx`

- [ ] **Step 1: Write the failing hook assertion**

Extend an existing status test to assert the backend version survives the hook refresh path.

```jsx
it('loads backend version from runtime status', async () => {
  api.client.fetchGet.mockImplementation(async (endpoint) => {
    if (endpoint === '/api/modes') {
      return {
        default_mode: 'cinematic',
        modes: {
          cinematic: { model: 'base-cinematic' },
        },
      };
    }

    if (endpoint === '/api/models/status') {
      return {
        current_mode: 'cinematic',
        is_loaded: true,
        backend_version: 'abc1234',
      };
    }

    throw new Error(`Unexpected endpoint: ${endpoint}`);
  });

  const { result } = renderHook(() => useModeConfig());

  await waitFor(() => expect(result.current.runtimeStatus?.backend_version).toBe('abc1234'));
});
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `cd lcm-sr-ui && yarn vitest run src/hooks/useModeConfig.test.jsx`
Expected: This may already PASS because the hook stores the whole status object. If it does, keep the test and proceed; the point is to lock the contract before any refactor.

- [ ] **Step 3: Update mocked status payloads for clarity**

Where the file currently returns `/api/models/status`, add `backend_version` to representative success payloads so future assertions use realistic route shapes.

```jsx
return {
  current_mode: 'portrait',
  is_loaded: true,
  backend_version: 'abc1234',
};
```

No hook implementation change should be required unless a regression appears.

- [ ] **Step 4: Run the full targeted frontend/backend verification**

Run: `pytest tests/test_model_routes.py tests/test_cuda_packaging_contract.py -q`
Expected: PASS

Run: `cd lcm-sr-ui && yarn vitest run src/hooks/useModeConfig.test.jsx src/components/chat/ChatHeader.test.jsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lcm-sr-ui/src/hooks/useModeConfig.test.jsx tests/test_model_routes.py tests/test_cuda_packaging_contract.py
git commit -m "test: cover shared sha version reporting"
```

## Self-Review

- Spec coverage:
  - Shared Docker SHA injection: Task 1
  - Backend version on `/api/models/status`: Task 2
  - Frontend embedded version plus header display: Task 3
  - Backend and frontend tests: Task 4
- Placeholder scan:
  - No `TODO`, `TBD`, or implicit “write tests later” steps remain.
- Type consistency:
  - `backend_version` is used consistently across backend payloads, hook state, and UI props.
  - `frontendVersion` and `backendVersion` are the prop names used consistently in the UI path.
