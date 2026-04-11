# Review: LLM Expert Integration Proposal

Reviewed proposal: [docs/superpowers/proposals/2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md)

## Findings

### 1. High: the proposal assigns the wrong backend sources to context injection

The proposal says expert context should come from `backends/model_registry.py` and `backends/styles.py`, including checkpoint lists, LoRA lists, scheduler options, and capability data. That boundary does not match the current codebase. `ModelRegistry` only tracks loaded models and VRAM, not the available checkpoint inventory or scheduler policy, and `STYLE_REGISTRY` is a static style mapping that is currently empty by default. See [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):37, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):101, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):157, [model_registry.py](/Users/darkbit1001/workspace/Stability-Toys/backends/model_registry.py#L17), and [styles.py](/Users/darkbit1001/workspace/Stability-Toys/backends/styles.py#L6).

Recommended correction: make `server/mode_config.py` and the current active mode the primary source for generation defaults and allowed scheduler/model policy; use inventory scanning and `scheduler_registry` for selectable capabilities; use `ModelRegistry` only for loaded-state and VRAM facts.

### 2. High: the frontend seam in the proposal duplicates UI that already exists instead of extending it

The proposal introduces a new `ChatPanel.jsx`, `useChatJob.js`, and `PromoteToGenerate.jsx`, but the app already has a substantial chat workspace and composer path. The current chat tab renders [ChatContainer.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/chat/ChatContainer.jsx#L9), and the composer in [App.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/App.jsx#L766) currently dispatches generation jobs directly through the existing selection/history flow. `useGenerateJob` does not exist in the repo, so the proposal is also naming a nonexistent integration seam. See [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):110, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):113, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):195, and [App.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/App.jsx#L854).

Recommended correction: keep the frontend owning conversation state and message rendering in the existing chat tab. Add LLM chat as another composer mode and another message kind inside the current chat surface, not as a parallel mini-app.

### 3. Medium: the expert output contract is internally inconsistent

Phase 2 says the expert returns free text plus a fenced JSON block that the orchestrator or frontend parses. Section 9 then chooses `pydantic-ai` typed outputs and explicitly says this avoids fenced-JSON parsing. Those are materially different contracts and imply different `job:complete` shapes. See [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):224, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):257, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):345, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):360, and [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):426.

Recommended correction: lock one transport contract in the follow-up spec. The cleanest boundary here is `job:complete = { text, suggested_params, meta }`, with `suggested_params` validated server-side before it reaches the browser.

### 4. Medium: “expert” is doing three jobs at once and the loop story is still underbaked

The proposal currently uses “expert” to mean:

- a reusable preset/persona
- a persisted user-editable config file
- a multi-step loop runner that may replace the specialized SDXL prose model

Those are separate concepts with different boundaries. A Phase 2 expert defined as a preset is solid. A Phase 3 looped workflow is not the same thing, and “create an expert” is not actually specified as an end-user feature here because the only authoring path described is “drop a YAML file and restart.” See [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):22, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):41, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):217, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):232, [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):275, and [2026-04-10-llm-expert-integration.md](/Users/darkbit1001/workspace/Stability-Toys/docs/superpowers/proposals/2026-04-10-llm-expert-integration.md):448.

Recommended correction: split the design into:

- `expert profile`: prompt template + context policy + output schema
- `workflow`: single-shot or looped execution strategy
- `authoring`: who can create/edit profiles, and through what surface

## Decision Weighting

I do not agree with a browser-only implementation beyond plain chat.

For your original request, the weighting looks like this:

- Plain “talk to an LLM” chat: frontend-heavy is fine.
- Model-aware prompt refinement and parameter suggestion: backend should own capability injection and validation.
- Vision input using existing gallery/upload state: backend is the better seam in this repo, even if browser-direct is technically possible.
- Latent alignment: not a near-term API requirement. Treat it as research until there is concrete user pain.

So the practical split is:

- Frontend owns conversation UX, local history, selection state, and promotion actions.
- Backend owns third-party LLM transport, current-mode capability context, structured param synthesis, and any loop orchestration.

## Recommendation

Approve the general direction, but not this proposal as an implementation-ready architecture.

The next spec should narrow to Phase 1 and Phase 2 only, with these constraints:

- Reuse the existing chat tab and composer surface instead of adding a second chat UI.
- Define one stable `chat` job result shape before any expert work begins.
- Define expert profiles without loops first.
- Treat looped “expert workflows” as a later extension, not as the initial meaning of “expert.”
- Keep latent alignment out of the core initiative scope until there is a concrete benchmark and a real integration need.
