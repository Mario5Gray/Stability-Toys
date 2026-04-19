# Operations Surface Design

## Summary

This design replaces ad-hoc action rows and scattered animated status labels with a reusable control-surface system for `lcm-sr-ui`.

The system has three shared primitives:

- `SurfaceHeader`: title, durable meta, and calm summary state for a major surface
- `PendingOperationsPane`: the single place where active work is rendered and animated
- `PanelActionBar`: a resilient, unmistakably clickable footer for multi-action panels

The immediate trigger is the Advisor UI, where equal-weight footer buttons overflow narrow layouts and transient status is rendered as plain text inside the panel. The design intentionally expands beyond Advisor because the same problem already exists in chat and generation surfaces.

This is a product-UI design, not a utility-tool skin. It should feel intentionally composed, remain legible on narrow panels, and centralize motion so the interface stops glowing in arbitrary places.

## Goals

- Define a reusable `PanelActionBar` for multi-action panel footers
- Make actions visually identifiable as buttons without hover
- Optimize panel actions for mobile and narrow-width resilience
- Introduce a top-of-chat `PendingOperationsPane` for active work
- Centralize animation and glow in the operations pane only
- Normalize status presentation around structured operation feedback
- Support a low-friction status handle API so feature code can update status without rendering UI directly
- Use real structured backend or job feedback when available, with coarse fallback updates when it is not

## Non-Goals

- Rebuild the entire visual design system in one pass
- Change backend business logic beyond exposing structured operation progress
- Make the operations pane the source of truth for job state
- Eliminate local error or recency messaging inside panels
- Introduce a full notification center, timeline, or activity log

## Current State

- [AdvisorPanel.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/options/AdvisorPanel.jsx) renders three equal-weight actions in a single `flex gap-2` row. On narrow widths this overflows, compresses labels, and makes actions feel like stray text rather than explicit controls.
- [AdvisorPanel.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/options/AdvisorPanel.jsx) also renders active rebuild state inline as plain text (`Building digest...`) inside the panel body.
- [ChatContainer.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/chat/ChatContainer.jsx) already contains a sticky strip above the message stream, but it currently renders placeholder content (`[]`) instead of a real shared status surface.
- [MessageBubble.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/chat/MessageBubble.jsx) renders animated `dreaming` and `generating` badges directly on image content.
- Generation work and advisor rebuilds have lifecycle state, but the UI is not yet driven by a unified status presentation model.
- Generation can expose structured progress later through `callback_on_step_end`, which is the right source for numeric denoising progress (`step / num_inference_steps`).

## Design Principles

### One Surface For Active Work

Active work should be rendered in one shared surface, not duplicated as animated labels across panels, bubbles, and headers.

### Actions Must Look Like Actions

Buttons must read as buttons at rest. Hover may enhance affordance, but affordance must not depend on hover.

### Narrow Width First

The system should survive narrow panels and mobile widths without truncation, overflow, or equal-priority horizontal button piles.

### Real Feedback Over Invented Feedback

When an operation already exposes structured progress or phase data, the UI should consume that data rather than fabricating coarse labels. The operations surface is a display layer over operation state, not the source of truth.

### One Primary Action Per Surface

Every multi-action panel must expose exactly one primary action. Supporting actions are secondary and visually subordinate.

## Proposed Architecture

### SurfaceHeader

`SurfaceHeader` is used for major surfaces such as chat. It owns:

- title
- durable metadata such as version or mode chips
- a calm summary line

It does not own animated working state. That moves to `PendingOperationsPane`.

### Operations Store

The operations store should use React context plus `useReducer`, not a new external state library.

Rationale:

- `lcm-sr-ui` does not currently depend on Zustand or another shared store library
- the operations surface is app-global UI state, which fits a provider mounted near the app shell
- this avoids introducing a new dependency for a narrowly scoped cross-cutting concern
- context plus reducer is sufficient for keyed upsert, completion expiry, and cancellation metadata

Recommended shape:

- `OperationsProvider`
- `useOperationsStore()` for read access
- `useOperationsController()` for write access and status-handle creation

Responsibilities:

- keep a keyed map of operations
- expose deterministic ordering for rendering
- own completion and error expiry timers
- support upsert-by-key
- keep operation metadata serializable and UI-oriented

Non-responsibilities:

- own backend job state
- parse transport payloads directly inside the reducer
- bypass feature-specific adapters

### PendingOperationsPane

`PendingOperationsPane` lives directly under the header and above the scrollable chat transcript.

Responsibilities:

- render active operations in a consistent visual language
- own pulse, glow, fade, completion landing, and expiry timing
- group and order operations deterministically
- expose cancellation affordances when supported

Non-responsibilities:

- invent semantic job state
- replace contextual inline errors
- act as the source of truth for operation progress

### PanelActionBar

`PanelActionBar` is the required footer for any panel with more than one action.

Responsibilities:

- make actions unmistakably clickable
- establish primary versus secondary hierarchy
- remain resilient on narrow widths
- support `icon`, `label`, and short helper `subtext`

## PanelActionBar Design

### Structure

The action bar sits at the bottom of the panel with a top divider and a slight tonal separation from the content area above it.

Each action renders as a full button surface with:

- icon
- short verb label
- one short helper line

The helper line explains effect, not mechanism.

### Layout Rules

- default layout is stacked for narrow widths
- two secondary actions may sit side-by-side when width allows
- the primary action remains visually dominant and may span full width even on larger layouts
- three equal-width horizontal buttons are not allowed
- button labels must not be truncated into ambiguity

Recommended responsive pattern:

- mobile: stacked buttons
- narrow tablet: two secondary buttons on one row, primary full width below
- wide: secondary group inline, primary still larger or more visually dominant

### Hierarchy Rules

- exactly one `primary`
- all other visible footer actions are `secondary`
- a tertiary action, if needed, should be rendered outside the main action bar as a link-style escape hatch rather than a third equal-weight button

### Advisor Mapping

Advisor should map into the pattern as:

- `Rebuild`
  - helper: `Refresh digest from gallery`
- `Reset`
  - helper: `Restore digest text`
- `Apply`
  - helper: depends on apply mode, such as `Append to prompt` or `Replace prompt`

The current `Apply Mode` selector should be visually grouped with the `Apply` action rather than floating as an unrelated input above the button row.

## SurfaceHeader Design

### Structure

`SurfaceHeader` is a calm top band for major surfaces. It should feel informative and stable, not animated or busy.

It contains:

- title
- durable meta chips such as frontend version, backend version, and current mode summary
- one short summary line for persistent guidance or recency

### Layout Rules

- header content must wrap cleanly on narrow widths
- meta chips may wrap to additional lines
- header content must not compete visually with the operations pane below it
- active status badges should not live here once the operations pane exists

### Chat Application

For chat, `SurfaceHeader` should replace the current mixed header behavior in [ChatHeader.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/chat/ChatHeader.jsx):

- keep title and durable version chips
- keep the keyboard tip or other calm helper text
- remove animated dream-mode presentation from the header once dream state is represented in `PendingOperationsPane`

The header remains in scope for this work because it is part of the control-surface stack:

`SurfaceHeader -> PendingOperationsPane -> content`

## PendingOperationsPane Design

### Placement

The pane lives in the sticky strip already present inside [ChatContainer.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/chat/ChatContainer.jsx).

This gives the app a single visible place for current work without forcing status to compete with panel content or image content.

### Operation Item Shape

Each rendered operation item may show:

- icon
- operation title
- short state text
- optional progress value
- optional count or queue position
- optional cancel affordance

Example items:

- `Generating image` / `Step 8 of 28`
- `Queue` / `2 waiting`
- `Advisor rebuild` / `Refreshing digest`
- `Dream mode` / `Exploring variations`

### Motion Rules

- only the operations pane may pulse or glow for active work
- completion uses a short landing state, then auto-removes
- error stops active animation immediately and lingers longer than completion
- panels, image badges, and headers should remain visually calm while an operation is active elsewhere

### Duplication Rules

When an operation is represented in the pane, animated duplicates should be removed from local surfaces.

That means:

- remove animated `generating` and `dreaming` badges from image content in [MessageBubble.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/chat/MessageBubble.jsx)
- replace advisor `Building digest...` text in [AdvisorPanel.jsx](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/components/options/AdvisorPanel.jsx) with durable local state such as recency or error

Local surfaces may still show:

- last updated timestamps
- inline errors
- static contextual labels

## Operation Status Model

### Status System Role

The shared status system is a normalized display layer over structured operation feedback.

It is not authoritative state.

Preferred flow:

`operation events -> adapter -> operations store -> pending operations pane`

Fallback flow:

`local lifecycle events -> coarse adapter update -> operations store -> pending operations pane`

### Status Controller API

Feature code should not render pane rows directly. It should receive a lightweight handle that updates a shared operation record.

Representative shape:

```ts
const status = operations.start({
  key: 'advisor-rebuild:gal_1',
  kind: 'advisor',
  icon: 'sparkles',
  tone: 'active',
  text: 'Rebuilding',
  detail: 'Refreshing digest',
  cancellable: false,
});

status.setText('Analyzing evidence');
status.setDetail('12 images');
status.setProgress({ current: 6, total: 12 });
status.complete({ text: 'Digest updated' });
```

Supported semantics:

- `setText()`
- `setDetail()`
- `setProgress()`
- `setTone()`
- `cancel()`
- `complete()`
- `error()`
- `remove()`

The handle updates semantic state only. The pane owns visual policy such as glow, fade, landing style, and auto-retirement.

`cancel()` is optional and only present when the underlying operation exposes a real cancellation callback or cancellable operation key. The controller should not invent cancellation where none exists.

### Keyed Updates

Operations should support a stable `key` so callers can upsert one visible item instead of creating duplicates.

Examples:

- `advisor-rebuild:<gallery_id>`
- `generation:<message_id>`
- `dream:<session_id>`

If a keyed operation already exists, `start()` should update or reuse that record rather than spraying new indicators.

## Structured Feedback Integration

### Generation

Generation progress should be sourced from structured backend or job events rather than inferred from UI state alone.

Planned generation payload:

```json
{
  "operation_id": "gen_123",
  "kind": "generation",
  "phase": "denoising",
  "step": 8,
  "total_steps": 28,
  "message": "Generating",
  "cancellable": true
}
```

The natural source for these updates is `callback_on_step_end`, which can emit numeric progress derived from `step / num_inference_steps`.

The UI adapter translates this into pane-friendly display state:

- title: `Generating image`
- detail: `Step 8 of 28`
- progress: `8 / 28`

Later phases such as `artifacting` or `finalizing` may update `phase` and `message` without changing operation identity.

### Advisor

Advisor rebuild should use a keyed operation and move through coarse phases such as:

- `Collecting gallery evidence`
- `Analyzing evidence`
- `Refreshing digest`
- `Digest updated`

If future backend work exposes finer-grained advisor progress, the same adapter path should consume it.

### Dream Mode

Dream mode should enter the operations system from the existing lifecycle in [useImageGeneration.js](/Users/darkbit1001/workspace/Stability-Toys/lcm-sr-ui/src/hooks/useImageGeneration.js), not from presentation components.

Concrete entry points:

- `startDreaming()` creates or upserts a keyed dream operation
- `restartDreamInterval()` updates recurring detail such as cadence
- `guideDream()` may update detail text to reflect a newly guided target
- `saveDreamAndContinue()` keeps the same dream operation but refreshes its detail to reflect the new active message
- `stopDreaming()` completes or removes the dream operation

Representative dream operation shape:

- key: `dream:active`
- title: `Dream mode`
- detail: `Exploring variations`
- optional detail updates: `Every 5s`, `Guided to selected image`

Once dream lifecycle is routed through the operations store, animated dream badges should be removed from both image content and chat header.

## Visual Language

### Product UI Direction

The control surface should look designed rather than merely functional.

Guidelines:

- use clear containers and deliberate spacing so actions never resemble inline text
- use icons to accelerate scanning
- use helper text to disambiguate consequence
- keep the panel body calmer than the action and operations surfaces

### Affordance Requirements

Every action button must have:

- visible container shape
- fill or outline at rest
- icon plus label
- sufficient padding
- disabled styling that still reads as a control

Prohibited patterns:

- text-only footer actions
- hover-only affordance
- three same-weight buttons in one compressed row

## Integration Plan

### Phase 1: Shared Primitives

- add `SurfaceHeader`
- add `PanelActionBar`
- add `PendingOperationsPane`
- add an `OperationsProvider` based on React context plus `useReducer`
- add a status controller abstraction exposed through controller hooks

### Phase 2: Advisor Migration

- migrate Advisor actions into `PanelActionBar`
- group apply mode with the apply action
- remove animated rebuild text from Advisor body
- source advisor rebuild status through the shared operations layer

### Phase 3: Chat And Generation Migration

- migrate chat header to `SurfaceHeader`
- replace the sticky placeholder strip in chat with `PendingOperationsPane`
- route generation lifecycle into the operations store
- route dream lifecycle from `useImageGeneration()` into the operations store
- remove animated `generating` and `dreaming` badges from message content

### Phase 4: Structured Generation Progress

- expose structured denoising progress from generation logic
- translate that progress into pane updates
- add queue and cancellation metadata where available

Phase 4 depends on backend work to emit structured progress from generation callbacks such as `callback_on_step_end`. It should be planned as a dependency-sensitive track that can slip independently of the earlier UI migration work.

## Validation Criteria

The design is successful when:

- Advisor footer actions remain legible and inside bounds on narrow widths
- a first-time user can identify all footer actions as buttons without hover
- active work appears in one shared pane rather than in scattered animated labels
- generation and advisor statuses can be updated through handles or adapters without manual pane rendering
- local surfaces retain contextual errors and durable summary information without duplicating active-state animation

## Review Checklist

Every new UI feature that introduces work state or multi-action footers should answer:

- What is the one primary action?
- How does the action area behave at narrow width?
- Does active work flow through the shared operations surface?
- If not, is there real structured feedback that should be adapted first?
- Are actions recognizable as buttons without hover?
- Is glow or pulse limited to the pending operations surface?
