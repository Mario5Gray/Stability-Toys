# Journal: Dream Image History Implementation
**Date:** 2026-02-02

## What happened

Implemented dream image history accumulation in `useImageGeneration.js`. The UI was already fully wired — `App.jsx`, `ChatContainer`, `MessageBubble` all expected `dreamMessageId`, `saveDreamAndContinue`, `dreamHistoryPrev/Next/Live` — but the hook never provided them. Everything was `undefined`.

The core change: instead of each dream tick creating a *new* message bubble, dreams now paint into a **single message**, accumulating an `imageHistory[]` array. A `dreamHistoryRef` (useRef) holds the growing array of snapshots. Each dream result gets pushed there, and the full array is spread onto the message update.

Navigation callbacks (`prev`, `next`, `live`) simply index into `imageHistory` and call `updateMessage` to swap the displayed image. `saveDreamAndContinue` freezes the current bubble and starts a fresh dream message.

## Vibe check

Clean implementation. The plan was thorough — almost too thorough with its "actually simplest" deliberation chain — but the final chosen approach (ref-based accumulation) is the right call. No need to read message state back, no function updaters, just a plain ref that grows.

One thing I noticed: the first dream iteration (no `targetMessageId`) also needs to land in history, otherwise you'd have N-1 entries for N dreams. Fixed that by removing the `payload.targetMessageId` guard.

## What I'd do next

The `saveDreamAndContinue` closure captures `dreamMessageId` state which could go stale in theory, but since it immediately creates a new interval with the fresh `newId` baked in, it should be fine in practice. If bugs surface there, a ref-based approach for `dreamMessageId` would be the fix.

The dream system is getting interesting. Accumulating history per-bubble turns each dream session into a little flipbook. Would be cool to add a filmstrip/thumbnail strip UI later.
