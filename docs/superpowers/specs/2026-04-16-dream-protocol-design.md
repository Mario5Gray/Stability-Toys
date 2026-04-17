# Dream Protocol Design

## Summary

This design replaces the current ad hoc dream loop with a state-transition control plane and a pluggable navigator protocol.

The core distinction is:

- the server owns dream session identity, policy, buffering, storage, evaluation hooks, and transition reconciliation
- navigator plugins own generation behavior and internal navigation logic

The protocol is intentionally not built around "submit and wait." It is built around observable state transitions and append-only events. A navigator may run in-process today, but the contract must also support out-of-process and non-Python implementations later, including long-running Rust dreamers and separate execution engines.

The first backend issue should deliver the protocol, the core operator boundary, runtime switching semantics, and a reference in-process adapter. It should not deliver the separate short-environment executor.

## Goals

- Replace the current loop-shaped dream implementation with a typed state-machine protocol
- Introduce generic named navigators such as `random` and `langevin`
- Let the core switch navigators at runtime without redefining the dream API
- Separate session/state carriage from navigator-owned generation logic
- Support manifests and runtime contracts that can map to local Python plugins now and non-Python or out-of-process runtimes later
- Preserve the current dream feature as a baseline navigator behind the new contract
- Make `langevin` a first-class navigator target rather than a UI alias or heuristic shortcut

## Non-Goals

- Building the separate short-environment or containerized execution engine in this issue
- Adding frontend controls for navigator selection in this issue
- Generalizing the protocol into a full training and deployment lifecycle system
- Solving cluster scheduling, GPU orchestration, or external runtime reconciliation in this repo change
- Designing the final LLM-assisted UX for dream guidance in this issue

## Current State

- [`yume_lab/yume/dream_worker.py`](/Users/darkbit1001/workspace/Stability-Toys/yume_lab/yume/dream_worker.py:102) runs a background loop that generates one candidate, scores it, conditionally stores it, and repeats.
- [`yume_lab/server/ws_routes.py`](/Users/darkbit1001/workspace/Stability-Toys/yume_lab/server/ws_routes.py:29) exposes `dream:start`, `dream:stop`, `dream:top`, and `dream:guide`, but those handlers target a single mutable worker shape rather than a protocol boundary.
- Navigator behavior is implicitly embedded in the worker. There is no manifest, no runtime contract, no session transition model, and no supported way to switch generation procedures while preserving core session state.
- [`yume_lab/yume/strategies.py`](/Users/darkbit1001/workspace/Stability-Toys/yume_lab/yume/strategies.py:113) contains exploration strategy scaffolding, but it is not the control boundary and does not provide a plugin protocol.

The current shape is useful as an experiment, but it encodes the wrong ownership boundary. The core should coordinate dream lifecycle. The navigator should define how the dream advances.

## Proposed Approach

Introduce a dream protocol with three explicit layers:

- `Dream Protocol`: typed session state, event stream, navigator manifests, runtime messages, and transition rules
- `Dream Operator`: server-owned reconciler that stores session state, routes control messages, buffers outputs, applies evaluation policy, and manages navigator attachment or switching
- `Navigator Runtime`: plugin-owned implementation that accepts state and control inputs, emits events, and advances generation

The canonical flow becomes:

`session created -> navigator resolved -> navigator attached -> navigator initialized -> stepping -> candidate emitted -> candidate evaluated -> state updated -> navigator switched -> draining -> stopped | failed`

This keeps the server responsible for durable control-plane behavior while allowing generation procedures to vary by navigator implementation.

## Protocol Model

### DreamSessionSpec

`DreamSessionSpec` defines the session at creation time.

Proposed fields:

```json
{
  "version": 1,
  "session_id": "dream_123",
  "base_prompt": "cinematic photograph of a foggy city",
  "navigator": "random",
  "navigator_config": {},
  "evaluation_policy": {
    "scorer": "clip",
    "similarity_threshold": 0.7
  },
  "buffer_policy": {
    "top_k": 100,
    "render_interval": 100
  },
  "storage_policy": {
    "persist_candidates": true
  },
  "isolation_hints": {
    "runtime_class": "in_process"
  }
}
```

Rules:

- the core owns `session_id`, policy, and storage semantics
- the navigator name is a logical routing key, not a UI label
- `navigator_config` is namespaced plugin input
- `isolation_hints` are advisory protocol fields, not guarantees in this issue

### DreamSessionState

`DreamSessionState` is the reconciled state the operator maintains over time.

Minimum fields:

- lifecycle phase
- attached navigator identity and version
- last accepted navigator snapshot
- evaluation summary
- buffered candidate counts
- last error
- pending switch request if one exists

This state must be updateable by events. It must not depend on blocking call completion.

### DreamEvent

All progression should occur through append-only events.

Minimum event classes:

- `session_created`
- `navigator_resolved`
- `navigator_attached`
- `navigator_initialized`
- `step_requested`
- `candidate_emitted`
- `candidate_evaluated`
- `state_updated`
- `input_applied`
- `navigator_switch_requested`
- `navigator_switched`
- `paused`
- `stopped`
- `failed`

Events are the transport truth. Derived state is a projection.

## Navigator Contract

### Manifest

Each navigator exposes a first-class manifest so the protocol can support non-Python and out-of-process implementations later.

Minimum manifest fields:

```json
{
  "name": "langevin",
  "version": "0.1.0",
  "display_name": "Langevin Navigator",
  "runtime": {
    "kind": "rpc",
    "entrypoint": "navigator://langevin"
  },
  "capabilities": {
    "supports_runtime_input": true,
    "supports_snapshot": true,
    "supports_handoff": true,
    "supports_streaming_candidates": true
  },
  "state_schema_version": 1,
  "config_schema_version": 1
}
```

In-process Python navigators may use a local adapter, but they should still present this logical manifest shape.

### Runtime Interface

The runtime contract is transition-oriented.

Required operations:

- `initialize(session_spec) -> events`
- `step(operator_state, navigator_state) -> events`
- `apply_input(input_event, operator_state, navigator_state) -> events`
- `snapshot(operator_state, navigator_state) -> snapshot`
- `handoff(snapshot, target_navigator) -> events`
- `shutdown(reason) -> events`

Notes:

- operations may yield multiple events
- `step` may emit zero, one, or many candidates
- the operator is allowed to schedule steps repeatedly or pause them
- `handoff` exists so navigator switching can preserve useful state when supported

### Named Navigators

The protocol should expose generic named navigators, not hard-coded mode labels.

Initial registry target:

- `random`
- `langevin`

Later navigators may include `evolutionary`, `cluster`, or external engines without changing the top-level dream API.

UI simplification can map these to "Dream Mode 1" and "Dream Mode 2" later, but the protocol should retain the generic names.

## Langevin Navigator

`langevin` is a protocol target for real score-guided navigation, not a heuristic seed mutator.

For this design, the important boundary is:

- the protocol must allow a navigator to own iterative score-guided state evolution
- the core must not assume generation is a single blocking request with one score result
- the navigator may require richer internal state, repeated score evaluations, and noise-injected transition steps

The first implementation may begin with an adapter layer in this repo, but the contract must remain suitable for a future external runtime with stronger numerical and systems guarantees.

## Operator Responsibilities

The dream operator should own:

- session creation and identity
- manifest resolution
- navigator attachment and lifecycle supervision
- event ingestion and state reconciliation
- candidate buffering and top-k selection
- score and evaluation policy routing
- persistence to Redis or later storage backends
- runtime switching requests and switch safety checks

The operator should not own:

- navigator-specific generation procedure
- navigator-specific internal search state semantics
- container orchestration for external runtimes

## Runtime Switching

Runtime switching is a first-class protocol feature.

Required semantics:

- switching is expressed as a state transition request, not a hard reset
- the operator may request a snapshot from the current navigator
- the target navigator may accept or reject handoff
- failed handoff must surface as an explicit event and preserve session integrity
- the session should remain queryable throughout switch progress

Switch flow:

`navigator_switch_requested -> snapshot_requested -> snapshot_received -> target_attached -> handoff_applied -> navigator_switched`

If handoff is unsupported, the operator may fall back to a cold initialize path, but that fallback must be explicit in state and events.

## WebSocket and API Shape

The existing `dream:start`, `dream:stop`, and `dream:guide` messages should evolve into protocol-oriented control messages.

Desired direction:

- `dream:start` creates a session from `DreamSessionSpec`
- `dream:input` applies live runtime input such as prompt updates or navigator-scoped controls
- `dream:switch` requests navigator replacement by logical name
- `dream:status` returns current projected session state
- `dream:events` or streamed event pushes expose transition progress

The important change is semantic, not just naming. Requests should express desired state changes. Responses should expose accepted transitions and projected state rather than imply synchronous completion.

## Execution Model

This issue should support two conceptual runtime classes in the contract:

- `in_process`: local adapter for immediate implementation
- `external`: reserved runtime class for future RPC or isolated executors

Only `in_process` needs to be implemented now. `external` exists so the protocol does not later need a breaking redesign when Rust dreamers or a separate workload operator arrive.

The separate short-environment execution engine is explicitly deferred to another project or issue set.

## Failure Model

Failures should be evented and typed.

Initial failure classes:

- manifest resolution failure
- navigator initialization failure
- step execution failure
- snapshot or handoff failure
- evaluation failure
- persistence failure
- protocol violation

The operator should surface these as dream-session state, not only as logs.

## Testing

The first implementation should include:

- state transition tests for session start, step, stop, and failure paths
- registry tests for manifest resolution and navigator lookup
- switching tests for successful and failed handoff
- compatibility tests that place the current baseline dream worker behavior behind the navigator contract
- protocol serialization tests for control messages and events

## Acceptance Criteria

- Dream sessions are modeled and driven as state transitions rather than blocking generation requests
- The core exposes a navigator registry keyed by generic names
- The baseline current dream behavior runs behind the navigator contract
- A `langevin` navigator is represented as a first-class protocol target
- Runtime switching between navigators is part of the backend contract
- The contract allows future out-of-process or non-Python navigators without a top-level API redesign
- The separate short-environment executor remains out of scope for this issue

## Follow-On Work

The first issue created from this design should remain backend-first.

Later work may cover:

- UI abstractions that hide generic navigator names behind simpler mode labels
- finalized interaction design for live dream guidance, including if and how LLM assistance participates
- a separate execution engine or workload operator for isolated short-environments
- richer evaluation and feedback loops beyond the current candidate scoring flow
