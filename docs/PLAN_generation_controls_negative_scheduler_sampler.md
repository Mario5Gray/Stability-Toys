# Generation Controls Scope: Negative Prompt, Scheduler, Sampler

## Goal

Add first-class generation controls for:

- negative prompting
- scheduler / sampler selection

The design should work across backend, mode config, and frontend without baking model-specific special cases into the UI.

## Current State

### Backend request surface

`server/lcm_sr_server.py` `GenerateRequest` currently supports:

- `prompt`
- `size`
- `num_inference_steps`
- `guidance_scale`
- `seed`
- style LoRA
- superres
- denoise strength

There is no request field for:

- `negative_prompt`
- `scheduler`
- `sampler`

### Worker execution

`backends/cuda_worker.py` currently passes only:

- `prompt`
- `width`
- `height`
- `num_inference_steps`
- `guidance_scale`
- `generator`

No negative prompt is forwarded, and there is no per-request scheduler override.

### Mode config

`server/mode_config.py` supports model defaults and capability hints, but not:

- default negative prompt template
- allowed negative prompt templates
- default scheduler choice
- allowed schedulers / samplers

### Frontend

The UI currently exposes:

- prompt
- size
- steps
- cfg
- seed

There are no controls for negative prompt or scheduler / sampler.

## Important Design Decision

Treat "scheduler" and "sampler" as one backend control in v1.

Reason:

- In Diffusers, what users often call a "sampler" is usually implemented as a scheduler class/config choice.
- If we model both independently now, we create an API surface that looks flexible but is internally ambiguous.
- We can still label the UI as `Sampler` if that is the user-facing language you want.

Recommended canonical backend field:

- `scheduler_id`

Examples of UI labels:

- `Sampler`
- `Scheduler`
- `Sampler / Scheduler`

## Product Recommendation

### Negative prompt

Config-side authority, UI-side selection.

Use mode config to define:

- a default negative prompt template
- a small list of named negative prompt templates
- optional permission for freeform negative prompt editing

Recommended UX:

- simple dropdown in the UI for template selection
- optional advanced textarea for raw negative prompt override

This matches your preference: curated templates live in config, users choose from the front end.

### Scheduler / sampler

Mode-constrained frontend selection.

Use mode config to define:

- `default_scheduler_id`
- `allowed_scheduler_ids`

Recommended UX:

- dropdown in the UI
- only show options allowed for the active mode

This keeps unsafe or nonsensical combinations out of the user path.

## Proposed Backend Shape

### 1. Request schema

Extend `GenerateRequest` with:

- `negative_prompt: Optional[str] = None`
- `scheduler_id: Optional[str] = None`

Do not add a separate `sampler` field in v1.

### 2. Mode config schema

Extend `ModeConfig` with:

- `default_negative_prompt_template: Optional[str]`
- `negative_prompt_templates: Dict[str, str]`
- `allow_custom_negative_prompt: bool`
- `default_scheduler_id: Optional[str]`
- `allowed_scheduler_ids: List[str]`

These should serialize through:

- YAML loading
- YAML save
- `/api/modes`

### 3. Scheduler registry

Add a backend scheduler registry module, for example:

- `backends/scheduler_registry.py`

Responsibilities:

- map `scheduler_id` -> Diffusers scheduler class
- expose allowed scheduler IDs
- filter scheduler compatibility by pipeline family / mode

Do not scatter scheduler construction across workers.

### 4. Worker application

Before generation, resolve the effective scheduler:

- request override if present and allowed
- else mode default
- else current pipeline scheduler

Apply by rebuilding from existing config:

- `SchedulerClass.from_config(pipe.scheduler.config)`

Negative prompt should be passed directly into:

- txt2img pipeline calls
- img2img pipeline calls

### 5. Compatibility rules

Compatibility must be mode-aware.

Examples:

- LCM-tuned modes should not expose arbitrary schedulers by default
- SDXL modes may allow a broader set
- Some checkpoint families may require sticking to native scheduler defaults

This is why scheduler options belong in config/mode policy, not as a global hardcoded UI list.

## Proposed Frontend Shape

### Base controls

Add to generation params and request serialization:

- `negativePrompt`
- `schedulerId`

### UI behavior

For the active mode:

- fetch `negative_prompt_templates`
- fetch `default_negative_prompt_template`
- fetch `allow_custom_negative_prompt`
- fetch `allowed_scheduler_ids`
- fetch `default_scheduler_id`

Render:

- `Negative Prompt Template` dropdown
- optional `Negative Prompt` textarea when custom editing is allowed
- `Sampler` dropdown backed by `scheduler_id`

### Recommended defaults

Keep the default UI collapsed/simple:

- dropdown for negative template
- dropdown for sampler
- freeform negative prompt behind an advanced affordance

## API / Config Contract

Recommended `/api/modes` additions per mode:

- `negative_prompt_templates`
- `default_negative_prompt_template`
- `allow_custom_negative_prompt`
- `allowed_scheduler_ids`
- `default_scheduler_id`

This lets the frontend stay dumb:

- server tells UI what is legal
- UI only renders what the current mode supports

## Testing Scope

### Backend tests

- mode config parse/save round-trip for new fields
- `/api/modes` response includes new scheduler/template metadata
- request schema accepts `negative_prompt` and `scheduler_id`
- worker forwards `negative_prompt`
- worker swaps scheduler correctly for allowed scheduler IDs
- invalid scheduler IDs are rejected cleanly

### Frontend tests

- params hook stores `negativePrompt` and `schedulerId`
- request payload includes them
- mode switch refreshes available templates/schedulers
- disabled/hidden UI behavior matches config

## Recommended Delivery Order

### Phase 1

Backend negative prompt plumbing.

Ship:

- request field
- worker forwarding
- mode config templates
- API exposure

This is the lowest-risk, highest-value first slice.

### Phase 2

Backend scheduler registry + mode-constrained scheduler selection.

Ship:

- `scheduler_id`
- registry
- allowed/default scheduler config
- worker application

### Phase 3

Frontend controls.

Ship:

- negative template selection
- optional custom negative prompt
- sampler dropdown

## Risks

### 1. Scheduler explosion

If the UI gets a raw global list of every Diffusers scheduler, users will create invalid combinations and produce hard-to-debug complaints.

Mitigation:

- mode-constrained `allowed_scheduler_ids`

### 2. Terminology mismatch

Users say "sampler", Diffusers says "scheduler".

Mitigation:

- backend canonical field: `scheduler_id`
- UI label can say `Sampler`

### 3. LCM incompatibility

LCM modes are especially sensitive to scheduler choice.

Mitigation:

- config defaults and allowlists per mode

## Recommendation

Build this as:

- config-driven policy
- backend canonical `scheduler_id`
- frontend selection from server-provided allowlists

Do not ship a freeform sampler/scheduler picker first.
Do not split `scheduler` and `sampler` into two backend fields in v1.

That gets you a stable foundation for negative prompt templates now, and controlled scheduler selection without turning the UI into a compatibility trap.
