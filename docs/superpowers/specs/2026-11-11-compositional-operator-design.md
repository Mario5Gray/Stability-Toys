# Compositional Operator Design

- Status: draft
- Date: 2026-11-11
- FP issue: STABL-wdfpqfcg
- Related: `docs/vector-map-operator.md`, `docs/vector-map-recipes/rose-heart/README.md`, STABL-kczspmud, STABL-pfaepjbz
- External: `github.com/Mario5Gray/dagdir` (GAP recovery only. Not vendored.)

## 1. Intent

One command takes a query and returns an output. The query names an input asset, a target artifact, constraints, and acceptance. The operator finds a chain of existing tools. It runs the chain. It returns the trail and the final output.

Rose Heart proved the shape for one chain. This design generalizes that shape to the whole suite.

## 2. Why now

No single tool solves image to origami pieces. The chain crosses tool boundaries. Manual chaining loses provenance, hashes, and failure reasons. A cold agent cannot replay it.

## 3. Goals

- G1. A query returns a result. The result is a bundle or a gap spec.
- G2. Each node runs as an isolated process.
- G3. Each node accepts its neighbour's artifact as input.
- G4. Every step records input hash, params, output hashes, and verdicts.
- G5. Replay needs no planner, no skill, and no LLM.
- G6. The result bundle keeps the Rose Heart shape and rules.
- G7. Discovery uses the tools themselves, not a hand-written catalog.

## 4. Non-goals

- The operator does not schedule work across machines.
- The operator does not provide a user interface beyond the CLI.
- The operator does not host models. It calls the existing services.
- The operator does not write missing nodes. It returns their specs.

## 5. Anti-goals

- An LLM inside replay. This hides or distorts facts.
- A node that names its partners. This recouples the nodes.
- A PASS built from sub-step sums. This hides a broken function.
- A silent fallback when a gap exists. This fakes a result.
- A calibrated score without calibration data. This overstates certainty.

## 6. Constraints

- No intelligence runs downstream of the query. Nodes do arithmetic and geometry.
- Each node is its own service to any other node.
- A node accepts the output of its nearest neighbour.
- Node tools use CPython 3.12 in the `stability-toys` conda env.
- The `st` CLI uses Go and cobra.
- Untracked output types never land in the repository.
- A node keeps its declared determinism class. It never claims a stronger one.
- An unverified fact stays unverified. `not_run` and `pending` never become `pass`.

## 7. Terms

| Term | Meaning |
|---|---|
| Node | One tool with typed input and output ports |
| Port | One artifact kind, with a schema and a hash |
| Edge | One port match between two nodes |
| Exact candidate | A node whose input port matches the held artifact |
| Likely neighbour | A node the producer proposes from its output evaluation |
| Trail | Ordered record of the steps taken |
| Bundle | Trail, recipe, manifest, artifacts, result, README |
| Gap spec | Full spec of the one missing node |
| Noul | Probability that an edge succeeds. Value 0 to 1 |
| Sentiment | Preference-weighted desirability. Value 0 to 1 |
| Determinism class | How replay compares two runs |

## 8. Architecture

```text
query.json
    |
    v
+-----------------------------+
|  operator (st compose)      |
|  discovery -> decision ->   |
|  run -> emit                |
+-----------------------------+
    |            |            |
    v            v            v
 node A       node B       node C      each an isolated process
    |            |            |
    +---artifact--+--artifact--+
                 |
                 v
        bundle: trail + output
```

Rules:

- The operator calls nodes. Nodes never call each other.
- A node consumes the artifact its neighbour produced.
- A node publishes a manifest for its own outputs.
- The operator composes an outer manifest. The outer manifest lists every artifact below it.
- No manifest lists itself.

## 9. Two phases

| Phase | Planner | Reproducible |
|---|---|---|
| Explore | Operator decides each edge | No |
| Replay | Trail decides every edge | Yes, to its class |

Explore resolves every guess. Replay walks only confirmed steps.

A skill may front Explore with prose. The skill writes its structured query into the bundle. Replay then needs no skill.

## 10. Node contract

Each node emits a descriptor from its own tool. The descriptor has eight parts.

| Part | Content |
|---|---|
| Identity | Node name and `schema_version` |
| Determinism | Class and replay comparison rule |
| Input | One port. Accepted artifact kind, requirements, expected input hash |
| Output | Artifact names, kinds, hashes |
| Evidence | Named keys with units. The node measures each |
| Acceptance | One functional contract over the whole output |
| Limits | Refusals and known failure modes |
| Replay | Exact command |

### 10.1 Acceptance is functional

Acceptance is one predicate on the whole output. Sub-step evidence sits beside it. Sub-step evidence never sums into PASS.

Good: "flat panels reassemble to the source silhouette at `width_mm`, and each fold edge is shared by exactly two panels."

Bad: "panel count above zero and coverage delta below one percent."

### 10.2 One input port

A node has exactly one input port. A node has one or more output artifacts.

A need for two artifacts becomes a join node upstream. Example: source image plus mask. The join node accepts one input and emits one combined artifact. The operator composes the join like any other node.

Never add a second input port. This keeps every edge simple and hashable.

### 10.3 Output evaluation

A node judges its own output. It reports named measurements. It also proposes likely consumers, with a reason from those measurements.

Example: the mesh is closed and manifold. Likely consumer: a slicer. Reason: slicers accept closed manifold solids.

### 10.3 Affordance, not partner

A node names an affordance of its output. It never names a partner tool.

- Allowed: "closed manifold solid accepts a slicer."
- Forbidden: "call Bambu Studio next."

## 11. Describe is scoped

`<tool> --describe` takes the held artifact and returns a local picture.

| Tier | Content | Weight |
|---|---|---|
| Self | Params, evidence keys, limits, replay | Contract |
| Exact | Candidates whose input port matches the held artifact | Fact |
| Likely | Neighbours proposed by output evaluation, each with a reason | Hypothesis |

Rules:

- `--describe` never dumps the whole suite. It shows the neighbourhood.
- Exact candidates come from port matching.
- Likely neighbours come from the producer's output evaluation.
- A likely neighbour without a reason is dropped.
- A likely edge becomes exact only after a run confirms it.
- The operator records which tier each edge used.

This keeps operator context bounded. The picture widens one ring per step.

## 12. Decision record

Each edge decision follows the project decision model.

| Input | Source |
|---|---|
| Objective | Query `objective` and `target` |
| Current state | Trail so far and the held artifact |
| Constraints | Query `constraints` |
| Alternatives | Exact candidates and likely neighbours |
| Evidence and uncertainty | Verdicts and determinism class |
| Preferences | Query `preferences` |
| Authority | Query `authority` |

Output:

- A chosen edge, or an explicit request for more information.
- `noul`, `sentiment`, `why`, `assumes`, `expects`, `accepts`, `next`.
- `null_reason` when information is missing.

### 12.1 Noul and sentiment

- Noul answers "will this edge work?" It is epistemic.
- Sentiment answers "do we want this edge, given preferences?" It is evaluative.
- Neither comes from an LLM. Both are functions of node evidence and query preferences.

### 12.2 Calibration limit

A deterministic score is not a calibrated probability. Calibration needs fit data and a validation set. Until that exists, the field is a declared score.

The trail names the kind: `noul_kind: "declared"` or `noul_kind: "calibrated"`. Only `calibrated` may use the word Noul without a qualifier.

## 13. Trail

The trail is linear. It records the steps taken. It does not record rejected branches. (Decision: Mario, 2026-11-11.)

Each entry:

```json
{
  "step": 3,
  "node": "st-vector-map",
  "version": "0.9.1",
  "input": {"path": "source.jpg", "sha256": "03bfbbaf..."},
  "params": {"threshold": 50, "width_mm": 120},
  "outputs": [{"path": "rose-heart.svg", "sha256": "da49fe7c..."}],
  "edge": {"tier": "exact", "reason": "image port match"},
  "verdicts": {"conversion": "converted", "cad_render": "not_run"},
  "decision": {
    "choice": "trace silhouette",
    "noul": 0.97,
    "noul_kind": "declared",
    "sentiment": 0.8,
    "why": "reviewed mask absent. Threshold matches this image.",
    "assumes": ["source stays unchanged"],
    "expects": ["one closed silhouette path"],
    "accepts": ["one closed path with 78 vertices"],
    "next": "render SCAD"
  },
  "null_reason": null
}
```

Rules:

- Each entry pins its input hash. A changed upstream artifact breaks the chain at a known step.
- An entry never holds another node's verdict.
- Replay writes its own trail. Compare trails entry by entry.

## 14. Result bundle

The bundle keeps the Rose Heart shape.

```text
<run-dir>/
  source.<ext>              input asset, hashed
  recipe.json               replay file, linear steps
  trail.json                decision trail
  <stem>.manifest.json      composed manifest
  <stem>.<ext>              terminal artifacts
  result.json               external gate verdicts
  README.md                 commands, evidence, change control, versions
```

| Rose file | Operator file | Change |
|---|---|---|
| `source.jpg` | `source.<ext>` | None |
| `recipe.json` | `recipe.json` | Adds `steps` |
| `*.vector.json` | `*.manifest.json` | Lists every artifact below it |
| `result.json` | `result.json` | Same role |
| `README.md` | `README.md` | Same six sections |
| none | `trail.json` | New |

Inherited rules:

- The manifest lists every other artifact with a hash.
- The manifest cannot list itself.
- Trust only artifacts the current manifest lists with matching hashes.
- External gates stay `not_run` or `pending` until proven.
- Three clean runs must agree on every bit-exact artifact.

### 14.1 Recipe versus trail

- `recipe.json` is the compact replay list. It holds node, input, input hash, params, and expected outputs.
- `trail.json` is the richer record. It adds edge tier, verdicts, and the decision block.
- Replay reads `recipe.json`. Audit reads `trail.json`.

## 15. Result kinds

| Kind | Meaning | Returns |
|---|---|---|
| SOLVED | Path found. Terminals meet `accepts`. | Full bundle |
| FAILED | A node exits nonzero, or its acceptance fails | Bundle plus the failing step |
| GAP | Path stops at a real artifact | The one missing node spec |
| NULL | Objective or constraints are incoherent | Bundle plus `null_reason` |

A `FAILED` result names four things: the step, the exit code, the stderr tail, and the predicate that failed.

Every query writes a result. No query exits without a verdict.

## 16. Gap spec

A gap result returns one node, not a partial chain.

```json
{
  "schema_version": 1,
  "node": "panelize",
  "determinism": "bit_exact",
  "input": {
    "artifact": "svg",
    "kind": "silhouette",
    "requires": {"closed_paths": true}
  },
  "gap": "split a silhouette into flat foldable panels with fold lines",
  "output": {
    "artifacts": [
      {"kind": "panel graph", "path": "<stem>.panels.json"},
      {"kind": "preview", "path": "<stem>.panels.svg"}
    ]
  },
  "sub_steps": [
    {"step": 1, "goal": "segment the silhouette into planar regions"},
    {"step": 2, "goal": "set fold edges between adjacent regions"},
    {"step": 3, "goal": "emit panel outlines plus the fold graph"}
  ],
  "evidence": [
    {"key": "panel_count", "unit": "count"},
    {"key": "fold_count", "unit": "count"},
    {"key": "coverage_delta", "unit": "pixels"},
    {"key": "fold_graph_connected", "unit": "bool"}
  ],
  "acceptance": {
    "contract": "flat panels reassemble to the source silhouette at width_mm, and each fold edge is shared by exactly two panels"
  },
  "limits": []
}
```

Rules:

- Three fields define the node: input, gap, output.
- `sub_steps` states coarse goals only. No implementation detail.
- `evidence` names the keys and units. The node measures each one.
- `acceptance` is one functional contract. It is not a sum of sub-steps.
- The spec has the same shape as `--describe`. Build the node, fill the stub, drop it into the suite, rerun the query.

### 16.1 Optional GAP recovery through dagdir

One possible motivation for dagdir is GAP recovery. A gap spec maps to one dagdir work package.

| Gap field | dagdir `WorkPackage` field |
|---|---|
| `node` | `id` |
| `gap` | `title` and `output_contract` context |
| `acceptance.contract` | `output_contract` |
| `evidence[].key` | `done_checks` |
| replay command | `test_command` |
| `input.kind` and `output.kind` | `scope_files` context |

Rules:

- dagdir is external. It is not vendored, copied, or placed inside this repo.
- The operator emits the gap spec. A separate human or agent action turns it into a `dag.yaml`.
- The operator never calls dagdir. The bridge is a file handoff.
- dagdir plans the work to build the node. The operator does not.
- The semantics fit. dagdir treats a confirmed absence as evidence and a halt as a result. A gap spec is both.

## 17. Determinism classes

| Class | Replay compares | Example |
|---|---|---|
| `bit_exact` | Output hashes | SVG, SCAD, STL geometry |
| `outcome_stable` | Acceptance predicate | SDXL and HunyuanDiT on MPS |
| `unclassified` | Nothing. Replay returns NULL | ControlNet map path without a generator |

Replay verdict:

| Verdict | Condition |
|---|---|
| PASS | Hashes match, or the predicate holds inside the class |
| FAIL | The predicate fails |
| NULL | No class, or inputs are not comparable |

The lean is a measured distance from the predicate. Examples: coverage delta in pixels, band width delta in mm, box error in mm. The trail stores the number.

NULL is the honest default. A node that declares no class makes replay NULL, not PASS.

## 18. Query interface

```bash
st compose --query query.json --out runs/rose-heart
st compose replay runs/rose-heart/recipe.json
```

`query.json`:

```json
{
  "schema_version": 1,
  "objective": "photographed object into 3d-printed origami pieces",
  "input": {"path": "source.jpg", "kind": "image"},
  "target": {"artifact": "stl", "accepts": ["closed manifold", "1 component"]},
  "constraints": {"width_mm": 120, "no_llm": true},
  "preferences": {"reliability": 1.0, "speed": 0.2, "reversibility": 0.5},
  "authority": {"may_run": ["*"]}
}
```

`objective` is free text. Every other field is structured. The operator does not parse prose into fields.

stdout summary:

```json
{
  "run": "runs/rose-heart",
  "kind": "SOLVED",
  "path": ["st-canny-map", "st-vector-map", "openscad", "bambu-info"],
  "verdict": "PASS",
  "terminal": "rose-heart.stl",
  "noul": 0.97,
  "noul_kind": "declared",
  "next": "operator slices and prints"
}
```

## 19. Existing tool map

| Node | Input | Output | Determinism | Descriptor today |
|---|---|---|---|---|
| `st gen` | text, image | image | outcome_stable | no |
| `st-canny-map` | image | edge PNG | bit_exact | no |
| `st-depth-map` | image | depth PNG | bit_exact | no |
| `st-pose-map` | image | pose PNG | bit_exact | no |
| `st-vector-map` | image, mask, edges | SVG, layers, SCAD, preview | bit_exact | no. Has `--json` evidence |
| OpenSCAD | SCAD | STL | bit_exact | no. Has `--version` |
| Bambu Studio | STL | acceptance, G-code | bit_exact | no. Has `--info` |

`st-vector-map --json` already emits `status`, `artifacts`, `counts`, `diagnostics`, and `evidence`. That is most of the descriptor payload. The descriptor adds ports, determinism, acceptance, and likely neighbours.

## 20. First slice

Scope: one chain, one descriptor, one gap.

1. Add `st-vector-map --describe`. Emit the eight-part descriptor.
2. Add `st compose` with structured query input.
3. Implement discovery over two nodes: `st-vector-map` and OpenSCAD.
4. Emit the bundle for the Rose Heart query. Compare with the committed recipe.
5. Return a gap spec for the origami query at the `panelize` port.

Success for the slice:

- Rose Heart replay matches committed hashes.
- Origami query returns the `panelize` gap spec.
- Trail and recipe differences are explainable.
- No node calls an LLM.

## 21. Open questions

| Question | Owner | Current position |
|---|---|---|
| CLI form: `st compose` subcommand or separate binary | Builder | `st compose` |
| Descriptor flag name and schema version | Builder | `--describe`, `schema_version: 1` |
| Evidence key naming | Builder | snake_case, units suffix |
| Noul calibration data | Mario | Declared score only for now |
| Authority semantics | Mario | `may_run` names allowed nodes |
| Exit codes for GAP and NULL | Builder | 0 SOLVED, 3 GAP, 4 NULL, 5 FAILED |
| Skill ships with the tool or after it | Mario | After the tool |
| Rejected branches in the trail | Mario | Not stored |
| GAP bridge to dagdir in the first slice | Mario | No. Later. |

## 22. What is left to the builder

- Choose the `st compose` package layout.
- Define the JSON schemas in one place.
- Decide how descriptors are found. Options: a flag on each tool, or a registry file.
- Define the manifest composition rule.
- Write the replay comparison.
- Keep prose out of the trail.

## 23. Decision log

| Decision | Owner | Date |
|---|---|---|
| Result = trail plus final output | Mario | 2026-11-11 |
| Gap returns one missing node | Mario | 2026-11-11 |
| Acceptance is the functional contract | Mario | 2026-11-11 |
| Node declares likely neighbours from output evaluation | Mario | 2026-11-11 |
| Describe is capability-driven and scoped | Mario | 2026-11-11 |
| Bundle keeps the Rose Heart shape | Mario | 2026-11-11 |
| Sentiment is preference-weighted desirability | Mario | 2026-11-11 |
| dagdir stays external to this repo | Mario | 2026-11-11 |
| dagdir use case is GAP recovery | Mario | 2026-11-11 |
