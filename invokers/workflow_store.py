# invokers/workflow_store.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

Json = Dict[str, Any]


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    prompt_path: str

    # Node IDs to patch (per-workflow contract)
    load_image_node: Optional[str] = None      # node id that has inputs.image (LoadImage)
    sampler_node: Optional[str] = None         # node id for KSampler (seed/steps/cfg/denoise)
    pos_text_node: Optional[str] = None        # CLIPTextEncode positive (optional)
    neg_text_node: Optional[str] = None        # CLIPTextEncode negative (optional)


class WorkflowStore:
    def __init__(self, specs: Dict[str, WorkflowSpec]) -> None:
        self.specs = specs
        self._cache: Dict[str, Json] = {}

    def load_prompt(self, workflow_id: str) -> Json:
        if workflow_id in self._cache:
            return self._cache[workflow_id]
        spec = self.specs.get(workflow_id)
        if not spec:
            raise WorkflowError(f"Unknown workflow_id={workflow_id}")
        with open(spec.prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise WorkflowError(f"{spec.prompt_path} must be a JSON object (prompt graph)")
        self._cache[workflow_id] = data
        return data

    def make_prompt(
        self,
        workflow_id: str,
        *,
        uploaded_filename: Optional[str] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        denoise: Optional[float] = None,
        seed: Optional[int] = None,
        prompt_text: Optional[str] = None,
        negative_text: Optional[str] = None,
    ) -> Json:
        base = self.load_prompt(workflow_id)
        pg: Json = copy.deepcopy(base)

        spec = self.specs[workflow_id]

        def _ensure_inputs(nid: str) -> Dict[str, Any]:
            if nid not in pg:
                raise WorkflowError(f"node {nid} not in prompt graph")
            pg[nid].setdefault("inputs", {})
            return pg[nid]["inputs"]

        def _is_link(v: Any) -> bool:
            # Comfy links look like ["node_id", output_index]
            return (
                isinstance(v, (list, tuple))
                and len(v) == 2
                and isinstance(v[0], str)
                and isinstance(v[1], int)
            )

        def _patch_linked_value(link: Any, value: Any, *, preferred_keys: tuple[str, ...]) -> bool:
            """
            Try to patch the source node for a linked input.
            Returns True if we successfully patched a plausible input key.
            """
            src_nid = link[0]
            src_inputs = _ensure_inputs(src_nid)

            # Try common keys in order; only write keys that exist OR are reasonable to add.
            # For Seed Generator (Image Saver) in your graph, key is "seed".
            for k in preferred_keys:
                if k in src_inputs:
                    src_inputs[k] = value
                    return True

            # If none exist, fall back to: if node has a single numeric-like input, patch that.
            numeric_keys = []
            for k, v in src_inputs.items():
                if isinstance(v, (int, float)) and not _is_link(v):
                    numeric_keys.append(k)

            if len(numeric_keys) == 1:
                src_inputs[numeric_keys[0]] = value
                return True

            return False

        def _set_input(nid: str, key: str, value: Any, *, link_keys: tuple[str, ...]) -> None:
            """
            Set pg[nid].inputs[key] to value.
            If pg[nid].inputs[key] is a link, patch the source node instead (best-effort),
            and leave the link intact.
            """
            inputs = _ensure_inputs(nid)
            cur = inputs.get(key)

            if _is_link(cur):
                # Patch the upstream node and keep the link, so graph wiring remains valid.
                patched = _patch_linked_value(cur, value, preferred_keys=link_keys)
                if not patched:
                    # Last resort: overwrite the sampler input directly.
                    # (Keeps behavior similar to your previous implementation if upstream patching fails.)
                    inputs[key] = value
            else:
                inputs[key] = value

        # Patch image (always overwrite if provided)
        if uploaded_filename and spec.load_image_node:
            _set_input(spec.load_image_node, "image", uploaded_filename, link_keys=("image",))

        # Patch sampler parameters
        if spec.sampler_node:
            sn = spec.sampler_node

            if steps is not None:
                _set_input(sn, "steps", int(steps), link_keys=("steps", "value", "int", "number"))
            if cfg is not None:
                _set_input(sn, "cfg", float(cfg), link_keys=("cfg", "value", "float", "number"))
            if denoise is not None:
                _set_input(sn, "denoise", float(denoise), link_keys=("denoise", "strength", "value", "float", "number"))

            if seed is not None:
                # In your graphs, seed is wired to node 29 ("Seed Generator (Image Saver)") which uses inputs.seed
                _set_input(sn, "seed", int(seed), link_keys=("seed", "value", "int", "number"))

        # Optional prompt text
        if prompt_text is not None and spec.pos_text_node:
            _set_input(spec.pos_text_node, "text", str(prompt_text), link_keys=("text",))

        if negative_text is not None and spec.neg_text_node:
            _set_input(spec.neg_text_node, "text", str(negative_text), link_keys=("text",))

        return pg