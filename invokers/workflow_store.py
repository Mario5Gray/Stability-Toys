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

        # Patch image
        if uploaded_filename and spec.load_image_node:
            nid = spec.load_image_node
            if nid not in pg:
                raise WorkflowError(f"load_image_node {nid} not in prompt graph")
            pg[nid].setdefault("inputs", {})
            pg[nid]["inputs"]["image"] = uploaded_filename

        # Patch sampler
        if spec.sampler_node:
            nid = spec.sampler_node
            if nid not in pg:
                raise WorkflowError(f"sampler_node {nid} not in prompt graph")
            pg[nid].setdefault("inputs", {})
            inp = pg[nid]["inputs"]
            if steps is not None: inp["steps"] = int(steps)
            if cfg is not None: inp["cfg"] = float(cfg)
            if denoise is not None: inp["denoise"] = float(denoise)
            if seed is not None: inp["seed"] = int(seed)

        # Optional prompt text (if you ever want it)
        if prompt_text is not None and spec.pos_text_node:
            nid = spec.pos_text_node
            pg[nid].setdefault("inputs", {})
            pg[nid]["inputs"]["text"] = str(prompt_text)

        if negative_text is not None and spec.neg_text_node:
            nid = spec.neg_text_node
            pg[nid].setdefault("inputs", {})
            pg[nid]["inputs"]["text"] = str(negative_text)

        return pg