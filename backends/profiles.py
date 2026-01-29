# server/profiles.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

Json = Dict[str, Any]

@dataclass(frozen=True)
class WorkflowProfile:
    id: str
    title: str
    # node_id -> inputs patch
    node_inputs: Dict[str, Dict[str, Any]]
    notes: Optional[str] = None


PROFILES: Dict[str, WorkflowProfile] = {
    "default": WorkflowProfile(
        id="default",
        title="Default",
        node_inputs={
            # You can omit fields you don't want overridden by "default".
            # Keeping empty is fine.
        },
    ),

    # Example "papercut" profile — swap ckpt, swap LoRA, set texts.
    "papercut": WorkflowProfile(
        id="papercut",
        title="Papercut",
        node_inputs={
            "4": {  # CheckpointLoaderSimple
                "ckpt_name": "papercut_xl_v1.safetensors",
            },
            "55": {  # LoraLoader
                "lora_name": "papercut_style_xl.safetensors",
                "strength_model": 0.85,
                "strength_clip": 0.65,
            },
            # Positive prompt nodes
            "6":  {"text": "papercut, layered paper, crisp cut edges, high contrast, studio lighting"},
            "20": {"text": "papercut style, depth layers, clean silhouettes, minimal texture"},
            # Negative prompt nodes
            "7":  {"text": "blurry, soft edges, painterly, noisy texture, smudged"},
            "21": {"text": "low contrast, muddy edges, watercolor, oil paint, sketch"},
        },
        notes="High-contrast paper-cut look",
    ),
}