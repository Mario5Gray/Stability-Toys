# backends/styles.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

@dataclass(frozen=True)
class StyleDef:
    """
    A named style that corresponds to exactly one LoRA (exclusive selection).
    `levels` is a 1-indexed ladder of adapter weights. level=1 -> levels[0].
    """
    id: str
    title: str
    lora_path: str
    adapter_name: str  # stable internal name used in diffusers
    levels: Sequence[float]  # e.g. [0.35, 0.60, 0.85, 1.15]

    required_cross_attention_dim: Optional[int] = None  # e.g. 768 (SD1.x) or 2048 (SDXL)

# Tiny request parsing helper (no pydantic)
def parse_style_request(params: dict) -> StyleRequest:
    s = params.get("style")  # allow "style": "papercut"
    obj = params.get("style_lora")  # or allow "style_lora": {style, level}
    if isinstance(obj, dict):
        s = obj.get("style") or obj.get("id") or s
        lvl = obj.get("level", 0)
    else:
        lvl = params.get("style_level", 0)

    # normalize
    if s in (None, "", "none", "off", False):
        s = None
    try:
        lvl_i = int(lvl or 0)
    except Exception:
        lvl_i = 0

    return StyleRequest(style_id=s, level=lvl_i)

@dataclass(frozen=True)
class StyleRequest:
    """
    What the API receives.
    - style_id selects which LoRA (exclusive). None/"none" means off.
    - level: 0 => off, 1..N => preset strength index
    """
    style_id: Optional[str] = None
    level: int = 0

    def is_enabled(self) -> bool:
        return bool(self.style_id) and self.level > 0

    def weight(self, registry: Dict[str, StyleDef]) -> Optional[float]:
        if not self.is_enabled():
            return None
        sd = registry.get(self.style_id or "")
        if not sd:
            return None
        # clamp 1..len(levels)
        idx = max(1, min(int(self.level), len(sd.levels))) - 1
        return float(sd.levels[idx])


# ---- Your registry of styles (add more over time) ----
STYLE_REGISTRY: Dict[str, StyleDef] = {
    "papercut": StyleDef(
        id="papercut",
        title="Papercut",
        lora_path="/models/loras/PaperCut_SDXL.safetensors",
        adapter_name="style_papercut",
        levels=[0.80, 0.90, 1.00, 1.15]  # level 1..4
    ),

    # Example: another style
    # "clay": StyleDef(
    #     id="clay",
    #     title="Clay",
    #     lora_path="/models/loras/claymation_xl.safetensors",
    #     adapter_name="style_clay",
    #     levels=[0.30, 0.55, 0.80, 1.05],
    # ),
}