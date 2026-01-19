# backends/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional

@dataclass
class GenSpec:
    prompt: str
    size: str
    steps: int
    cfg: float
    seed: Optional[int] = None

class PipelineWorker(Protocol):
    worker_id: int
    def run_job(self, spec: GenSpec) -> Tuple[bytes, int]:
        """Return (png_bytes, seed_used)."""