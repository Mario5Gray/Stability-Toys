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

    def run_job_with_latents(self, spec: GenSpec) -> Tuple[bytes, int, bytes]:
        """
        Return (png_bytes, seed_used, latents_bytes) where latents_bytes is a raw tensor
        of shape [1,4,8,8] (NCHW) serialized as little-endian float16 bytes.
        """