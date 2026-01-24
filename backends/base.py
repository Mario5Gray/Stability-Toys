# backends/base.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional

@dataclass
class Job:
    req: GenerateRequest
    fut: Future
    submitted_at: float


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

@dataclass(frozen=True)
class ModelPaths:
    root: str

    @property
    def scheduler_config(self) -> str:
        return os.path.join(self.root, "scheduler", "scheduler_config.json")

    @property
    def text_encoder(self) -> str:
        return os.path.join(self.root, "text_encoder")

    @property
    def unet(self) -> str:
        return os.path.join(self.root, "unet")

    @property
    def vae_decoder(self) -> str:
        return os.path.join(self.root, "vae_decoder")
