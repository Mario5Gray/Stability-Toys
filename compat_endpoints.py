# compat_endpoints.py
"""
Compatibility endpoints for external callers:
- AUTOMATIC1111: /sdapi/v1/txt2img
- OpenAI Images: /v1/images/generations

This module is intentionally "thin":
- It maps external schemas -> your internal GenerateRequest
- It calls a supplied runner callable to perform generation
- It returns JSON with base64 images (common for these APIs)
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field


# -----------------------------
# External request schemas
# -----------------------------

class A1111Txt2ImgRequest(BaseModel):
    prompt: str = Field(default="")
    negative_prompt: Optional[str] = Field(default=None)

    steps: int = Field(default=4, ge=1, le=50)
    cfg_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    width: int = Field(default=512, ge=64, le=4096)
    height: int = Field(default=512, ge=64, le=4096)
    seed: int = Field(default=-1)  # A1111 uses -1 for random

    # allow some common A1111 fields without breaking:
    sampler_name: Optional[str] = None
    batch_size: int = Field(default=1, ge=1, le=8)
    n_iter: int = Field(default=1, ge=1, le=8)

    # your extras:
    superres: bool = False
    superres_magnitude: int = Field(default=2, ge=1, le=3)


class OpenAIImagesRequest(BaseModel):
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: str = Field(default="512x512", pattern=r"^\d+x\d+$")

    # non-standard but useful for you:
    seed: Optional[int] = None
    steps: int = Field(default=4, ge=1, le=50)
    cfg: float = Field(default=1.0, ge=0.0, le=20.0)

    superres: bool = False
    superres_magnitude: int = Field(default=2, ge=1, le=3)


# -----------------------------
# Minimal response helpers
# -----------------------------

def _b64_png(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")



def get_rknn_model_paths() -> str:
    """
    Best-effort model name extraction.
    Uses MODEL_ROOT folder name as the canonical model identifier.
    """
    return "RKNN_LCM"


# -----------------------------
# Public class
# -----------------------------

@dataclass
class CompatEndpoints:
    """
    Mounts compatibility endpoints onto an existing FastAPI app.

    You provide a single callable that performs "generate and return final image bytes":
        run_generate(req_dict) -> (image_bytes, seed_used, meta_headers)

    Where:
      - req_dict is the dict you would use to build your internal GenerateRequest
      - meta_headers can include SR metadata if available (optional)
    """

    app: FastAPI
    run_generate: Callable[[Dict[str, Any]], Tuple[bytes, int, Dict[str, str]]]
    router_prefix: str = ""

    def mount(self) -> None:
        r = APIRouter(prefix=self.router_prefix)

        @r.get("/sdapi/v1/sd-models")
        def sdapi_v1_sd_models():
            """
            Automatic1111-compatible model listing.
            Returns the currently loaded LCM model.
            """
            model_name = get_rknn_model_paths()

            return [
                {
                    "title": model_name,
                    "model_name": model_name,
                    "filename": model_name,
                    "hash": None,
                    "sha256": None,
                    "config": None,
                }
            ]
            
        @r.get("/sdapi/v1/options")
        def sdapi_v1_options():
            return {
                "sd_model_checkpoint": get_rknn_model_paths(),
                "sd_checkpoint_hash": "",
                "sd_model_hash": "",                
            }

        # Optional probes (some clients ask these)
        @r.get("/sdapi/v1/samplers")
        def sdapi_samplers():
            # LCM scheduler only; return a minimal "sampler list"
            return [{"name": "LCM", "aliases": ["lcm"], "options": {}}]
            
        @r.post("/sdapi/v1/txt2img")
        def sdapi_txt2img(req: A1111Txt2ImgRequest):
            # A1111 semantics:
            # - seed = -1 means random
            seed_in = None if req.seed is None or int(req.seed) < 0 else int(req.seed)

            gen_req = {
                "prompt": req.prompt or "",
                "size": f"{int(req.width)}x{int(req.height)}",
                "num_inference_steps": int(req.steps),
                "guidance_scale": float(req.cfg_scale),
                "seed": seed_in,
                "superres": bool(req.superres),
                "superres_format": "png",
                "superres_quality": 92,
                "superres_magnitude": int(req.superres_magnitude),
            }

            img_bytes, seed_used, meta = self.run_generate(gen_req)

            # A1111 returns list of base64 images (no data: prefix)
            b64 = _b64_png(img_bytes)

            # Keep it simple but compatible
            info_obj = {
                "seed": seed_used,
                "steps": req.steps,
                "cfg_scale": req.cfg_scale,
                "width": req.width,
                "height": req.height,
                "superres": bool(req.superres),
                "superres_magnitude": int(req.superres_magnitude),
            }
            # include any SR headers you set, when present
            if meta:
                info_obj["headers"] = dict(meta)

            return {
                "images": [b64],
                "parameters": {
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt,
                    "steps": req.steps,
                    "cfg_scale": req.cfg_scale,
                    "width": req.width,
                    "height": req.height,
                    "seed": seed_used,
                },
                "info": json.dumps(info_obj),
            }

        @r.post("/v1/images/generations")
        def openai_images(req: OpenAIImagesRequest):
            gen_req = {
                "prompt": req.prompt,
                "size": req.size,
                "num_inference_steps": int(req.steps),
                "guidance_scale": float(req.cfg),
                "seed": req.seed,
                "superres": bool(req.superres),
                "superres_format": "png",
                "superres_quality": 92,
                "superres_magnitude": int(req.superres_magnitude),
            }

            img_bytes, seed_used, meta = self.run_generate(gen_req)
            b64 = _b64_png(img_bytes)

            return {
                "created": int(time.time()),
                "data": [{"b64_json": b64}],
                "meta": {
                    "seed": seed_used,
                    "size": req.size,
                    "steps": req.steps,
                    "cfg": req.cfg,
                    "headers": dict(meta or {}),
                },
            }

        self.app.include_router(r)
