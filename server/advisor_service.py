"""
Gallery advisor digest service.
"""

import hashlib
import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from backends.chat_client import ChatCompletionsClient, ChatConfig
from server.mode_config import get_mode_config


class AdvisorDigestRequest(BaseModel):
    gallery_id: str
    evidence: Dict[str, Any]
    mode: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    length_limit: Optional[int] = Field(default=None, ge=1, le=4096)


def build_evidence_fingerprint(evidence: Dict[str, Any]) -> str:
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _build_messages(req: AdvisorDigestRequest, fingerprint: str, effective_limit: Optional[int]):
    evidence_json = json.dumps(req.evidence, sort_keys=True, ensure_ascii=True)
    user_prompt = (
        "Analyze this gallery evidence and produce a concise reusable style digest.\n"
        f"gallery_id: {req.gallery_id}\n"
        f"evidence_fingerprint: {fingerprint}\n"
        f"length_limit: {effective_limit}\n"
        f"evidence_json: {evidence_json}"
    )
    return [{"role": "user", "content": user_prompt}]


async def generate_digest(req: AdvisorDigestRequest) -> Dict[str, Any]:
    mode_config = get_mode_config()
    mode_name = req.mode or mode_config.get_default_mode()
    mode = mode_config.get_mode(mode_name)
    chat_cfg = getattr(mode, "chat", None)
    if chat_cfg is None:
        raise ValueError("advisor digest requires chat configuration on the active mode")

    client = ChatCompletionsClient(
        ChatConfig(
            endpoint=chat_cfg.endpoint,
            model=chat_cfg.model,
            api_key_env=chat_cfg.api_key_env,
            max_tokens=chat_cfg.max_tokens,
            temperature=chat_cfg.temperature,
            system_prompt=chat_cfg.system_prompt,
        )
    )

    effective_limit = req.length_limit
    mode_limit = getattr(mode, "maximum_len", None)
    if mode_limit is not None:
        effective_limit = min(effective_limit if effective_limit is not None else int(mode_limit), int(mode_limit))

    fingerprint = build_evidence_fingerprint(req.evidence)
    messages = _build_messages(req, fingerprint, effective_limit)
    if chat_cfg.system_prompt:
        messages.insert(0, {"role": "system", "content": chat_cfg.system_prompt})

    digest_text = await client.complete(
        messages,
        max_tokens=effective_limit,
        temperature=req.temperature,
    )

    return {
        "gallery_id": req.gallery_id,
        "evidence_fingerprint": fingerprint,
        "digest_text": digest_text,
        "mode": mode_name,
        "length_limit": effective_limit,
    }
