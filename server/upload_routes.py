"""
upload_routes.py — Temporary file upload for WS clients.

POST /v1/upload  →  multipart file  →  {"fileRef": "uuid"}

In-memory store with 5-minute TTL, background cleanup task.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Tuple

from fastapi import APIRouter, File, UploadFile, HTTPException

logger = logging.getLogger(__name__)

upload_router = APIRouter()

# fileRef → (bytes, created_at)
UPLOADS: Dict[str, Tuple[bytes, float]] = {}
TTL_S = 300  # 5 minutes


@upload_router.post("/v1/upload")
async def upload_temp_file(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    ref = uuid.uuid4().hex
    UPLOADS[ref] = (data, time.time())
    logger.info("Upload stored: %s (%d bytes)", ref, len(data))
    return {"fileRef": ref}


def resolve_file_ref(ref: str) -> bytes:
    """Resolve a fileRef to bytes. Raises KeyError if expired/missing."""
    entry = UPLOADS.get(ref)
    if entry is None:
        raise KeyError(f"fileRef {ref!r} not found or expired")
    return entry[0]


async def cleanup_uploads_loop():
    """Background task that purges expired uploads every 30s."""
    while True:
        await asyncio.sleep(30)
        now = time.time()
        expired = [k for k, (_, ts) in UPLOADS.items() if now - ts > TTL_S]
        for k in expired:
            UPLOADS.pop(k, None)
        if expired:
            logger.debug("Cleaned %d expired uploads", len(expired))
