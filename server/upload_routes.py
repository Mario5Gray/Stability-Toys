"""
upload_routes.py — Temporary file upload for WS clients.

POST /v1/upload  →  multipart file  →  {"fileRef": "uuid"}

Backed by the module-level AssetStore (server/asset_store.py).
Upload entries have kind="upload" and a 5-minute TTL enforced by cleanup_uploads_loop.
"""

import asyncio
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException

from server.asset_store import get_store

logger = logging.getLogger(__name__)

upload_router = APIRouter()

TTL_S = 300  # 5 minutes


@upload_router.post("/v1/upload")
async def upload_temp_file(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    ref = get_store().insert("upload", data)
    logger.info("Upload stored: %s (%d bytes)", ref, len(data))
    return {"fileRef": ref}


def resolve_file_ref(ref: str) -> bytes:
    """Resolve a fileRef to bytes. Raises KeyError if expired/missing."""
    return get_store().resolve(ref).data


async def cleanup_uploads_loop():
    """Background task that purges expired upload entries every 30s."""
    while True:
        await asyncio.sleep(30)
        expired = get_store().cleanup_expired(ttl_s=TTL_S)
        if expired:
            logger.debug("Cleaned %d expired uploads", len(expired))
