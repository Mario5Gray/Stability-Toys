"""
upload_routes.py — File upload for WS/CLI clients, routed by type.

POST /v1/upload  →  multipart file [+ type]  →  {fileRef, bucket, width?, height?}

Backed by the module-level AssetStore (server/asset_store.py). The optional
`type` field routes the file to a store bucket: canny/depth/pose control maps
land in the durable "control_map" bucket, image/ref in "ref_image", and any
other or missing type in the ephemeral "upload" bucket (5-minute TTL enforced
by cleanup_uploads_loop). Routed buckets are validated as decodable images.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from server.asset_store import get_store, image_metadata

logger = logging.getLogger(__name__)

upload_router = APIRouter()

# type label -> store bucket. Deliberately NOT derived from the ControlNet
# registry: upload must not depend on registry load.
_TYPE_TO_BUCKET = {
    "canny": "control_map",
    "depth": "control_map",
    "pose": "control_map",
    "image": "ref_image",
    "ref": "ref_image",
}
_VALIDATED_BUCKETS = {"control_map", "ref_image"}


class UploadResponse(BaseModel):
    fileRef: str
    bucket: str
    width: Optional[int] = None
    height: Optional[int] = None


@upload_router.post("/v1/upload", response_model=UploadResponse, response_model_exclude_none=True)
async def upload_temp_file(
    file: UploadFile = File(...),
    type: Optional[str] = Form(default=None),
) -> UploadResponse:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    bucket = _TYPE_TO_BUCKET.get((type or "").strip(), "upload")

    metadata = None
    width = height = None
    if bucket in _VALIDATED_BUCKETS:
        try:
            meta = image_metadata(data)
        except ValueError:
            raise HTTPException(400, f"{bucket} upload must be a decodable image")
        metadata = meta
        width, height = meta["width"], meta["height"]

    try:
        ref = get_store().write(bucket, data, metadata)
    except ValueError as exc:  # e.g. exceeds bucket byte budget
        raise HTTPException(400, str(exc))

    logger.info("Upload stored: %s -> bucket=%s (%d bytes)", ref, bucket, len(data))
    return UploadResponse(fileRef=ref, bucket=bucket, width=width, height=height)


def resolve_file_ref(ref: str) -> bytes:
    """Resolve a fileRef to bytes. Raises KeyError if expired/missing."""
    return get_store().resolve(ref).data


async def cleanup_uploads_loop():
    """Background task that purges expired upload entries every 30s."""
    while True:
        await asyncio.sleep(30)
        expired = get_store().cleanup_expired()
        if expired:
            logger.debug("Cleaned %d expired uploads", len(expired))
