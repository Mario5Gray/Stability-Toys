"""
Telemetry proxy endpoints.
Receives UI telemetry and forwards it to an OTLP/HTTP collector.
"""

import asyncio
import logging
import os
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

from fastapi import APIRouter, Request, Response, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["telemetry"])


def _post_bytes(url: str, body: bytes, content_type: str) -> int:
    req = urlrequest.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=5) as resp:
        return resp.status


@router.post("/telemetry")
async def ingest_telemetry(request: Request):
    """
    Proxy UI telemetry to an OTLP/HTTP collector.
    Set OTEL_PROXY_ENDPOINT (e.g. http://otel-collector:4318/v1/traces).
    """
    endpoint = os.environ.get("OTEL_PROXY_ENDPOINT", "").strip()
    if not endpoint:
        # No-op when collector isn't configured
        return Response(status_code=204)

    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    try:
        status = await asyncio.to_thread(_post_bytes, endpoint, body, content_type)
    except HTTPError as e:
        logger.warning("[telemetry] collector error %s", e)
        raise HTTPException(status_code=502, detail="collector error")
    except URLError as e:
        logger.warning("[telemetry] collector unavailable %s", e)
        raise HTTPException(status_code=503, detail="collector unavailable")

    return Response(status_code=status)
