"""
Advisor HTTP routes.
"""

from fastapi import APIRouter, HTTPException

from server.advisor_service import AdvisorDigestRequest, generate_digest

router = APIRouter(prefix="/api/advisors", tags=["advisor"])


@router.post("/digest")
async def digest(req: AdvisorDigestRequest):
    try:
        return await generate_digest(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
