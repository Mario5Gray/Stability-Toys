"""
yume/dream_endpoints.py

FastAPI endpoints for dream system.
Add these to your lcm_server.py or lcm_sr_server.py
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

# Initialize router
dream_router = APIRouter(prefix="/dreams", tags=["dreams"])

# Global dream worker instance (set via init_dream_worker)
_dream_worker: Optional['DreamWorker'] = None


def init_dream_worker(worker):
    """Initialize the global dream worker instance."""
    global _dream_worker
    _dream_worker = worker
    return worker


def get_dream_worker():
    """Get the global dream worker instance."""
    if _dream_worker is None:
        raise HTTPException(500, "Dream worker not initialized. Call init_dream_worker() first.")
    return _dream_worker


class DreamStartRequest(BaseModel):
    """Request to start dreaming."""
    prompt: str = Field(..., description="Base prompt to explore")
    duration_hours: float = Field(1.0, ge=0.1, le=24, description="How long to dream")
    temperature: float = Field(0.5, ge=0.0, le=1.0, description="Exploration randomness")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Min score to keep")
    render_interval: int = Field(100, ge=1, description="Render every N high-scoring dreams")
    top_k: int = Field(100, ge=10, le=1000, description="Keep top K candidates")
    


class DreamStatus(BaseModel):
    """Dream session status."""
    is_dreaming: bool
    dream_count: int
    dreams_per_second: float
    candidates: int
    top_k: int
    elapsed_seconds: float


class DreamCandidateResponse(BaseModel):
    """A dream candidate result."""
    seed: int
    prompt: str
    score: float
    timestamp: float
    rendered: bool
    image_data: Optional[str] = None


@dream_router.post("/start")
async def start_dream_session(request: DreamStartRequest):
    """
    Start a background dream session.
    
    The worker will:
    1. Generate latents at low resolution (fast)
    2. Score with CLIP (similarity to prompt)
    3. Keep top-K highest scoring candidates
    4. Periodically render top candidates to full PNG
    5. Store in Redis for later retrieval
    
    Returns immediately - session runs in background.
    """
    worker = get_dream_worker()
    
    result = await worker.start_dreaming(
        base_prompt=request.prompt,
        duration_hours=request.duration_hours,
        temperature=request.temperature,
        similarity_threshold=request.similarity_threshold,
        render_interval=request.render_interval,
    )
    
    if "error" in result:
        raise HTTPException(400, result["error"])
    
    return result


@dream_router.post("/stop")
async def stop_dream_session():
    """Stop the current dream session."""
    worker = get_dream_worker()
    result = worker.stop_dreaming()
    return result


@dream_router.get("/status", response_model=DreamStatus)
async def get_dream_status():
    """Get current dream session status."""
    worker = get_dream_worker()
    status = worker.get_status()
    return status


@dream_router.get("/top", response_model=List[DreamCandidateResponse])
async def get_top_dreams(
    limit: int = 50,
    min_score: float = 0.0,
    rendered_only: bool = False,
):
    """
    Get top N dreams by score.
    
    Args:
        limit: Maximum number of results
        min_score: Minimum score threshold
        rendered_only: Only return fully rendered images
    
    Returns:
        List of dream candidates sorted by score (descending)
    """
    worker = get_dream_worker()
    results = await worker.get_top_dreams(limit, min_score)
    
    if rendered_only:
        results = [r for r in results if r['rendered']]
    
    return results


@dream_router.get("/recent", response_model=List[DreamCandidateResponse])
async def get_recent_dreams(limit: int = 20):
    """
    Get most recent dreams (by timestamp).
    Useful for watching the dream session in real-time.
    """
    worker = get_dream_worker()
    results = await worker.get_top_dreams(limit=1000, min_score=0.0)
    
    # Sort by timestamp desc
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return results[:limit]


@dream_router.get("/stats")
async def get_dream_stats():
    """
    Get aggregate statistics about dream sessions.
    """
    worker = get_dream_worker()
    status = worker.get_status()
    
    # Calculate additional stats
    if status['elapsed_seconds'] > 0:
        dreams_per_hour = (status['dream_count'] / status['elapsed_seconds']) * 3600
    else:
        dreams_per_hour = 0
    
    return {
        **status,
        "dreams_per_hour": dreams_per_hour,
    }


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================
"""
# In your lcm_sr_server.py:

from yume import DreamWorker, dream_router, init_dream_worker
from yume.scoring import CLIPScorer, CompositeScorer
import redis.asyncio as redis

# Initialize Redis
redis_client = await redis.from_url("redis://localhost:6379")

# Initialize CLIP scorer
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_scorer = CLIPScorer(clip_model)

# Initialize dream worker
dream_worker = DreamWorker(
    model=your_lcm_pipeline,
    scorer=clip_scorer,
    redis_client=redis_client,
    config={'top_k': 100}
)

# Register with endpoints
init_dream_worker(dream_worker)

# Add router to app
app.include_router(dream_router)
"""