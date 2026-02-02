"""
Dream System Initialization Library

Provides a reusable function to initialize the Yume dream system
with Redis, CLIP scoring, and dream workers.
"""
import os
import sys
from typing import Optional, Dict, Any
import redis.asyncio as redis

YUME_ENABLED = os.environ.get("YUME_ENABLED", "false").lower().strip()
YUME_CLIP_MODEL = os.environ.get("YUME_CLIP_MODEL", "openai/clip-vit-base-patch32").lower().strip()

from transformers import logging
logging.disable_progress_bar()


async def initialize_dream_system(
    app_state: Any,
    service: Any,  # FIXED: removed stray 'f'
    backend: str = "cpu",
    redis_url: Optional[str] = None,
    clip_model_name: str = YUME_CLIP_MODEL,
    dream_config: Optional[Dict[str, Any]] = None,
    yume_available: bool = True,
    worker_pool: Any = None
) -> bool:
    """
    Initialize the Yume dream system with Redis, CLIP, and dream workers.
    
    Args:
        app_state: FastAPI app.state object to attach dream_worker to
        service: Service object containing workers list
        backend: Backend type ("cuda", "cpu", "rknn", etc.)
        redis_url: Redis connection URL (defaults to REDIS_URL env or localhost)
        clip_model_name: HuggingFace CLIP model identifier
        dream_config: Configuration dict for DreamWorker (defaults to {'top_k': 100})
        yume_available: Whether Yume system is available/enabled
    
    Returns:
        bool: True if initialization succeeded, False otherwise
    
    Example:
        ```python
        from yume.dream_init import initialize_dream_system
        
        @app.on_event("startup")
        async def startup():
            success = await initialize_dream_system(
                app_state=app.state,
                service=app.state.service,
                backend="cuda",
                dream_config={'top_k': 200, 'explore_temperature': 0.8}
            )
            if success: 
                print("Dream system ready!")
        ```
    """

    # Check if Yume is enabled via environment variable
    if YUME_ENABLED != "true":
        print("🌚Yume disabled via YUME_ENABLED env var")
        return False

    # Yume dream system
    try:
        import redis.asyncio as redis
        YUME_AVAILABLE = True
    except ImportError as e:
        YUME_AVAILABLE = False
        print(f"Yume not available (redis import failed): {e}")
        return False

    # Set default config
    if dream_config is None:
        dream_config = {'top_k': 100}
    
    # Initialize to None
    app_state.dream_worker = None
    
    if not yume_available or not YUME_AVAILABLE:
        print("⚠️  Yume system not available")
        return False
    
    try:
        # ================================================================
        # Step 1: Connect to Redis
        # ================================================================
        if redis_url is None:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        
        redis_client = None
        try:
            redis_client = await redis.from_url(redis_url, decode_responses=False)
            await redis_client.ping()
            print("✅ Redis connected for dreams")
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
            return False
        
        # ================================================================
        # Step 2: Load CLIP scorer (optional)
        # ================================================================
        clip_scorer = None
        try:
            from transformers import CLIPModel, CLIPProcessor  # FIXED: Added CLIPProcessor
            from yume.scoring import CLIPScorer
            
            print(f"Loading CLIP model: {clip_model_name}...")
            clip_model = CLIPModel.from_pretrained(clip_model_name)
            clip_processor = CLIPProcessor.from_pretrained(clip_model_name)  # FIXED: Load processor
            
            device = "cuda" if backend == "cuda" else "cpu"
            
            # FIXED: Pass both model and processor
            clip_scorer = CLIPScorer(
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device
            )
            print(f"✅ CLIP loaded on {device}")
            
        except Exception as e:
            print(f"⚠️  CLIP not available, using heuristic scoring: {e}")
            import traceback
            traceback.print_exc()
        
        # ================================================================
        # Step 3: Get LCM worker and initialize dream system
        # ================================================================
        lcm_worker = None
        if worker_pool is not None and getattr(worker_pool, '_worker', None) is not None:
            lcm_worker = worker_pool._worker
        elif service is not None and getattr(service, 'workers', None):
            lcm_worker = service.workers[0]

        if lcm_worker is None:
            print("⚠️  No workers available")
            return False
        
        # Import DreamWorker
        from yume.dream_worker import DreamWorker
        
        # Create dream worker
        dream_worker = DreamWorker(
            model=lcm_worker,
            redis_client=redis_client,
            clip_scorer=clip_scorer,
            config=dream_config
        )
        
        # FIXED: Make init_dream_worker optional
        try:
            from yume.dream_worker import init_dream_worker
            init_dream_worker(dream_worker)
            print("✅ Global dream worker initialized")
        except ImportError:
            print("ℹ️  init_dream_worker not found (optional)")
            pass
        
        app_state.dream_worker = dream_worker
        
        print("✅ Dream system initialized successfully")
        return True
        
    except Exception as e:
        print(f"⚠️  Dream setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def shutdown_dream_system(app_state: Any) -> None:
    """
    Gracefully shutdown the dream system.
    
    Args:
        app_state: FastAPI app.state object containing dream_worker
    
    Example:
        ```python
        @app.on_event("shutdown")
        async def shutdown():
            await shutdown_dream_system(app.state)
        ```
    """
    if hasattr(app_state, 'dream_worker') and app_state.dream_worker:
        try:
            # Close Redis connection
            if app_state.dream_worker.redis_client:
                await app_state.dream_worker.redis_client.close()
                print("✅ Dream system Redis connection closed")
        except Exception as e:
            print(f"⚠️  Error during dream system shutdown: {e}")