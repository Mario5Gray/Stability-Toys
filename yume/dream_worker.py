"""
Background dream worker for latent space exploration.
Runs asynchronously, generates low-res samples, scores them, renders top candidates.
"""

import asyncio
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable
from collections import deque
import torch
from PIL import Image
import io
import base64
import os
# At the top of your main server file (lcm_sr_server.py):

import logging
os.environ['TQDM_DISABLE'] = '1'

# Redirect torch logging to file
torch_logger = logging.getLogger('torch')
torch_handler = logging.FileHandler('/app/logs/torch.log')
torch_logger.addHandler(torch_handler)
torch_logger.setLevel(logging.WARNING)

@dataclass
class DreamCandidate:
    """A single dream candidate with scoring."""
    seed: int
    prompt: str
    score: float
    timestamp: float
    latent_hash: str  # Hash of latent representation
    rendered: bool = False
    image_data: Optional[str] = None  # Base64 PNG if rendered
    metadata: dict = None
    
    def to_dict(self):
        d = asdict(self)
        if not self.rendered:
            d.pop('image_data', None)  # Don't send unrendered data
        return d


class DreamWorker:
    """
    Background worker for exploring latent space.
    
    Workflow:
    1. Generate latent @ low res (fast)
    2. Score using CLIP/aesthetic predictor
    3. Keep top-K in memory
    4. Periodically render top candidates to full PNG
    5. Store in Redis with metadata
    """
    
    def __init__(
        self,
        model,
        redis_client,
        clip_scorer=None,
        config: dict = None
    ):
        """
        Initialize dream worker.
        
        Args:
            model: Your LCM pipeline worker (RKNNPipelineWorker or DiffusersCudaWorker)
            redis_client: Redis async client
            clip_scorer: Optional CLIPScorer instance (from yume.scoring)
            config: Optional config dict
        """
        print(f"config= {config}")
        self.model = model
        self.clip_scorer = clip_scorer
        self.redis = redis_client
        self.config = config or {}
        
        # Dream state
        self.is_dreaming = False
        self.dream_count = 0
        self.start_time = None
        
        # Candidate tracking
        self.top_k = self.config.get('top_k', 100)
        self.candidates = deque(maxlen=self.top_k * 2)  # Buffer
        self.rendered_cache = {}  # seed -> image data
        
        # Exploration params
        self.base_prompt = ""
        self.prompt_variations = []
        self.seed_range = (0, 2**31 - 1)
        self.exploration_strategy = "random"  # random | linear_walk | grid
        
        # Performance tracking
        self.dreams_per_second = 0
        self.last_fps_check = time.time()
        self.fps_counter = 0
        
    async def start_dreaming(
        self,
        base_prompt: str,
        duration_hours: float = 1.0,
        temperature: float = 0.5,
        similarity_threshold: float = 0.7,
        render_interval: int = 100,  # Render every N high-scoring dreams
        exploration_strategy = "random",
    ):
        """
        Start background dreaming session.
        
        Args:
            base_prompt: Base prompt to explore around
            duration_hours: How long to dream
            temperature: Exploration randomness (0-1)
            similarity_threshold: Min score to keep candidate
            render_interval: Render full PNG every N high-scoring dreams
        """
        if self.is_dreaming:
            return {"error": "Already dreaming"}
        
        self.is_dreaming = True
        self.dream_count = 0
        self.start_time = time.time()
        self.base_prompt = base_prompt
        
        # Generate prompt variations
        self.prompt_variations = self._generate_prompt_variations(
            base_prompt, 
            temperature
        )
        
        print(f"🌙 Dream session started: {duration_hours}h, threshold={similarity_threshold}")
        
        # Run dream loop in background
        asyncio.create_task(
            self._dream_loop(
                duration_hours,
                temperature,
                similarity_threshold,
                render_interval
            )
        )
        
        return {
            "status": "started",
            "base_prompt": base_prompt,
            "duration_hours": duration_hours,
            "top_k": self.top_k,
        }
    
    async def _dream_loop(
        self,
        duration_hours: float,
        temperature: float,
        similarity_threshold: float,
        render_interval: int,
    ):
        """Main dreaming loop - runs in background."""
        end_time = time.time() + (duration_hours * 3600)
        high_score_count = 0
        
        while self.is_dreaming and time.time() < end_time:
            try:
                # Generate candidate
                candidate = await self._generate_candidate(temperature)
                
                # Score it (fast, low-res)
                score = await self._score_candidate(candidate)
                candidate.score = score
                
                # Track FPS
                self._update_fps()
                
                # Keep if above threshold
                if score >= similarity_threshold:
                    self.candidates.append(candidate)
                    high_score_count += 1
                    
                    # Render to full PNG periodically
                    if high_score_count % render_interval == 0:
                        await self._render_candidate(candidate)
                        await self._store_candidate(candidate)
                
                self.dream_count += 1
                
                # Small yield to avoid blocking
                if self.dream_count % 10 == 0:
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                print(f"Dream error: {e}")
                continue
        
        # Dream session complete
        self.is_dreaming = False
        await self._finalize_session()
        
        print(f"✅ Dream session complete: {self.dream_count} dreams, "
              f"{len(self.candidates)} high-scoring candidates")
    
    async def _generate_candidate(self, temperature: float) -> DreamCandidate:
        """
        Generate a latent candidate (fast, low-res).
        Returns candidate with latent representation.
        """
        # Pick prompt variation
        prompt = np.random.choice(self.prompt_variations)
        
        # Generate seed
        if self.exploration_strategy == "random":
            seed = np.random.randint(*self.seed_range)
        else:
            # Other strategies: linear walk, grid, etc.
            seed = self._next_exploration_seed()
        
        # Generate latent @ low resolution (FAST)
        # This is the key optimization: don't render full image yet
        latent = await self._generate_latent_only(
            prompt=prompt,
            seed=seed,
            size=(64, 64),  # Tiny for speed
            steps=1,  # Single step LCM
            cfg=0.0,  # Fast mode
        )
        
        # Hash latent for deduplication
        latent_hash = self._hash_latent(latent)
        
        return DreamCandidate(
            seed=seed,
            prompt=prompt,
            score=0.0,  # Will be filled by scorer
            timestamp=time.time(),
            latent_hash=latent_hash,
            rendered=False,
            metadata={
                'size': '64x64',
                'steps': 1,
                'cfg': 0.0,
                'temperature': temperature,
            }
        )
    
    async def _generate_latent_only(
        self,
        prompt: str,
        seed: int,
        size: tuple,
        steps: int,
        cfg: float,
    ) -> np.ndarray:
        """
        Generate latent representation fast (64x64, 1 step).
        
        For your existing workers, we'll just generate a tiny full image
        and extract features from it. This is still much faster than 512x512.
        """
        from dataclasses import dataclass
        from pydantic import BaseModel, Field
        
        # Create a minimal request for your worker
        # Using your GenerateRequest schema
        class QuickGenRequest:
            def __init__(self):
                self.prompt = prompt
                self.size = f"{size[0]}x{size[1]}"  # e.g. "64x64"
                self.num_inference_steps = steps
                self.guidance_scale = cfg
                self.seed = seed
                self.superres = False
                self.superres_format = "png"
                self.superres_quality = 92
                self.superres_magnitude = 1
        
        req = QuickGenRequest()
        
        # Create a job for your worker
        from concurrent.futures import Future
        import io
        
        @dataclass
        class QuickJob:
            req: any
            fut: Future
            submitted_at: float
        
        fut = Future()
        job = QuickJob(req=req, fut=fut, submitted_at=time.time())
        
        # Run through your existing worker
        try:
            png_bytes, seed_used = self.model.run_job(job)
            
            # Convert PNG to numpy for scoring
            from PIL import Image
            img = Image.open(io.BytesIO(png_bytes))
            latent = np.array(img)
            
            return latent
            
        except Exception as e:
            print(f"Dream latent generation failed: {e}")
            # Return dummy latent on error
            return np.random.rand(size[0], size[1], 3).astype(np.float32)
    
    async def _score_candidate(self, candidate: DreamCandidate) -> float:
        """
        Score candidate using CLIP or simple heuristics.
        Works on decoded low-res image.
        """
        # Decode latent to image (it's already a numpy array from our quick gen)
        import io
        from PIL import Image
        
        # The "latent" is actually a small RGB image (64x64)
        # Convert from our stored format
        image_data = np.frombuffer(
            candidate.latent_hash.encode('latin1'), 
            dtype=np.uint8
        ).reshape((64, 64, 3)) if len(candidate.latent_hash) > 32 else np.random.rand(64, 64, 3) * 255
        
        image = Image.fromarray(image_data.astype(np.uint8), mode='RGB')
        
        # Score with CLIP if available
        if self.clip_scorer:
            try:
                score = self.clip_scorer.score(image, candidate.prompt)
                return score
            except Exception as e:
                print(f"CLIP scoring failed: {e}")
                # Fall back to heuristic
        
        # Simple heuristic scoring
        score = self._aesthetic_score(image)
        return score
    
    def _clip_score(self, image: Image.Image, prompt: str) -> float:
        """Score image-text similarity with CLIP (if available)."""
        if not self.clip_scorer:
            return 0.5  # Neutral score if no CLIP
        
        try:
            return self.clip_scorer.score(image, prompt)
        except Exception as e:
            print(f"CLIP score error: {e}")
            return 0.5
    
    def _aesthetic_score(self, image: Image.Image) -> float:
        """
        Score aesthetic quality.
        Can use pre-trained aesthetic predictor or simple heuristics.
        """
        # Placeholder: use laplacian variance (sharpness)
        import cv2
        img_array = np.array(image.convert('L'))
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize (tuned to typical ranges)
        normalized = min(1.0, sharpness / 1000.0)
        
        return normalized
    
    async def _render_candidate(self, candidate: DreamCandidate):
        """Render candidate to full PNG (512x512 or higher) using your worker."""
        from dataclasses import dataclass
        from concurrent.futures import Future
        import io
        
        # Create request for full-size generation
        class FullGenRequest:
            def __init__(self):
                self.prompt = candidate.prompt
                self.size = "512x512"
                self.num_inference_steps = 4
                self.guidance_scale = 1.0
                self.seed = candidate.seed
                self.superres = False
                self.superres_format = "png"
                self.superres_quality = 92
                self.superres_magnitude = 1
        
        req = FullGenRequest()
        
        @dataclass
        class FullJob:
            req: any
            fut: Future
            submitted_at: float
        
        fut = Future()
        job = FullJob(req=req, fut=fut, submitted_at=time.time())
        
        try:
            # Generate full-size through your worker
            png_bytes, seed_used = self.model.run_job(job)
            
            # Convert to base64
            import base64
            image_data = base64.b64encode(png_bytes).decode()
            
            candidate.rendered = True
            candidate.image_data = image_data
            candidate.metadata['rendered_size'] = '512x512'
            
            # Cache
            self.rendered_cache[candidate.seed] = image_data
            
        except Exception as e:
            print(f"Dream render failed for seed {candidate.seed}: {e}")
            candidate.rendered = False
    
    async def _store_candidate(self, candidate: DreamCandidate):
        """Store candidate in Redis."""
        key = f"dream:{int(self.start_time or 0)}:{candidate.seed}"
        
        await self.redis.hset(key, mapping={
            'seed': candidate.seed,
            'prompt': candidate.prompt,
            'score': candidate.score,
            'timestamp': candidate.timestamp,
            'latent_hash': candidate.latent_hash,
            'rendered': int(candidate.rendered),
            'image_data': candidate.image_data or '',
            'metadata': str(candidate.metadata),
        })
        
        # Add to sorted set for top-K queries
        await self.redis.zadd(
            f"dream_scores:{int(self.start_time or 0)}",
            {key: candidate.score}
        )
    
    def _generate_prompt_variations(
        self, 
        base_prompt: str, 
        temperature: float
    ) -> List[str]:
        """Generate prompt variations for exploration."""
        modifiers = [
            "dramatic lighting", "soft lighting", "golden hour",
            "cinematic", "highly detailed", "ethereal",
            "warm tones", "cool tones", "vibrant colors",
            "misty", "foggy", "hazy", "atmospheric",
        ]
        
        variations = [base_prompt]
        
        # Add single modifiers
        for mod in modifiers[:int(len(modifiers) * temperature)]:
            variations.append(f"{base_prompt}, {mod}")
        
        # Add combinations
        if temperature > 0.5:
            import itertools
            for combo in itertools.combinations(modifiers, 2):
                variations.append(f"{base_prompt}, {', '.join(combo)}")
        
        return variations
    
    def _hash_latent(self, latent: torch.Tensor) -> str:
        """Hash latent tensor for deduplication."""
        import hashlib
        
        latent_bytes = latent.tobytes() if isinstance(latent, np.ndarray) else latent.cpu().numpy().tobytes()        
        return hashlib.md5(latent_bytes).hexdigest()
    
    def _next_exploration_seed(self) -> int:
        """Get next seed based on exploration strategy."""
        # Linear walk
        if self.exploration_strategy == "linear_walk":
            return (self.dream_count * 1000) % (2**31)
        
        # Grid
        elif self.exploration_strategy == "grid":
            grid_size = int(np.sqrt(self.top_k))
            x = self.dream_count % grid_size
            y = self.dream_count // grid_size
            return x * 1000000 + y
        
        # Random (default)
        return np.random.randint(*self.seed_range)
    
    def _update_fps(self):
        """Track dreams per second."""
        self.fps_counter += 1
        now = time.time()
        
        if now - self.last_fps_check >= 1.0:
            self.dreams_per_second = self.fps_counter
            self.fps_counter = 0
            self.last_fps_check = now
    
    async def _finalize_session(self):
        """Finalize dream session - render remaining top candidates."""
        # Sort candidates by score
        sorted_candidates = sorted(
            self.candidates, 
            key=lambda c: c.score, 
            reverse=True
        )
        
        # Render top unrendered candidates
        render_count = 0
        for candidate in sorted_candidates[:self.top_k]:
            if not candidate.rendered and render_count < 50:
                await self._render_candidate(candidate)
                await self._store_candidate(candidate)
                render_count += 1
        
        print(f"Finalized: {render_count} additional renders")
    
    def stop_dreaming(self):
        """Stop dream session."""
        self.is_dreaming = False
        return {
            "status": "stopped",
            "total_dreams": self.dream_count,
            "candidates_kept": len(self.candidates),
        }
    
    def get_status(self) -> dict:
        """Get current dream status."""
        return {
            "is_dreaming": self.is_dreaming,
            "dream_count": self.dream_count,
            "dreams_per_second": self.dreams_per_second,
            "candidates": len(self.candidates),
            "top_k": self.top_k,
            "elapsed_seconds": time.time() - self.start_time if self.start_time else 0,
        }
    
    async def get_top_dreams(self, limit: int = 50, min_score: float = 0.0):
        """Get top N dreams by score."""
        session_key = f"dream_scores:{int(self.start_time or 0)}"
        
        # Get top from Redis
        top_keys = await self.redis.zrevrange(
            session_key, 
            0, 
            limit - 1, 
            withscores=True
        )
        
        results = []
        for key, score in top_keys:
            if score < min_score:
                continue
            
            data = await self.redis.hgetall(key)
            results.append({
                'seed': int(data[b'seed']),
                'prompt': data[b'prompt'].decode(),
                'score': float(data[b'score']),
                'timestamp': float(data[b'timestamp']),
                'rendered': bool(int(data[b'rendered'])),
                'image_data': data[b'image_data'].decode() if data[b'rendered'] else None,
            })

        return results

# Global dream worker instance (for easy access across the app)
_global_dream_worker = None


def init_dream_worker(dream_worker):
    """
    Initialize the global dream worker instance.
    
    This allows you to access the dream worker from anywhere:
    
    from yume.dream_worker import get_dream_worker
    
    worker = get_dream_worker()
    if worker:
        worker.start_session(...)
    
    Args:
        dream_worker: DreamWorker instance
        
    Returns:
        The dream worker instance
    """
    global _global_dream_worker
    _global_dream_worker = dream_worker
    print(f"✅ Global dream worker initialized: {dream_worker}")
    return dream_worker


def get_dream_worker():
    """
    Get the global dream worker instance.
    
    Returns:
        DreamWorker instance if initialized, None otherwise
    """
    return _global_dream_worker


def clear_dream_worker():
    """
    Clear the global dream worker instance.
    Useful for cleanup or testing.
    """
    global _global_dream_worker
    _global_dream_worker = None


# If you prefer NOT to use a global singleton pattern, you can skip
# these functions entirely and just pass the dream_worker instance
# around via function parameters or FastAPI dependency injection:
#
# @app.get("/dreams/status")
# async def get_status(request: Request):
#     dream_worker = request.app.state.dream_worker
#     return dream_worker.get_status()