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
        clip_model,
        redis_client,
        config: dict
    ):
        self.model = model
        self.clip_model = clip_model
        self.redis = redis_client
        self.config = config
        
        # Dream state
        self.is_dreaming = False
        self.dream_count = 0
        self.start_time = None
        
        # Candidate tracking
        self.top_k = config.get('top_k', 100)
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
    ) -> torch.Tensor:
        """
        Generate latent representation WITHOUT decoding to image.
        This is 10-100x faster than full generation.
        """
        generator = torch.manual_seed(seed)
        
        # Use your model's encode-only path
        # For LCM: run scheduler steps but don't decode
        with torch.no_grad():
            # Encode prompt
            prompt_embeds = self.model.encode_prompt(prompt)
            
            # Run diffusion in latent space
            latent = self.model.generate_latent(
                prompt_embeds=prompt_embeds,
                height=size[0],
                width=size[1],
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )
        
        return latent
    
    async def _score_candidate(self, candidate: DreamCandidate) -> float:
        """
        Score candidate using CLIP or aesthetic predictor.
        Works on latent or decoded low-res image.
        """
        # Decode latent to low-res image (still fast)
        with torch.no_grad():
            # Get latent from somewhere (you'll need to store it in candidate)
            # For now, assume we decode to 64x64
            image = self.model.decode_latent_fast(candidate.latent_hash)
        
        # Score with CLIP
        score = self._clip_score(image, candidate.prompt)
        
        # Optional: aesthetic predictor
        aesthetic_score = self._aesthetic_score(image)
        
        # Combined score
        final_score = 0.7 * score + 0.3 * aesthetic_score
        
        return final_score
    
    def _clip_score(self, image: Image.Image, prompt: str) -> float:
        """Score image-text similarity with CLIP."""
        # Encode image
        image_features = self.clip_model.encode_image(image)
        
        # Encode text
        text_features = self.clip_model.encode_text(prompt)
        
        # Cosine similarity
        similarity = torch.cosine_similarity(
            image_features, 
            text_features, 
            dim=-1
        ).item()
        
        # Normalize to 0-1
        return (similarity + 1) / 2
    
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
        """Render candidate to full PNG (512x512 or higher)."""
        # Full generation with stored seed
        generator = torch.manual_seed(candidate.seed)
        
        with torch.no_grad():
            image = self.model.generate(
                prompt=candidate.prompt,
                height=512,
                width=512,
                num_inference_steps=4,
                guidance_scale=1.0,
                generator=generator,
            ).images[0]
        
        # Convert to base64 PNG
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        candidate.rendered = True
        candidate.image_data = image_data
        candidate.metadata['rendered_size'] = '512x512'
        
        # Cache
        self.rendered_cache[candidate.seed] = image_data
    
    async def _store_candidate(self, candidate: DreamCandidate):
        """Store candidate in Redis."""
        key = f"dream:{int(self.start_time)}:{candidate.seed}"
        
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
            f"dream_scores:{int(self.start_time)}",
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
        latent_bytes = latent.cpu().numpy().tobytes()
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
        session_key = f"dream_scores:{int(self.start_time)}"
        
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