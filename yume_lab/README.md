# 🌙 Yume (夢) - Background Dream System

** Directive (02/01/2026)
** By no means is this project deprecated I think it can still 
** be useful in exploration of the prospects of LCM
** However, as a front-end it has litte use beyond novel research
** which is why we have this sub-project that utilizies the 
** lcm-sd server API's! It can change and we wont have a problem 
** with getting API's off sync unless in active development

> Server-side latent space exploration with CLIP scoring

---

## 📁 Structure

```
yume/
├── __init__.py           # Package exports
├── dream_worker.py       # Core dream worker (latent generation + scoring)
├── dream_endpoints.py    # FastAPI REST endpoints
├── scoring.py            # CLIP & aesthetic scoring
├── strategies.py         # Exploration strategies (random, evolutionary, etc.)
└── README.md            # This file
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install torch torchvision transformers redis pillow opencv-python
```

### Step 2: Initialize in Your Server

**In `lcm_sr_server.py`:**

```python
from yume import DreamWorker, dream_router, init_dream_worker
from yume.scoring import CLIPScorer, AestheticScorer, CompositeScorer
import redis.asyncio as redis

# Initialize Redis
redis_client = await redis.from_url("redis://localhost:6379")

# Initialize CLIP
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_scorer = CLIPScorer(clip_model, device="cuda")

# Optional: Add aesthetic scorer
aesthetic_scorer = AestheticScorer()
composite_scorer = CompositeScorer(clip_scorer, aesthetic_scorer)

# Initialize dream worker
dream_worker = DreamWorker(
    model=your_lcm_pipeline,  # Your existing LCM model
    scorer=composite_scorer,   # Or just clip_scorer
    redis_client=redis_client,
    config={
        'top_k': 100,
        'device': 'cuda',
    }
)

# Register with endpoints
init_dream_worker(dream_worker)

# Add routes
app.include_router(dream_router)
```

### Step 3: Start Dreaming

```bash
# Start 2-hour dream session
curl -X POST http://localhost:8000/dreams/start \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cinematic photograph of a futuristic city",
    "duration_hours": 2.0,
    "temperature": 0.5,
    "similarity_threshold": 0.7,
    "render_interval": 100,
    "strategy": "random"
  }'

# Check status
curl http://localhost:8000/dreams/status

# Get top 50 dreams
curl http://localhost:8000/dreams/top?limit=50&min_score=0.8

# Stop session
curl -X POST http://localhost:8000/dreams/stop
```

---

## 🎯 How It Works

### Phase 1: Fast Latent Generation
```python
# Generate @ 64x64, 1 step (FAST)
latent = model.generate_latent(
    prompt=variation,
    size=(64, 64),
    steps=1,
    cfg=0.0,
)
# Speed: 500-2000/sec on RTX 4090
```

### Phase 2: CLIP Scoring
```python
# Score in milliseconds
score = clip_scorer.score(
    image=decode_latent(latent),
    text=prompt
)
# Keep if score > threshold
```

### Phase 3: Selective Rendering
```python
# Only render high-scoring candidates
if high_score_count % 100 == 0:
    full_image = model.generate(
        prompt=prompt,
        seed=seed,
        size=(512, 512),
        steps=4,
    )
    redis.set(f"dream:{session}:{seed}", image)
```

---

## 🧪 Exploration Strategies

### Random (Default)
```python
strategy = "random"
# Uniform sampling across latent space
# Good for: Broad coverage
```

### Linear Walk
```python
strategy = "linear_walk"
# Sequential seed traversal
# Good for: Smooth transitions, animations
```

### Evolutionary
```python
strategy = "evolutionary"
# Genetic algorithm with mutation/crossover
# Good for: Finding optimal regions
```

### Temperature Schedule
```python
strategy = "temperature"
# Simulated annealing (starts wild, focuses)
# Good for: Exploration → refinement
```

### Cluster
```python
strategy = "cluster"
# Maintains multiple good regions
# Good for: Finding diverse solutions
```

---

## 📊 Scoring

### CLIP Scorer
```python
from yume.scoring import CLIPScorer

scorer = CLIPScorer(clip_model, device="cuda")
score = scorer.score(image, prompt)  # Returns 0-1
```

**Features:**
- Text embedding caching (faster)
- Batch scoring support
- Cosine similarity normalized to [0, 1]

### Aesthetic Scorer
```python
from yume.scoring import AestheticScorer

scorer = AestheticScorer()
score = scorer.score(image)  # Returns 0-1
```

**Heuristics:**
- Sharpness (Laplacian variance)
- Contrast (std deviation)
- Color variety (unique colors)

### Composite Scorer
```python
from yume.scoring import CompositeScorer

scorer = CompositeScorer(
    clip_scorer,
    aesthetic_scorer,
    clip_weight=0.7,
    aesthetic_weight=0.3,
)

scores = scorer.score(image, prompt)
# Returns: {'clip': 0.85, 'aesthetic': 0.72, 'final': 0.81}
```

---

## ⚙️ Configuration

### DreamWorker Config
```python
config = {
    'top_k': 100,              # Keep top N candidates
    'device': 'cuda',          # Device for scoring
    'latent_size': (64, 64),   # Low-res for speed
    'latent_steps': 1,         # Single LCM step
    'render_size': (512, 512), # Full res for final
    'render_steps': 4,         # Quality for finals
}
```

### Session Parameters
```python
{
    "prompt": "base prompt",
    "duration_hours": 2.0,        # How long to run
    "temperature": 0.5,           # 0=subtle, 1=wild
    "similarity_threshold": 0.7,  # Min score to keep
    "render_interval": 100,       # Render every N high-scoring
    "strategy": "random",         # Exploration method
    "top_k": 100,                 # Max candidates
}
```

---

## 📈 Performance

### Expected Speeds

| Hardware | Dreams/sec | 2-hour session |
|----------|------------|----------------|
| RTX 4090 | 1000-2000  | 7.2M - 14.4M   |
| RTX 3090 | 500-1000   | 3.6M - 7.2M    |
| RK3588   | 50-200     | 360K - 1.44M   |

### Optimization Tips

**For max speed:**
```python
latent_size = (32, 32)  # Even smaller
latent_steps = 1
cfg = 0.0
batch_size = 8  # If memory allows
```

**For max quality:**
```python
latent_size = (128, 128)
latent_steps = 2
cfg = 0.5
similarity_threshold = 0.85
render_interval = 50
```

---

## 🐛 Troubleshooting

### Issue: Low dreams/sec

**Check:**
```bash
# GPU utilization
nvidia-smi

# Reduce latent size
latent_size = (32, 32)

# Single step only
latent_steps = 1
```

### Issue: All scores similar

**Fix:**
```python
# Increase temperature
temperature = 0.8

# Lower threshold
similarity_threshold = 0.6

# Try different strategy
strategy = "evolutionary"
```

### Issue: Redis OOM

**Fix:**
```python
# Limit candidates
top_k = 50

# Add TTL
await redis.expire(key, 86400)  # 24h

# Clear old sessions
await redis.delete(f"dream_scores:{old_session}")
```

---

## 🔌 API Reference

### POST /dreams/start
Start dream session
```json
{
  "prompt": "string",
  "duration_hours": 2.0,
  "temperature": 0.5,
  "similarity_threshold": 0.7,
  "render_interval": 100,
  "strategy": "random"
}
```

### GET /dreams/status
Get current status
```json
{
  "is_dreaming": true,
  "dream_count": 145832,
  "dreams_per_second": 807.3,
  "candidates": 94,
  "elapsed_seconds": 180.7
}
```

### GET /dreams/top?limit=50&min_score=0.8
Get top dreams
```json
[
  {
    "seed": 42857391,
    "prompt": "...",
    "score": 0.923,
    "image_data": "base64...",
    "rendered": true
  }
]
```

### POST /dreams/stop
Stop session
```json
{
  "status": "stopped",
  "total_dreams": 145832,
  "candidates_kept": 94
}
```

---

## 🎨 Example Workflows

### Workflow 1: Find Best Variations
```python
# 1. Start 1-hour exploration
POST /dreams/start {
  "prompt": "cyberpunk city",
  "duration_hours": 1.0,
  "temperature": 0.5,
  "strategy": "random"
}

# 2. Check after 1 hour
GET /dreams/top?limit=50&min_score=0.85

# 3. Download favorites
```

### Workflow 2: Evolutionary Refinement
```python
# 1. Start with evolutionary strategy
POST /dreams/start {
  "prompt": "portrait",
  "strategy": "evolutionary",
  "duration_hours": 2.0
}

# 2. Population evolves toward best scoring
# 3. Get refined results
GET /dreams/top?limit=20&min_score=0.9
```

### Workflow 3: Animation Frames
```python
# 1. Linear walk for smooth transitions
POST /dreams/start {
  "prompt": "landscape at dawn",
  "strategy": "linear_walk",
  "duration_hours": 0.5
}

# 2. Get sequential frames
GET /dreams/recent?limit=100

# 3. Create video from frames
```

---

## 📝 TODO / Future Ideas

- [ ] Multi-GPU support (distribute scoring)
- [ ] Better aesthetic predictor (LAION model)
- [ ] Latent interpolation (smooth walks)
- [ ] User feedback integration (learn preferences)
- [ ] Dream visualization (t-SNE of latent space)
- [ ] Dream evolution replay
- [ ] Collaborative dreaming (multi-user)

---

**Happy Dreaming! 🌙✨**
