# 🔌 Yume Integration Guide

## Step-by-Step: Add Dream System to Your Server

---

## 📦 Step 1: Create Yume Folder

```bash
cd your-project-root
mkdir yume
cd yume
```

**Create these files** (copy from artifacts):
- `__init__.py`
- `dream_worker.py`
- `dream_endpoints.py`
- `scoring.py`
- `strategies.py`
- `README.md`

---

## 🔧 Step 2: Update lcm_sr_server.py

Add this to your existing `lcm_sr_server.py`:

```python
# ============================================================================
# AT THE TOP (with other imports)
# ============================================================================

from yume import DreamWorker, dream_router, init_dream_worker
from yume.scoring import CLIPScorer, CompositeScorer
import redis.asyncio as redis

# ============================================================================
# AFTER YOUR EXISTING MODEL INITIALIZATION
# ============================================================================

# Your existing code:
# app = FastAPI()
# your_lcm_model = initialize_your_model()
# ...

# Initialize Redis (for dream storage)
try:
    redis_client = await redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=False  # Keep binary for images
    )
    print("✅ Redis connected for dream storage")
except Exception as e:
    print(f"⚠️  Redis not available: {e}")
    print("Dream system will run without persistence")
    redis_client = None

# Initialize CLIP (for dream scoring)
try:
    from transformers import CLIPModel, CLIPProcessor
    
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_scorer = CLIPScorer(clip_model, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("✅ CLIP loaded for dream scoring")
except Exception as e:
    print(f"⚠️  CLIP not available: {e}")
    print("Dream system will use simple heuristics")
    clip_scorer = None

# Initialize Dream Worker
if redis_client and clip_scorer:
    dream_worker = DreamWorker(
        model=your_lcm_model,  # Your existing LCM pipeline
        scorer=clip_scorer,
        redis_client=redis_client,
        config={
            'top_k': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
    )
    
    # Register with endpoints
    init_dream_worker(dream_worker)
    
    # Add dream routes to app
    app.include_router(dream_router)
    
    print("✅ Dream system initialized")
else:
    print("⚠️  Dream system disabled (requires Redis + CLIP)")

# ============================================================================
# REST OF YOUR EXISTING CODE
# ============================================================================

# Your existing endpoints:
# @app.post("/generate")
# @app.post("/superres")
# ...
```

---

## 🐳 Step 3: Update Docker (if using)

### Add to requirements.txt:
```
# Existing deps...
# torch
# diffusers
# ...

# New deps for dream system
redis>=5.0.0
transformers>=4.35.0
opencv-python>=4.8.0
```

### Add to docker-compose.yml:
```yaml
services:
  lcm-server:
    # ... your existing config
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

volumes:
  redis-data:
```

---

## 🎨 Step 4: Frontend Integration

### Add Tabs Component

If you haven't already:
```bash
cd lcm-sr-ui
npm install @radix-ui/react-tabs
```

### Update App.jsx

Replace your current `App.jsx` with the tabbed version (from artifacts):
- Tab 1: Main Chat (existing functionality)
- Tab 2: Dream Gallery (new dream view)

### Create Dream Gallery

Copy `DreamGallery.jsx` to `src/components/dreams/DreamGallery.jsx`

---

## ✅ Step 5: Test It!

### Start Redis
```bash
# Local
redis-server

# Or Docker
docker-compose up redis -d
```

### Start Your Server
```bash
python lcm_sr_server.py
```

**Check logs for:**
```
✅ Redis connected for dream storage
✅ CLIP loaded for dream scoring
✅ Dream system initialized
```

### Start Frontend
```bash
cd lcm-sr-ui
npm run dev
```

### Test Dream Endpoints

```bash
# Check status
curl http://localhost:8000/dreams/status

# Should return:
{
  "is_dreaming": false,
  "dream_count": 0,
  "dreams_per_second": 0.0,
  ...
}
```

### Start a Test Dream

In the frontend:
1. Click "Dream Gallery" tab
2. Enter prompt: "a cinematic photograph"
3. Set duration: 0.1 hours (6 minutes for testing)
4. Click "Start Dreaming"
5. Watch real-time stats update
6. After 6 minutes, view results in gallery

---

## 🔍 Verify Installation

### Check Redis Keys
```bash
redis-cli
> KEYS dream:*
> GET dream:1234567890:42857391
```

### Check Backend Logs
```bash
# Should see:
🌙 Dream session started: 0.1h, threshold=0.7
Dreams/sec: 127.3
Dreams/sec: 243.8
...
✅ Dream session complete: 3420 dreams, 27 high-scoring candidates
```

### Check Frontend
```bash
# Browser console should show:
Dream session started: {status: "started", ...}
```

---

## 🐛 Common Issues

### Issue: "Dream worker not initialized"

**Cause:** Redis or CLIP not loaded

**Fix:**
```bash
# Check Redis
redis-cli ping
# Should return: PONG

# Check CLIP
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Issue: "ModuleNotFoundError: No module named 'yume'"

**Cause:** Python can't find yume folder

**Fix:**
```bash
# Make sure yume/ is in same directory as lcm_sr_server.py
ls -la
# Should see:
# yume/
# lcm_sr_server.py
```

### Issue: Low dreams/sec

**Cause:** Settings too high quality

**Fix in `dream_worker.py`:**
```python
# Change:
size=(64, 64)  → size=(32, 32)
steps=1        → (keep at 1)
cfg=0.0        → (keep at 0)
```

---

## 📊 Performance Expectations

| Setup | Dreams/sec | 1-hour yield |
|-------|------------|--------------|
| RTX 4090 + Full | 1000-2000 | 3.6M - 7.2M |
| RTX 3090 + Med | 500-1000 | 1.8M - 3.6M |
| RK3588 + Low | 50-200 | 180K - 720K |

---

## 🎯 Next Steps

Once working:
1. ✅ Test 1-hour session
2. ✅ Tune `similarity_threshold` for your use case
3. ✅ Try different strategies (evolutionary, cluster)
4. ✅ Add favorites to Redis
5. ✅ Integrate with main chat (use dream seeds)

---

## 📝 File Checklist

```
✅ yume/__init__.py
✅ yume/dream_worker.py
✅ yume/dream_endpoints.py
✅ yume/scoring.py
✅ yume/strategies.py
✅ yume/README.md
✅ lcm_sr_server.py (updated)
✅ requirements.txt (updated)
✅ docker-compose.yml (updated, optional)
✅ lcm-sr-ui/src/components/dreams/DreamGallery.jsx
✅ lcm-sr-ui/src/components/ui/tabs.jsx
✅ lcm-sr-ui/src/App.jsx (updated with tabs)
```

---

**You're ready to dream! 🌙✨**
