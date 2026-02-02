"""
Yume (夢) - Background Dream System
Server-side latent space exploration with CLIP scoring.
"""
from .dream_worker import DreamWorker, DreamCandidate, init_dream_worker  # FIXED: Added init_dream_worker
from .dream_endpoints import dream_router
from .scoring import CLIPScorer, AestheticScorer
from .strategies import ExplorationStrategy

__version__ = "0.1.0"

__all__ = [
    "DreamWorker",
    "DreamCandidate",
    "dream_router",
    "init_dream_worker",  # This was in __all__ but not imported!
    "get_dream_worker",
    "clear_dream_worker",
    "CLIPScorer",
    "AestheticScorer",
    "ExplorationStrategy",
]