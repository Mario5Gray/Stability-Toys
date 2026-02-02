"""
Yume - Latent Space Exploration Library

Background dream worker for exploring latent space in LCM-based image generation.
Generates, scores, and curates through cfg, clip.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Import main classes for easy access
try:
    from .dream_worker import DreamWorker, DreamCandidate
    from .scoring import CLIPScorer, AestheticScorer, CompositeScorer
    
    __all__ = [
        "DreamWorker",
        "DreamCandidate",
        "CLIPScorer",
        "AestheticScorer",
        "CompositeScorer",
    ]
except ImportError:
    # Allow package to be imported even if dependencies aren't installed
    # Useful for setup.py installation
    __all__ = []
