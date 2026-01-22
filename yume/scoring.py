"""
Scoring functions for dream candidates.
Includes CLIP similarity and aesthetic quality prediction.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional
import cv2


class CLIPScorer:
    """
    Score image-text similarity using CLIP.
    Fast evaluation for latent space exploration.
    """
    
    def __init__(self, clip_model, device="cuda"):
        self.model = clip_model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Cache for text embeddings (avoid recomputing)
        self.text_cache = {}
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to CLIP embedding (with caching)."""
        if text in self.text_cache:
            return self.text_cache[text]
        
        with torch.no_grad():
            text_tokens = self.model.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        self.text_cache[text] = text_features
        return text_features
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to CLIP embedding."""
        # Resize to CLIP input size
        image = image.convert('RGB').resize((224, 224))
        
        # Preprocess (normalize)
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor / 255.0
        
        # Normalize with CLIP stats
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def score(self, image: Image.Image, text: str) -> float:
        """
        Compute CLIP similarity score between image and text.
        Returns float in [0, 1] where higher is more similar.
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Cosine similarity
        similarity = torch.cosine_similarity(
            image_features, 
            text_features, 
            dim=-1
        ).item()
        
        # Normalize to [0, 1]
        # CLIP similarity is typically in [-1, 1] but usually [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def batch_score(self, images: list[Image.Image], text: str) -> list[float]:
        """Score multiple images against single text (faster)."""
        text_features = self.encode_text(text)
        
        scores = []
        for image in images:
            image_features = self.encode_image(image)
            similarity = torch.cosine_similarity(
                image_features, 
                text_features, 
                dim=-1
            ).item()
            scores.append(max(0.0, min(1.0, similarity)))
        
        return scores


class AestheticScorer:
    """
    Score image aesthetic quality.
    Uses simple heuristics or pre-trained aesthetic predictor.
    """
    
    def __init__(self, use_predictor: bool = False, device="cuda"):
        self.use_predictor = use_predictor
        self.device = device
        self.predictor = None
        
        if use_predictor:
            # Load aesthetic predictor (e.g., LAION aesthetic predictor)
            # For now, placeholder - you can integrate a real model
            pass
    
    def score(self, image: Image.Image) -> float:
        """
        Score image aesthetic quality.
        Returns float in [0, 1] where higher is better.
        """
        if self.use_predictor and self.predictor:
            return self._predictor_score(image)
        else:
            return self._heuristic_score(image)
    
    def _heuristic_score(self, image: Image.Image) -> float:
        """
        Simple heuristic scoring:
        - Sharpness (Laplacian variance)
        - Color variety (histogram entropy)
        - Contrast (std deviation)
        """
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(1.0, sharpness / 1000.0)  # Normalize
        
        # Contrast (std deviation)
        contrast = gray.std() / 127.5  # Normalize to [0, 1]
        contrast_score = min(1.0, contrast)
        
        # Color variety (simple metric: unique colors)
        h, w, c = img_array.shape
        total_pixels = h * w
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        variety_score = min(1.0, unique_colors / (total_pixels * 0.1))
        
        # Weighted combination
        final_score = (
            0.5 * sharpness_score +
            0.3 * contrast_score +
            0.2 * variety_score
        )
        
        return max(0.0, min(1.0, final_score))
    
    def _predictor_score(self, image: Image.Image) -> float:
        """Score using pre-trained aesthetic predictor."""
        # Placeholder for real aesthetic predictor
        # e.g., LAION aesthetic predictor v2
        with torch.no_grad():
            # Preprocess image
            # Run through predictor
            # Return score
            pass
        
        return 0.5  # Placeholder


class CompositeScorer:
    """
    Combine multiple scoring methods.
    """
    
    def __init__(
        self,
        clip_scorer: CLIPScorer,
        aesthetic_scorer: Optional[AestheticScorer] = None,
        clip_weight: float = 0.7,
        aesthetic_weight: float = 0.3,
    ):
        self.clip_scorer = clip_scorer
        self.aesthetic_scorer = aesthetic_scorer or AestheticScorer()
        self.clip_weight = clip_weight
        self.aesthetic_weight = aesthetic_weight
    
    def score(self, image: Image.Image, text: str) -> dict:
        """
        Score image with both CLIP and aesthetic.
        Returns dict with individual scores and weighted final.
        """
        clip_score = self.clip_scorer.score(image, text)
        aesthetic_score = self.aesthetic_scorer.score(image)
        
        final_score = (
            self.clip_weight * clip_score +
            self.aesthetic_weight * aesthetic_score
        )
        
        return {
            'clip': clip_score,
            'aesthetic': aesthetic_score,
            'final': final_score,
        }
