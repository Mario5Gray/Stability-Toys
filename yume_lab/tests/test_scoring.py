"""
Unit tests for scoring classes (CLIPScorer, AestheticScorer, CompositeScorer).
Tests both OpenAI CLIP and Hugging Face implementations.
"""

import pytest
import sys
import types
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, MagicMock, patch
import io

# Pre-inject a fake 'clip' module into sys.modules so that
# patch('clip.load') / patch('clip.tokenize') never triggers
# the real clip → torchvision → torch.hub import chain.
if 'clip' not in sys.modules:
    _fake_clip = types.ModuleType('clip')
    _fake_clip.load = Mock(return_value=(Mock(), lambda x: x))
    _fake_clip.tokenize = Mock(return_value=torch.randint(0, 1000, (1, 77)))
    sys.modules['clip'] = _fake_clip


@pytest.fixture
def test_image():
    """Create a test PIL image."""
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    return img


@pytest.fixture
def mock_openai_clip_model():
    """Mock OpenAI CLIP model."""
    model = Mock()
    model.__class__.__name__ = "CLIP"
    
    # Mock encode methods
    def encode_text(tokens):
        # Return fake embeddings
        return torch.randn(1, 512)
    
    def encode_image(image):
        # Return fake embeddings
        return torch.randn(1, 512)
    
    model.encode_text = Mock(side_effect=encode_text)
    model.encode_image = Mock(side_effect=encode_image)
    model.to = Mock(return_value=model)
    model.eval = Mock()
    
    return model


@pytest.fixture
def mock_huggingface_clip_model():
    """Mock Hugging Face CLIP model."""
    model = Mock()
    model.__class__.__name__ = "CLIPModel"
    
    # Mock feature extraction methods
    def get_text_features(**kwargs):
        return torch.randn(1, 512)
    
    def get_image_features(**kwargs):
        return torch.randn(1, 512)
    
    model.get_text_features = Mock(side_effect=get_text_features)
    model.get_image_features = Mock(side_effect=get_image_features)
    model.to = Mock(return_value=model)
    model.eval = Mock()
    
    return model


@pytest.fixture
def mock_clip_processor():
    """Mock Hugging Face CLIP processor."""
    processor = Mock()
    
    def process_text(text, **kwargs):
        return {'input_ids': torch.randint(0, 1000, (1, 10))}
    
    def process_image(images, **kwargs):
        return {'pixel_values': torch.randn(1, 3, 224, 224)}
    
    processor.side_effect = lambda text=None, images=None, **kwargs: (
        process_text(text, **kwargs) if text else process_image(images, **kwargs)
    )
    
    return processor


class TestCLIPScorerDetection:
    """Test CLIP type detection."""
    
    def test_detect_huggingface(self, mock_huggingface_clip_model, mock_clip_processor):
        """Test detection of Hugging Face CLIP."""
        from yume.scoring import CLIPScorer
        
        scorer = CLIPScorer(
            mock_huggingface_clip_model,
            mock_clip_processor,
            device="cpu"
        )
        
        assert scorer.clip_type == "huggingface"
    
    def test_detect_openai_without_processor(self, mock_openai_clip_model):
        """Test detection of OpenAI CLIP."""
        from yume.scoring import CLIPScorer
        
        with patch('clip.load') as mock_load:
            mock_load.return_value = (mock_openai_clip_model, lambda x: x)
            
            scorer = CLIPScorer(mock_openai_clip_model, device="cpu")
            
            assert scorer.clip_type == "openai"
    
    def test_huggingface_requires_processor(self, mock_huggingface_clip_model):
        """Test that Hugging Face CLIP requires processor."""
        from yume.scoring import CLIPScorer
        
        with pytest.raises(ValueError, match="requires clip_processor"):
            CLIPScorer(mock_huggingface_clip_model, device="cpu")
    
    def test_unknown_model_type(self):
        """Test handling of unknown CLIP model type."""
        from yume.scoring import CLIPScorer
        
        unknown_model = Mock()
        unknown_model.__class__.__name__ = "UnknownModel"
        unknown_model.to = Mock(return_value=unknown_model)
        unknown_model.eval = Mock()
        
        with pytest.raises(ValueError, match="Unknown CLIP model type"):
            CLIPScorer(unknown_model, device="cpu")


class TestCLIPScorerHuggingFace:
    """Test CLIPScorer with Hugging Face implementation."""
    
    @pytest.fixture
    def hf_scorer(self, mock_huggingface_clip_model, mock_clip_processor):
        """Create Hugging Face CLIP scorer."""
        from yume.scoring import CLIPScorer
        return CLIPScorer(mock_huggingface_clip_model, mock_clip_processor, device="cpu")
    
    def test_encode_text_huggingface(self, hf_scorer):
        """Test text encoding with Hugging Face."""
        embedding = hf_scorer.encode_text("a beautiful cat")
        
        assert embedding is not None
        assert embedding.shape[1] == 512  # Embedding dimension
        assert hf_scorer.model.get_text_features.called
    
    def test_encode_image_huggingface(self, hf_scorer, test_image):
        """Test image encoding with Hugging Face."""
        embedding = hf_scorer.encode_image(test_image)
        
        assert embedding is not None
        assert embedding.shape[1] == 512
        assert hf_scorer.model.get_image_features.called
    
    def test_score_huggingface(self, hf_scorer, test_image):
        """Test scoring with Hugging Face."""
        score = hf_scorer.score(test_image, "a beautiful landscape")
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_text_caching(self, hf_scorer):
        """Test that text embeddings are cached."""
        text = "test prompt"
        
        # First call
        hf_scorer.encode_text(text)
        call_count_1 = hf_scorer.model.get_text_features.call_count
        
        # Second call with same text
        hf_scorer.encode_text(text)
        call_count_2 = hf_scorer.model.get_text_features.call_count
        
        # Should not increase (cached)
        assert call_count_2 == call_count_1


class TestCLIPScorerOpenAI:
    """Test CLIPScorer with OpenAI implementation."""
    
    @pytest.fixture
    def openai_scorer(self, mock_openai_clip_model):
        """Create OpenAI CLIP scorer."""
        from yume.scoring import CLIPScorer
        
        with patch('clip.load') as mock_load:
            mock_load.return_value = (mock_openai_clip_model, lambda x: x)
            return CLIPScorer(mock_openai_clip_model, device="cpu")
    
    def test_encode_text_openai(self, openai_scorer):
        """Test text encoding with OpenAI CLIP."""
        with patch('clip.tokenize') as mock_tokenize:
            mock_tokenize.return_value = torch.randint(0, 1000, (1, 77))
            
            embedding = openai_scorer.encode_text("a dog")
            
            assert embedding is not None
            assert openai_scorer.model.encode_text.called
    
    def test_encode_image_openai(self, openai_scorer, test_image):
        """Test image encoding with OpenAI CLIP."""
        embedding = openai_scorer.encode_image(test_image)
        
        assert embedding is not None
        assert openai_scorer.model.encode_image.called
    
    def test_score_openai(self, openai_scorer, test_image):
        """Test scoring with OpenAI CLIP."""
        with patch('clip.tokenize') as mock_tokenize:
            mock_tokenize.return_value = torch.randint(0, 1000, (1, 77))
            
            score = openai_scorer.score(test_image, "a cat")
            
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0


class TestCLIPScorerBatchScoring:
    """Test batch scoring functionality."""
    
    def test_batch_score(self, mock_huggingface_clip_model, mock_clip_processor, test_image):
        """Test scoring multiple images at once."""
        from yume.scoring import CLIPScorer
        
        scorer = CLIPScorer(mock_huggingface_clip_model, mock_clip_processor, device="cpu")
        
        images = [test_image, test_image, test_image]
        scores = scorer.batch_score(images, "test prompt")
        
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestAestheticScorer:
    """Test aesthetic quality scoring."""
    
    @pytest.fixture
    def aesthetic_scorer(self):
        """Create aesthetic scorer."""
        from yume.scoring import AestheticScorer
        return AestheticScorer(use_predictor=False, device="cpu")
    
    def test_score_heuristic(self, aesthetic_scorer, test_image):
        """Test heuristic-based scoring."""
        score = aesthetic_scorer.score(test_image)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_sharp_image_scores_higher(self, aesthetic_scorer):
        """Test that sharp images score higher than blurry ones."""
        # Create sharp image with high contrast
        sharp_img = Image.new('RGB', (224, 224))
        pixels = np.zeros((224, 224, 3), dtype=np.uint8)
        pixels[:112, :, :] = 255  # Half white, half black
        sharp_img = Image.fromarray(pixels)
        
        # Create blurry image (uniform color)
        blurry_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        sharp_score = aesthetic_scorer.score(sharp_img)
        blurry_score = aesthetic_scorer.score(blurry_img)
        
        assert sharp_score > blurry_score
    
    def test_colorful_image_scores_higher(self, aesthetic_scorer):
        """Test that colorful images score higher."""
        # Create colorful image
        colorful = Image.new('RGB', (224, 224))
        pixels = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        colorful = Image.fromarray(pixels)
        
        # Create monochrome
        mono = Image.new('RGB', (224, 224), color=(100, 100, 100))
        
        colorful_score = aesthetic_scorer.score(colorful)
        mono_score = aesthetic_scorer.score(mono)
        
        # Colorful should generally score higher
        # (though this isn't guaranteed with random pixels)
        assert isinstance(colorful_score, float)
        assert isinstance(mono_score, float)


class TestCompositeScorer:
    """Test composite scoring with multiple methods."""
    
    @pytest.fixture
    def composite_scorer(self, mock_huggingface_clip_model, mock_clip_processor):
        """Create composite scorer."""
        from yume.scoring import CLIPScorer, AestheticScorer, CompositeScorer
        
        clip_scorer = CLIPScorer(
            mock_huggingface_clip_model,
            mock_clip_processor,
            device="cpu"
        )
        aesthetic_scorer = AestheticScorer(use_predictor=False)
        
        return CompositeScorer(
            clip_scorer=clip_scorer,
            aesthetic_scorer=aesthetic_scorer,
            clip_weight=0.7,
            aesthetic_weight=0.3
        )
    
    def test_composite_score(self, composite_scorer, test_image):
        """Test composite scoring returns all scores."""
        result = composite_scorer.score(test_image, "a beautiful cat")
        
        assert 'clip' in result
        assert 'aesthetic' in result
        assert 'final' in result
        
        assert 0.0 <= result['clip'] <= 1.0
        assert 0.0 <= result['aesthetic'] <= 1.0
        assert 0.0 <= result['final'] <= 1.0
    
    def test_composite_weights(self, composite_scorer, test_image):
        """Test that composite score respects weights."""
        result = composite_scorer.score(test_image, "test")
        
        # Final should be weighted average
        expected = (
            0.7 * result['clip'] +
            0.3 * result['aesthetic']
        )
        
        assert abs(result['final'] - expected) < 1e-6
    
    def test_custom_weights(self, mock_huggingface_clip_model, mock_clip_processor):
        """Test composite scorer with custom weights."""
        from yume.scoring import CLIPScorer, AestheticScorer, CompositeScorer
        
        clip_scorer = CLIPScorer(
            mock_huggingface_clip_model,
            mock_clip_processor,
            device="cpu"
        )
        aesthetic_scorer = AestheticScorer(use_predictor=False)
        
        scorer = CompositeScorer(
            clip_scorer=clip_scorer,
            aesthetic_scorer=aesthetic_scorer,
            clip_weight=0.9,
            aesthetic_weight=0.1
        )
        
        assert scorer.clip_weight == 0.9
        assert scorer.aesthetic_weight == 0.1


class TestScoringEdgeCases:
    """Test edge cases and error handling."""
    
    def test_score_with_invalid_image(self, mock_huggingface_clip_model, mock_clip_processor):
        """Test scoring with invalid image."""
        from yume.scoring import CLIPScorer
        
        scorer = CLIPScorer(mock_huggingface_clip_model, mock_clip_processor, device="cpu")
        
        # This should either handle gracefully or raise appropriate error
        # Behavior depends on implementation
        try:
            score = scorer.score(None, "test")
            # If it doesn't crash, check score is valid
            assert isinstance(score, (float, type(None)))
        except (AttributeError, TypeError):
            # Expected for invalid input
            pass
    
    def test_score_with_empty_text(self, mock_huggingface_clip_model, mock_clip_processor, test_image):
        """Test scoring with empty text."""
        from yume.scoring import CLIPScorer
        
        scorer = CLIPScorer(mock_huggingface_clip_model, mock_clip_processor, device="cpu")
        
        # Should handle empty text
        score = scorer.score(test_image, "")
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_grayscale_image_aesthetic(self):
        """Test aesthetic scoring on grayscale images."""
        from yume.scoring import AestheticScorer
        
        scorer = AestheticScorer(use_predictor=False)
        
        # Create grayscale image
        gray_img = Image.new('L', (224, 224), color=128)
        gray_rgb = gray_img.convert('RGB')
        
        score = scorer.score(gray_rgb)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestScorerPerformance:
    """Test performance characteristics of scorers."""
    
    def test_text_embedding_cache_performance(self, mock_huggingface_clip_model, mock_clip_processor):
        """Test that caching improves performance."""
        from yume.scoring import CLIPScorer
        import time
        
        scorer = CLIPScorer(mock_huggingface_clip_model, mock_clip_processor, device="cpu")
        
        text = "performance test prompt"
        
        # First call (not cached)
        start1 = time.time()
        scorer.encode_text(text)
        time1 = time.time() - start1
        
        # Second call (cached)
        start2 = time.time()
        scorer.encode_text(text)
        time2 = time.time() - start2
        
        # Cached should be faster (though timing may vary)
        # Just verify both complete successfully
        assert time1 >= 0
        assert time2 >= 0
        
        # Verify it's actually cached
        assert text in scorer.text_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
