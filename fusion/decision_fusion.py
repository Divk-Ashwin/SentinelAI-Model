"""
Decision Fusion Module - Late Fusion of Multi-modal Predictions
"""
from typing import Dict, Tuple, Optional
from config import FUSION_WEIGHTS, SPAM_THRESHOLD


class DecisionFusion:
    """
    Late fusion strategy for combining predictions from multiple modalities
    """
    
    def __init__(
        self,
        text_weight: float = FUSION_WEIGHTS["text"],
        metadata_weight: float = FUSION_WEIGHTS["metadata"],
        image_weight: float = FUSION_WEIGHTS["image"],
        threshold: float = SPAM_THRESHOLD
    ):
        """
        Initialize fusion module
        
        Args:
            text_weight: Weight for text model predictions (default: 0.5)
            metadata_weight: Weight for metadata model predictions (default: 0.3)
            image_weight: Weight for image model predictions (default: 0.2)
            threshold: Decision threshold for SPAM classification (default: 0.6)
        """
        # Validate weights sum to 1.0
        total_weight = text_weight + metadata_weight + image_weight
        assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"
        
        self.weights = {
            "text": text_weight,
            "metadata": metadata_weight,
            "image": image_weight
        }
        self.threshold = threshold
    
    def fuse(
        self,
        text_score: float,
        metadata_score: float,
        image_score: float
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Perform late fusion on multi-modal predictions
        
        Args:
            text_score: Prediction score from text model (0.0 - 1.0)
            metadata_score: Prediction score from metadata model (0.0 - 1.0)
            image_score: Prediction score from image model (0.0 - 1.0)
            
        Returns:
            Tuple of (label, confidence, scores_dict):
                - label: "SPAM" or "HAM"
                - confidence: Final confidence score (0.0 - 1.0)
                - scores_dict: Individual model scores
        """
        # Weighted combination
        final_score = (
            self.weights["text"] * text_score +
            self.weights["metadata"] * metadata_score +
            self.weights["image"] * image_score
        )
        
        # Make decision
        label = "SPAM" if final_score > self.threshold else "HAM"
        
        # Prepare scores dictionary
        scores = {
            "text": round(text_score, 4),
            "metadata": round(metadata_score, 4),
            "image": round(image_score, 4)
        }
        
        return label, round(final_score, 4), scores
    
    def adjust_weights(
        self,
        text_weight: Optional[float] = None,
        metadata_weight: Optional[float] = None,
        image_weight: Optional[float] = None
    ):
        """
        Dynamically adjust fusion weights
        
        Args:
            text_weight: New weight for text model
            metadata_weight: New weight for metadata model
            image_weight: New weight for image model
        """
        if text_weight is not None:
            self.weights["text"] = text_weight
        if metadata_weight is not None:
            self.weights["metadata"] = metadata_weight
        if image_weight is not None:
            self.weights["image"] = image_weight
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total
    
    def set_threshold(self, threshold: float):
        """
        Adjust decision threshold
        
        Args:
            threshold: New threshold value (0.0 - 1.0)
        """
        assert 0.0 <= threshold <= 1.0, "Threshold must be between 0.0 and 1.0"
        self.threshold = threshold
    
    def get_config(self) -> Dict:
        """Get current fusion configuration."""
        return {
            "weights": self.weights.copy(),
            "threshold": self.threshold
        }