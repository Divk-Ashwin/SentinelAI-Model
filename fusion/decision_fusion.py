"""
Decision Fusion Module - Late Fusion of Multi-modal Predictions

Implements dynamic weight redistribution when modalities are missing,
avoiding the bias introduced by defaulting missing scores to 0.5.
"""
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

from config import FUSION_WEIGHTS, SPAM_THRESHOLD, CONFIDENCE_THRESHOLDS


class ConfidenceLevel(Enum):
    """Confidence levels for fusion predictions."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class FusionResult:
    """Result of multi-modal fusion."""
    label: str                          # "SPAM" or "HAM"
    score: float                        # Fused probability score (0.0 - 1.0)
    confidence: ConfidenceLevel         # HIGH, MEDIUM, or LOW
    modalities_used: List[str]          # Which modalities contributed
    individual_scores: Dict[str, float] # Per-modality scores (only those provided)
    weights_applied: Dict[str, float]   # Actual weights used after redistribution

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence.value,
            "modalities_used": self.modalities_used,
            "individual_scores": self.individual_scores,
            "weights_applied": self.weights_applied
        }


class DecisionFusion:
    """
    Late fusion strategy for combining predictions from multiple modalities.

    Key improvements:
    - Missing modalities are EXCLUDED from fusion (not defaulted to 0.5)
    - Weights are dynamically redistributed to maintain relative ratios
    - Confidence levels indicate prediction reliability

    Base weight ratios (text:metadata:image = 0.5:0.3:0.2):
    - All three available: text=0.50, metadata=0.30, image=0.20
    - Image missing:       text=0.625, metadata=0.375
    - Metadata missing:    text=0.714, image=0.286
    - Image+metadata miss: text=1.0
    - Text missing:        metadata=0.60, image=0.40
    - Text+image missing:  metadata=1.0
    - Text+metadata miss:  image=1.0
    """

    # Base weights (relative ratios)
    BASE_WEIGHTS = {
        "text": FUSION_WEIGHTS.get("text", 0.5),
        "metadata": FUSION_WEIGHTS.get("metadata", 0.3),
        "image": FUSION_WEIGHTS.get("image", 0.2)
    }

    def __init__(self, threshold: float = SPAM_THRESHOLD):
        """
        Initialize fusion module.

        Args:
            threshold: Decision threshold for SPAM classification (default from config)
        """
        self.threshold = threshold
        self._validate_threshold(threshold)

    def _validate_threshold(self, threshold: float):
        """Validate threshold is in valid range."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    def _calculate_confidence(self, score: float) -> ConfidenceLevel:
        """
        Determine confidence level based on fused score.

        HIGH:   score > 0.75 (confident spam) or score < 0.25 (confident ham)
        MEDIUM: score in [0.40, 0.75] or [0.25, 0.40]
        LOW:    score very close to threshold (uncertain)
        """
        high_upper = CONFIDENCE_THRESHOLDS.get("high_upper", 0.75)
        high_lower = CONFIDENCE_THRESHOLDS.get("high_lower", 0.25)
        medium_upper = CONFIDENCE_THRESHOLDS.get("medium_upper", 0.60)
        medium_lower = CONFIDENCE_THRESHOLDS.get("medium_lower", 0.40)

        # HIGH confidence: very clear spam or very clear ham
        if score > high_upper or score < high_lower:
            return ConfidenceLevel.HIGH

        # MEDIUM confidence: reasonably confident
        if score >= medium_upper or score <= medium_lower:
            return ConfidenceLevel.MEDIUM

        # LOW confidence: score is in the uncertain zone (0.40 - 0.60)
        return ConfidenceLevel.LOW

    def _redistribute_weights(
        self,
        available_modalities: List[str]
    ) -> Dict[str, float]:
        """
        Redistribute weights among available modalities, preserving relative ratios.

        Args:
            available_modalities: List of modality names that have scores

        Returns:
            Dictionary of modality -> redistributed weight
        """
        if not available_modalities:
            raise ValueError("At least one modality must be available for fusion")

        # Get base weights for available modalities
        available_weights = {
            mod: self.BASE_WEIGHTS[mod]
            for mod in available_modalities
            if mod in self.BASE_WEIGHTS
        }

        # Normalize to sum to 1.0
        total = sum(available_weights.values())
        if total == 0:
            raise ValueError(f"Unknown modalities: {available_modalities}")

        normalized = {
            mod: weight / total
            for mod, weight in available_weights.items()
        }

        return normalized

    def fuse(
        self,
        text_score: Optional[float] = None,
        metadata_score: Optional[float] = None,
        image_score: Optional[float] = None,
        url_text_score: Optional[float] = None
    ) -> FusionResult:
        """
        Perform late fusion on multi-modal predictions.

        Missing modalities (None) are EXCLUDED from fusion entirely.
        Weights are redistributed among available modalities.

        Args:
            text_score: Prediction score from text model (0.0 - 1.0), or None
            metadata_score: Prediction score from metadata model (0.0 - 1.0), or None
            image_score: Prediction score from image model (0.0 - 1.0), or None
            url_text_score: Optional URL text risk score (0.0 - 1.0) from metadata pipeline

        Returns:
            FusionResult with label, score, confidence, and details

        Raises:
            ValueError: If no modality scores are provided
        """
        # Collect available scores
        scores = {}
        if text_score is not None:
            scores["text"] = text_score
        if metadata_score is not None:
            scores["metadata"] = metadata_score
        if image_score is not None:
            scores["image"] = image_score

        if not scores:
            raise ValueError("At least one modality score must be provided")

        # Special case: If only text is available and it's very high confidence (> 0.85),
        # use it directly without dilution
        if len(scores) == 1 and "text" in scores and text_score > 0.85:
            fused_score = text_score
            weights = {"text": 1.0}
            available_modalities = ["text"]
        else:
            # Get redistributed weights for available modalities
            available_modalities = list(scores.keys())
            weights = self._redistribute_weights(available_modalities)

            # Compute weighted fusion
            fused_score = sum(
                weights[mod] * scores[mod]
                for mod in available_modalities
            )

        # Apply URL text score boost based on risk level
        if url_text_score is not None:
            if url_text_score >= 0.75:
                # Very high risk URL - boost by 20%
                fused_score = min(fused_score + 0.20, 1.0)
            elif url_text_score >= 0.5:
                # Moderate risk URL - boost by 10%
                fused_score = min(fused_score + 0.10, 1.0)

        # Make classification decision
        label = "SPAM" if fused_score > self.threshold else "HAM"

        # Determine confidence level
        confidence = self._calculate_confidence(fused_score)

        return FusionResult(
            label=label,
            score=round(fused_score, 4),
            confidence=confidence,
            modalities_used=available_modalities,
            individual_scores={k: round(v, 4) for k, v in scores.items()},
            weights_applied={k: round(v, 4) for k, v in weights.items()}
        )

    def fuse_simple(
        self,
        text_score: Optional[float] = None,
        metadata_score: Optional[float] = None,
        image_score: Optional[float] = None,
        url_text_score: Optional[float] = None
    ) -> Tuple[str, float, str]:
        """
        Simplified fusion returning just (label, score, confidence).

        Args:
            text_score: Prediction score from text model (0.0 - 1.0), or None
            metadata_score: Prediction score from metadata model (0.0 - 1.0), or None
            image_score: Prediction score from image model (0.0 - 1.0), or None
            url_text_score: Optional URL text risk score (0.0 - 1.0)

        Returns:
            Tuple of (label, fused_score, confidence_level)
        """
        result = self.fuse(text_score, metadata_score, image_score, url_text_score)
        return result.label, result.score, result.confidence.value

    def set_threshold(self, threshold: float):
        """
        Adjust decision threshold.

        Args:
            threshold: New threshold value (0.0 - 1.0)
        """
        self._validate_threshold(threshold)
        self.threshold = threshold

    def get_config(self) -> Dict:
        """Get current fusion configuration."""
        return {
            "base_weights": self.BASE_WEIGHTS.copy(),
            "threshold": self.threshold,
            "confidence_thresholds": CONFIDENCE_THRESHOLDS.copy()
        }

    @staticmethod
    def compute_weights_for_modalities(modalities: List[str]) -> Dict[str, float]:
        """
        Utility to preview what weights would be applied for given modalities.

        Args:
            modalities: List of available modality names

        Returns:
            Dictionary of modality -> weight

        Example:
            >>> DecisionFusion.compute_weights_for_modalities(["text", "metadata"])
            {"text": 0.625, "metadata": 0.375}
        """
        base = DecisionFusion.BASE_WEIGHTS
        available = {mod: base[mod] for mod in modalities if mod in base}
        total = sum(available.values())
        return {mod: round(w / total, 4) for mod, w in available.items()}


# Convenience function for quick fusion
def fuse_predictions(
    text_score: Optional[float] = None,
    metadata_score: Optional[float] = None,
    image_score: Optional[float] = None,
    url_text_score: Optional[float] = None,
    threshold: float = SPAM_THRESHOLD
) -> Dict:
    """
    Convenience function for one-off fusion.

    Args:
        text_score: Text model score (or None if unavailable)
        metadata_score: Metadata model score (or None if unavailable)
        image_score: Image model score (or None if unavailable)
        url_text_score: Optional URL text risk score (or None if unavailable)
        threshold: Decision threshold

    Returns:
        Dictionary with fusion result
    """
    fusion = DecisionFusion(threshold=threshold)
    result = fusion.fuse(text_score, metadata_score, image_score, url_text_score)
    return result.to_dict()
