"""
Metadata Analysis Pipeline - FFNN for Phishing Detection (Pipeline 3)
Returns feature importance scores based on first layer weights for explainability.
"""
import os
import pickle
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
import numpy as np
import logging

from config import METADATA_MODEL_CONFIG, DEVICE, METADATA_FEATURES, METADATA_MODEL_PATH, METADATA_SCALER_PATH
from utils.preprocessing import extract_metadata_features, detect_spam_in_metadata

logger = logging.getLogger(__name__)


def check_spam_number(phone: str) -> float:
    """
    Check if number is reported as spam.
    Uses heuristic approach to detect suspicious sender patterns.

    Args:
        phone: Sender phone number or ID

    Returns:
        Spam probability (0.0-1.0) where higher = more likely spam
    """
    if not phone:
        return 0.5

    try:
        # Known safe service headers
        KNOWN_SAFE = [
            "HDFCBK", "SBIBNK", "ICICIB", "AXISBK",
            "AMAZON", "PAYTM", "JIONET", "AIRTEL",
            "BSNL", "IRCTC", "FLIPKART", "SWIGGY",
            "ZOMATO", "GOOGLE", "NETFLIX", "NSDL",
            "UIDAI", "TRAI", "LIC", "HDFC", "ICICI",
            "SBI", "AXIS", "KOTAK", "INDUSIND"
        ]
        phone_upper = phone.upper().strip()

        # Known safe sender header
        if any(safe in phone_upper for safe in KNOWN_SAFE):
            return 0.05

        # Pure numeric and 10+ digits = suspicious
        clean = phone.replace("+", "").replace("-", "").replace(" ", "")
        if clean.isdigit() and len(clean) >= 10:
            return 0.65

        # Mix of random chars and numbers
        if (any(c.isdigit() for c in phone) and
            any(c.isalpha() for c in phone) and
            not any(safe in phone_upper for safe in KNOWN_SAFE)):
            return 0.55

        return 0.5
    except Exception:
        return 0.5


def check_url_text(url: str) -> float:
    """
    Analyze URL text content for phishing indicators.
    Returns spam probability 0.0 to 1.0
    """
    if not url:
        return 0.5
    try:
        import re
        url_lower = url.lower()
        score = 0.5

        # High risk URL keywords
        PHISHING_URL_KEYWORDS = [
            "verify", "secure", "update", "confirm",
            "login", "signin", "account", "banking",
            "suspend", "block", "urgent", "alert",
            "kyc", "claim", "prize", "winner", "free",
            "lucky", "reward", "offer", "gift",
            "paypal", "sbi", "hdfc", "icici", "uidai",
            "aadhaar", "pan", "upi", "refund", "tax"
        ]

        # Suspicious TLDs
        SUSPICIOUS_TLDS = [
            ".xyz", ".tk", ".ml", ".ga", ".cf",
            ".gq", ".top", ".club", ".online",
            ".site", ".web", ".info", ".biz"
        ]

        # Safe domains
        SAFE_DOMAINS = [
            "amazon.in", "amazon.com", "flipkart.com",
            "paytm.com", "phonepe.com", "googlepay.com",
            "hdfcbank.com", "sbi.co.in", "icicibank.com",
            "irctc.co.in", "jio.com", "airtel.in",
            "netflix.com", "google.com", "apple.com",
            "incometax.gov.in",
            "uidai.gov.in",
            "myaadhaar.uidai.gov.in",
            "tsspdcl.co.in",
            "bsnl.co.in",
            "licindia.in",
            "jntuh.ac.in",
            "mgit.ac.in",
            "gov.in",
            "nic.in",
            "ac.in", "bescom.co.in",
            "tneb.in",
            "msedcl.in", 
            "tsspdcl.co.in",
            "apspdcl.in",
        ]

        # Check government domains first (highest trust)
        if any(domain in url_lower for domain in ['.gov.in', '.nic.in', '.ac.in', 'uidai', 'irctc']):
            return 0.05

        # Check safe domains
        if any(domain in url_lower for domain in SAFE_DOMAINS):
            return 0.05

        # Count phishing keywords in URL
        keyword_hits = sum(
            1 for kw in PHISHING_URL_KEYWORDS
            if kw in url_lower
        )

        # Count suspicious TLDs
        tld_hits = sum(
            1 for tld in SUSPICIOUS_TLDS
            if url_lower.endswith(tld) or tld + "/" in url_lower
        )

        # Suspicious patterns
        has_ip = bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url))
        has_multiple_subdomains = url.count('.') > 3
        has_suspicious_port = bool(re.search(r':\d{4,5}/', url))
        is_shortened = any(s in url_lower for s in [
            'bit.ly', 't.ly', 'tinyurl', 'goo.gl',
            'ow.ly', 'short.link', 'tiny.cc'])

        # Calculate score
        if has_ip:
            score = min(score + 0.35, 1.0)
        if keyword_hits >= 2:
            score = min(score + 0.20, 1.0)
        elif keyword_hits == 1:
            score = min(score + 0.10, 1.0)
        if tld_hits >= 1:
            score = min(score + 0.20, 1.0)
        if has_multiple_subdomains:
            score = min(score + 0.10, 1.0)
        if has_suspicious_port:
            score = min(score + 0.15, 1.0)
        if is_shortened:
            score = min(score + 0.15, 1.0)

        return round(score, 3)

    except Exception:
        return 0.5


@dataclass
class FeatureContribution:
    """A feature and its importance score."""
    feature: str
    score: float
    value: float  # The actual input value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "score": round(self.score, 4),
            "value": round(self.value, 4)
        }


@dataclass
class MetadataAnalysisResult:
    """Complete result from metadata analysis with explainability."""
    score: float                                    # Spam probability
    label: str                                      # "SPAM" or "HAM"
    contributing_features: List[FeatureContribution]  # Top features by importance
    suspicious_indicators: Dict[str, Any]           # Rule-based suspicious features
    explanation: str                                # Human-readable explanation
    url_text_score: Optional[float] = None          # URL text risk score (0.0-1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "label": self.label,
            "contributing_features": [f.to_dict() for f in self.contributing_features],
            "suspicious_indicators": self.suspicious_indicators,
            "explanation": self.explanation,
            "url_text_score": self.url_text_score
        }


class MetadataClassifier(nn.Module):
    """
    Feed Forward Neural Network for metadata-based phishing detection.
    Architecture: 15 -> 64 -> 32 -> 16 -> 2 (with ReLU and Dropout 0.3)

    IMPORTANT: This must match the MetadataFFNN class in train/train_metadata_model.py exactly.
    """

    def __init__(
        self,
        input_dim: int = METADATA_MODEL_CONFIG["input_dim"],
        hidden_dims: list = METADATA_MODEL_CONFIG["hidden_dims"],
        output_dim: int = METADATA_MODEL_CONFIG["output_dim"],
        dropout: float = METADATA_MODEL_CONFIG["dropout"]
    ):
        super(MetadataClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers: Linear -> ReLU -> Dropout (NO BatchNorm - must match training)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Feature tensor (batch_size, input_dim)

        Returns:
            Logits (batch_size, output_dim)
        """
        return self.network(x)

    def get_first_layer_weights(self) -> torch.Tensor:
        """Get weights from the first linear layer for feature importance."""
        # First layer is at index 0 in the Sequential
        first_layer = self.network[0]
        if isinstance(first_layer, nn.Linear):
            return first_layer.weight.data
        return None


class MetadataPipeline:
    """
    Complete metadata analysis pipeline with feature importance explainability.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize metadata pipeline.

        Args:
            model_path: Path to saved model weights
        """
        self.device = torch.device(DEVICE)
        self.model_loaded = False
        self.scaler = None
        self.feature_names = METADATA_FEATURES

        # Initialize model
        self.model = MetadataClassifier()

        # Load scaler if available
        scaler_path = METADATA_SCALER_PATH
        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✓ Loaded feature scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"✗ Could not load scaler from {scaler_path}: {e}")
                logger.warning("Feature normalization will not be applied")
        else:
            logger.warning(f"✗ Scaler file not found: {scaler_path}")
            logger.warning("Feature normalization will not be applied")

        # Load trained weights if available
        model_path = model_path or METADATA_MODEL_PATH
        if model_path and os.path.exists(model_path):
            self._load_model_weights(model_path)
        elif model_path:
            logger.warning(f"✗ Metadata model file not found: {model_path}")
            logger.warning("Using untrained model (random weights)")

        self.model.to(self.device)
        self.model.eval()

    def _load_model_weights(self, model_path: str) -> bool:
        """
        Load model weights with handling for key prefix mismatches.

        Args:
            model_path: Path to the saved model checkpoint

        Returns:
            True if loading succeeded, False otherwise
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)

            # Get model's expected keys
            model_keys = set(self.model.state_dict().keys())
            loaded_keys = set(state_dict.keys())

            # Check for exact match
            if model_keys == loaded_keys:
                self.model.load_state_dict(state_dict)
                logger.info(f"✓ Loaded metadata model from {model_path}")

            # Check for 'model.' prefix in saved keys
            elif all(k.startswith('model.') for k in loaded_keys):
                new_state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)
                logger.info(f"✓ Loaded metadata model from {model_path} (removed 'model.' prefix)")

            # Check for 'module.' prefix (nn.DataParallel)
            elif all(k.startswith('module.') for k in loaded_keys):
                new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)
                logger.info(f"✓ Loaded metadata model from {model_path} (removed 'module.' prefix)")

            # Check if model needs 'network.' prefix added
            elif all(k.startswith('network.') for k in model_keys) and not any(k.startswith('network.') for k in loaded_keys):
                new_state_dict = {'network.' + k: v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)
                logger.info(f"✓ Loaded metadata model from {model_path} (added 'network.' prefix)")

            else:
                missing_keys = model_keys - loaded_keys
                unexpected_keys = loaded_keys - model_keys
                if missing_keys:
                    logger.error(f"Missing keys in checkpoint: {list(missing_keys)[:5]}...")
                if unexpected_keys:
                    logger.error(f"Unexpected keys in checkpoint: {list(unexpected_keys)[:5]}...")
                raise RuntimeError("State dict keys do not match model architecture")

            # Verify model with dummy forward pass
            self._verify_model_loading()
            self.model_loaded = True
            return True

        except Exception as e:
            logger.error(f"✗ Could not load metadata model from {model_path}: {e}")
            logger.warning("Using untrained model (random weights)")
            self.model_loaded = False
            return False

    def _verify_model_loading(self):
        """Verify model loads correctly by running a dummy forward pass."""
        input_dim = METADATA_MODEL_CONFIG["input_dim"]
        try:
            self.model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, input_dim, device=self.device)
                output = self.model(dummy_input)

                expected_output_dim = METADATA_MODEL_CONFIG["output_dim"]
                if output.shape != (1, expected_output_dim):
                    raise RuntimeError(f"Output shape mismatch: expected (1, {expected_output_dim}), got {output.shape}")

                logger.info(f"✓ Model verification passed: input {dummy_input.shape} -> output {output.shape}")

        except Exception as e:
            logger.error(f"✗ Model verification failed: {e}")
            raise

    def preprocess(
        self,
        url: Optional[str] = None,
        sender: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract and preprocess metadata features.

        Args:
            url: URL from the message
            sender: Sender ID
            timestamp: Message timestamp

        Returns:
            Tuple of (feature_tensor, raw_features_array)
        """
        # Extract features
        features = extract_metadata_features(url, sender, timestamp)

        # Convert to array in correct order
        features_array = np.array([features[feature_name] for feature_name in self.feature_names], dtype=np.float32)
        raw_features = features_array.copy()
        features_array = features_array.reshape(1, -1)

        # Normalize features if scaler is available
        if self.scaler is not None:
            try:
                features_array = self.scaler.transform(features_array)
            except Exception as e:
                logger.warning(f"Could not normalize features: {e}")

        # Convert to tensor
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        return features_tensor.to(self.device), raw_features

    def _compute_feature_importance(
        self,
        features_tensor: torch.Tensor,
        raw_features: np.ndarray,
        top_k: int = 5
    ) -> List[FeatureContribution]:
        """
        Compute feature importance using first layer weights.

        Importance = abs(input_value) * abs(weight_norm)

        Args:
            features_tensor: Normalized feature tensor
            raw_features: Raw feature values (for display)
            top_k: Number of top features to return

        Returns:
            List of FeatureContribution objects
        """
        try:
            # Get first layer weights: shape (hidden_dim, input_dim)
            first_layer_weights = self.model.get_first_layer_weights()
            if first_layer_weights is None:
                return []

            # Compute L2 norm of weights for each input feature
            # This gives us a (input_dim,) tensor
            weight_norms = torch.norm(first_layer_weights, dim=0).cpu().numpy()

            # Get input values (normalized)
            input_values = features_tensor[0].cpu().numpy()

            # Compute importance: |input_value| * |weight_norm|
            importance_scores = np.abs(input_values) * np.abs(weight_norms)

            # Create feature-score pairs
            feature_scores = []
            for idx, (feature_name, importance, raw_value) in enumerate(
                zip(self.feature_names, importance_scores, raw_features)
            ):
                feature_scores.append(FeatureContribution(
                    feature=feature_name,
                    score=float(importance),
                    value=float(raw_value)
                ))

            # Sort by importance descending
            feature_scores.sort(key=lambda x: x.score, reverse=True)

            # Take top k
            top_features = feature_scores[:top_k]

            # Normalize scores to 0-1 range
            if top_features:
                max_score = max(f.score for f in top_features)
                if max_score > 0:
                    for f in top_features:
                        f.score = f.score / max_score

            return top_features

        except Exception as e:
            logger.warning(f"Failed to compute feature importance: {e}")
            return []

    @torch.no_grad()
    def predict(
        self,
        url: Optional[str] = None,
        sender: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> Optional[float]:
        """
        Predict phishing probability from metadata.

        Args:
            url: URL from the message
            sender: Sender ID
            timestamp: Message timestamp

        Returns:
            Probability score (0.0 - 1.0) where higher = more likely spam,
            or None if no URL is provided (to exclude from fusion)
        """
        # Return None if no URL provided - this tells fusion to ignore metadata
        if not url or url.strip() == "":
            return None

        features_tensor, _ = self.preprocess(url, sender, timestamp)
        logits = self.model(features_tensor)
        probs = torch.softmax(logits, dim=1)
        spam_prob = probs[0, 1].item()

        # Sanity check: if model gives extreme/broken output, return None
        # This prevents untrained/broken models from corrupting fusion
        if spam_prob < 0.01 or spam_prob > 0.99:
            logger.warning(f"Metadata model returned extreme score {spam_prob:.6f}, excluding from fusion")
            return None

        return spam_prob

    @torch.no_grad()
    def predict_with_explanation(
        self,
        url: Optional[str] = None,
        sender: Optional[str] = None,
        timestamp: Optional[str] = None,
        time: Optional[str] = None,
        date: Optional[str] = None,
        mobile_number: Optional[str] = None,
        top_k: int = 5
    ) -> Optional[MetadataAnalysisResult]:
        """
        Predict with full explainability including feature importance.

        Args:
            url: URL from the message
            sender: Sender ID
            timestamp: Full timestamp (ISO format)
            time: Time message was sent (HH:MM:SS)
            date: Date message was sent (YYYY-MM-DD)
            mobile_number: Receiver's mobile number
            top_k: Number of top contributing features to return

        Returns:
            MetadataAnalysisResult with score, contributing features, and explanation,
            or None if no URL is provided (to exclude from fusion)
        """
        # Return None if no URL provided - this tells fusion to ignore metadata
        if not url or url.strip() == "":
            return None

        # Check if sender number is reported as spam
        number_score = check_spam_number(sender or mobile_number or "")

        # Analyze URL text content for phishing indicators
        url_text_score = check_url_text(url or "")

        # Preprocess and get raw features
        features_tensor, raw_features = self.preprocess(url, sender, timestamp)

        # Forward pass
        logits = self.model(features_tensor)
        probs = torch.softmax(logits, dim=1)
        spam_prob = probs[0, 1].item()

        # Sanity check: if model gives extreme/broken output, return None
        if spam_prob < 0.01 or spam_prob > 0.99:
            logger.warning(f"Metadata model returned extreme score {spam_prob:.6f}, excluding from fusion")
            return None

        label = "SPAM" if spam_prob > 0.5 else "HAM"

        # Compute feature importance
        contributing_features = self._compute_feature_importance(
            features_tensor, raw_features, top_k=top_k
        )

        # Add spam number score if suspicious
        if number_score > 0.6:
            contributing_features.append(FeatureContribution(
                feature="suspicious_sender_number",
                score=number_score,
                value=number_score
            ))

        # Add URL text risk score if suspicious
        if url_text_score > 0.6:
            contributing_features.append(FeatureContribution(
                feature="suspicious_url_content",
                score=url_text_score,
                value=url_text_score
            ))

        # Detect rule-based suspicious indicators
        suspicious_indicators = detect_spam_in_metadata(
            url=url,
            sender=sender,
            time=time,
            date=date,
            mobile_number=mobile_number
        )

        # Generate explanation
        explanation = self._generate_explanation(spam_prob, contributing_features, suspicious_indicators)

        return MetadataAnalysisResult(
            score=spam_prob,
            label=label,
            contributing_features=contributing_features,
            suspicious_indicators=suspicious_indicators,
            explanation=explanation,
            url_text_score=url_text_score
        )

    @torch.no_grad()
    def predict_with_indicators(
        self,
        url: Optional[str] = None,
        sender: Optional[str] = None,
        time: Optional[str] = None,
        date: Optional[str] = None,
        mobile_number: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any], str]:
        """
        Predict phishing probability with detailed suspicious features (Pipeline 3).
        Legacy method for backwards compatibility.

        Args:
            url: URL from the message
            sender: Sender ID
            time: Time message was sent (HH:MM:SS)
            date: Date message was sent (YYYY-MM-DD)
            mobile_number: Receiver's mobile number
            timestamp: Full timestamp (ISO format)

        Returns:
            Tuple of (score, suspicious_features_dict, explanation)
        """
        result = self.predict_with_explanation(
            url=url, sender=sender, timestamp=timestamp,
            time=time, date=date, mobile_number=mobile_number
        )
        return result.score, result.suspicious_indicators, result.explanation

    def _generate_explanation(
        self,
        score: float,
        contributing_features: List[FeatureContribution],
        suspicious_indicators: Dict[str, Any]
    ) -> str:
        """Generate explanation for metadata-based prediction."""
        parts = []

        if score > 0.7:
            parts.append("High-risk metadata detected")
        elif score > 0.5:
            parts.append("Moderate-risk metadata")
        else:
            parts.append("Metadata appears legitimate")

        # Add top contributing features
        if contributing_features:
            top_features = [f.feature for f in contributing_features[:3]]
            parts.append(f"Key factors: {', '.join(top_features)}")

        # Add suspicious indicators
        if suspicious_indicators:
            indicator_names = list(suspicious_indicators.keys())[:3]
            readable_names = []
            for name in indicator_names:
                readable = name.replace('_', ' ').replace('is ', '').replace('has ', '')
                readable_names.append(readable)
            parts.append(f"Suspicious: {', '.join(readable_names)}")

        return " | ".join(parts)

    def _generate_metadata_explanation(self, score: float, suspicious_features: Dict[str, Any]) -> str:
        """Generate explanation for metadata-based prediction (legacy)."""
        if not suspicious_features:
            return "No suspicious metadata features detected"

        features_list = []
        for feature_name, feature_value in suspicious_features.items():
            if feature_name == "is_shortened_url":
                features_list.append("shortened URL")
            elif feature_name == "has_ip_address":
                features_list.append("IP address in URL")
            elif feature_name == "suspicious_tld":
                features_list.append("suspicious TLD")
            elif feature_name == "no_https":
                features_list.append("uses HTTP instead of HTTPS")
            elif feature_name == "high_url_entropy":
                features_list.append(f"high URL entropy ({feature_value:.2f})")
            elif feature_name == "sender_has_numbers":
                features_list.append("numbers in sender ID")
            elif feature_name == "sender_has_special_chars":
                features_list.append("special characters in sender")
            elif feature_name == "unusual_sender_length":
                features_list.append(f"unusual sender length ({feature_value})")
            elif feature_name == "suspicious_send_time":
                features_list.append(f"unusual send time ({feature_value})")
            elif feature_name == "invalid_mobile_length":
                features_list.append("invalid mobile number length")

        if features_list:
            return f"Suspicious metadata detected: {', '.join(features_list)}"
        else:
            return f"Metadata analysis: Score {score:.2f}"

    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Metadata model saved to {path}")
