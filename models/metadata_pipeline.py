import os
import pickle
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
import logging

from config import METADATA_MODEL_CONFIG, DEVICE, METADATA_FEATURES, METADATA_MODEL_PATH, METADATA_SCALER_PATH
from utils.preprocessing import extract_metadata_features, detect_spam_in_metadata

logger = logging.getLogger(__name__)


class MetadataClassifier(nn.Module):
    """
    Feed Forward Neural Network for metadata-based phishing detection
    Analyzes URLs, sender IDs, and temporal patterns
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
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Feature tensor (batch_size, input_dim)
            
        Returns:
            Logits (batch_size, output_dim)
        """
        return self.network(x)


class MetadataPipeline:
    """
    Complete metadata analysis pipeline
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize metadata pipeline
        
        Args:
            model_path: Path to saved model weights
        """
        self.device = torch.device(DEVICE)
        self.model_loaded = False
        self.scaler = None
        
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
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"✓ Loaded trained metadata model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                logger.warning(f"✗ Could not load trained metadata model from {model_path}: {e}")
                logger.warning("Using untrained model (random weights)")
        elif model_path:
            logger.warning(f"✗ Metadata model file not found: {model_path}")
            logger.warning("Using untrained model (random weights)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(
        self,
        url: Optional[str] = None,
        sender: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract and preprocess metadata features
        
        Args:
            url: URL from the message
            sender: Sender ID
            timestamp: Message timestamp
            
        Returns:
            Feature tensor
        """
        # Extract features
        features = extract_metadata_features(url, sender, timestamp)
        
        # Convert to array in correct order
        features_array = np.array([features[feature_name] for feature_name in METADATA_FEATURES], dtype=np.float32)
        features_array = features_array.reshape(1, -1)
        
        # Normalize features if scaler is available
        if self.scaler is not None:
            try:
                features_array = self.scaler.transform(features_array)
            except Exception as e:
                logger.warning(f"Could not normalize features: {e}")
        
        # Convert to tensor
        features_tensor = torch.tensor(features_array, dtype=torch.float32)
        
        return features_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        url: Optional[str] = None,
        sender: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> float:
        """
        Predict phishing probability from metadata
        
        Args:
            url: URL from the message
            sender: Sender ID
            timestamp: Message timestamp
            
        Returns:
            Probability score (0.0 - 1.0) where higher = more likely spam
        """
        # If no metadata provided, return neutral score
        if not any([url, sender, timestamp]):
            return 0.5
        
        # Preprocess
        features = self.preprocess(url, sender, timestamp)
        
        # Forward pass
        logits = self.model(features)
        
        # Get probability for SPAM class (assuming index 1)
        probs = torch.softmax(logits, dim=1)
        spam_prob = probs[0, 1].item()
        
        return spam_prob
    
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
        Predict phishing probability with detailed suspicious features (Pipeline 3)
        
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
        # Get metadata-based spam score
        metadata_score = self.predict(url, sender, timestamp)
        
        # Detect suspicious features in metadata
        suspicious_features = detect_spam_in_metadata(
            url=url,
            sender=sender,
            time=time,
            date=date,
            mobile_number=mobile_number
        )
        
        # Generate explanation
        explanation = self._generate_metadata_explanation(metadata_score, suspicious_features)
        
        return metadata_score, suspicious_features, explanation
    
    def _generate_metadata_explanation(self, score: float, suspicious_features: Dict[str, Any]) -> str:
        """Generate explanation for metadata-based prediction."""
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
                features_list.append(f"invalid mobile number length")
        
        if features_list:
            return f"Suspicious metadata detected: {', '.join(features_list)}"
        else:
            return "Metadata analysis: Score {:.2f}".format(score)
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Metadata model saved to {path}")