"""
Image Analysis Pipeline - CNN for Phishing Image Detection with OCR
Extracts text from images and detects spam keywords (Pipeline 2)
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Tuple, List
import numpy as np
import logging

from config import IMAGE_MODEL_CONFIG, DEVICE, IMAGE_MODEL_PATH
from utils.preprocessing import detect_spam_keywords, extract_text_from_image_ocr

logger = logging.getLogger(__name__)


class ImageClassifier(nn.Module):
    """
    CNN-based image classifier for detecting phishing images
    (fake bank screenshots, UPI scams, fake login pages, etc.)
    """
    
    def __init__(
        self,
        architecture: str = IMAGE_MODEL_CONFIG["architecture"],
        num_classes: int = IMAGE_MODEL_CONFIG["num_classes"],
        pretrained: bool = IMAGE_MODEL_CONFIG["pretrained"]
    ):
        super(ImageClassifier, self).__init__()
        
        # Load pretrained model
        if architecture == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        elif architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Image tensor (batch_size, 3, H, W)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        return self.backbone(x)


class ImagePipeline:
    """
    Complete image analysis pipeline
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize image pipeline.

        Args:
            model_path: Path to saved model weights
        """
        self.device = torch.device(DEVICE)
        self.input_size = IMAGE_MODEL_CONFIG["input_size"]
        self.model_loaded = False

        self.model = ImageClassifier()

        model_path = model_path or IMAGE_MODEL_PATH
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"✓ Loaded trained image model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                logger.warning(f"✗ Could not load trained image model from {model_path}: {e}")
                logger.warning("Using pretrained MobileNetV2 base model (untrained classifier head)")
        elif model_path:
            logger.warning(f"✗ Image model file not found: {model_path}")
            logger.warning("Using pretrained MobileNetV2 base model (untrained classifier head)")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        """
        Predict phishing probability for image
        
        Args:
            image: PIL Image
            
        Returns:
            Probability score (0.0 - 1.0) where higher = more likely spam
        """
        if image is None:
            return 0.5  # Neutral score for missing image
        
        # Preprocess
        image_tensor = self.preprocess(image)
        
        # Forward pass
        logits = self.model(image_tensor)
        
        # Get probability for SPAM class (assuming index 1)
        probs = torch.softmax(logits, dim=1)
        spam_prob = probs[0, 1].item()
        
        return spam_prob
    
    @torch.no_grad()
    def predict_with_ocr(self, image: Image.Image) -> Tuple[float, str, List[str], str]:
        """
        Predict phishing probability with OCR text extraction (Pipeline 2)
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (score, extracted_text, detected_keywords, explanation)
        """
        if image is None:
            return 0.5, "", [], "No image provided"
        
        # Get image-based spam score
        image_score = self.predict(image)
        
        # Extract text from image using OCR
        extracted_text = extract_text_from_image_ocr(image)
        
        # Detect spam keywords in extracted text
        detected_keywords = detect_spam_keywords(extracted_text) if extracted_text else []
        
        # Generate explanation
        explanation = self._generate_image_explanation(image_score, extracted_text, detected_keywords)
        
        return image_score, extracted_text, detected_keywords, explanation
    
    def _generate_image_explanation(self, score: float, extracted_text: str, keywords: List[str]) -> str:
        """Generate explanation for image-based prediction."""
        reasons = []
        
        # Visual analysis score
        if score > 0.7:
            reasons.append("Suspicious image patterns detected")
        
        # Text-based analysis
        if extracted_text:
            if keywords:
                reasons.append(f"Extracted text contains suspicious keywords: {', '.join(keywords[:3])}")
            elif extracted_text.strip():
                reasons.append("Text extracted from image")
        else:
            reasons.append("No text extracted from image")
        
        if not reasons or (score <= 0.5 and not keywords):
            reasons = ["Image appears legitimate"]
        
        return " | ".join(reasons)
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Image model saved to {path}")