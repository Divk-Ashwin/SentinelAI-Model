"""
Text Analysis Pipeline - Multilingual BERT for SMS Classification
Returns detected spam keywords and explanations
"""
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Optional, Tuple, List
import numpy as np
import logging

from config import TEXT_MODEL_CONFIG, DEVICE, TEXT_TOKENIZER_PATH, TEXT_MODEL_PATH
from utils.preprocessing import detect_spam_keywords

logger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    """
    Multilingual BERT-based text classifier for phishing detection
    Supports English, Hindi, Hinglish, Telugu, and other languages
    """
    
    def __init__(
        self,
        model_name: str = TEXT_MODEL_CONFIG["model_name"],
        num_labels: int = TEXT_MODEL_CONFIG["num_labels"],
        dropout: float = TEXT_MODEL_CONFIG["dropout"]
    ):
        super(TextClassifier, self).__init__()
        
        # Load pretrained multilingual BERT
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_labels)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class TextPipeline:
    """
    Complete text analysis pipeline
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ):
        """
        Initialize text pipeline
        
        Args:
            model_path: Path to saved model weights
            tokenizer_path: Path to saved tokenizer
        """
        self.device = torch.device(DEVICE)
        self.max_length = TEXT_MODEL_CONFIG["max_length"]
        self.model_loaded = False
        
        # Determine tokenizer path
        if tokenizer_path is None:
            tokenizer_path = TEXT_TOKENIZER_PATH if TEXT_TOKENIZER_PATH else TEXT_MODEL_CONFIG["model_name"]
        
        # Load tokenizer (trained or pretrained)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {tokenizer_path}: {e}")
            logger.warning(f"Falling back to pretrained model: {TEXT_MODEL_CONFIG['model_name']}")
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_CONFIG["model_name"])
        
        # Initialize model
        self.model = TextClassifier()
        
        # Load trained weights if available
        model_path = model_path or TEXT_MODEL_PATH
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"✓ Loaded trained text model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                logger.warning(f"✗ Could not load trained text model from {model_path}: {e}")
                logger.warning("Using pretrained XLM-RoBERTa base model (untrained classifier head)")
        elif model_path:
            logger.warning(f"✗ Text model file not found: {model_path}")
            logger.warning("Using pretrained XLM-RoBERTa base model (untrained classifier head)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text input
        
        Args:
            text: Input SMS text
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    @torch.no_grad()
    def predict(self, text: str) -> float:
        """
        Predict phishing probability for text
        
        Args:
            text: Input SMS text
            
        Returns:
            Probability score (0.0 - 1.0) where higher = more likely spam
        """
        if not text or not text.strip():
            return 0.5  # Neutral score for empty text
        
        # Preprocess
        inputs = self.preprocess(text)
        
        # Forward pass
        logits = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Get probability for SPAM class (assuming index 1)
        probs = torch.softmax(logits, dim=1)
        spam_prob = probs[0, 1].item()
        
        return spam_prob
    
    @torch.no_grad()
    def predict_with_keywords(self, text: str) -> Tuple[float, List[str], str]:
        """
        Predict phishing probability and return detected keywords (Pipeline 1)
        
        Args:
            text: Input SMS text
            
        Returns:
            Tuple of (score, detected_keywords_list, explanation)
        """
        if not text or not text.strip():
            return 0.5, [], "Text is empty"
        
        # Get spam probability
        spam_score = self.predict(text)
        
        # Detect suspicious keywords
        detected_keywords = detect_spam_keywords(text)
        
        # Generate explanation
        explanation = self._generate_text_explanation(text, spam_score, detected_keywords)
        
        return spam_score, detected_keywords, explanation
    
    def _generate_text_explanation(self, text: str, score: float, keywords: List[str]) -> str:
        """Generate explanation for text-based prediction."""
        if score > 0.7:
            if keywords:
                return f"High-risk spam detected. Suspicious keywords: {', '.join(keywords[:5])}"
            else:
                return "High-risk patterns detected in text (model-based)"
        elif score > 0.5:
            if keywords:
                return f"Moderate spam probability. Keywords present: {', '.join(keywords[:3])}"
            else:
                return "Moderate-risk patterns detected"
        else:
            return "Text appears legitimate"
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")