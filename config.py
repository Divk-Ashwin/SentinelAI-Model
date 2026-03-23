"""
Configuration file for SentinelAI Backend
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "saved_models"
DATASETS_DIR = BASE_DIR / "datasets"

# API Configuration
API_TITLE = "SentinelAI Phishing Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Multi-modal ML system for real-time phishing/smishing detection"

# Model paths
TEXT_MODEL_PATH = MODELS_DIR / "text_model.pth"
IMAGE_MODEL_PATH = MODELS_DIR / "image_model.pth"
METADATA_MODEL_PATH = MODELS_DIR / "metadata_model.pth"
TEXT_TOKENIZER_PATH = MODELS_DIR / "text_tokenizer"
METADATA_SCALER_PATH = MODELS_DIR / "metadata_scaler.pkl"

# Model configurations
TEXT_MODEL_CONFIG = {
    "model_name": "xlm-roberta-base",  # Multilingual BERT
    "max_length": 256,
    "num_labels": 2,
    "dropout": 0.1
}

IMAGE_MODEL_CONFIG = {
    "architecture": "mobilenet_v2",
    "num_classes": 2,
    "pretrained": True,
    "input_size": (224, 224)
}

METADATA_MODEL_CONFIG = {
    "input_dim": 15,  # Number of engineered features
    "hidden_dims": [64, 32, 16],
    "output_dim": 2,
    "dropout": 0.3
}

# Fusion weights (base weights - will be redistributed if modalities missing)
FUSION_WEIGHTS = {
    "text": 0.45,
    "metadata": 0.40,
    "image": 0.15
}

# Decision threshold (configurable)
SPAM_THRESHOLD = 0.50  # Lowered threshold to catch more suspicious messages

# Confidence thresholds for fusion output
CONFIDENCE_THRESHOLDS = {
    "high_upper": 0.75,   # Score > 0.75 = HIGH confidence spam
    "high_lower": 0.25,   # Score < 0.25 = HIGH confidence ham
    "medium_upper": 0.60, # Score 0.40-0.75 or 0.25-0.40 = MEDIUM
    "medium_lower": 0.40,
}

# Default scores - DEPRECATED (missing modalities now excluded from fusion)
DEFAULT_SCORES = {
    "text": 0.5,
    "image": 0.5,
    "metadata": 0.5
}

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
TRAINING_CONFIG = {
    "epochs": 20,
    "batch_size": 8,
    "learning_rate": 3e-5,
    "max_sequence_length": 128,
    "validation_split": 0.2,
    "random_seed": 42,
    "warmup_steps": 100,
    "early_stopping_patience": 4
}

# Feature extraction settings
METADATA_FEATURES = [
    "url_length",
    "url_num_dots",
    "url_num_digits",
    "url_num_special_chars",
    "url_entropy",
    "is_shortened_url",
    "has_ip_address",
    "has_at_symbol",
    "sender_length",
    "sender_has_numbers",
    "sender_has_special_chars",
    "hour_of_day",
    "is_weekend",
    "url_has_https",
    "url_suspicious_tld"
]

# Suspicious patterns
SUSPICIOUS_KEYWORDS = [
    "verify", "urgent", "suspend", "confirm", "account", "update",
    "click", "prize", "winner", "congratulations", "blocked",
    "expire", "immediately", "act now", "limited time", "offer"
]

SHORTENED_URL_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "buff.ly", "is.gd", "cli.gs", "cutt.ly", "shorturl.at"
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".work", ".date", ".racing", ".download"
]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

# LLM Configuration for /justify endpoint (Groq API)
LLM_CONFIG = {
    "provider": "groq",
    "groq_api_key": os.getenv("GROQ_API_KEY"),  # Set in .env file
    "groq_model": "llama-3.1-8b-instant",
    "groq_api_url": "https://api.groq.com/openai/v1/chat/completions",
    "max_tokens": 150,
    "temperature": 0.3,
}

# System prompt for justification generation
JUSTIFY_SYSTEM_PROMPT = """You are a cybersecurity assistant. Based on the phishing detection analysis provided, give a clear 2-3 sentence explanation to a non-technical user about why this message was flagged or cleared. Mention specific words or URL patterns that were suspicious. Be factual, not alarmist. Respond in plain English only. IMPORTANT: Only mention URLs, words, or features 
that are explicitly listed in the analysis data.Never invent or assume URLs, keywords, or sender 
names that are not provided. If contributing_words 
is empty, say 'our text analysis flagged suspicious 
patterns' without mentioning specific words."""