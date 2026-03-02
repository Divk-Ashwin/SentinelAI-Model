"""
Utility functions for SentinelAI
"""
import re
import math
import base64
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Any
from collections import Counter
import tldextract
from PIL import Image
import numpy as np

from config import (
    SHORTENED_URL_DOMAINS,
    SUSPICIOUS_TLDS,
    SUSPICIOUS_KEYWORDS
)


def extract_url_from_text(text: str) -> Optional[str]:
    """Extract URL from text message"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls[0] if urls else None


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string"""
    if not text:
        return 0.0
    
    counter = Counter(text)
    length = len(text)
    entropy = 0.0
    
    for count in counter.values():
        probability = count / length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


def is_shortened_url(url: str) -> bool:
    """Check if URL is from a URL shortening service"""
    if not url:
        return False
    
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    
    return domain in SHORTENED_URL_DOMAINS


def has_ip_address(url: str) -> bool:
    """Check if URL contains IP address instead of domain"""
    if not url:
        return False
    
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return bool(re.search(ip_pattern, url))


def has_suspicious_tld(url: str) -> bool:
    """Check if URL has suspicious top-level domain"""
    if not url:
        return False
    
    extracted = tldextract.extract(url)
    tld = f".{extracted.suffix}"
    
    return tld in SUSPICIOUS_TLDS


def extract_metadata_features(
    url: Optional[str] = None,
    sender: Optional[str] = None,
    timestamp: Optional[str] = None
) -> Dict[str, float]:
    """
    Extract metadata features for the FFNN model
    
    Returns:
        Dictionary with 15 features
    """
    features = {}
    
    # URL features
    if url:
        features["url_length"] = len(url)
        features["url_num_dots"] = url.count('.')
        features["url_num_digits"] = sum(c.isdigit() for c in url)
        features["url_num_special_chars"] = len(re.findall(r'[^a-zA-Z0-9]', url))
        features["url_entropy"] = calculate_entropy(url)
        features["is_shortened_url"] = float(is_shortened_url(url))
        features["has_ip_address"] = float(has_ip_address(url))
        features["has_at_symbol"] = float('@' in url)
        features["url_has_https"] = float(url.startswith('https://'))
        features["url_suspicious_tld"] = float(has_suspicious_tld(url))
    else:
        features.update({
            "url_length": 0.0,
            "url_num_dots": 0.0,
            "url_num_digits": 0.0,
            "url_num_special_chars": 0.0,
            "url_entropy": 0.0,
            "is_shortened_url": 0.0,
            "has_ip_address": 0.0,
            "has_at_symbol": 0.0,
            "url_has_https": 0.0,
            "url_suspicious_tld": 0.0
        })
    
    # Sender features
    if sender:
        features["sender_length"] = len(sender)
        features["sender_has_numbers"] = float(any(c.isdigit() for c in sender))
        features["sender_has_special_chars"] = float(bool(re.search(r'[^a-zA-Z0-9]', sender)))
    else:
        features.update({
            "sender_length": 0.0,
            "sender_has_numbers": 0.0,
            "sender_has_special_chars": 0.0
        })
    
    # Temporal features
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            features["hour_of_day"] = dt.hour / 23.0  # Normalize to 0-1
            features["is_weekend"] = float(dt.weekday() >= 5)
        except:
            features["hour_of_day"] = 0.5
            features["is_weekend"] = 0.0
    else:
        features["hour_of_day"] = 0.5
        features["is_weekend"] = 0.0
    
    return features


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def generate_explanation(
    text: Optional[str],
    metadata: Optional[Dict[str, Any]],
    scores: Dict[str, float]
) -> str:
    """
    Generate human-readable explanation for the prediction
    """
    reasons = []
    
    # Text-based reasons
    if text and scores["text"] > 0.7:
        text_lower = text.lower()
        found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
        if found_keywords:
            reasons.append(f"Urgency keywords detected: {', '.join(found_keywords[:3])}")
    
    # Metadata-based reasons
    if metadata:
        url = metadata.get("url")
        sender = metadata.get("sender")
        
        if url:
            if is_shortened_url(url):
                reasons.append("Shortened URL detected")
            if has_ip_address(url):
                reasons.append("IP address in URL")
            if has_suspicious_tld(url):
                reasons.append("Suspicious domain extension")
        
        if sender and scores["metadata"] > 0.7:
            if any(c.isdigit() for c in sender):
                reasons.append("Suspicious sender ID pattern")
    
    # Image-based reasons
    if scores["image"] > 0.7:
        reasons.append("Suspicious image content detected")
    
    if not reasons:
        if scores["text"] > 0.5 or scores["metadata"] > 0.5:
            reasons.append("Multiple weak signals combined")
        else:
            reasons.append("No clear phishing indicators found")
    
    return " + ".join(reasons)


def normalize_features(features: Dict[str, float]) -> np.ndarray:
    """
    Normalize features to 0-1 range
    Returns numpy array in fixed order
    """
    # Define reasonable max values for normalization
    max_values = {
        "url_length": 200.0,
        "url_num_dots": 10.0,
        "url_num_digits": 50.0,
        "url_num_special_chars": 30.0,
        "url_entropy": 5.0,
        "sender_length": 20.0,
    }
    
    normalized = []
    feature_order = [
        "url_length", "url_num_dots", "url_num_digits", "url_num_special_chars",
        "url_entropy", "is_shortened_url", "has_ip_address", "has_at_symbol",
        "sender_length", "sender_has_numbers", "sender_has_special_chars",
        "hour_of_day", "is_weekend", "url_has_https", "url_suspicious_tld"
    ]
    
    for feature in feature_order:
        value = features[feature]
        
        # Normalize if max value is defined, otherwise assume already 0-1
        if feature in max_values:
            value = min(value / max_values[feature], 1.0)
        
        normalized.append(value)
    
    return np.array(normalized, dtype=np.float32)