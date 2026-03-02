"""
Pydantic models for API requests and responses.
"""
import re
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class MetadataInput(BaseModel):
    """Metadata information from SMS - Pipeline 3 Parameters."""
    # Time parameters
    time: Optional[str] = Field(None, description="Time message was sent (HH:MM:SS format)")
    date: Optional[str] = Field(None, description="Date message was sent (YYYY-MM-DD format)")
    timestamp: Optional[str] = Field(None, description="Full timestamp (ISO format, auto-generated if not provided)")
    
    # Sender information
    sender: Optional[str] = Field(None, description="Sender ID or phone number")
    mobile_number: Optional[str] = Field(None, description="Receiver's mobile number")
    
    # URL information
    url: Optional[str] = Field(None, description="URL extracted from message")

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp format."""
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError:
                raise ValueError("Timestamp must be in ISO format (YYYY-MM-DD HH:MM:SS)")
        return v
    
    @field_validator('time')
    @classmethod
    def validate_time(cls, v):
        """Validate time format."""
        if v is not None:
            try:
                datetime.strptime(v, '%H:%M:%S')
            except ValueError:
                raise ValueError("Time must be in HH:MM:SS format")
        return v
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        """Validate date format."""
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v
    
    @field_validator('mobile_number')
    @classmethod
    def validate_mobile(cls, v):
        """Validate mobile number format."""
        if v is not None:
            # Remove common separators
            clean = re.sub(r'[^\d+]', '', v)
            if len(clean) < 10:
                raise ValueError("Mobile number must have at least 10 digits")
        return v


class PredictionRequest(BaseModel):
    """Request model for /predict endpoint."""
    text: Optional[str] = Field(None, description="SMS text content")
    image: Optional[str] = Field(None, description="Base64-encoded image (optional)")
    metadata: Optional[MetadataInput] = Field(None, description="Message metadata")

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """Ensure at least some content is provided."""
        if v is not None and len(v.strip()) == 0:
            return None
        return v

    class Config:
        schema_extra = {
            "example": {
                "text": "Your account has been suspended. Click here to verify: bit.ly/verify123",
                "image": None,
                "metadata": {
                    "url": "https://bit.ly/verify123",
                    "sender": "VK-BANK",
                    "timestamp": "2026-01-20 14:30:00"
                }
            }
        }


class ModelScores(BaseModel):
    """Individual model prediction scores with detected spam indicators."""
    text: float = Field(..., ge=0.0, le=1.0, description="Text model score (Pipeline 1)")
    image: float = Field(..., ge=0.0, le=1.0, description="Image model score (Pipeline 2)")
    metadata: float = Field(..., ge=0.0, le=1.0, description="Metadata model score (Pipeline 3)")


class SpamIndicators(BaseModel):
    """Spam detection indicators from each pipeline."""
    # Pipeline 1: Text pipeline indicators
    text_keywords: List[str] = Field(default_factory=list, description="Suspicious keywords detected in text")
    text_explanation: Optional[str] = Field(None, description="Text pipeline explanation")
    
    # Pipeline 2: Image pipeline indicators
    extracted_text: Optional[str] = Field(None, description="Text extracted from image (OCR)")
    image_keywords: List[str] = Field(default_factory=list, description="Suspicious keywords detected in image text")
    image_explanation: Optional[str] = Field(None, description="Image pipeline explanation")
    
    # Pipeline 3: Metadata pipeline indicators
    suspicious_features: Dict[str, Any] = Field(default_factory=dict, description="Suspicious metadata features detected")
    metadata_explanation: Optional[str] = Field(None, description="Metadata pipeline explanation")


class PredictionResponse(BaseModel):
    """Response model for /predict endpoint with detailed indicators."""
    label: str = Field(..., description="Classification label: SPAM or HAM")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score (0.0-1.0)")
    scores: ModelScores = Field(..., description="Individual model scores")
    
    # Detailed spam indicators
    spam_indicators: SpamIndicators = Field(..., description="Detailed spam detection indicators from each pipeline")
    
    # Overall explanation
    reason: str = Field(..., description="Human-readable explanation of the decision")
    recommendation: str = Field(..., description="Action recommendation (e.g., 'Block', 'Warn', 'Allow')")

    class Config:
        schema_extra = {
            "example": {
                "label": "SPAM",
                "confidence": 0.89,
                "scores": {
                    "text": 0.87,
                    "image": 0.74,
                    "metadata": 0.91
                },
                "spam_indicators": {
                    "text_keywords": ["verify", "urgent", "account", "click"],
                    "text_explanation": "Multiple urgency and verification keywords detected",
                    "extracted_text": "Verify your account NOW!",
                    "image_keywords": ["verify", "urgent"],
                    "image_explanation": "Extracted text contains suspicious keywords",
                    "suspicious_features": {
                        "is_shortened_url": True,
                        "url_entropy": 4.2,
                        "suspicious_sender": True
                    },
                    "metadata_explanation": "Shortened URL detected with high entropy score"
                },
                "reason": "Urgency keywords + shortened URL + suspicious sender detected across multiple modalities",
                "recommendation": "Block - High confidence phishing attempt"
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str = Field(..., description="Service health status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
