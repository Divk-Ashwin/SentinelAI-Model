"""
API Data Models using Pydantic
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime


class MetadataInput(BaseModel):
    """Metadata information from SMS"""
    url: Optional[str] = Field(None, description="URL extracted from message")
    sender: Optional[str] = Field(None, description="Sender ID or phone number")
    timestamp: Optional[str] = Field(None, description="Message timestamp (ISO format)")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError:
                raise ValueError("Timestamp must be in ISO format (YYYY-MM-DD HH:MM:SS)")
        return v


class PredictionRequest(BaseModel):
    """Request model for /predict endpoint"""
    text: Optional[str] = Field(None, description="SMS text content")
    image: Optional[str] = Field(None, description="Base64-encoded image (optional)")
    metadata: Optional[MetadataInput] = Field(None, description="Message metadata")
    
    @validator('text')
    def validate_text(cls, v):
        """Ensure at least some content is provided"""
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
    """Individual model prediction scores"""
    text: float = Field(..., ge=0.0, le=1.0, description="Text model score")
    image: float = Field(..., ge=0.0, le=1.0, description="Image model score")
    metadata: float = Field(..., ge=0.0, le=1.0, description="Metadata model score")


class PredictionResponse(BaseModel):
    """Response model for /predict endpoint"""
    label: str = Field(..., description="Classification label: SPAM or HAM")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    scores: ModelScores = Field(..., description="Individual model scores")
    reason: str = Field(..., description="Human-readable explanation")
    
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
                "reason": "Urgency keywords + shortened URL + suspicious sender"
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str = Field(..., description="Service health status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")