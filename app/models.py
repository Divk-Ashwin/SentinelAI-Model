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
        json_schema_extra = {
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


class PipelineScores(BaseModel):
    """Individual pipeline scores - null if modality was not provided."""
    text_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Text model score (Pipeline 1)")
    metadata_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Metadata model score (Pipeline 3)")
    image_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Image model score (Pipeline 2)")


class ContributingWord(BaseModel):
    """A token and its attention-based contribution score."""
    word: str = Field(..., description="Token/word from text")
    score: float = Field(..., ge=0.0, le=1.0, description="Attention-based contribution score")


class ContributingFeature(BaseModel):
    """A metadata feature and its importance score."""
    feature: str = Field(..., description="Feature name")
    score: float = Field(..., ge=0.0, le=1.0, description="Weight-based importance score")


class Explainability(BaseModel):
    """Explainability information from all pipelines."""
    contributing_words: List[ContributingWord] = Field(
        default_factory=list,
        description="Top contributing tokens from text analysis (empty if no text provided)"
    )
    contributing_features: List[ContributingFeature] = Field(
        default_factory=list,
        description="Top contributing metadata features (empty if no metadata provided)"
    )
    ocr_extracted_text: Optional[str] = Field(
        None,
        description="Text extracted from image via OCR (null if no image or OCR failed)"
    )


class FusionWeightsUsed(BaseModel):
    """Actual fusion weights applied after dynamic redistribution."""
    text: float = Field(..., ge=0.0, le=1.0, description="Weight applied to text score")
    metadata: float = Field(..., ge=0.0, le=1.0, description="Weight applied to metadata score")
    image: float = Field(..., ge=0.0, le=1.0, description="Weight applied to image score")


class PredictionResponse(BaseModel):
    """Response model for /predict endpoint with explainability."""
    final_score: float = Field(..., ge=0.0, le=1.0, description="Fused probability score (0.0-1.0)")
    decision: str = Field(..., description="Classification decision: SPAM or HAM")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, or LOW")
    pipeline_scores: PipelineScores = Field(..., description="Individual pipeline scores (null if not provided)")
    explainability: Explainability = Field(..., description="Explainability information")
    fusion_weights_used: FusionWeightsUsed = Field(..., description="Dynamic weights actually applied")

    class Config:
        json_schema_extra = {
            "example": {
                "final_score": 0.82,
                "decision": "SPAM",
                "confidence": "HIGH",
                "pipeline_scores": {
                    "text_score": 0.91,
                    "metadata_score": 0.76,
                    "image_score": None
                },
                "explainability": {
                    "contributing_words": [
                        {"word": "verify", "score": 0.87},
                        {"word": "account", "score": 0.76},
                        {"word": "urgent", "score": 0.71}
                    ],
                    "contributing_features": [
                        {"feature": "url_length", "score": 0.91},
                        {"feature": "has_ip_address", "score": 0.85}
                    ],
                    "ocr_extracted_text": None
                },
                "fusion_weights_used": {
                    "text": 0.625,
                    "metadata": 0.375,
                    "image": 0.0
                }
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


class JustifyRequest(BaseModel):
    """Request model for /justify endpoint - accepts full prediction output."""
    final_score: float = Field(..., ge=0.0, le=1.0, description="Fused probability score")
    decision: str = Field(..., description="Classification decision: SPAM or HAM")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, or LOW")
    pipeline_scores: PipelineScores = Field(..., description="Individual pipeline scores")
    explainability: Explainability = Field(..., description="Explainability information")
    fusion_weights_used: FusionWeightsUsed = Field(..., description="Dynamic weights applied")

    class Config:
        json_schema_extra = {
            "example": {
                "final_score": 0.82,
                "decision": "SPAM",
                "confidence": "HIGH",
                "pipeline_scores": {
                    "text_score": 0.91,
                    "metadata_score": 0.76,
                    "image_score": None
                },
                "explainability": {
                    "contributing_words": [
                        {"word": "verify", "score": 0.87},
                        {"word": "account", "score": 0.76}
                    ],
                    "contributing_features": [
                        {"feature": "url_length", "score": 0.91}
                    ],
                    "ocr_extracted_text": None
                },
                "fusion_weights_used": {
                    "text": 0.625,
                    "metadata": 0.375,
                    "image": 0.0
                }
            }
        }


class JustifyResponse(BaseModel):
    """Response model for /justify endpoint."""
    justification: str = Field(..., description="Human-readable explanation of the detection result")

    class Config:
        json_schema_extra = {
            "example": {
                "justification": "This message was flagged because it contains urgent language like 'verify' and 'account', and the link has an unusually long URL with suspicious patterns. These are common signs of phishing."
            }
        }
