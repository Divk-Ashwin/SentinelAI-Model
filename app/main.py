"""
SentinelAI Backend - FastAPI Application
Multi-modal Phishing Detection System
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import traceback

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    TEXT_MODEL_PATH,
    IMAGE_MODEL_PATH,
    METADATA_MODEL_PATH,
    TEXT_TOKENIZER_PATH,
    DEFAULT_SCORES,
    LOG_LEVEL,
    LOG_FORMAT
)

from app.models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
    ModelScores,
    SpamIndicators
)

from models.text_pipeline import TextPipeline
from models.image_pipeline import ImagePipeline
from models.metadata_pipeline import MetadataPipeline
from fusion.decision_fusion import DecisionFusion
from utils.preprocessing import decode_base64_image, generate_explanation

# Configure logging
logger.remove()
logger.add(sys.stderr, format=LOG_FORMAT, level=LOG_LEVEL)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instances
text_pipeline: Optional[TextPipeline] = None
image_pipeline: Optional[ImagePipeline] = None
metadata_pipeline: Optional[MetadataPipeline] = None
fusion_module: Optional[DecisionFusion] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global text_pipeline, image_pipeline, metadata_pipeline, fusion_module
    
    logger.info("🚀 Starting SentinelAI Backend...")
    
    try:
        # Initialize text pipeline
        logger.info("Loading text model...")
        text_pipeline = TextPipeline(
            model_path=TEXT_MODEL_PATH if TEXT_MODEL_PATH.exists() else None,
            tokenizer_path=TEXT_TOKENIZER_PATH if TEXT_TOKENIZER_PATH.exists() else None
        )
        logger.success("✓ Text model loaded")
        
        # Initialize image pipeline
        logger.info("Loading image model...")
        image_pipeline = ImagePipeline(
            model_path=IMAGE_MODEL_PATH if IMAGE_MODEL_PATH.exists() else None
        )
        logger.success("✓ Image model loaded")
        
        # Initialize metadata pipeline
        logger.info("Loading metadata model...")
        metadata_pipeline = MetadataPipeline(
            model_path=METADATA_MODEL_PATH if METADATA_MODEL_PATH.exists() else None
        )
        logger.success("✓ Metadata model loaded")
        
        # Initialize fusion module
        fusion_module = DecisionFusion()
        logger.success("✓ Fusion module initialized")
        
        logger.success("🎉 All models loaded successfully!")
        logger.info(f"📊 Fusion weights: {fusion_module.get_config()}")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error(traceback.format_exc())
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down SentinelAI Backend...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "SentinelAI Phishing Detection API",
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "text_model": text_pipeline is not None,
            "image_model": image_pipeline is not None,
            "metadata_model": metadata_pipeline is not None,
            "fusion_module": fusion_module is not None
        },
        timestamp=datetime.now().isoformat()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Prediction"]
)
async def predict(request: PredictionRequest):
    """
    Multi-modal phishing detection endpoint
    
    Analyzes SMS text, image, and metadata to detect phishing attempts.
    Returns classification label, confidence score, and explanation.
    """
    try:
        logger.info("📨 New prediction request received")
        
        # Validate that at least one modality is provided
        if not any([request.text, request.image, request.metadata]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one of text, image, or metadata must be provided"
            )
        
        # Initialize scores with defaults
        text_score = DEFAULT_SCORES["text"]
        image_score = DEFAULT_SCORES["image"]
        metadata_score = DEFAULT_SCORES["metadata"]
        
        # Initialize spam indicators
        spam_indicators = SpamIndicators()
        
        # ==================== PIPELINE 1: TEXT ANALYSIS ====================
        if request.text and text_pipeline:
            logger.info("🔤 PIPELINE 1: Analyzing text...")
            try:
                text_score, text_keywords, text_explanation = text_pipeline.predict_with_keywords(request.text)
                spam_indicators.text_keywords = text_keywords
                spam_indicators.text_explanation = text_explanation
                logger.info(f"Text score: {text_score:.4f}")
                if text_keywords:
                    logger.info(f"Detected keywords: {text_keywords}")
            except Exception as e:
                logger.warning(f"Text analysis error: {e}")
                spam_indicators.text_explanation = "Text analysis error"
        
        # ==================== PIPELINE 2: IMAGE ANALYSIS ====================
        if request.image and image_pipeline:
            logger.info("🖼️  PIPELINE 2: Analyzing image...")
            try:
                # Decode base64 image
                image = decode_base64_image(request.image)
                image_score, extracted_text, image_keywords, image_explanation = image_pipeline.predict_with_ocr(image)
                spam_indicators.extracted_text = extracted_text
                spam_indicators.image_keywords = image_keywords
                spam_indicators.image_explanation = image_explanation
                logger.info(f"Image score: {image_score:.4f}")
                if extracted_text:
                    logger.info(f"Extracted text from image: {extracted_text[:50]}...")
                if image_keywords:
                    logger.info(f"Image keywords: {image_keywords}")
            except Exception as e:
                logger.warning(f"Image processing failed: {e}")
                spam_indicators.image_explanation = f"Image processing error: {str(e)}"
        
        # ==================== PIPELINE 3: METADATA ANALYSIS ====================
        if request.metadata and metadata_pipeline:
            logger.info("📊 PIPELINE 3: Analyzing metadata...")
            try:
                metadata_score, suspicious_features, metadata_explanation = metadata_pipeline.predict_with_indicators(
                    url=request.metadata.url,
                    sender=request.metadata.sender,
                    time=request.metadata.time,
                    date=request.metadata.date,
                    mobile_number=request.metadata.mobile_number,
                    timestamp=request.metadata.timestamp
                )
                spam_indicators.suspicious_features = suspicious_features
                spam_indicators.metadata_explanation = metadata_explanation
                logger.info(f"Metadata score: {metadata_score:.4f}")
                if suspicious_features:
                    logger.info(f"Suspicious features: {suspicious_features}")
            except Exception as e:
                logger.warning(f"Metadata analysis error: {e}")
                spam_indicators.metadata_explanation = f"Metadata analysis error: {str(e)}"
        
        # ==================== DECISION FUSION ====================
        logger.info("🔀 Performing decision fusion...")
        label, confidence, scores = fusion_module.fuse(
            text_score=text_score,
            metadata_score=metadata_score,
            image_score=image_score
        )
        
        # Generate recommendation
        if label == "SPAM":
            if confidence > 0.9:
                recommendation = "Block - Very high confidence phishing attempt"
            elif confidence > 0.7:
                recommendation = "Block - High confidence phishing attempt"
            else:
                recommendation = "Warn user - Suspected phishing, requires review"
        else:
            if confidence > 0.9:
                recommendation = "Allow - Very high confidence legitimate message"
            else:
                recommendation = "Allow - Likely legitimate message"
        
        # Generate detailed reason
        reason_parts = []
        if spam_indicators.text_keywords:
            reason_parts.append(f"Text: {', '.join(spam_indicators.text_keywords[:3])}")
        if spam_indicators.suspicious_features:
            feature_names = list(spam_indicators.suspicious_features.keys())[:3]
            reason_parts.append(f"Metadata: {', '.join(feature_names)}")
        if spam_indicators.image_keywords:
            reason_parts.append(f"Image: {', '.join(spam_indicators.image_keywords[:2])}")
        
        reason = " | ".join(reason_parts) if reason_parts else "Multiple weak signals combined"
        
        logger.success(f"✅ Prediction: {label} (confidence: {confidence:.4f})")
        logger.success(f"Recommendation: {recommendation}")
        logger.info(f"Reason: {reason}")
        
        # Prepare response with detailed spam indicators
        response = PredictionResponse(
            label=label,
            confidence=confidence,
            scores=ModelScores(**scores),
            spam_indicators=spam_indicators,
            reason=reason,
            recommendation=recommendation
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "details": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"exception": str(exc)}
        }
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting SentinelAI Backend Server...")
    logger.info("📝 API Documentation: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )