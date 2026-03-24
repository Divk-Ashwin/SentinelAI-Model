"""
SentinelAI Backend - FastAPI Application
Multi-modal Phishing Detection System
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
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
    LOG_LEVEL,
    LOG_FORMAT,
    LLM_CONFIG,
    JUSTIFY_SYSTEM_PROMPT
)

from app.models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
    PipelineScores,
    Explainability,
    FusionWeightsUsed,
    ContributingWord,
    ContributingFeature,
    JustifyRequest,
    JustifyResponse
)

from models.text_pipeline import TextPipeline
from models.image_pipeline import ImagePipeline
from models.metadata_pipeline import MetadataPipeline
from fusion.decision_fusion import DecisionFusion

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
        
        # Initialize image pipeline (OCR-based, no model needed)
        logger.info("Loading image pipeline...")
        image_pipeline = ImagePipeline(text_pipeline=text_pipeline)
        logger.success("✓ Image pipeline loaded")
        
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
    Multi-modal phishing detection endpoint.

    Analyzes SMS text, image (OCR), and metadata to detect phishing attempts.
    Returns final score, decision, confidence level, and explainability info.

    Missing modalities are excluded from fusion (not defaulted to 0.5).
    Weights are dynamically redistributed among available modalities.
    """
    try:
        logger.info("📨 New prediction request received")

        # Validate that at least one modality is provided
        if not any([request.text, request.image, request.metadata]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one of text, image, or metadata must be provided"
            )

        # Initialize scores as None (will remain None if modality not provided)
        text_score: Optional[float] = None
        metadata_score: Optional[float] = None
        image_score: Optional[float] = None
        url_text_score: Optional[float] = None

        # Initialize explainability components
        contributing_words: List[ContributingWord] = []
        contributing_features: List[ContributingFeature] = []
        ocr_extracted_text: Optional[str] = None

        # ==================== PIPELINE 1: TEXT ANALYSIS ====================
        if request.text and text_pipeline:
            logger.info("🔤 PIPELINE 1: Analyzing text...")
            try:
                text_result = text_pipeline.predict_with_explanation(request.text)
                text_score = text_result.score

                # Extract contributing words with scores
                contributing_words = [
                    ContributingWord(word=tc.word, score=tc.score)
                    for tc in text_result.contributing_words
                ]

                logger.info(f"Text score: {text_score:.4f}")
                if contributing_words:
                    logger.info(f"Top tokens: {[w.word for w in contributing_words[:3]]}")
            except Exception as e:
                logger.warning(f"Text analysis error: {e}")

        # ==================== PIPELINE 2: IMAGE ANALYSIS (OCR) ====================
        if request.image and image_pipeline:
            logger.info("🖼️  PIPELINE 2: Analyzing image (OCR)...")
            try:
                image_result = image_pipeline.analyze(request.image)
                image_score = image_result.score  # None if OCR failed or text too short
                ocr_extracted_text = image_result.ocr_text

                if image_score is not None:
                    logger.info(f"Image score: {image_score:.4f}")
                else:
                    logger.info(f"Image: {image_result.explanation}")
                if ocr_extracted_text:
                    logger.info(f"OCR text: {ocr_extracted_text[:50]}...")
            except Exception as e:
                logger.warning(f"Image processing failed: {e}")

        # ==================== PIPELINE 3: METADATA ANALYSIS ====================
        if request.metadata and metadata_pipeline:
            logger.info("📊 PIPELINE 3: Analyzing metadata...")
            try:
                metadata_result = metadata_pipeline.predict_with_explanation(
                    url=request.metadata.url,
                    sender=request.metadata.sender,
                    timestamp=request.metadata.timestamp,
                    time=request.metadata.time,
                    date=request.metadata.date,
                    mobile_number=request.metadata.mobile_number
                )

                # metadata_result will be None if no URL provided
                if metadata_result is not None:
                    metadata_score = metadata_result.score
                    url_text_score = metadata_result.url_text_score

                    # Extract contributing features with scores
                    contributing_features = [
                        ContributingFeature(feature=fc.feature, score=fc.score)
                        for fc in metadata_result.contributing_features
                    ]

                    logger.info(f"Metadata score: {metadata_score:.4f}")
                    if url_text_score is not None:
                        logger.info(f"URL text risk score: {url_text_score:.4f}")
                    if contributing_features:
                        logger.info(f"Top features: {[f.feature for f in contributing_features[:3]]}")
                else:
                    logger.info("Metadata: No URL provided, skipping metadata analysis")
            except Exception as e:
                logger.warning(f"Metadata analysis error: {e}")

        # ==================== DECISION FUSION ====================
        logger.info("🔀 Performing decision fusion...")

        # Extract url and sender for trust checking (optional parameters)
        url = request.metadata.url if request.metadata else None
        sender = request.metadata.sender if request.metadata else None

        # Fusion with None for missing modalities (dynamic weight redistribution)
        fusion_result = fusion_module.fuse(
            text_score=text_score,
            metadata_score=metadata_score,
            image_score=image_score,
            url_text_score=url_text_score,
            url=url,
            sender=sender
        )

        # Extract fusion weights (0.0 for modalities not used)
        weights_used = {
            "text": fusion_result.weights_applied.get("text", 0.0),
            "metadata": fusion_result.weights_applied.get("metadata", 0.0),
            "image": fusion_result.weights_applied.get("image", 0.0)
        }

        logger.success(f"✅ Decision: {fusion_result.label} (score: {fusion_result.score:.4f}, confidence: {fusion_result.confidence.value})")
        logger.info(f"Modalities used: {fusion_result.modalities_used}")
        logger.info(f"Weights applied: {weights_used}")

        # Build response
        response = PredictionResponse(
            final_score=fusion_result.score,
            decision=fusion_result.label,
            confidence=fusion_result.confidence.value,
            pipeline_scores=PipelineScores(
                text_score=text_score,
                metadata_score=metadata_score,
                image_score=image_score
            ),
            explainability=Explainability(
                contributing_words=contributing_words,
                contributing_features=contributing_features,
                ocr_extracted_text=ocr_extracted_text
            ),
            fusion_weights_used=FusionWeightsUsed(
                text=weights_used["text"],
                metadata=weights_used["metadata"],
                image=weights_used["image"]
            )
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


def build_justify_user_prompt(request: JustifyRequest) -> str:
    """
    Build user prompt from prediction JSON for LLM justification.

    Extracts key information: final_score, decision, contributing_words,
    contributing_features, and ocr_extracted_text.
    """
    parts = []

    # Decision summary
    parts.append(f"Detection Result: {request.decision} (confidence: {request.confidence})")
    parts.append(f"Risk Score: {request.final_score:.2f} out of 1.0")

    # Contributing words from text analysis
    if request.explainability.contributing_words:
        words = [f"'{w.word}' (importance: {w.score:.2f})"
                 for w in request.explainability.contributing_words[:5]]
        parts.append(f"Suspicious words detected: {', '.join(words)}")

    # Contributing features from metadata analysis
    if request.explainability.contributing_features:
        features = [f"{f.feature} (importance: {f.score:.2f})"
                    for f in request.explainability.contributing_features[:5]]
        parts.append(f"Suspicious metadata patterns: {', '.join(features)}")

    # OCR extracted text from image
    if request.explainability.ocr_extracted_text:
        ocr_text = request.explainability.ocr_extracted_text
        if len(ocr_text) > 200:
            ocr_text = ocr_text[:200] + "..."
        parts.append(f"Text extracted from image: \"{ocr_text}\"")

    # Pipeline scores context
    scores_info = []
    if request.pipeline_scores.text_score is not None:
        scores_info.append(f"text={request.pipeline_scores.text_score:.2f}")
    if request.pipeline_scores.metadata_score is not None:
        scores_info.append(f"metadata={request.pipeline_scores.metadata_score:.2f}")
    if request.pipeline_scores.image_score is not None:
        scores_info.append(f"image={request.pipeline_scores.image_score:.2f}")
    if scores_info:
        parts.append(f"Individual pipeline scores: {', '.join(scores_info)}")

    return "\n".join(parts)


async def call_groq_api(system_prompt: str, user_prompt: str) -> str:
    """
    Call Groq API for justification generation.

    Uses Llama 3 8B model via Groq's OpenAI-compatible endpoint.
    Includes fallback logic if API call fails.
    """
    import httpx

    api_key = LLM_CONFIG["groq_api_key"]
    api_url = LLM_CONFIG["groq_api_url"]
    model = LLM_CONFIG["groq_model"]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": LLM_CONFIG["max_tokens"],
                    "temperature": LLM_CONFIG["temperature"]
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logger.warning(f"Groq API call failed: {e}")
        # Return None to trigger fallback in generate_justification
        return None


def get_fallback_justification(decision: str, score: float) -> str:
    """
    Generate a fallback justification when LLM API fails.

    Args:
        decision: "SPAM" or "HAM"
        score: The final risk score

    Returns:
        A hardcoded but contextual justification string
    """
    if decision == "SPAM":
        if score > 0.8:
            return "This message was flagged as highly suspicious. It contains multiple indicators commonly associated with phishing attempts, such as urgent language, suspicious links, or unusual sender patterns. We recommend not clicking any links and deleting this message."
        elif score > 0.6:
            return "This message shows several warning signs of a potential phishing attempt. The combination of suspicious keywords and metadata patterns triggered our detection system. Exercise caution before taking any action."
        else:
            return "This message has been flagged as potentially suspicious. While the risk is moderate, we recommend verifying the sender through official channels before responding or clicking any links."
    else:
        if score < 0.3:
            return "This message appears to be legitimate. Our analysis found no significant indicators of phishing or spam. The sender and content patterns match expected communication norms."
        else:
            return "This message appears to be safe, though it contains some patterns that occasionally appear in spam. The overall risk is low, but always verify unexpected requests through official channels."


async def generate_justification(system_prompt: str, user_prompt: str, decision: str = "SPAM", score: float = 0.5) -> str:
    """
    Generate justification using Groq API with fallback.

    Args:
        system_prompt: The system prompt for the LLM
        user_prompt: The user prompt with detection details
        decision: The detection decision for fallback
        score: The final score for fallback

    Returns:
        LLM-generated justification or fallback string
    """
    # Try Groq API
    result = await call_groq_api(system_prompt, user_prompt)

    if result:
        return result

    # Fallback if API fails
    logger.warning("Using fallback justification due to API failure")
    return get_fallback_justification(decision, score)


@app.post(
    "/justify",
    response_model=JustifyResponse,
    responses={
        200: {"model": JustifyResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Justification"]
)
async def justify(request: JustifyRequest):
    """
    Generate human-readable justification for a phishing detection result.

    Accepts the full JSON output from /predict and uses Groq's Llama 3 model
    to generate a clear 2-3 sentence explanation for non-technical users.

    If the Groq API fails, returns a contextual fallback justification.
    """
    try:
        logger.info("📝 Justification request received")
        logger.info(f"Decision: {request.decision}, Score: {request.final_score:.4f}")

        # Build prompts
        system_prompt = JUSTIFY_SYSTEM_PROMPT
        user_prompt = build_justify_user_prompt(request)

        logger.debug(f"User prompt:\n{user_prompt}")

        # Call Groq API (with fallback)
        logger.info("Calling Groq API for justification...")

        justification = await generate_justification(
            system_prompt,
            user_prompt,
            decision=request.decision,
            score=request.final_score
        )

        logger.success(f"✅ Justification generated ({len(justification)} chars)")
        logger.debug(f"Justification: {justification}")

        return JustifyResponse(justification=justification)

    except Exception as e:
        logger.error(f"❌ Justification error: {e}")
        logger.error(traceback.format_exc())
        # Even if everything fails, return a generic fallback
        fallback = get_fallback_justification(request.decision, request.final_score)
        return JustifyResponse(justification=fallback)


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