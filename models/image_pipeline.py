"""
Image Analysis Pipeline - OCR-based Text Extraction for Phishing Detection (Pipeline 2)

Extracts text from images using Tesseract OCR and passes to text model for scoring.
This approach avoids the need for CNN training on phishing images.
"""
import os
import io
import base64
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Optional imports with graceful fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from config import DEVICE

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysisResult:
    """Result of image analysis via OCR."""
    score: Optional[float]       # Spam probability (None if no text extracted)
    ocr_text: Optional[str]      # Extracted text (None if OCR failed)
    text_length: int             # Length of extracted text
    explanation: str             # Human-readable explanation
    ocr_success: bool            # Whether OCR succeeded

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "score": self.score,
            "ocr_text": self.ocr_text,
            "text_length": self.text_length,
            "explanation": self.explanation,
            "ocr_success": self.ocr_success
        }


class ImagePipeline:
    """
    OCR-based image analysis pipeline for phishing detection.

    Instead of training a CNN on phishing images, this pipeline:
    1. Extracts text from images using Tesseract OCR
    2. Passes extracted text to the text model for spam classification
    3. Returns None if no meaningful text is found (avoids 0.5 bias)

    This approach works well for screenshot-based phishing attempts
    (fake bank alerts, UPI scam images, etc.)
    """

    # Minimum text length to consider for analysis
    MIN_TEXT_LENGTH = 10

    def __init__(self, text_pipeline=None):
        """
        Initialize image pipeline.

        Args:
            text_pipeline: Optional TextPipeline instance for scoring extracted text.
                          If not provided, will be lazily loaded on first use.
        """
        self._text_pipeline = text_pipeline
        self._text_pipeline_loaded = text_pipeline is not None

        # Check dependencies
        if not CV2_AVAILABLE:
            logger.warning("OpenCV (cv2) not available. Image preprocessing will be limited.")
        if not TESSERACT_AVAILABLE:
            logger.warning("pytesseract not available. OCR will not work.")
            logger.warning("Install with: pip install pytesseract")
            logger.warning("Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")

        logger.info("ImagePipeline initialized (OCR-based approach)")

    @property
    def text_pipeline(self):
        """Lazy load TextPipeline to avoid circular imports."""
        if not self._text_pipeline_loaded:
            try:
                from models.text_pipeline import TextPipeline
                self._text_pipeline = TextPipeline()
                self._text_pipeline_loaded = True
                logger.info("TextPipeline loaded for OCR text scoring")
            except Exception as e:
                logger.error(f"Failed to load TextPipeline: {e}")
                self._text_pipeline = None
                self._text_pipeline_loaded = True  # Don't retry
        return self._text_pipeline

    def _load_image(self, image_input) -> Optional[Image.Image]:
        """
        Load image from various input formats.

        Args:
            image_input: Can be:
                - PIL Image object
                - File path (str)
                - Base64 encoded string
                - Bytes

        Returns:
            PIL Image or None if loading failed
        """
        try:
            # Already a PIL Image
            if isinstance(image_input, Image.Image):
                return image_input

            # File path
            if isinstance(image_input, str):
                # Check if it's base64
                if image_input.startswith('data:image'):
                    # Data URL format: data:image/png;base64,<data>
                    _, base64_data = image_input.split(',', 1)
                    image_bytes = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(image_bytes))
                elif len(image_input) > 500 and not os.path.exists(image_input):
                    # Likely raw base64 string
                    try:
                        image_bytes = base64.b64decode(image_input)
                        return Image.open(io.BytesIO(image_bytes))
                    except Exception:
                        pass
                # File path
                if os.path.exists(image_input):
                    return Image.open(image_input)
                else:
                    logger.error(f"Image file not found: {image_input}")
                    return None

            # Bytes
            if isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input))

            logger.error(f"Unsupported image input type: {type(image_input)}")
            return None

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None

    def _preprocess_for_ocr(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.

        Steps:
        1. Convert to grayscale
        2. Apply adaptive thresholding
        3. Denoise using fastNlMeansDenoising

        Args:
            image: PIL Image

        Returns:
            Preprocessed image as numpy array
        """
        # Convert PIL to numpy array
        img_array = np.array(image)

        # If image has alpha channel, remove it
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        if not CV2_AVAILABLE:
            # Fallback: just convert to grayscale using PIL
            if len(img_array.shape) == 3:
                gray = np.array(image.convert('L'))
            else:
                gray = img_array
            return gray

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply adaptive thresholding for better text extraction
        # This helps with varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # Denoise to remove noise while preserving edges
        denoised = cv2.fastNlMeansDenoising(
            thresh,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

        return denoised

    def _extract_text_ocr(self, image: Image.Image) -> Optional[str]:
        """
        Extract text from image using Tesseract OCR.

        Args:
            image: PIL Image

        Returns:
            Extracted text or None if OCR failed
        """
        if not TESSERACT_AVAILABLE:
            logger.error("pytesseract not installed. Cannot perform OCR.")
            return None

        try:
            # Preprocess image for better OCR
            preprocessed = self._preprocess_for_ocr(image)

            # Convert back to PIL for pytesseract
            preprocessed_pil = Image.fromarray(preprocessed)

            # Run OCR with English + common Indian languages
            # Use --psm 6 for uniform block of text
            custom_config = r'--oem 3 --psm 6'

            text = pytesseract.image_to_string(
                preprocessed_pil,
                config=custom_config
            )

            # Clean up extracted text
            text = text.strip()

            # Remove excessive whitespace
            text = ' '.join(text.split())

            return text if text else None

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None

    def predict(self, image_input) -> Optional[float]:
        """
        Predict spam probability for an image.

        If meaningful text is extracted (>10 chars), passes to text model.
        Otherwise returns None (not 0.5) to indicate no score available.

        Args:
            image_input: Image as PIL Image, file path, base64 string, or bytes

        Returns:
            Spam probability (0.0-1.0) or None if no text extracted
        """
        result = self.analyze(image_input)
        return result.score

    def analyze(self, image_input) -> ImageAnalysisResult:
        """
        Full image analysis with OCR text extraction.

        Args:
            image_input: Image as PIL Image, file path, base64 string, or bytes

        Returns:
            ImageAnalysisResult with score, OCR text, and explanation
        """
        # Load image
        image = self._load_image(image_input)
        if image is None:
            return ImageAnalysisResult(
                score=None,
                ocr_text=None,
                text_length=0,
                explanation="Failed to load image",
                ocr_success=False
            )

        # Extract text via OCR
        ocr_text = self._extract_text_ocr(image)

        if ocr_text is None:
            return ImageAnalysisResult(
                score=None,
                ocr_text=None,
                text_length=0,
                explanation="OCR extraction failed",
                ocr_success=False
            )

        text_length = len(ocr_text)

        # Check if text is meaningful (> MIN_TEXT_LENGTH characters)
        if text_length <= self.MIN_TEXT_LENGTH:
            return ImageAnalysisResult(
                score=None,
                ocr_text=ocr_text if ocr_text else None,
                text_length=text_length,
                explanation=f"Extracted text too short ({text_length} chars, need >{self.MIN_TEXT_LENGTH})",
                ocr_success=True
            )

        # Get text pipeline for scoring
        if self.text_pipeline is None:
            return ImageAnalysisResult(
                score=None,
                ocr_text=ocr_text,
                text_length=text_length,
                explanation="Text pipeline not available for scoring",
                ocr_success=True
            )

        # Score the extracted text using the text model
        try:
            spam_score = self.text_pipeline.predict(ocr_text)

            # Generate explanation based on score
            if spam_score > 0.7:
                explanation = f"High-risk text detected in image ({text_length} chars extracted)"
            elif spam_score > 0.5:
                explanation = f"Moderate-risk text found in image ({text_length} chars extracted)"
            else:
                explanation = f"Image text appears legitimate ({text_length} chars extracted)"

            return ImageAnalysisResult(
                score=round(spam_score, 4),
                ocr_text=ocr_text,
                text_length=text_length,
                explanation=explanation,
                ocr_success=True
            )

        except Exception as e:
            logger.error(f"Text scoring failed: {e}")
            return ImageAnalysisResult(
                score=None,
                ocr_text=ocr_text,
                text_length=text_length,
                explanation=f"Text extracted but scoring failed: {e}",
                ocr_success=True
            )

    def predict_with_ocr(
        self,
        image_input
    ) -> Tuple[Optional[float], Optional[str], str]:
        """
        Predict with OCR details (simplified interface).

        Args:
            image_input: Image as PIL Image, file path, base64 string, or bytes

        Returns:
            Tuple of (score, ocr_text, explanation)
            - score: Spam probability or None if no meaningful text
            - ocr_text: Extracted text or None
            - explanation: Human-readable explanation
        """
        result = self.analyze(image_input)
        return result.score, result.ocr_text, result.explanation


# Convenience function for one-off analysis
def analyze_image(
    image_input,
    text_pipeline=None
) -> Dict[str, Any]:
    """
    Convenience function for analyzing a single image.

    Args:
        image_input: Image as PIL Image, file path, base64 string, or bytes
        text_pipeline: Optional TextPipeline instance

    Returns:
        Dictionary with analysis results
    """
    pipeline = ImagePipeline(text_pipeline=text_pipeline)
    result = pipeline.analyze(image_input)
    return result.to_dict()


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are available.

    Returns:
        Dictionary with dependency status
    """
    return {
        "opencv": CV2_AVAILABLE,
        "pytesseract": TESSERACT_AVAILABLE,
        "tesseract_path": _check_tesseract_installation()
    }


def _check_tesseract_installation() -> bool:
    """Check if Tesseract OCR is installed and accessible."""
    if not TESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
