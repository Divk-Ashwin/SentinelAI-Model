"""
Models package for real-time inference
Contains three specialized ML pipelines for multi-modal phishing detection
"""
from models.text_pipeline import TextPipeline
from models.image_pipeline import ImagePipeline
from models.metadata_pipeline import MetadataPipeline

__all__ = [
    "TextPipeline",
    "ImagePipeline", 
    "MetadataPipeline"
]