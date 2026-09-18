"""
Phase 2: Feature Inspection Pipeline

Now includes Phase 2A preprocessing and Phase 2B actual feature extraction.

Downstream matching, comparison, and inspection are not yet implemented.
"""

from .preprocessing import CanonicalPreprocessor, PreprocessingResult
from .actual import (
    ActualFeatureExtractor,
    ActualFeature,
    ActualFeatureType,
    ActualFeatureExtractionResult,
    extract_actual_features
)

__version__ = "2.0.0-preprocessing-and-extraction"

__all__ = [
    # Phase 2A: Canonical preprocessing
    "CanonicalPreprocessor",
    "PreprocessingResult",
    
    # Phase 2B: Actual feature extraction
    "ActualFeatureExtractor",
    "ActualFeature", 
    "ActualFeatureType",
    "ActualFeatureExtractionResult",
    "extract_actual_features"
]