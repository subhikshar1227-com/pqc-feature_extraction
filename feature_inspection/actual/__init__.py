"""
Actual Feature Extraction Module for Phase 2B

This module provides functionality for detecting and extracting geometric features
from preprocessed product images using image evidence only.

CRITICAL: This module does NOT use expected features for detection.
All feature detection is based purely on image evidence from preprocessing results.
"""

from .feature_models import (
    ActualFeature,
    ActualFeatureType, 
    GeometricProperties,
    EvidenceMetrics,
    ActualFeatureExtractionResult
)

from .actual_feature_extractor import ActualFeatureExtractor, extract_actual_features

__all__ = [
    'ActualFeature',
    'ActualFeatureType',
    'GeometricProperties', 
    'EvidenceMetrics',
    'ActualFeatureExtractionResult',
    'ActualFeatureExtractor',
    'extract_actual_features'
]