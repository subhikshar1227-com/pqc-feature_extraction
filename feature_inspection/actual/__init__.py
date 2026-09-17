"""
Actual Feature Detection

Implements detection of geometric features in real product images.
"""

from .detector import ActualFeatureDetector
from .canonical_preprocessor import CanonicalPreprocessor

__all__ = [
    "ActualFeatureDetector", 
    "CanonicalPreprocessor"
]