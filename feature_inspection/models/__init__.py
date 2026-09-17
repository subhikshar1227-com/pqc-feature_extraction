"""
Phase 2: Data Models

Core data structures for actual features, matching, and inspection results.
"""

from .actual_feature import ActualFeature, ActualFeatureSet
from .feature_match import FeatureMatch, FeatureMatchSet
from .inspection_result import InspectionResult, InspectionStatus
from .coordinate_transform import CoordinateTransform

__all__ = [
    "ActualFeature",
    "ActualFeatureSet",
    "FeatureMatch", 
    "FeatureMatchSet",
    "InspectionResult",
    "InspectionStatus",
    "CoordinateTransform"
]