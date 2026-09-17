"""
Expected Feature Detection Module

Detects prominent geometric features from DXF entities and reconstructed geometry.
Currently supports circles and through holes.

Public API:
    detect_circles(entities, reconstructed) -> List[ExpectedFeature]
    detect_through_holes(entities, reconstructed) -> List[ExpectedFeature]
    filter_significant_features(features) -> List[ExpectedFeature] 
    build_expected_feature_set(features, metadata) -> ExpectedFeatureSet
"""

from .feature_types import FeatureType, ExpectedFeature, ExpectedFeatureSet
from .circle_detector import CircleDetector
from .through_hole_detector import ThroughHoleDetector
from .square_hole_detector import SquareHoleDetector
from .significance_filter import SignificanceFilter
from .expected_feature_builder import build_expected_feature_set
from .semantic_grouping import SemanticFeatureGrouper

__all__ = [
    "FeatureType",
    "ExpectedFeature", 
    "ExpectedFeatureSet",
    "CircleDetector",
    "ThroughHoleDetector",
    "SquareHoleDetector",
    "SignificanceFilter",
    "build_expected_feature_set"
]