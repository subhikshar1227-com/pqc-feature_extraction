"""
Feature Extraction Module

Provides DXF-to-expected-features pipeline for the Peenya Project MSME
quality inspection system.

Public API:
    extract_expected_features(dxf_path: Path) -> ExpectedFeatureSet
    FeatureType: Enumeration of supported feature types
    ExpectedFeature: Feature data structure
    ExpectedFeatureSet: Collection of features with metadata
"""

from .expected_feature_extractor import extract_expected_features
from .expected.feature_types import FeatureType, ExpectedFeature, ExpectedFeatureSet

__all__ = [
    "extract_expected_features", 
    "FeatureType", 
    "ExpectedFeature", 
    "ExpectedFeatureSet"
]
__version__ = "0.1.0"