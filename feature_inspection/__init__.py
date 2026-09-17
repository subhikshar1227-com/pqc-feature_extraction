"""
Phase 2: Feature Inspection Pipeline

Implements actual feature detection, expected-to-actual matching,
and quality inspection for product manufacturing verification.
"""

from .models.actual_feature import ActualFeature, ActualFeatureSet
from .models.feature_match import FeatureMatch, FeatureMatchSet
from .models.inspection_result import InspectionResult, InspectionStatus
from .models.coordinate_transform import CoordinateTransform
from .pipeline.orchestrator import inspect_product_quality, inspect_product_with_phase1_integration
from .pipeline.coordinate_pipeline import CoordinateTransformationPipeline
from .actual.detector import ActualFeatureDetector
from .matching.matcher import FeatureMatcher
from .inspection.inspector import QualityInspector
from .output.persistence import OutputPersistence
from .output.visualizer import ActualFeatureVisualizer

__version__ = "2.0.0"

__all__ = [
    # Data models
    "ActualFeature",
    "ActualFeatureSet", 
    "FeatureMatch",
    "FeatureMatchSet",
    "InspectionResult",
    "InspectionStatus",
    "CoordinateTransform",
    
    # Main pipeline
    "inspect_product_quality",
    "inspect_product_with_phase1_integration",
    
    # Core components
    "ActualFeatureDetector",
    "FeatureMatcher",
    "QualityInspector",
    "CoordinateTransformationPipeline",
    
    # Output handling
    "OutputPersistence",
    "ActualFeatureVisualizer"
]