"""
Actual Feature Data Models

Data structures representing features detected in real product images.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

from feature_extraction.expected.feature_types import FeatureType, Point2D


class DetectionMethod(Enum):
    """Methods used for actual feature detection."""
    HOUGH_CIRCLES = "hough_circles"
    CONTOUR_ANALYSIS = "contour_analysis"
    EDGE_DETECTION = "edge_detection"
    TEMPLATE_MATCHING = "template_matching"
    HYBRID = "hybrid"


@dataclass
class ActualFeature:
    """
    Represents a feature detected in an actual product image.
    
    This is the Phase 2 counterpart to ExpectedFeature, representing
    what was actually observed in the physical product image rather
    than what was specified in the CAD design.
    """
    feature_id: str                           # Generated unique identifier
    feature_type: FeatureType                 # Type of detected feature
    confidence: float                         # Detection confidence (0.0 to 1.0)
    
    # Geometric properties (in image pixel coordinates)
    center: Point2D                           # Feature center in image pixels
    radius: Optional[float] = None            # For circular features (pixels)
    width: Optional[float] = None             # For rectangular features (pixels)
    height: Optional[float] = None            # For rectangular features (pixels)
    
    # Detection metadata
    detection_method: DetectionMethod = DetectionMethod.CONTOUR_ANALYSIS
    coordinate_system: str = "image_pixels"   # Coordinate system identifier
    
    # Detection evidence and quality metrics
    detection_evidence: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Source information
    contour_area: Optional[float] = None      # Area of source contour (pixels²)
    bounding_box: Optional[tuple] = None      # (x, y, width, height) in pixels
    
    def __post_init__(self):
        """Validate and compute derived properties."""
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Validate coordinate system
        if self.coordinate_system not in ["image_pixels", "dxf_mm"]:
            raise ValueError(f"Invalid coordinate system: {self.coordinate_system}")
        
        # Compute bounding box if not provided
        if self.bounding_box is None and self.radius is not None:
            self.bounding_box = (
                self.center.x - self.radius,
                self.center.y - self.radius, 
                self.radius * 2,
                self.radius * 2
            )
        elif self.bounding_box is None and self.width is not None and self.height is not None:
            self.bounding_box = (
                self.center.x - self.width / 2,
                self.center.y - self.height / 2,
                self.width,
                self.height
            )
    
    @property
    def area(self) -> Optional[float]:
        """Calculate feature area."""
        if self.radius is not None:
            import math
            return math.pi * (self.radius ** 2)
        elif self.width is not None and self.height is not None:
            return self.width * self.height
        return None
    
    @property
    def perimeter(self) -> Optional[float]:
        """Calculate feature perimeter."""
        if self.radius is not None:
            import math
            return 2 * math.pi * self.radius
        elif self.width is not None and self.height is not None:
            return 2 * (self.width + self.height)
        return None
    
    def get_dimensions(self) -> Dict[str, float]:
        """Get feature dimensions as a dictionary."""
        dims = {}
        if self.radius is not None:
            dims["radius"] = self.radius
            dims["diameter"] = self.radius * 2
        if self.width is not None:
            dims["width"] = self.width
        if self.height is not None:
            dims["height"] = self.height
        return dims


@dataclass
class ActualDetectionStatistics:
    """Statistics about the actual feature detection process."""
    total_contours_found: int
    contours_after_filtering: int
    circles_detected: int
    through_holes_detected: int
    rectangular_holes_detected: int
    total_features_detected: int
    average_confidence: float
    detection_time_seconds: float
    preprocessing_time_seconds: float


@dataclass
class ActualFeatureSet:
    """
    Complete set of actual features detected from a product image.
    
    This represents the output of Phase 2 actual feature detection,
    containing all features observed in the physical product image.
    """
    source_image_path: Path                    # Source product image
    features: List[ActualFeature]              # All detected actual features
    detection_timestamp: str                   # When detection was performed
    detection_statistics: ActualDetectionStatistics
    configuration_snapshot: Dict[str, Any]     # Configuration used for detection
    
    # Image processing metadata  
    image_dimensions: tuple                    # (width, height) in pixels - IMPORTANT: width first, height second
    preprocessing_applied: List[str]           # List of preprocessing steps
    
    # Coordinate system information
    coordinate_system: str = "image_pixels"    # Primary coordinate system
    transform_matrix: Optional[Any] = None     # Transform to DXF coordinates if available
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[ActualFeature]:
        """Get all features of a specific type."""
        return [f for f in self.features if f.feature_type == feature_type]
    
    @property
    def circle_count(self) -> int:
        """Number of circular features."""
        return len(self.get_features_by_type(FeatureType.CIRCLE))
    
    @property
    def through_hole_count(self) -> int:
        """Number of through hole features."""
        return len(self.get_features_by_type(FeatureType.THROUGH_HOLE))
    
    @property
    def rectangular_hole_count(self) -> int:
        """Number of rectangular hole features."""
        rect_types = [FeatureType.SQUARE_HOLE, FeatureType.RECTANGULAR_HOLE]
        return len([f for f in self.features if f.feature_type in rect_types])
    
    @property
    def square_hole_count(self) -> int:
        """Number of square hole features."""
        return len(self.get_features_by_type(FeatureType.SQUARE_HOLE))
    
    @property
    def total_feature_count(self) -> int:
        """Total number of detected features."""
        return len(self.features)
    
    @property
    def average_confidence(self) -> float:
        """Average confidence across all detected features."""
        if not self.features:
            return 0.0
        return sum(f.confidence for f in self.features) / len(self.features)
    
    @property
    def high_confidence_features(self) -> List[ActualFeature]:
        """Features with confidence >= 0.8."""
        return [f for f in self.features if f.confidence >= 0.8]
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get a summary of detected features."""
        return {
            "source_image": self.source_image_path.name,
            "coordinate_system": self.coordinate_system,
            "total_features": self.total_feature_count,
            "circles": self.circle_count,
            "through_holes": self.through_hole_count,
            "rectangular_holes": self.rectangular_hole_count,
            "average_confidence": round(self.average_confidence, 3),
            "high_confidence_count": len(self.high_confidence_features),
            "detection_time": self.detection_statistics.detection_time_seconds,
            "preprocessing_time": self.detection_statistics.preprocessing_time_seconds
        }