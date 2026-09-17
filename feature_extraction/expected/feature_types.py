"""
Expected Feature Type Definitions

Defines the data structures for representing expected geometric features
extracted from DXF files.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..dxf.entity_models import Point2D
from ..config import FEATURE_HIGH_CONFIDENCE_THRESHOLD


class FeatureType(Enum):
    """Supported prominent feature types."""
    CIRCLE = "circle"
    THROUGH_HOLE = "through_hole"
    SQUARE_HOLE = "square_hole"
    RECTANGULAR_HOLE = "rectangular_hole"


@dataclass
class ExpectedFeature:
    """
    Represents a single expected geometric feature extracted from DXF.
    
    This is the primary data structure for representing prominent features
    that will later be matched against actual features detected in product images.
    """
    feature_id: str                           # Unique feature identifier
    feature_type: FeatureType                 # Type of geometric feature
    confidence: float                         # Detection confidence (0.0 to 1.0)
    
    # Geometric properties (in DXF coordinate system, mm)
    center: Point2D                           # Feature center point
    radius: Optional[float] = None            # For circular features
    diameter: Optional[float] = None          # For circular features
    
    # Source information
    source_entity_ids: List[str] = field(default_factory=list)  # Contributing DXF entities
    source_type: str = "direct"               # "direct", "reconstructed", or "inferred"
    
    # Evidence and metadata
    detection_evidence: Dict[str, Any] = field(default_factory=dict)  # Evidence for this feature
    geometric_properties: Dict[str, Any] = field(default_factory=dict)  # Additional geometric data
    relationships: List[str] = field(default_factory=list)  # Related feature IDs
    
    def __post_init__(self):
        """Validate and compute derived properties."""
        if self.radius is not None and self.diameter is None:
            self.diameter = self.radius * 2
        elif self.diameter is not None and self.radius is None:
            self.radius = self.diameter / 2
        
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def area(self) -> Optional[float]:
        """Calculate feature area (for circular features)."""
        if self.radius is not None:
            import math
            return math.pi * (self.radius ** 2)
        return None
    
    @property
    def perimeter(self) -> Optional[float]:
        """Calculate feature perimeter (for circular features).""" 
        if self.radius is not None:
            import math
            return 2 * math.pi * self.radius
        return None


@dataclass
class FeatureDetectionStatistics:
    """Statistics about the feature detection process."""
    total_entities_processed: int
    entities_by_type: Dict[str, int] 
    circles_detected: int
    through_holes_detected: int
    reconstructed_features: int
    filtered_features: int
    final_feature_count: int
    confidence_distribution: Dict[str, int]  # Binned confidence scores
    
    
@dataclass
class ExpectedFeatureSet:
    """
    Complete set of expected features extracted from a DXF file.
    
    This represents the final output of Phase 1: all prominent expected features
    that should be present in a manufactured part based on its CAD design.
    """
    source_dxf_path: Path                     # Source DXF file
    dxf_units: str                           # Units from DXF (e.g., "mm")
    features: List[ExpectedFeature]          # All detected expected features
    extraction_timestamp: str                # When extraction was performed
    processing_statistics: FeatureDetectionStatistics
    configuration_snapshot: Dict[str, Any]   # Configuration used for extraction
    raw_entity_count: int                    # Number of raw DXF entities
    normalized_entity_count: int             # Number successfully normalized
    reconstructed_geometry_count: int        # Number of reconstructed structures
    
    # Reference geometry for visualization (complete DXF geometry) - optional fields last
    reference_geometry: Optional[List[Any]] = None  # Complete normalized DXF entities
    reconstructed_geometry: Optional[Any] = None    # Reconstructed geometry structures
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[ExpectedFeature]:
        """Get all features of a specific type."""
        return [f for f in self.features if f.feature_type == feature_type]
    
    @property
    def circle_count(self) -> int:
        """Number of circle features."""
        return len(self.get_features_by_type(FeatureType.CIRCLE))
    
    @property 
    def through_hole_count(self) -> int:
        """Number of through hole features."""
        return len(self.get_features_by_type(FeatureType.THROUGH_HOLE))
    
    @property
    def total_feature_count(self) -> int:
        """Total number of features."""
        return len(self.features)
    
    @property
    def average_confidence(self) -> float:
        """Average confidence across all features."""
        if not self.features:
            return 0.0
        return sum(f.confidence for f in self.features) / len(self.features)
    
    @property
    def high_confidence_features(self) -> List[ExpectedFeature]:
        """Features with confidence >= configured threshold."""
        return [f for f in self.features if f.confidence >= FEATURE_HIGH_CONFIDENCE_THRESHOLD]
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of detected features."""
        return {
            "source_file": self.source_dxf_path.name,
            "units": self.dxf_units,
            "total_features": self.total_feature_count,
            "circles": self.circle_count,
            "through_holes": self.through_hole_count,
            "average_confidence": round(self.average_confidence, 3),
            "high_confidence_count": len(self.high_confidence_features),
            "raw_entities": self.raw_entity_count,
            "extraction_timestamp": self.extraction_timestamp
        }