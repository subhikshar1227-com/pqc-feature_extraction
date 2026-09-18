"""
Actual Feature Models for Phase 2B Feature Extraction

Defines the data structures for features detected from actual product images.
These models capture geometric and evidence information needed for later 
matching and comparison stages.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
from pathlib import Path


class ActualFeatureType(Enum):
    """Types of features that can be detected from actual product images."""
    CIRCLE = "CIRCLE"
    RECTANGLE = "RECTANGLE" 
    SQUARE = "SQUARE"
    GENERAL_CONTOUR = "GENERAL_CONTOUR"
    HOLE = "HOLE"
    

@dataclass
class GeometricProperties:
    """Geometric properties extracted from detected features."""
    center: Tuple[float, float]           # (x, y) center coordinates
    area: float                           # Feature area in pixels
    perimeter: float                      # Feature perimeter in pixels
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height) bounding rectangle
    
    # Shape-specific properties (None if not applicable)
    radius: Optional[float] = None        # Circle radius
    diameter: Optional[float] = None      # Circle diameter
    width: Optional[float] = None         # Rectangle/square width
    height: Optional[float] = None        # Rectangle/square height
    aspect_ratio: Optional[float] = None  # Width/height ratio
    
    # Geometric quality metrics
    circularity: float = 0.0              # 4π*area/perimeter² (1.0 = perfect circle)
    solidity: float = 0.0                 # Area/convex_hull_area
    extent: float = 0.0                   # Area/bounding_box_area
    convexity: float = 0.0                # Convex_hull_perimeter/perimeter


@dataclass
class EvidenceMetrics:
    """Evidence and confidence metrics for detected features."""
    confidence: float                     # Overall confidence score (0.0 - 1.0)
    edge_support: float                  # Fraction of perimeter supported by edges
    contour_quality: float               # Quality of contour fit
    intensity_consistency: float         # Consistency of internal intensity
    geometric_consistency: float         # Consistency with expected geometry
    
    # Evidence source contributions
    internal_edge_evidence: float = 0.0   # Evidence from internal geometry edges
    raw_edge_evidence: float = 0.0       # Evidence from raw internal edges
    boundary_evidence: float = 0.0       # Evidence from outer boundary
    intensity_evidence: float = 0.0      # Evidence from isolated product image
    
    # Quality flags
    is_border_feature: bool = False      # Feature touches image border
    is_partial_feature: bool = False     # Feature appears partially occluded
    has_noise_artifacts: bool = False    # Feature may contain noise artifacts


@dataclass  
class ActualFeature:
    """
    Represents a feature detected from actual product images.
    
    Contains complete geometric and evidence information needed for
    matching, comparison, and inspection stages.
    """
    # Feature identification
    feature_id: str                      # Unique identifier for this detection
    feature_type: ActualFeatureType     # Type of detected feature
    
    # Geometric information
    geometry: GeometricProperties        # Complete geometric measurements
    contour: np.ndarray                 # Original contour points (Nx2 array)
    
    # Evidence and confidence
    evidence: EvidenceMetrics           # Detection evidence and quality metrics
    
    # Source information
    source_representation: str          # Primary representation used for detection
    detection_method: str               # Algorithm/method used for detection
    
    # Coordinate system information  
    coordinate_system: str = "processed" # "processed" or "original" coordinates
    scale_factor: float = 1.0           # Scale factor from original to processed
    
    # Processing metadata
    detection_timestamp: Optional[str] = None  # When feature was detected
    processing_parameters: Dict[str, Any] = None  # Detection parameters used
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.processing_parameters is None:
            self.processing_parameters = {}
    
    def to_original_coordinates(self, roi_offset: Tuple[int, int] = (0, 0)) -> 'ActualFeature':
        """
        Convert feature coordinates to original image coordinate system.
        
        Args:
            roi_offset: ROI offset if cropping was used during preprocessing
            
        Returns:
            New ActualFeature with coordinates in original image space
        """
        if self.coordinate_system == "original":
            return self  # Already in original coordinates
        
        # Convert center coordinates
        orig_x = (self.geometry.center[0] + roi_offset[0]) / self.scale_factor
        orig_y = (self.geometry.center[1] + roi_offset[1]) / self.scale_factor
        
        # Convert contour coordinates
        orig_contour = self.contour.copy().astype(np.float32)
        if len(orig_contour.shape) == 3:
            # Nx1x2 format 
            orig_contour[:, 0, 0] = (orig_contour[:, 0, 0] + roi_offset[0]) / self.scale_factor
            orig_contour[:, 0, 1] = (orig_contour[:, 0, 1] + roi_offset[1]) / self.scale_factor
        else:
            # Nx2 format
            orig_contour[:, 0] = (orig_contour[:, 0] + roi_offset[0]) / self.scale_factor
            orig_contour[:, 1] = (orig_contour[:, 1] + roi_offset[1]) / self.scale_factor
        
        # Convert bounding box
        x, y, w, h = self.geometry.bounding_box
        orig_x_box = int((x + roi_offset[0]) / self.scale_factor)
        orig_y_box = int((y + roi_offset[1]) / self.scale_factor)
        orig_w_box = int(w / self.scale_factor)
        orig_h_box = int(h / self.scale_factor)
        
        # Create new geometry with original coordinates
        orig_geometry = GeometricProperties(
            center=(orig_x, orig_y),
            area=self.geometry.area / (self.scale_factor ** 2),
            perimeter=self.geometry.perimeter / self.scale_factor,
            bounding_box=(orig_x_box, orig_y_box, orig_w_box, orig_h_box),
            radius=self.geometry.radius / self.scale_factor if self.geometry.radius else None,
            diameter=self.geometry.diameter / self.scale_factor if self.geometry.diameter else None,
            width=self.geometry.width / self.scale_factor if self.geometry.width else None,
            height=self.geometry.height / self.scale_factor if self.geometry.height else None,
            aspect_ratio=self.geometry.aspect_ratio,  # Ratio is scale-invariant
            circularity=self.geometry.circularity,    # Ratio is scale-invariant
            solidity=self.geometry.solidity,         # Ratio is scale-invariant
            extent=self.geometry.extent,             # Ratio is scale-invariant
            convexity=self.geometry.convexity        # Ratio is scale-invariant
        )
        
        # Create new feature with original coordinates
        return ActualFeature(
            feature_id=self.feature_id,
            feature_type=self.feature_type,
            geometry=orig_geometry,
            contour=orig_contour.astype(np.int32),
            evidence=self.evidence,  # Evidence metrics are coordinate-independent
            source_representation=self.source_representation,
            detection_method=self.detection_method,
            coordinate_system="original",
            scale_factor=1.0,  # Now in original coordinates
            detection_timestamp=self.detection_timestamp,
            processing_parameters=self.processing_parameters.copy()
        )


@dataclass
class ActualFeatureExtractionResult:
    """
    Complete result of actual feature extraction from a preprocessed image.
    """
    # Source information
    source_image_path: Path             # Original image path
    preprocessing_successful: bool       # Whether preprocessing was successful
    
    # Detected features
    features: List[ActualFeature]       # All detected features
    
    # Detection statistics  
    total_candidates_generated: int     # Total candidates before validation
    candidates_by_type: Dict[str, int]  # Candidate counts by feature type
    features_by_type: Dict[str, int]    # Final feature counts by type
    duplicate_candidates_suppressed: int # Number of duplicates removed
    
    # Quality metrics
    average_confidence: float           # Average confidence of detected features
    confidence_by_type: Dict[str, float] # Average confidence by feature type
    
    # Processing information
    detection_methods_used: List[str]   # Detection algorithms used
    representations_used: List[str]     # Preprocessing representations used
    processing_time_seconds: float      # Total processing time
    
    # Coordinate system information
    coordinate_system: str = "processed" # Coordinate system of detected features
    scale_factor: float = 1.0           # Scale factor to original coordinates
    roi_offset: Tuple[int, int] = (0, 0) # ROI offset if applicable
    
    def get_features_by_type(self, feature_type: ActualFeatureType) -> List[ActualFeature]:
        """Get all features of a specific type."""
        return [f for f in self.features if f.feature_type == feature_type]
    
    def get_feature_count_by_type(self, feature_type: ActualFeatureType) -> int:
        """Get count of features of a specific type."""
        return len(self.get_features_by_type(feature_type))
    
    def to_original_coordinates(self) -> 'ActualFeatureExtractionResult':
        """
        Convert all features to original image coordinates.
        
        Returns:
            New result with all features in original coordinate system
        """
        if self.coordinate_system == "original":
            return self  # Already in original coordinates
        
        # Convert all features to original coordinates
        original_features = [
            feature.to_original_coordinates(self.roi_offset) 
            for feature in self.features
        ]
        
        # Create new result with original coordinates
        return ActualFeatureExtractionResult(
            source_image_path=self.source_image_path,
            preprocessing_successful=self.preprocessing_successful,
            features=original_features,
            total_candidates_generated=self.total_candidates_generated,
            candidates_by_type=self.candidates_by_type.copy(),
            features_by_type=self.features_by_type.copy(),
            duplicate_candidates_suppressed=self.duplicate_candidates_suppressed,
            average_confidence=self.average_confidence,
            confidence_by_type=self.confidence_by_type.copy(),
            detection_methods_used=self.detection_methods_used.copy(),
            representations_used=self.representations_used.copy(),
            processing_time_seconds=self.processing_time_seconds,
            coordinate_system="original",
            scale_factor=1.0,
            roi_offset=(0, 0)
        )