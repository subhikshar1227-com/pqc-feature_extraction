"""
Geometric Feature Matching

Implements geometric similarity matching between expected and actual features.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

from feature_extraction.expected.feature_types import ExpectedFeature, FeatureType
from ..models.actual_feature import ActualFeature
from ..models.feature_match import FeatureMatch, MatchType, GeometricComparison
from ..models.coordinate_transform import CoordinateTransform
from ..config import (
    MATCH_CENTER_TOLERANCE_MM, MATCH_RADIUS_TOLERANCE_ABSOLUTE_MM, MATCH_RADIUS_TOLERANCE_RELATIVE,
    MATCH_MIN_CONFIDENCE, MATCH_TYPE_COMPATIBILITY_BONUS,
    MATCH_DISTANCE_WEIGHT, MATCH_SIZE_WEIGHT, MATCH_TYPE_WEIGHT,
    COMPARISON_CENTER_TOLERANCE_MM, COMPARISON_RADIUS_TOLERANCE_MM, COMPARISON_RADIUS_TOLERANCE_RELATIVE,
    COMPARISON_EXCELLENT_THRESHOLD, COMPARISON_GOOD_THRESHOLD, COMPARISON_ACCEPTABLE_THRESHOLD
)

logger = logging.getLogger(__name__)


class GeometricMatcher:
    """
    Matches expected and actual features based on geometric similarity.
    
    Uses coordinate transformation to compare features in a common coordinate system
    and calculates similarity scores based on position, size, and type compatibility.
    """
    
    def __init__(self, coordinate_transform: Optional[CoordinateTransform] = None):
        """
        Initialize geometric matcher.
        
        Args:
            coordinate_transform: Transform between DXF and image coordinates
        """
        self.coordinate_transform = coordinate_transform
        self.match_cache = {}
    
    def calculate_match_score(self, expected: ExpectedFeature, 
                            actual: ActualFeature) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate geometric similarity score between expected and actual features.
        
        Args:
            expected: Expected feature from DXF
            actual: Detected actual feature
            
        Returns:
            Tuple of (match_score, evidence_dict)
            - match_score: Overall similarity score (0.0 to 1.0)
            - evidence_dict: Dictionary with detailed scoring breakdown
        """
        evidence = {}
        
        # Type compatibility check
        type_compatible = expected.feature_type == actual.feature_type
        type_score = 1.0 if type_compatible else 0.0
        evidence["type_compatible"] = type_compatible
        evidence["type_score"] = type_score
        
        # If not type compatible, return low score unless it's a reasonable substitution
        if not type_compatible:
            # Allow some reasonable type substitutions
            substitutable = self._are_types_substitutable(expected.feature_type, actual.feature_type)
            if not substitutable:
                evidence["substitutable"] = False
                return 0.1, evidence
            else:
                type_score = 0.7  # Penalty for type mismatch but still valid
                evidence["substitutable"] = True
        
        # Transform coordinates to common system (DXF mm)
        try:
            actual_center_dxf, actual_radius_mm = self._transform_actual_to_dxf(actual)
            evidence["coordinate_transform_success"] = True
        except Exception as e:
            logger.warning(f"Coordinate transformation failed: {e}")
            evidence["coordinate_transform_success"] = False
            evidence["transform_error"] = str(e)
            return 0.0, evidence
        
        # Calculate center distance
        center_distance_mm = expected.center.distance_to(actual_center_dxf)
        distance_score = self._calculate_distance_score(center_distance_mm)
        evidence["center_distance_mm"] = center_distance_mm
        evidence["distance_score"] = distance_score
        
        # Calculate size similarity
        size_score = self._calculate_size_score(expected, actual_radius_mm, actual)
        evidence["size_score"] = size_score
        
        # Apply type compatibility bonus if exact match
        if type_compatible:
            type_score += MATCH_TYPE_COMPATIBILITY_BONUS
            type_score = min(1.0, type_score)
        
        evidence["final_type_score"] = type_score
        
        # Weighted combination
        match_score = (
            distance_score * MATCH_DISTANCE_WEIGHT +
            size_score * MATCH_SIZE_WEIGHT +
            type_score * MATCH_TYPE_WEIGHT
        )
        
        evidence["weighted_score"] = match_score
        evidence["weights"] = {
            "distance": MATCH_DISTANCE_WEIGHT,
            "size": MATCH_SIZE_WEIGHT,
            "type": MATCH_TYPE_WEIGHT
        }
        
        return match_score, evidence
    
    def create_geometric_comparison(self, expected: ExpectedFeature, 
                                  actual: ActualFeature) -> GeometricComparison:
        """
        Create detailed geometric comparison between matched features.
        
        Args:
            expected: Expected feature from DXF
            actual: Detected actual feature
            
        Returns:
            GeometricComparison with detailed metrics
        """
        # Transform actual feature to DXF coordinates
        actual_center_dxf, actual_radius_mm = self._transform_actual_to_dxf(actual)
        
        # Calculate center distance
        center_distance_mm = expected.center.distance_to(actual_center_dxf)
        center_distance_pixels = expected.center.distance_to(actual.center)  # This is approximate
        
        comparison = GeometricComparison(
            center_distance_mm=center_distance_mm,
            center_distance_pixels=center_distance_pixels
        )
        
        # Size comparison for circular features
        if expected.radius is not None and actual_radius_mm is not None:
            comparison.radius_difference_mm = expected.radius - actual_radius_mm
            comparison.radius_difference_relative = comparison.radius_difference_mm / expected.radius
        
        # Size comparison for rectangular features
        if hasattr(actual, 'width') and actual.width is not None:
            if self.coordinate_transform and self.coordinate_transform.is_available():
                # Transform dimensions (approximate)
                scale_factor = self.coordinate_transform.get_scale_factor(actual_center_dxf)
                if scale_factor:
                    actual_width_mm = actual.width / scale_factor
                    actual_height_mm = actual.height / scale_factor if actual.height else None
                    
                    # Compare with expected dimensions if available
                    if hasattr(expected, 'width') and expected.width:
                        comparison.width_difference_mm = expected.width - actual_width_mm
                    if hasattr(expected, 'height') and expected.height and actual_height_mm:
                        comparison.height_difference_mm = expected.height - actual_height_mm
        
        # Calculate accuracy scores
        comparison.position_accuracy = self._calculate_position_accuracy(center_distance_mm)
        comparison.size_accuracy = self._calculate_size_accuracy(expected, actual_radius_mm, actual)
        
        # Overall accuracy
        comparison.overall_accuracy = (comparison.position_accuracy + comparison.size_accuracy) / 2.0
        
        return comparison
    
    def determine_match_type(self, match_score: float, 
                           geometric_comparison: GeometricComparison) -> MatchType:
        """
        Determine the type/quality of a feature match.
        
        Args:
            match_score: Overall match score
            geometric_comparison: Detailed geometric comparison
            
        Returns:
            MatchType indicating match quality
        """
        if match_score < MATCH_MIN_CONFIDENCE:
            return MatchType.NO_MATCH
        
        overall_accuracy = geometric_comparison.overall_accuracy
        
        if overall_accuracy >= COMPARISON_EXCELLENT_THRESHOLD:
            return MatchType.EXACT_MATCH
        elif overall_accuracy >= COMPARISON_GOOD_THRESHOLD:
            return MatchType.GOOD_MATCH
        elif overall_accuracy >= COMPARISON_ACCEPTABLE_THRESHOLD:
            return MatchType.ACCEPTABLE_MATCH
        else:
            return MatchType.POOR_MATCH
    
    def _transform_actual_to_dxf(self, actual: ActualFeature) -> Tuple[Any, Optional[float]]:
        """Transform actual feature from image coordinates to DXF coordinates."""
        if not self.coordinate_transform or not self.coordinate_transform.is_available():
            # If no transform available, assume features are already in compatible coordinates
            logger.warning("No coordinate transformation available, assuming compatible coordinates")
            return actual.center, actual.radius
        
        # Transform center point
        actual_center_dxf = self.coordinate_transform.image_to_dxf(actual.center)
        
        # Transform radius if present
        actual_radius_mm = None
        if actual.radius is not None:
            actual_radius_mm = self.coordinate_transform.transform_radius_image_to_dxf(
                actual.radius, actual.center
            )
        
        return actual_center_dxf, actual_radius_mm
    
    def _are_types_substitutable(self, expected_type: FeatureType, 
                               actual_type: FeatureType) -> bool:
        """Check if feature types can be reasonably substituted."""
        # Allow circle and through hole to be matched (holes are often detected as circles)
        substitutable_pairs = [
            (FeatureType.CIRCLE, FeatureType.THROUGH_HOLE),
            (FeatureType.THROUGH_HOLE, FeatureType.CIRCLE),
            (FeatureType.SQUARE_HOLE, FeatureType.RECTANGULAR_HOLE),
            (FeatureType.RECTANGULAR_HOLE, FeatureType.SQUARE_HOLE)
        ]
        
        return (expected_type, actual_type) in substitutable_pairs
    
    def _calculate_distance_score(self, distance_mm: float) -> float:
        """Calculate score based on center distance."""
        if distance_mm <= MATCH_CENTER_TOLERANCE_MM:
            return 1.0
        else:
            # Linear decay beyond tolerance
            decay_rate = 1.0 / (MATCH_CENTER_TOLERANCE_MM * 3)  # Score reaches 0 at 3x tolerance
            score = max(0.0, 1.0 - (distance_mm - MATCH_CENTER_TOLERANCE_MM) * decay_rate)
            return score
    
    def _calculate_size_score(self, expected: ExpectedFeature, 
                            actual_radius_mm: Optional[float],
                            actual: ActualFeature) -> float:
        """Calculate score based on size similarity."""
        if expected.radius is None:
            return 0.8  # Default score when size comparison not possible
        
        if actual_radius_mm is None:
            return 0.5  # Penalty for missing size information
        
        # Calculate relative and absolute differences
        abs_diff = abs(expected.radius - actual_radius_mm)
        rel_diff = abs_diff / expected.radius
        
        # Score based on both absolute and relative tolerances
        abs_score = 1.0 if abs_diff <= MATCH_RADIUS_TOLERANCE_ABSOLUTE_MM else 0.0
        rel_score = 1.0 if rel_diff <= MATCH_RADIUS_TOLERANCE_RELATIVE else 0.0
        
        # If within either tolerance, give partial credit
        if abs_score == 1.0 or rel_score == 1.0:
            return max(abs_score, rel_score)
        
        # Linear decay for sizes outside tolerance
        score = max(0.0, 1.0 - rel_diff / (MATCH_RADIUS_TOLERANCE_RELATIVE * 2))
        return score
    
    def _calculate_position_accuracy(self, center_distance_mm: float) -> float:
        """Calculate position accuracy score."""
        if center_distance_mm <= COMPARISON_CENTER_TOLERANCE_MM:
            return 1.0
        else:
            # Exponential decay
            decay_factor = center_distance_mm / COMPARISON_CENTER_TOLERANCE_MM
            accuracy = np.exp(-decay_factor + 1)  # Normalized exponential decay
            return max(0.0, min(1.0, accuracy))
    
    def _calculate_size_accuracy(self, expected: ExpectedFeature,
                               actual_radius_mm: Optional[float],
                               actual: ActualFeature) -> float:
        """Calculate size accuracy score."""
        if expected.radius is None or actual_radius_mm is None:
            return 0.8  # Default when comparison not possible
        
        # Calculate relative difference
        rel_diff = abs(expected.radius - actual_radius_mm) / expected.radius
        
        if rel_diff <= COMPARISON_RADIUS_TOLERANCE_RELATIVE:
            return 1.0
        else:
            # Linear decay
            accuracy = max(0.0, 1.0 - (rel_diff - COMPARISON_RADIUS_TOLERANCE_RELATIVE) / 
                          (COMPARISON_RADIUS_TOLERANCE_RELATIVE * 2))
            return accuracy