"""
Feature Matching Data Models

Data structures representing the relationship between expected and actual features.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from feature_extraction.expected.feature_types import ExpectedFeature
from .actual_feature import ActualFeature


class MatchType(Enum):
    """Type of feature match."""
    EXACT_MATCH = "exact_match"         # Perfect geometric correspondence
    GOOD_MATCH = "good_match"          # Good geometric correspondence  
    ACCEPTABLE_MATCH = "acceptable_match"  # Acceptable correspondence
    POOR_MATCH = "poor_match"          # Poor but valid correspondence
    NO_MATCH = "no_match"              # No valid correspondence found


@dataclass
class GeometricComparison:
    """Detailed geometric comparison between matched features."""
    center_distance_mm: float              # Distance between centers (mm in DXF space)
    center_distance_pixels: float          # Distance between centers (pixels in image space)
    
    # Size comparison (when applicable)
    radius_difference_mm: Optional[float] = None     # Expected - Actual radius (mm)
    radius_difference_relative: Optional[float] = None  # Relative difference (fraction)
    width_difference_mm: Optional[float] = None      # Width difference for rectangular features
    height_difference_mm: Optional[float] = None     # Height difference for rectangular features
    
    # Area and shape comparison
    area_ratio: Optional[float] = None      # Actual area / Expected area
    shape_similarity: Optional[float] = None  # Shape similarity metric (0-1)
    
    # Overall geometric metrics
    position_accuracy: float = 0.0          # Position accuracy score (0-1)
    size_accuracy: float = 0.0             # Size accuracy score (0-1)
    overall_accuracy: float = 0.0          # Combined accuracy score (0-1)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        # Ensure all scores are in valid range
        self.position_accuracy = max(0.0, min(1.0, self.position_accuracy))
        self.size_accuracy = max(0.0, min(1.0, self.size_accuracy))
        self.overall_accuracy = max(0.0, min(1.0, self.overall_accuracy))


@dataclass
class FeatureMatch:
    """
    Represents a match between an expected feature and an actual feature.
    
    Contains the geometric correspondence and quality metrics for
    evaluating how well the actual feature matches the expected design.
    """
    expected_feature: ExpectedFeature        # Expected feature from DXF
    actual_feature: ActualFeature           # Detected actual feature
    match_type: MatchType                   # Quality of the match
    match_confidence: float                 # Confidence in the match (0.0 to 1.0)
    
    # Geometric comparison details
    geometric_comparison: GeometricComparison
    
    # Matching algorithm metadata
    match_method: str = "geometric_similarity"  # Algorithm used for matching
    match_score: float = 0.0                # Raw matching score
    
    # Evidence and validation
    match_evidence: Dict[str, Any] = field(default_factory=dict)
    validation_flags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate match data."""
        # Ensure confidence and score are in valid range
        self.match_confidence = max(0.0, min(1.0, self.match_confidence))
        self.match_score = max(0.0, min(1.0, self.match_score))
        
        # Validate feature type compatibility
        if hasattr(self.expected_feature, 'feature_type') and hasattr(self.actual_feature, 'feature_type'):
            if self.expected_feature.feature_type != self.actual_feature.feature_type:
                self.validation_flags.append("type_mismatch")
    
    @property
    def is_acceptable_match(self) -> bool:
        """Check if this match meets acceptable quality standards."""
        return (self.match_type in [MatchType.EXACT_MATCH, MatchType.GOOD_MATCH, MatchType.ACCEPTABLE_MATCH] 
                and self.match_confidence >= 0.6)
    
    @property
    def is_excellent_match(self) -> bool:
        """Check if this is an excellent quality match."""
        return (self.match_type in [MatchType.EXACT_MATCH, MatchType.GOOD_MATCH]
                and self.match_confidence >= 0.9
                and self.geometric_comparison.overall_accuracy >= 0.9)
    
    def get_deviation_summary(self) -> Dict[str, Any]:
        """Get summary of geometric deviations."""
        comp = self.geometric_comparison
        summary = {
            "center_deviation_mm": comp.center_distance_mm,
            "position_accuracy": comp.position_accuracy,
            "size_accuracy": comp.size_accuracy,
            "overall_accuracy": comp.overall_accuracy
        }
        
        if comp.radius_difference_mm is not None:
            summary["radius_deviation_mm"] = comp.radius_difference_mm
            summary["radius_deviation_relative"] = comp.radius_difference_relative
        
        if comp.width_difference_mm is not None:
            summary["width_deviation_mm"] = comp.width_difference_mm
        
        if comp.height_difference_mm is not None:
            summary["height_deviation_mm"] = comp.height_difference_mm
            
        return summary


@dataclass
class MatchingStatistics:
    """Statistics about the feature matching process."""
    total_expected_features: int
    total_actual_features: int
    successful_matches: int
    exact_matches: int
    good_matches: int
    acceptable_matches: int
    poor_matches: int
    unmatched_expected: int
    unmatched_actual: int
    average_match_confidence: float
    average_geometric_accuracy: float
    matching_time_seconds: float


@dataclass
class FeatureMatchSet:
    """
    Complete set of feature matches between expected and actual features.
    
    Represents the result of the matching algorithm, including all
    successful matches and unmatched features from both sides.
    """
    matches: List[FeatureMatch]              # Successful feature matches
    unmatched_expected: List[ExpectedFeature]  # Expected features with no match
    unmatched_actual: List[ActualFeature]    # Actual features with no match
    
    # Matching metadata
    matching_timestamp: str                  # When matching was performed
    matching_statistics: MatchingStatistics
    matching_algorithm: str                  # Algorithm used for matching
    configuration_snapshot: Dict[str, Any]   # Configuration used for matching
    
    @property
    def total_matches(self) -> int:
        """Total number of feature matches."""
        return len(self.matches)
    
    @property
    def acceptable_matches(self) -> List[FeatureMatch]:
        """Matches that meet acceptable quality standards."""
        return [m for m in self.matches if m.is_acceptable_match]
    
    @property
    def excellent_matches(self) -> List[FeatureMatch]:
        """Matches that meet excellent quality standards."""
        return [m for m in self.matches if m.is_excellent_match]
    
    @property
    def poor_matches(self) -> List[FeatureMatch]:
        """Matches with poor quality."""
        return [m for m in self.matches if not m.is_acceptable_match]
    
    @property
    def missing_features_count(self) -> int:
        """Number of expected features that were not found."""
        return len(self.unmatched_expected)
    
    @property
    def extra_features_count(self) -> int:
        """Number of actual features that were not expected."""
        return len(self.unmatched_actual)
    
    @property
    def match_completeness_ratio(self) -> float:
        """Ratio of successfully matched expected features."""
        total_expected = self.matching_statistics.total_expected_features
        if total_expected == 0:
            return 1.0
        return len(self.acceptable_matches) / total_expected
    
    @property
    def match_precision_ratio(self) -> float:
        """Ratio of actual features that were expected."""
        total_actual = self.matching_statistics.total_actual_features
        if total_actual == 0:
            return 1.0
        return len(self.acceptable_matches) / total_actual
    
    def get_matches_by_type(self, match_type: MatchType) -> List[FeatureMatch]:
        """Get all matches of a specific type."""
        return [m for m in self.matches if m.match_type == match_type]
    
    def get_matching_summary(self) -> Dict[str, Any]:
        """Get a summary of matching results."""
        stats = self.matching_statistics
        return {
            "total_expected": stats.total_expected_features,
            "total_actual": stats.total_actual_features,
            "total_matches": self.total_matches,
            "acceptable_matches": len(self.acceptable_matches),
            "excellent_matches": len(self.excellent_matches),
            "poor_matches": len(self.poor_matches),
            "missing_features": self.missing_features_count,
            "extra_features": self.extra_features_count,
            "match_completeness": round(self.match_completeness_ratio, 3),
            "match_precision": round(self.match_precision_ratio, 3),
            "average_confidence": round(stats.average_match_confidence, 3),
            "average_accuracy": round(stats.average_geometric_accuracy, 3),
            "matching_time": stats.matching_time_seconds
        }