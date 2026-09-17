"""
Quality Inspection Result Data Models

Data structures representing the final quality inspection outcome.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

from feature_extraction.expected.feature_types import ExpectedFeatureSet
from .actual_feature import ActualFeatureSet
from .feature_match import FeatureMatchSet, FeatureMatch


class InspectionStatus(Enum):
    """Overall inspection status."""
    PASS = "PASS"               # Product meets quality standards
    FAIL = "FAIL"               # Product does not meet quality standards
    REVIEW = "REVIEW"           # Product requires manual review
    ERROR = "ERROR"             # Inspection could not be completed


class FeatureInspectionResult(Enum):
    """Individual feature inspection result."""
    ACCEPTABLE = "acceptable"        # Feature meets requirements
    DEVIATION = "deviation"          # Feature has acceptable deviation
    DEFECTIVE = "defective"          # Feature has excessive deviations
    MISSING = "missing"             # Expected feature not found
    UNEXPECTED = "unexpected"       # Actual feature not expected
    NOT_DETERMINED = "not_determined"  # Could not determine due to blocked comparison


@dataclass
class FeatureInspectionDetail:
    """Detailed inspection result for a single feature."""
    feature_id: str                        # Feature identifier
    inspection_result: FeatureInspectionResult
    quality_score: Optional[float]             # Feature quality score (0.0 to 1.0) or None if not determinable
    
    # References to source data
    expected_feature: Optional[Any] = None  # ExpectedFeature if applicable
    actual_feature: Optional[Any] = None   # ActualFeature if applicable
    feature_match: Optional[FeatureMatch] = None  # FeatureMatch if applicable
    
    # Quality assessment details
    deviations: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    inspection_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate inspection detail data."""
        if self.quality_score is not None:
            self.quality_score = max(0.0, min(1.0, self.quality_score))


@dataclass
class QualityMetrics:
    """Overall quality metrics for the inspection."""
    completeness_score: Optional[float]   # How many expected features were found (0-1) or None if not determinable
    accuracy_score: Optional[float]       # How accurate the found features are (0-1) or None if not determinable
    precision_score: Optional[float]      # How many found features were expected (0-1) or None if not determinable
    confidence_score: float               # Average detection confidence (0-1)
    overall_quality_score: Optional[float] # Combined quality score (0-1) or None if not determinable
    
    # Detailed breakdowns
    feature_count_accuracy: Optional[float] = None   # Accuracy of feature counts or None if not determinable
    geometric_accuracy: Optional[float] = None       # Accuracy of geometric properties or None if not determinable
    detection_reliability: float = 0.0               # Reliability of detection process
    
    def __post_init__(self):
        """Validate quality metrics."""
        # Ensure all scores are in valid range (only validate non-None values)
        for field_name in ['completeness_score', 'accuracy_score', 'precision_score', 
                          'confidence_score', 'overall_quality_score',
                          'feature_count_accuracy', 'geometric_accuracy', 'detection_reliability']:
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, max(0.0, min(1.0, value)))


@dataclass
class InspectionSummary:
    """High-level summary of inspection results."""
    total_expected_features: int
    total_actual_features: int
    matched_features: int
    missing_features: int
    extra_features: int
    acceptable_features: int
    defective_features: int
    
    # Feature type breakdowns
    expected_circles: int = 0
    actual_circles: int = 0
    expected_holes: int = 0
    actual_holes: int = 0
    
    # Quality indicators
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class InspectionResult:
    """
    Complete quality inspection result for a product.
    
    Represents the final output of Phase 2: a comprehensive assessment
    of product quality based on comparison of expected vs actual features.
    """
    # Input data references
    source_image_path: Path                    # Original product image
    expected_feature_set: ExpectedFeatureSet   # Phase 1 expected features
    actual_feature_set: ActualFeatureSet       # Phase 2 detected features
    feature_match_set: FeatureMatchSet         # Feature matching results
    
    # Inspection outcome
    inspection_status: InspectionStatus        # Overall pass/fail/review
    quality_metrics: QualityMetrics           # Detailed quality scores
    inspection_summary: InspectionSummary     # High-level summary
    
    # Metadata
    inspection_timestamp: str                  # When inspection was performed
    inspection_duration_seconds: float        # Total inspection time
    configuration_snapshot: Dict[str, Any]    # Configuration used
    
    # Optional fields with defaults
    feature_details: List[FeatureInspectionDetail] = field(default_factory=list)
    blueprint_name: Optional[str] = None       # Identified blueprint
    dxf_path: Optional[Path] = None           # Source DXF file
    processing_notes: List[str] = field(default_factory=list)
    
    @property
    def passed_inspection(self) -> bool:
        """Check if product passed inspection."""
        return self.inspection_status == InspectionStatus.PASS
    
    @property
    def failed_inspection(self) -> bool:
        """Check if product failed inspection."""
        return self.inspection_status == InspectionStatus.FAIL
    
    @property
    def needs_review(self) -> bool:
        """Check if product needs manual review."""
        return self.inspection_status == InspectionStatus.REVIEW
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if inspection found critical issues."""
        return len(self.inspection_summary.critical_issues) > 0
    
    @property
    def feature_accuracy_percentage(self) -> float:
        """Get feature accuracy as percentage."""
        return self.quality_metrics.accuracy_score * 100
    
    @property
    def feature_completeness_percentage(self) -> float:
        """Get feature completeness as percentage.""" 
        return self.quality_metrics.completeness_score * 100
    
    def get_missing_features(self) -> List[Any]:
        """Get list of missing expected features."""
        return self.feature_match_set.unmatched_expected
    
    def get_extra_features(self) -> List[Any]:
        """Get list of unexpected actual features."""
        return self.feature_match_set.unmatched_actual
    
    def get_defective_features(self) -> List[FeatureInspectionDetail]:
        """Get list of defective features."""
        return [d for d in self.feature_details 
                if d.inspection_result == FeatureInspectionResult.DEFECTIVE]
    
    def get_acceptable_features(self) -> List[FeatureInspectionDetail]:
        """Get list of acceptable features."""
        return [d for d in self.feature_details 
                if d.inspection_result == FeatureInspectionResult.ACCEPTABLE]
    
    def get_inspection_report(self) -> Dict[str, Any]:
        """Generate a comprehensive inspection report."""
        summary = self.inspection_summary
        metrics = self.quality_metrics
        
        return {
            "inspection_overview": {
                "status": self.inspection_status.value,
                "overall_quality_score": round(metrics.overall_quality_score, 3),
                "passed": self.passed_inspection,
                "timestamp": self.inspection_timestamp,
                "duration_seconds": self.inspection_duration_seconds
            },
            "source_data": {
                "image_path": str(self.source_image_path),
                "blueprint_name": self.blueprint_name,
                "dxf_path": str(self.dxf_path) if self.dxf_path else None
            },
            "feature_summary": {
                "expected_features": summary.total_expected_features,
                "actual_features": summary.total_actual_features,
                "matched_features": summary.matched_features,
                "missing_features": summary.missing_features,
                "extra_features": summary.extra_features,
                "acceptable_features": summary.acceptable_features,
                "defective_features": summary.defective_features
            },
            "quality_metrics": {
                "completeness": round(metrics.completeness_score, 3),
                "accuracy": round(metrics.accuracy_score, 3),
                "precision": round(metrics.precision_score, 3),
                "confidence": round(metrics.confidence_score, 3),
                "geometric_accuracy": round(metrics.geometric_accuracy, 3)
            },
            "feature_type_breakdown": {
                "circles": {
                    "expected": summary.expected_circles,
                    "actual": summary.actual_circles
                },
                "holes": {
                    "expected": summary.expected_holes,
                    "actual": summary.actual_holes
                }
            },
            "issues_and_recommendations": {
                "critical_issues": summary.critical_issues,
                "warnings": summary.warnings,
                "recommendations": summary.recommendations
            },
            "processing_notes": self.processing_notes
        }