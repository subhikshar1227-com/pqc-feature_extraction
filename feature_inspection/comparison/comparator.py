"""
Feature Comparison Engine

Performs detailed geometric comparison of matched features.
"""

from typing import List, Dict, Any
import logging

from ..models.feature_match import FeatureMatch, FeatureMatchSet
from ..models.inspection_result import FeatureInspectionDetail, FeatureInspectionResult
from ..config import (
    COMPARISON_CENTER_TOLERANCE_MM, COMPARISON_RADIUS_TOLERANCE_MM, 
    COMPARISON_RADIUS_TOLERANCE_RELATIVE, COMPARISON_EXCELLENT_THRESHOLD,
    COMPARISON_GOOD_THRESHOLD, COMPARISON_ACCEPTABLE_THRESHOLD
)

logger = logging.getLogger(__name__)


class FeatureComparator:
    """
    Performs detailed comparison of matched expected and actual features.
    
    Evaluates geometric accuracy and determines whether each matched
    feature meets acceptable quality standards.
    """
    
    def __init__(self):
        """Initialize feature comparator."""
        pass
    
    def analyze_matches(self, match_set: FeatureMatchSet) -> List[FeatureInspectionDetail]:
        """
        Analyze all matches in a FeatureMatchSet and generate inspection details.
        
        Args:
            match_set: Set of feature matches to analyze
            
        Returns:
            List of detailed inspection results for each match
        """
        inspection_details = []
        
        # Check if matching was actually performed
        config = match_set.configuration_snapshot
        matching_performed = True
        
        if isinstance(config, dict):
            if "matching_performed" in config:
                matching_performed = config["matching_performed"]
            elif config.get("matching_blocked", False):  # Legacy check
                matching_performed = False
        
        if not matching_performed:
            logger.info("Matching was not performed - skipping geometric comparison")
            return []  # Return empty list - let inspector handle blocked case
        
        logger.info(f"Analyzing {len(match_set.matches)} feature matches")
        
        for match in match_set.matches:
            detail = self._analyze_single_match(match)
            inspection_details.append(detail)
        
        # Add details for missing features (only if matching was performed)
        for missing_expected in match_set.unmatched_expected:
            detail = FeatureInspectionDetail(
                feature_id=missing_expected.feature_id,
                inspection_result=FeatureInspectionResult.MISSING,
                quality_score=0.0,
                expected_feature=missing_expected,
                actual_feature=None,
                feature_match=None,
                inspection_notes=["Expected feature not detected in actual image"]
            )
            inspection_details.append(detail)
        
        # Add details for extra features (only if matching was performed)
        for extra_actual in match_set.unmatched_actual:
            detail = FeatureInspectionDetail(
                feature_id=extra_actual.feature_id,
                inspection_result=FeatureInspectionResult.UNEXPECTED,
                quality_score=0.5,  # Partial score - feature exists but wasn't expected
                expected_feature=None,
                actual_feature=extra_actual,
                feature_match=None,
                inspection_notes=["Detected feature not present in expected design"]
            )
            inspection_details.append(detail)
        
        logger.info(f"Generated {len(inspection_details)} inspection details")
        return inspection_details
    
    def _analyze_single_match(self, match: FeatureMatch) -> FeatureInspectionDetail:
        """Analyze a single feature match."""
        comparison = match.geometric_comparison
        
        # Evaluate geometric deviations
        deviations = self._evaluate_deviations(comparison)
        
        # Determine inspection result
        inspection_result = self._determine_inspection_result(match, deviations)
        
        # Calculate quality score
        quality_score = self._calculate_match_quality_score(match, deviations)
        
        # Generate inspection notes
        notes = self._generate_inspection_notes(match, deviations, inspection_result)
        
        # Create quality metrics
        quality_metrics = {
            "position_accuracy": comparison.position_accuracy,
            "size_accuracy": comparison.size_accuracy,
            "overall_accuracy": comparison.overall_accuracy,
            "match_confidence": match.match_confidence
        }
        
        return FeatureInspectionDetail(
            feature_id=match.expected_feature.feature_id,
            inspection_result=inspection_result,
            quality_score=quality_score,
            expected_feature=match.expected_feature,
            actual_feature=match.actual_feature,
            feature_match=match,
            deviations=deviations,
            quality_metrics=quality_metrics,
            inspection_notes=notes
        )
    
    def _evaluate_deviations(self, comparison) -> Dict[str, float]:
        """Evaluate geometric deviations and classify their severity."""
        deviations = {}
        
        # Center position deviation
        center_dev = comparison.center_distance_mm
        deviations["center_distance_mm"] = center_dev
        deviations["center_excessive"] = center_dev > COMPARISON_CENTER_TOLERANCE_MM
        
        # Radius deviation (if applicable)
        if comparison.radius_difference_mm is not None:
            radius_dev_abs = abs(comparison.radius_difference_mm)
            radius_dev_rel = abs(comparison.radius_difference_relative) if comparison.radius_difference_relative is not None else 0
            
            deviations["radius_deviation_abs_mm"] = radius_dev_abs
            deviations["radius_deviation_relative"] = radius_dev_rel
            deviations["radius_excessive_abs"] = radius_dev_abs > COMPARISON_RADIUS_TOLERANCE_MM
            deviations["radius_excessive_rel"] = radius_dev_rel > COMPARISON_RADIUS_TOLERANCE_RELATIVE
            deviations["radius_excessive"] = deviations["radius_excessive_abs"] or deviations["radius_excessive_rel"]
        
        # Width/height deviations (if applicable)
        if comparison.width_difference_mm is not None:
            width_dev = abs(comparison.width_difference_mm)
            deviations["width_deviation_mm"] = width_dev
            deviations["width_excessive"] = width_dev > COMPARISON_RADIUS_TOLERANCE_MM  # Reuse radius tolerance
        
        if comparison.height_difference_mm is not None:
            height_dev = abs(comparison.height_difference_mm)
            deviations["height_deviation_mm"] = height_dev
            deviations["height_excessive"] = height_dev > COMPARISON_RADIUS_TOLERANCE_MM  # Reuse radius tolerance
        
        return deviations
    
    def _determine_inspection_result(self, match: FeatureMatch, 
                                   deviations: Dict[str, float]) -> FeatureInspectionResult:
        """Determine the inspection result based on match quality and deviations."""
        overall_accuracy = match.geometric_comparison.overall_accuracy
        
        # Check for excessive deviations
        has_excessive_deviations = (
            deviations.get("center_excessive", False) or
            deviations.get("radius_excessive", False) or
            deviations.get("width_excessive", False) or 
            deviations.get("height_excessive", False)
        )
        
        # Determine result based on accuracy and deviations
        if has_excessive_deviations:
            return FeatureInspectionResult.DEFECTIVE
        elif overall_accuracy >= COMPARISON_EXCELLENT_THRESHOLD:
            return FeatureInspectionResult.ACCEPTABLE
        elif overall_accuracy >= COMPARISON_ACCEPTABLE_THRESHOLD:
            return FeatureInspectionResult.DEVIATION
        else:
            return FeatureInspectionResult.DEFECTIVE
    
    def _calculate_match_quality_score(self, match: FeatureMatch, 
                                     deviations: Dict[str, float]) -> float:
        """Calculate overall quality score for a matched feature."""
        base_score = match.geometric_comparison.overall_accuracy
        
        # Apply penalties for excessive deviations
        penalty = 0.0
        
        if deviations.get("center_excessive", False):
            penalty += 0.3
        
        if deviations.get("radius_excessive", False):
            penalty += 0.2
        
        if deviations.get("width_excessive", False) or deviations.get("height_excessive", False):
            penalty += 0.2
        
        # Apply confidence factor
        confidence_factor = match.match_confidence
        
        # Calculate final score
        quality_score = (base_score * confidence_factor) - penalty
        return max(0.0, min(1.0, quality_score))
    
    def _generate_inspection_notes(self, match: FeatureMatch, 
                                 deviations: Dict[str, float],
                                 inspection_result: FeatureInspectionResult) -> List[str]:
        """Generate human-readable inspection notes."""
        notes = []
        
        # Match type information
        notes.append(f"Match type: {match.match_type.value}")
        notes.append(f"Match confidence: {match.match_confidence:.3f}")
        
        # Position deviation notes
        center_dist = deviations["center_distance_mm"]
        if center_dist > COMPARISON_CENTER_TOLERANCE_MM:
            notes.append(f"Excessive center position deviation: {center_dist:.2f}mm (tolerance: {COMPARISON_CENTER_TOLERANCE_MM}mm)")
        elif center_dist > COMPARISON_CENTER_TOLERANCE_MM * 0.5:
            notes.append(f"Moderate center position deviation: {center_dist:.2f}mm")
        else:
            notes.append(f"Good center position accuracy: {center_dist:.2f}mm")
        
        # Size deviation notes
        if "radius_deviation_abs_mm" in deviations:
            radius_dev = deviations["radius_deviation_abs_mm"]
            radius_rel = deviations.get("radius_deviation_relative", 0) * 100
            
            if deviations.get("radius_excessive", False):
                notes.append(f"Excessive radius deviation: {radius_dev:.2f}mm ({radius_rel:.1f}%)")
            else:
                notes.append(f"Acceptable radius deviation: {radius_dev:.2f}mm ({radius_rel:.1f}%)")
        
        # Dimension deviation notes
        if "width_deviation_mm" in deviations:
            width_dev = deviations["width_deviation_mm"]
            if deviations.get("width_excessive", False):
                notes.append(f"Excessive width deviation: {width_dev:.2f}mm")
        
        if "height_deviation_mm" in deviations:
            height_dev = deviations["height_deviation_mm"]
            if deviations.get("height_excessive", False):
                notes.append(f"Excessive height deviation: {height_dev:.2f}mm")
        
        # Overall assessment
        overall_accuracy = match.geometric_comparison.overall_accuracy
        if overall_accuracy >= COMPARISON_EXCELLENT_THRESHOLD:
            notes.append("Excellent geometric accuracy")
        elif overall_accuracy >= COMPARISON_GOOD_THRESHOLD:
            notes.append("Good geometric accuracy") 
        elif overall_accuracy >= COMPARISON_ACCEPTABLE_THRESHOLD:
            notes.append("Acceptable geometric accuracy with minor deviations")
        else:
            notes.append("Poor geometric accuracy - requires attention")
        
        return notes