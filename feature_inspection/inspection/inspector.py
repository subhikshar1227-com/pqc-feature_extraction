"""
Quality Inspection Engine

Performs overall quality assessment and generates inspection results.
"""

import time
from typing import List, Dict, Any, Optional
import logging

from feature_extraction.expected.feature_types import ExpectedFeatureSet, FeatureType
from ..models.actual_feature import ActualFeatureSet
from ..models.feature_match import FeatureMatchSet
from ..models.inspection_result import (
    InspectionResult, InspectionStatus, InspectionSummary, QualityMetrics,
    FeatureInspectionDetail, FeatureInspectionResult
)
from ..comparison.comparator import FeatureComparator
from ..config import (
    INSPECTION_MATCHED_FEATURE_WEIGHT, INSPECTION_MISSING_FEATURE_PENALTY, 
    INSPECTION_EXTRA_FEATURE_PENALTY, INSPECTION_POOR_MATCH_PENALTY,
    INSPECTION_PASS_THRESHOLD, INSPECTION_REVIEW_THRESHOLD,
    QUALITY_COMPLETENESS_WEIGHT, QUALITY_ACCURACY_WEIGHT, QUALITY_CONFIDENCE_WEIGHT,
    MISSING_CRITICAL_FEATURE_PENALTY, MISSING_STANDARD_FEATURE_PENALTY, MISSING_MINOR_FEATURE_PENALTY,
    EXTRA_MAJOR_FEATURE_PENALTY, EXTRA_MINOR_FEATURE_PENALTY,
    FEATURE_CRITICAL_MIN_RADIUS, FEATURE_MAJOR_MIN_RADIUS, FEATURE_MINOR_MAX_RADIUS,
    FEATURE_TYPE_PRIORITIES
)

logger = logging.getLogger(__name__)


class QualityInspector:
    """
    Performs comprehensive quality inspection of manufactured products.
    
    Evaluates feature completeness, geometric accuracy, and overall quality
    to determine pass/fail status and generate detailed inspection reports.
    """
    
    def __init__(self):
        """Initialize quality inspector."""
        self.comparator = FeatureComparator()
    
    def inspect_product(self, expected_set: ExpectedFeatureSet,
                       actual_set: ActualFeatureSet,
                       match_set: FeatureMatchSet,
                       blueprint_name: Optional[str] = None) -> InspectionResult:
        """
        Perform complete quality inspection of a product.
        
        Args:
            expected_set: Expected features from DXF analysis
            actual_set: Detected features from image analysis  
            match_set: Feature matching results
            blueprint_name: Name of identified blueprint
            
        Returns:
            Complete inspection result with pass/fail determination
        """
        start_time = time.time()
        
        logger.info(f"Starting quality inspection for {actual_set.source_image_path.name}")
        
        # Check if matching was actually performed
        matching_performed = self._was_matching_performed(match_set)
        
        if matching_performed:
            # Normal inspection flow - matching was performed
            return self._perform_geometric_inspection(
                expected_set, actual_set, match_set, blueprint_name, start_time
            )
        else:
            # Blocked inspection flow - matching was not performed
            return self._perform_blocked_inspection(
                expected_set, actual_set, match_set, blueprint_name, start_time
            )
    
    def _was_matching_performed(self, match_set: FeatureMatchSet) -> bool:
        """Check if geometric matching was actually performed."""
        # Check configuration for explicit matching_performed flag
        config = match_set.configuration_snapshot
        if isinstance(config, dict):
            if "matching_performed" in config:
                return config["matching_performed"]
            # Legacy check for blocked matching
            if config.get("matching_blocked", False):
                return False
        
        # Check algorithm for blocked matching indicators
        if "blocked" in match_set.matching_algorithm.lower():
            return False
        
        # If there are actual matches, matching was definitely performed
        if len(match_set.matches) > 0:
            return True
        
        # If no matches but unmatched lists are populated, matching was performed
        if len(match_set.unmatched_expected) > 0 or len(match_set.unmatched_actual) > 0:
            return True
        
        # Default: assume matching was not performed
        return False
    
    def _perform_geometric_inspection(self, expected_set: ExpectedFeatureSet,
                                    actual_set: ActualFeatureSet,
                                    match_set: FeatureMatchSet,
                                    blueprint_name: Optional[str],
                                    start_time: float) -> InspectionResult:
        """Perform normal geometric inspection when matching was successful."""
        
        # Analyze feature matches in detail
        feature_details = self.comparator.analyze_matches(match_set)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            expected_set, actual_set, match_set, feature_details
        )
        
        # Generate inspection summary
        inspection_summary = self._generate_inspection_summary(
            expected_set, actual_set, match_set, feature_details
        )
        
        # Determine overall inspection status
        inspection_status = self._determine_inspection_status(
            quality_metrics, inspection_summary, feature_details
        )
        
        # Generate processing notes
        processing_notes = self._generate_processing_notes(
            expected_set, actual_set, match_set, quality_metrics
        )
        
        inspection_duration = time.time() - start_time
        
        # Create inspection result
        result = InspectionResult(
            source_image_path=actual_set.source_image_path,
            expected_feature_set=expected_set,
            actual_feature_set=actual_set,
            feature_match_set=match_set,
            inspection_status=inspection_status,
            quality_metrics=quality_metrics,
            inspection_summary=inspection_summary,
            feature_details=feature_details,
            inspection_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            inspection_duration_seconds=inspection_duration,
            configuration_snapshot=self._get_configuration_snapshot(),
            blueprint_name=blueprint_name,
            dxf_path=expected_set.source_dxf_path,
            processing_notes=processing_notes
        )
        
        logger.info(f"Geometric quality inspection complete: {inspection_status.value} "
                   f"(score: {quality_metrics.overall_quality_score:.3f}) "
                   f"in {inspection_duration:.2f}s")
        
        return result
    
    def _perform_blocked_inspection(self, expected_set: ExpectedFeatureSet,
                                  actual_set: ActualFeatureSet,
                                  match_set: FeatureMatchSet,
                                  blueprint_name: Optional[str],
                                  start_time: float) -> InspectionResult:
        """Perform inspection when matching was blocked due to coordinate system issues."""
        logger.info("Performing blocked inspection - geometric matching was not possible")
        
        # Create feature details for expected and actual features without comparison
        feature_details = []
        
        # Add details for expected features - not missing, just not comparable
        for expected in expected_set.features:
            detail = FeatureInspectionDetail(
                feature_id=expected.feature_id,
                inspection_result=FeatureInspectionResult.NOT_DETERMINED,  # Use appropriate enum value
                quality_score=None,  # No score available
                expected_feature=expected,
                actual_feature=None,
                feature_match=None,
                deviations={},
                quality_metrics={},
                inspection_notes=["Geometric comparison not performed due to coordinate system incompatibility"]
            )
            feature_details.append(detail)
        
        # Add details for actual features - not extra, just not comparable
        for actual in actual_set.features:
            detail = FeatureInspectionDetail(
                feature_id=actual.feature_id,
                inspection_result=FeatureInspectionResult.NOT_DETERMINED,  # Use appropriate enum value
                quality_score=None,  # No score available
                expected_feature=None,
                actual_feature=actual,
                feature_match=None,
                deviations={},
                quality_metrics={},
                inspection_notes=["Geometric comparison not performed due to coordinate system incompatibility"]
            )
            feature_details.append(detail)
        
        # Create blocked quality metrics
        quality_metrics = QualityMetrics(
            completeness_score=None,  # Cannot determine
            accuracy_score=None,      # Cannot determine
            precision_score=None,     # Cannot determine
            confidence_score=sum(f.confidence for f in actual_set.features) / max(1, len(actual_set.features)) if actual_set.features else 0.0,
            overall_quality_score=None,  # Cannot determine
            feature_count_accuracy=None,  # Cannot determine
            geometric_accuracy=None,      # Cannot determine
            detection_reliability=sum(f.confidence for f in actual_set.features) / max(1, len(actual_set.features)) if actual_set.features else 0.0
        )
        
        # Create blocked inspection summary
        inspection_summary = InspectionSummary(
            total_expected_features=len(expected_set.features),
            total_actual_features=len(actual_set.features),
            matched_features=0,
            missing_features=0,  # Not missing - not compared
            extra_features=0,    # Not extra - not compared
            acceptable_features=0,
            defective_features=0,
            expected_circles=len(expected_set.get_features_by_type(FeatureType.CIRCLE)),
            actual_circles=len(actual_set.get_features_by_type(FeatureType.CIRCLE)),
            expected_holes=len(expected_set.get_features_by_type(FeatureType.THROUGH_HOLE)),
            actual_holes=len(actual_set.get_features_by_type(FeatureType.THROUGH_HOLE)),
            critical_issues=["Geometric matching blocked due to coordinate system incompatibility"],
            warnings=[],
            recommendations=["Establish valid pixel-to-CAD/mm coordinate transformation to enable geometric comparison"]
        )
        
        # Status should be ERROR or REVIEW, not PASS/FAIL
        inspection_status = InspectionStatus.REVIEW  # Cannot determine physical quality
        
        # Generate processing notes
        processing_notes = [
            f"Expected features extracted: {len(expected_set.features)} from {expected_set.source_dxf_path.name}",
            f"Actual features detected: {len(actual_set.features)} from {actual_set.source_image_path.name}",
            f"Expected coordinate system: {getattr(expected_set, 'coordinate_system', 'dxf_mm')}",
            f"Actual coordinate system: {actual_set.coordinate_system}",
            "Geometric matching not performed due to coordinate system incompatibility",
            "Physical quality determination requires valid pixel-to-CAD/mm transformation"
        ]
        
        inspection_duration = time.time() - start_time
        
        # Create inspection result
        result = InspectionResult(
            source_image_path=actual_set.source_image_path,
            expected_feature_set=expected_set,
            actual_feature_set=actual_set,
            feature_match_set=match_set,
            inspection_status=inspection_status,
            quality_metrics=quality_metrics,
            inspection_summary=inspection_summary,
            feature_details=feature_details,
            inspection_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            inspection_duration_seconds=inspection_duration,
            configuration_snapshot=self._get_configuration_snapshot(),
            blueprint_name=blueprint_name,
            dxf_path=expected_set.source_dxf_path,
            processing_notes=processing_notes
        )
        
        logger.info(f"Blocked inspection complete: {inspection_status.value} - physical quality not determined")
        
        return result
    
    def _calculate_quality_metrics(self, expected_set: ExpectedFeatureSet,
                                 actual_set: ActualFeatureSet,
                                 match_set: FeatureMatchSet,
                                 feature_details: List[FeatureInspectionDetail]) -> QualityMetrics:
        """Calculate detailed quality metrics."""
        
        # Completeness: ratio of expected features that were successfully found
        total_expected = len(expected_set.features)
        successfully_matched = len([d for d in feature_details 
                                  if d.inspection_result == FeatureInspectionResult.ACCEPTABLE])
        
        completeness_score = successfully_matched / max(1, total_expected)
        
        # Accuracy: average geometric accuracy of matched features
        matched_details = [d for d in feature_details if d.feature_match is not None]
        if matched_details:
            accuracy_score = sum(d.quality_metrics.get("overall_accuracy", 0.0) 
                               for d in matched_details) / len(matched_details)
        else:
            accuracy_score = 0.0
        
        # Precision: ratio of detected features that were expected
        total_actual = len(actual_set.features)
        precision_score = successfully_matched / max(1, total_actual) if total_actual > 0 else 1.0
        
        # Confidence: average detection confidence
        if actual_set.features:
            confidence_score = sum(f.confidence for f in actual_set.features) / len(actual_set.features)
        else:
            confidence_score = 0.0
        
        # Feature count accuracy: how close actual count is to expected count
        count_ratio = min(total_actual, total_expected) / max(total_actual, total_expected, 1)
        feature_count_accuracy = count_ratio if total_expected > 0 else (1.0 if total_actual == 0 else 0.0)
        
        # Geometric accuracy: average of position and size accuracy
        if matched_details:
            position_accuracies = [d.quality_metrics.get("position_accuracy", 0.0) for d in matched_details]
            size_accuracies = [d.quality_metrics.get("size_accuracy", 0.0) for d in matched_details]
            geometric_accuracy = (sum(position_accuracies) + sum(size_accuracies)) / (2 * len(matched_details))
        else:
            geometric_accuracy = 0.0
        
        # Detection reliability: based on confidence distribution and consistency
        detection_reliability = confidence_score  # Simplified for now
        
        # Overall quality score (weighted combination)
        overall_quality_score = (
            completeness_score * QUALITY_COMPLETENESS_WEIGHT +
            accuracy_score * QUALITY_ACCURACY_WEIGHT +
            confidence_score * QUALITY_CONFIDENCE_WEIGHT
        )
        
        return QualityMetrics(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            precision_score=precision_score,
            confidence_score=confidence_score,
            overall_quality_score=overall_quality_score,
            feature_count_accuracy=feature_count_accuracy,
            geometric_accuracy=geometric_accuracy,
            detection_reliability=detection_reliability
        )
    
    def _generate_inspection_summary(self, expected_set: ExpectedFeatureSet,
                                   actual_set: ActualFeatureSet,
                                   match_set: FeatureMatchSet,
                                   feature_details: List[FeatureInspectionDetail]) -> InspectionSummary:
        """Generate high-level inspection summary."""
        
        # Count features by inspection result
        acceptable_count = len([d for d in feature_details 
                              if d.inspection_result == FeatureInspectionResult.ACCEPTABLE])
        defective_count = len([d for d in feature_details 
                             if d.inspection_result == FeatureInspectionResult.DEFECTIVE])
        missing_count = len([d for d in feature_details 
                           if d.inspection_result == FeatureInspectionResult.MISSING])
        extra_count = len([d for d in feature_details 
                         if d.inspection_result == FeatureInspectionResult.UNEXPECTED])
        
        # Count by feature type
        expected_circles = len(expected_set.get_features_by_type(FeatureType.CIRCLE))
        actual_circles = len(actual_set.get_features_by_type(FeatureType.CIRCLE))
        expected_holes = len(expected_set.get_features_by_type(FeatureType.THROUGH_HOLE))
        actual_holes = len(actual_set.get_features_by_type(FeatureType.THROUGH_HOLE))
        
        # Generate issues and recommendations
        critical_issues = []
        warnings = []
        recommendations = []
        
        if missing_count > 0:
            critical_issues.append(f"{missing_count} expected features not detected")
        
        if defective_count > 0:
            critical_issues.append(f"{defective_count} features have excessive geometric deviations")
        
        if extra_count > 0:
            warnings.append(f"{extra_count} unexpected features detected")
        
        if expected_circles != actual_circles:
            warnings.append(f"Circle count mismatch: expected {expected_circles}, found {actual_circles}")
        
        if expected_holes != actual_holes:
            warnings.append(f"Hole count mismatch: expected {expected_holes}, found {actual_holes}")
        
        # Generate recommendations
        if missing_count > 0:
            recommendations.append("Verify manufacturing process for missing features")
        
        if defective_count > 0:
            recommendations.append("Check dimensional accuracy and tooling calibration")
        
        if extra_count > 0:
            recommendations.append("Review manufacturing process for unexpected features")
        
        return InspectionSummary(
            total_expected_features=len(expected_set.features),
            total_actual_features=len(actual_set.features),
            matched_features=len(match_set.matches),
            missing_features=missing_count,
            extra_features=extra_count,
            acceptable_features=acceptable_count,
            defective_features=defective_count,
            expected_circles=expected_circles,
            actual_circles=actual_circles,
            expected_holes=expected_holes,
            actual_holes=actual_holes,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _determine_inspection_status(self, quality_metrics: QualityMetrics,
                                   inspection_summary: InspectionSummary,
                                   feature_details: List[FeatureInspectionDetail]) -> InspectionStatus:
        """Determine overall inspection status based on quality metrics and issues."""
        
        # Check for critical issues
        if len(inspection_summary.critical_issues) > 0:
            # Check if issues are severe enough to fail
            missing_critical = self._count_missing_critical_features(feature_details)
            if missing_critical > 0 or inspection_summary.defective_features > 1:
                return InspectionStatus.FAIL
        
        # Use overall quality score for decision
        overall_score = quality_metrics.overall_quality_score
        
        if overall_score >= INSPECTION_PASS_THRESHOLD:
            return InspectionStatus.PASS
        elif overall_score >= INSPECTION_REVIEW_THRESHOLD:
            return InspectionStatus.REVIEW
        else:
            return InspectionStatus.FAIL
    
    def _count_missing_critical_features(self, feature_details: List[FeatureInspectionDetail]) -> int:
        """Count missing features that are considered critical."""
        count = 0
        
        for detail in feature_details:
            if (detail.inspection_result == FeatureInspectionResult.MISSING and 
                detail.expected_feature is not None):
                
                # Determine if this is a critical feature
                if self._is_critical_feature(detail.expected_feature):
                    count += 1
        
        return count
    
    def _is_critical_feature(self, expected_feature) -> bool:
        """Determine if an expected feature is critical."""
        # Size-based criticality
        if expected_feature.radius and expected_feature.radius >= FEATURE_CRITICAL_MIN_RADIUS:
            return True
        
        # Type-based criticality
        feature_priority = FEATURE_TYPE_PRIORITIES.get(expected_feature.feature_type.value, 0.5)
        return feature_priority >= 1.0
    
    def _calculate_feature_penalty(self, feature_detail: FeatureInspectionDetail) -> float:
        """Calculate quality penalty for a feature issue."""
        if feature_detail.inspection_result == FeatureInspectionResult.MISSING:
            if feature_detail.expected_feature:
                if self._is_critical_feature(feature_detail.expected_feature):
                    return MISSING_CRITICAL_FEATURE_PENALTY
                elif self._is_major_feature(feature_detail.expected_feature):
                    return MISSING_STANDARD_FEATURE_PENALTY
                else:
                    return MISSING_MINOR_FEATURE_PENALTY
            return MISSING_STANDARD_FEATURE_PENALTY
        
        elif feature_detail.inspection_result == FeatureInspectionResult.UNEXPECTED:
            if feature_detail.actual_feature:
                if self._is_major_actual_feature(feature_detail.actual_feature):
                    return EXTRA_MAJOR_FEATURE_PENALTY
                else:
                    return EXTRA_MINOR_FEATURE_PENALTY
            return EXTRA_MINOR_FEATURE_PENALTY
        
        elif feature_detail.inspection_result == FeatureInspectionResult.DEFECTIVE:
            return INSPECTION_POOR_MATCH_PENALTY
        
        return 0.0
    
    def _is_major_feature(self, expected_feature) -> bool:
        """Determine if an expected feature is major (but not critical)."""
        if expected_feature.radius:
            return expected_feature.radius >= FEATURE_MAJOR_MIN_RADIUS
        return True  # Default to major if size unknown
    
    def _is_major_actual_feature(self, actual_feature) -> bool:
        """Determine if an actual feature is major.""" 
        if actual_feature.radius:
            return actual_feature.radius >= 10.0  # Converted to pixels approximately
        return True  # Default to major if size unknown
    
    def _generate_processing_notes(self, expected_set: ExpectedFeatureSet,
                                 actual_set: ActualFeatureSet,
                                 match_set: FeatureMatchSet,
                                 quality_metrics: QualityMetrics) -> List[str]:
        """Generate processing notes for the inspection."""
        notes = []
        
        # Processing summary
        notes.append(f"Processed {expected_set.source_dxf_path.name} against {actual_set.source_image_path.name}")
        notes.append(f"Expected features: {len(expected_set.features)}, Detected features: {len(actual_set.features)}")
        notes.append(f"Successful matches: {len(match_set.acceptable_matches)}/{len(match_set.matches)}")
        
        # Quality assessment
        if quality_metrics.overall_quality_score >= INSPECTION_PASS_THRESHOLD:
            notes.append("Product meets quality standards")
        elif quality_metrics.overall_quality_score >= INSPECTION_REVIEW_THRESHOLD:
            notes.append("Product requires review - minor quality concerns")
        else:
            notes.append("Product does not meet quality standards")
        
        # Detection performance
        if actual_set.detection_statistics.average_confidence >= 0.8:
            notes.append("High confidence detections")
        elif actual_set.detection_statistics.average_confidence >= 0.6:
            notes.append("Moderate confidence detections")
        else:
            notes.append("Low confidence detections - image quality may be poor")
        
        return notes
    
    def _get_configuration_snapshot(self) -> Dict[str, Any]:
        """Get current configuration for reproducibility."""
        return {
            "pass_threshold": INSPECTION_PASS_THRESHOLD,
            "review_threshold": INSPECTION_REVIEW_THRESHOLD,
            "matched_feature_weight": INSPECTION_MATCHED_FEATURE_WEIGHT,
            "missing_feature_penalty": INSPECTION_MISSING_FEATURE_PENALTY,
            "extra_feature_penalty": INSPECTION_EXTRA_FEATURE_PENALTY,
            "poor_match_penalty": INSPECTION_POOR_MATCH_PENALTY,
            "quality_weights": {
                "completeness": QUALITY_COMPLETENESS_WEIGHT,
                "accuracy": QUALITY_ACCURACY_WEIGHT,
                "confidence": QUALITY_CONFIDENCE_WEIGHT
            }
        }