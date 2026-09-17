"""
JSON Serialization for Phase 2 Data Structures

Handles serialization of Phase 2 data models to JSON format.
"""

import json
import numpy as np
from typing import Any, Dict, List, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from feature_extraction.expected.feature_types import Point2D, FeatureType
from ..models.actual_feature import ActualFeature, ActualFeatureSet, DetectionMethod
from ..models.feature_match import FeatureMatch, FeatureMatchSet, MatchType
from ..models.inspection_result import InspectionResult, InspectionStatus


class Phase2JSONEncoder:
    """
    Custom JSON encoder for Phase 2 data structures.
    
    Converts Python objects to JSON-serializable format while preserving
    all necessary information for audit and reproduction.
    """
    
    @staticmethod
    def encode_point2d(point: Point2D) -> Dict[str, float]:
        """Encode Point2D to JSON."""
        return {"x": float(point.x), "y": float(point.y)}
    
    @staticmethod
    def encode_enum(enum_value: Enum) -> str:
        """Encode enum to JSON string."""
        return enum_value.value if enum_value is not None else None
    
    @staticmethod
    def encode_numpy_array(array: np.ndarray) -> List[List[float]]:
        """Encode numpy array to nested JSON list."""
        return array.tolist() if array is not None else None
    
    @staticmethod
    def encode_path(path: Path) -> str:
        """Encode Path to JSON string."""
        return str(path) if path is not None else None
    
    @staticmethod
    def encode_actual_feature(feature: ActualFeature) -> Dict[str, Any]:
        """Encode ActualFeature to JSON."""
        return {
            "feature_id": feature.feature_id,
            "feature_type": Phase2JSONEncoder.encode_enum(feature.feature_type),
            "confidence": float(feature.confidence),
            "center": Phase2JSONEncoder.encode_point2d(feature.center),
            "radius": float(feature.radius) if feature.radius is not None else None,
            "width": float(feature.width) if feature.width is not None else None,
            "height": float(feature.height) if feature.height is not None else None,
            "detection_method": Phase2JSONEncoder.encode_enum(feature.detection_method),
            "coordinate_system": feature.coordinate_system,
            "detection_evidence": feature.detection_evidence,
            "quality_metrics": feature.quality_metrics,
            "contour_area": float(feature.contour_area) if feature.contour_area is not None else None,
            "bounding_box": list(feature.bounding_box) if feature.bounding_box is not None else None,
            "area": float(feature.area) if hasattr(feature, 'area') and feature.area is not None else None,
            "perimeter": float(feature.perimeter) if hasattr(feature, 'perimeter') and feature.perimeter is not None else None
        }
    
    @staticmethod
    def encode_actual_feature_set(feature_set: ActualFeatureSet) -> Dict[str, Any]:
        """Encode ActualFeatureSet to JSON."""
        return {
            "source_image_path": Phase2JSONEncoder.encode_path(feature_set.source_image_path),
            "features": [Phase2JSONEncoder.encode_actual_feature(f) for f in feature_set.features],
            "detection_timestamp": feature_set.detection_timestamp,
            "detection_statistics": {
                "total_contours_found": feature_set.detection_statistics.total_contours_found,
                "contours_after_filtering": feature_set.detection_statistics.contours_after_filtering,
                "circles_detected": feature_set.detection_statistics.circles_detected,
                "through_holes_detected": feature_set.detection_statistics.through_holes_detected,
                "rectangular_holes_detected": feature_set.detection_statistics.rectangular_holes_detected,
                "total_features_detected": feature_set.detection_statistics.total_features_detected,
                "average_confidence": float(feature_set.detection_statistics.average_confidence),
                "detection_time_seconds": float(feature_set.detection_statistics.detection_time_seconds),
                "preprocessing_time_seconds": float(feature_set.detection_statistics.preprocessing_time_seconds)
            },
            "configuration_snapshot": feature_set.configuration_snapshot,
            "image_dimensions": list(feature_set.image_dimensions) if feature_set.image_dimensions is not None else None,
            "preprocessing_applied": feature_set.preprocessing_applied,
            "coordinate_system": feature_set.coordinate_system,
            "transform_matrix": Phase2JSONEncoder.encode_numpy_array(feature_set.transform_matrix),
            
            # Computed properties
            "total_feature_count": feature_set.total_feature_count,
            "circle_count": feature_set.circle_count,
            "through_hole_count": feature_set.through_hole_count,
            "rectangular_hole_count": feature_set.rectangular_hole_count,
            "square_hole_count": feature_set.square_hole_count,
            "average_confidence": float(feature_set.average_confidence)
        }
    
    @staticmethod 
    def encode_feature_match(match: FeatureMatch) -> Dict[str, Any]:
        """Encode FeatureMatch to JSON."""
        return {
            "expected_feature": {
                "feature_id": match.expected_feature.feature_id,
                "feature_type": Phase2JSONEncoder.encode_enum(match.expected_feature.feature_type),
                "center": Phase2JSONEncoder.encode_point2d(match.expected_feature.center),
                "radius": float(match.expected_feature.radius) if match.expected_feature.radius is not None else None,
                "confidence": float(match.expected_feature.confidence)
            } if match.expected_feature is not None else None,
            
            "actual_feature": Phase2JSONEncoder.encode_actual_feature(match.actual_feature) if match.actual_feature is not None else None,
            
            "match_type": Phase2JSONEncoder.encode_enum(match.match_type),
            "match_confidence": float(match.match_confidence),
            "match_method": match.match_method,
            "match_score": float(match.match_score),
            "match_evidence": match.match_evidence,
            
            "geometric_comparison": {
                "center_deviation_mm": float(match.geometric_comparison.center_deviation_mm),
                "radius_deviation_mm": float(match.geometric_comparison.radius_deviation_mm) if match.geometric_comparison.radius_deviation_mm is not None else None,
                "radius_deviation_relative": float(match.geometric_comparison.radius_deviation_relative) if match.geometric_comparison.radius_deviation_relative is not None else None,
                "position_accuracy": float(match.geometric_comparison.position_accuracy),
                "size_accuracy": float(match.geometric_comparison.size_accuracy),
                "overall_accuracy": float(match.geometric_comparison.overall_accuracy)
            } if match.geometric_comparison is not None else None
        }
    
    @staticmethod
    def encode_feature_match_set(match_set: FeatureMatchSet) -> Dict[str, Any]:
        """Encode FeatureMatchSet to JSON."""
        return {
            "matches": [Phase2JSONEncoder.encode_feature_match(m) for m in match_set.matches],
            "unmatched_expected": [
                {
                    "feature_id": f.feature_id,
                    "feature_type": Phase2JSONEncoder.encode_enum(f.feature_type),
                    "center": Phase2JSONEncoder.encode_point2d(f.center),
                    "radius": float(f.radius) if f.radius is not None else None
                }
                for f in match_set.unmatched_expected
            ],
            "unmatched_actual": [Phase2JSONEncoder.encode_actual_feature(f) for f in match_set.unmatched_actual],
            "matching_timestamp": match_set.matching_timestamp,
            "matching_statistics": {
                "total_expected_features": match_set.matching_statistics.total_expected_features,
                "total_actual_features": match_set.matching_statistics.total_actual_features,
                "successful_matches": match_set.matching_statistics.successful_matches,
                "exact_matches": match_set.matching_statistics.exact_matches,
                "good_matches": match_set.matching_statistics.good_matches,
                "acceptable_matches": match_set.matching_statistics.acceptable_matches,
                "poor_matches": match_set.matching_statistics.poor_matches,
                "unmatched_expected": match_set.matching_statistics.unmatched_expected,
                "unmatched_actual": match_set.matching_statistics.unmatched_actual,
                "average_match_confidence": float(match_set.matching_statistics.average_match_confidence),
                "average_geometric_accuracy": float(match_set.matching_statistics.average_geometric_accuracy),
                "matching_time_seconds": float(match_set.matching_statistics.matching_time_seconds)
            },
            "matching_algorithm": match_set.matching_algorithm,
            "configuration_snapshot": match_set.configuration_snapshot,
            
            # Computed properties
            "total_matches": len(match_set.matches),
            "acceptable_matches": len(match_set.acceptable_matches)
        }
    
    @staticmethod
    def encode_inspection_result(result: InspectionResult) -> Dict[str, Any]:
        """Encode InspectionResult to JSON."""
        return {
            "source_image_path": Phase2JSONEncoder.encode_path(result.source_image_path),
            "dxf_path": Phase2JSONEncoder.encode_path(result.dxf_path) if result.dxf_path is not None else None,
            "blueprint_name": result.blueprint_name,
            
            "inspection_status": Phase2JSONEncoder.encode_enum(result.inspection_status),
            "inspection_timestamp": result.inspection_timestamp,
            "inspection_duration_seconds": float(result.inspection_duration_seconds),
            
            "quality_metrics": {
                "completeness_score": float(result.quality_metrics.completeness_score) if result.quality_metrics.completeness_score is not None else None,
                "accuracy_score": float(result.quality_metrics.accuracy_score) if result.quality_metrics.accuracy_score is not None else None,
                "precision_score": float(result.quality_metrics.precision_score) if result.quality_metrics.precision_score is not None else None,
                "confidence_score": float(result.quality_metrics.confidence_score),
                "overall_quality_score": float(result.quality_metrics.overall_quality_score) if result.quality_metrics.overall_quality_score is not None else None,
                "feature_count_accuracy": float(result.quality_metrics.feature_count_accuracy) if result.quality_metrics.feature_count_accuracy is not None else None,
                "geometric_accuracy": float(result.quality_metrics.geometric_accuracy) if result.quality_metrics.geometric_accuracy is not None else None,
                "detection_reliability": float(result.quality_metrics.detection_reliability)
            },
            
            "inspection_summary": {
                "total_expected_features": result.inspection_summary.total_expected_features,
                "total_actual_features": result.inspection_summary.total_actual_features,
                "matched_features": result.inspection_summary.matched_features,
                "missing_features": result.inspection_summary.missing_features,
                "extra_features": result.inspection_summary.extra_features,
                "acceptable_features": result.inspection_summary.acceptable_features,
                "defective_features": result.inspection_summary.defective_features,
                "expected_circles": result.inspection_summary.expected_circles,
                "actual_circles": result.inspection_summary.actual_circles,
                "expected_holes": result.inspection_summary.expected_holes,
                "actual_holes": result.inspection_summary.actual_holes,
                "critical_issues": result.inspection_summary.critical_issues,
                "warnings": result.inspection_summary.warnings,
                "recommendations": result.inspection_summary.recommendations
            },
            
            "feature_details": [
                {
                    "feature_id": detail.feature_id,
                    "inspection_result": Phase2JSONEncoder.encode_enum(detail.inspection_result),
                    "quality_score": float(detail.quality_score) if detail.quality_score is not None else None,
                    "expected_feature_id": detail.expected_feature.feature_id if detail.expected_feature is not None else None,
                    "actual_feature_id": detail.actual_feature.feature_id if detail.actual_feature is not None else None,
                    "quality_metrics": detail.quality_metrics,
                    "deviations": detail.deviations,
                    "inspection_notes": detail.inspection_notes
                }
                for detail in result.feature_details
            ],
            
            "configuration_snapshot": result.configuration_snapshot,
            "processing_notes": result.processing_notes,
            
            # Include referenced data
            "expected_feature_set_summary": {
                "total_features": len(result.expected_feature_set.features),
                "source_dxf": Phase2JSONEncoder.encode_path(result.expected_feature_set.source_dxf_path),
                "extraction_timestamp": result.expected_feature_set.extraction_timestamp
            },
            
            "actual_feature_set_summary": {
                "total_features": len(result.actual_feature_set.features),
                "coordinate_system": result.actual_feature_set.coordinate_system,
                "detection_timestamp": result.actual_feature_set.detection_timestamp,
                "average_confidence": float(result.actual_feature_set.average_confidence)
            },
            
            "matching_summary": {
                "total_matches": len(result.feature_match_set.matches),
                "matching_algorithm": result.feature_match_set.matching_algorithm,
                "matching_timestamp": result.feature_match_set.matching_timestamp
            }
        }


def save_json(data: Any, output_path: Path, encoder_method: str = None) -> None:
    """
    Save data to JSON file using appropriate encoder.
    
    Args:
        data: Data object to serialize
        output_path: Path where JSON will be saved
        encoder_method: Specific encoder method to use
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine encoder method if not specified
    if encoder_method is None:
        if isinstance(data, ActualFeatureSet):
            encoder_method = "encode_actual_feature_set"
        elif isinstance(data, FeatureMatchSet):
            encoder_method = "encode_feature_match_set"  
        elif isinstance(data, InspectionResult):
            encoder_method = "encode_inspection_result"
        else:
            raise ValueError(f"No encoder method available for type {type(data)}")
    
    # Encode data
    encoder_func = getattr(Phase2JSONEncoder, encoder_method)
    json_data = encoder_func(data)
    
    # Add metadata
    json_data["_metadata"] = {
        "serialization_timestamp": datetime.now().isoformat(),
        "serializer_version": "1.0.0",
        "data_type": type(data).__name__,
        "encoder_method": encoder_method
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {type(data).__name__} to {output_path}")


def load_json(input_path: Path) -> Dict[str, Any]:
    """
    Load JSON data from file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)