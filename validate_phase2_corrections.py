#!/usr/bin/env python3
"""
Phase 2 Corrections Validation Script

Test the corrected Phase 2 actual feature detector with real images
to validate all the fixes are working properly.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging

from feature_inspection.actual.detector import ActualFeatureDetector
from feature_extraction.expected.feature_types import FeatureType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_detection_corrections():
    """Validate all the Phase 2 corrections with real images."""
    
    print("=" * 70)
    print("Phase 2 Actual Feature Detection Corrections Validation")
    print("=" * 70)
    
    # Test with real images
    input_dir = Path("data/inputs")
    if not input_dir.exists():
        print("ERROR: Input directory not found")
        return
    
    image_files = list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not image_files:
        print("ERROR: No image files found")
        return
    
    detector = ActualFeatureDetector()
    
    print(f"\nTesting {len(image_files)} real image(s):\n")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 50)
        
        try:
            # Run detection
            feature_set = detector.detect_features(image_path)
            
            # Validate corrections
            validation_results = validate_feature_set_corrections(feature_set, image_path)
            
            # Report results
            print_detection_results(feature_set, validation_results)
            
        except Exception as e:
            print(f"ERROR: Detection failed - {e}")
            continue
        
        print()


def validate_feature_set_corrections(feature_set, image_path):
    """Validate that all corrections are working in the feature set."""
    
    results = {
        "coordinate_system_correct": False,
        "confidence_bounds_ok": False,
        "local_contrast_evidence": False,
        "original_coordinates": False,
        "no_fake_transforms": False,
        "preprocessing_successful": False
    }
    
    # Check coordinate system
    if feature_set.coordinate_system == "image_pixels":
        results["coordinate_system_correct"] = True
    
    # Check confidence bounds
    if all(0.0 <= f.confidence <= 1.0 for f in feature_set.features):
        results["confidence_bounds_ok"] = True
    
    # Check for local contrast evidence in holes
    hole_features = [f for f in feature_set.features 
                    if f.feature_type in [FeatureType.THROUGH_HOLE, FeatureType.SQUARE_HOLE, FeatureType.RECTANGULAR_HOLE]]
    
    if hole_features:
        has_contrast_evidence = all(
            "local_contrast" in f.detection_evidence for f in hole_features
        )
        results["local_contrast_evidence"] = has_contrast_evidence
    else:
        results["local_contrast_evidence"] = True  # No holes to check
    
    # Check original image coordinates (should be reasonable for image size)
    if feature_set.image_dimensions and feature_set.features:
        width, height = feature_set.image_dimensions
        coords_reasonable = all(
            0 <= f.center.x <= width and 0 <= f.center.y <= height
            for f in feature_set.features
        )
        results["original_coordinates"] = coords_reasonable
    else:
        results["original_coordinates"] = True  # No features to check
    
    # Check no fake transforms (transform_matrix should be None without valid calibration)
    results["no_fake_transforms"] = feature_set.transform_matrix is None
    
    # Check preprocessing success
    results["preprocessing_successful"] = "phase2_preprocessing" in feature_set.preprocessing_applied
    
    return results


def print_detection_results(feature_set, validation_results):
    """Print comprehensive detection results and validation status."""
    
    print(f"Image: {feature_set.source_image_path.name}")
    print(f"Dimensions: {feature_set.image_dimensions[0]}x{feature_set.image_dimensions[1]} (W×H)")
    print(f"Coordinate System: {feature_set.coordinate_system}")
    print()
    
    # Detection statistics
    stats = feature_set.detection_statistics
    print("Detection Results:")
    print(f"  Total features: {stats.total_features_detected}")
    print(f"  Circles: {stats.circles_detected}")
    print(f"  Through holes: {stats.through_holes_detected}") 
    print(f"  Rectangular holes: {stats.rectangular_holes_detected}")
    print(f"  Average confidence: {stats.average_confidence:.3f}")
    print(f"  Detection time: {stats.detection_time_seconds:.2f}s")
    print(f"  Preprocessing time: {stats.preprocessing_time_seconds:.2f}s")
    print()
    
    # Feature details
    if feature_set.features:
        print("Feature Details:")
        for i, feature in enumerate(feature_set.features, 1):
            print(f"  {i}. {feature.feature_type.value}")
            print(f"     Center: ({feature.center.x:.1f}, {feature.center.y:.1f})")
            print(f"     Confidence: {feature.confidence:.3f}")
            print(f"     Method: {feature.detection_method.value}")
            
            # Show hole evidence if available
            if (feature.feature_type in [FeatureType.THROUGH_HOLE, FeatureType.SQUARE_HOLE, FeatureType.RECTANGULAR_HOLE] 
                and feature.detection_evidence):
                if "local_contrast" in feature.detection_evidence:
                    contrast = feature.detection_evidence["local_contrast"]
                    print(f"     Local contrast: {contrast:.1f}")
                if "contrast_ratio" in feature.detection_evidence:
                    ratio = feature.detection_evidence["contrast_ratio"]
                    print(f"     Contrast ratio: {ratio:.3f}")
            print()
    
    # Validation results
    print("Correction Validation:")
    checks = {
        "coordinate_system_correct": "✓ Coordinate system is image_pixels",
        "confidence_bounds_ok": "✓ All confidences in [0,1]", 
        "local_contrast_evidence": "✓ Holes use local contrast evidence",
        "original_coordinates": "✓ Features in original image coordinates",
        "no_fake_transforms": "✓ No fake pixel→mm transforms",
        "preprocessing_successful": "✓ Phase 2 preprocessing applied"
    }
    
    all_passed = True
    for check, message in checks.items():
        if validation_results[check]:
            print(f"  {message}")
        else:
            print(f"  ✗ {message.replace('✓', 'FAILED:')}")
            all_passed = False
    
    print()
    if all_passed:
        print("✓ ALL CORRECTIONS VALIDATED")
    else:
        print("✗ SOME CORRECTIONS FAILED") 
    
    print()


if __name__ == "__main__":
    validate_detection_corrections()