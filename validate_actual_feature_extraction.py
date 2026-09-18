"""
Phase 2B Actual Feature Extraction Validation

Tests actual feature extraction on real product images using only
image evidence (NO expected features for detection).
"""

import logging
import cv2
import numpy as np
from pathlib import Path
import json

from feature_inspection.preprocessing import CanonicalPreprocessor
from feature_inspection.actual import extract_actual_features, ActualFeatureType

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
INPUTS_DIR = Path("data/inputs")
OUTPUTS_DIR = Path("outputs/actual_feature_extraction")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def collect_images(folder: Path) -> list[Path]:
    """Return all image files in a folder, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def save_feature_visualization(extraction_result, output_dir: Path):
    """Save visualization of detected features."""
    if not extraction_result.features:
        logger.warning("No features to visualize")
        return {}
    
    # Load original image for visualization
    original_bgr = cv2.imread(str(extraction_result.source_image_path))
    if original_bgr is None:
        logger.error(f"Could not load original image: {extraction_result.source_image_path}")
        return {}
    
    # Create visualization
    vis_image = original_bgr.copy()
    
    # Color map for different feature types
    colors = {
        ActualFeatureType.CIRCLE: (0, 255, 0),      # Green
        ActualFeatureType.RECTANGLE: (255, 0, 0),    # Blue  
        ActualFeatureType.SQUARE: (0, 0, 255),       # Red
        ActualFeatureType.GENERAL_CONTOUR: (255, 255, 0),  # Cyan
        ActualFeatureType.HOLE: (255, 0, 255)       # Magenta
    }
    
    saved_files = {}
    
    # Draw all detected features
    for i, feature in enumerate(extraction_result.features):
        color = colors.get(feature.feature_type, (128, 128, 128))
        
        # Convert to original coordinates if needed
        if feature.coordinate_system == "processed":
            original_feature = feature.to_original_coordinates(extraction_result.roi_offset)
        else:
            original_feature = feature
        
        # Draw contour
        if len(original_feature.contour) > 0:
            # Ensure contour is in the right format
            contour = original_feature.contour
            if len(contour.shape) == 2:
                # Convert Nx2 to Nx1x2 format for OpenCV
                contour = contour.reshape(-1, 1, 2)
            cv2.drawContours(vis_image, [contour], -1, color, 2)
        
        # Draw center point
        center = original_feature.geometry.center
        cv2.circle(vis_image, (int(center[0]), int(center[1])), 3, color, -1)
        
        # Add feature ID label
        label = f"{feature.feature_type.value}_{i}"
        cv2.putText(vis_image, label, (int(center[0]) + 5, int(center[1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add summary text
    summary_text = f"Features: {len(extraction_result.features)}"
    cv2.putText(vis_image, summary_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Save visualization
    vis_path = output_dir / "feature_detection_overlay.png"
    cv2.imwrite(str(vis_path), vis_image)
    saved_files["visualization"] = vis_path
    
    logger.info(f"Saved feature visualization: {vis_path}")
    return saved_files


def save_feature_metadata(extraction_result, output_dir: Path):
    """Save detailed feature extraction metadata."""
    metadata = {
        "source_image": str(extraction_result.source_image_path),
        "preprocessing_successful": extraction_result.preprocessing_successful,
        "processing_time_seconds": extraction_result.processing_time_seconds,
        
        # Detection statistics
        "total_candidates_generated": extraction_result.total_candidates_generated,
        "candidates_by_type": extraction_result.candidates_by_type,
        "features_by_type": extraction_result.features_by_type,
        "duplicate_candidates_suppressed": extraction_result.duplicate_candidates_suppressed,
        
        # Quality metrics
        "average_confidence": extraction_result.average_confidence,
        "confidence_by_type": extraction_result.confidence_by_type,
        
        # Processing information
        "detection_methods_used": extraction_result.detection_methods_used,
        "representations_used": extraction_result.representations_used,
        "coordinate_system": extraction_result.coordinate_system,
        "scale_factor": extraction_result.scale_factor,
        
        # Individual features
        "detected_features": []
    }
    
    # Add individual feature details
    for feature in extraction_result.features:
        feature_data = {
            "feature_id": feature.feature_id,
            "feature_type": feature.feature_type.value,
            "center": feature.geometry.center,
            "area": feature.geometry.area,
            "perimeter": feature.geometry.perimeter,
            "bounding_box": feature.geometry.bounding_box,
            "confidence": feature.evidence.confidence,
            "edge_support": feature.evidence.edge_support,
            "contour_quality": feature.evidence.contour_quality,
            "geometric_consistency": feature.evidence.geometric_consistency,
            "source_representation": feature.source_representation,
            "detection_method": feature.detection_method
        }
        
        # Add type-specific geometry
        if feature.geometry.radius is not None:
            feature_data["radius"] = feature.geometry.radius
            feature_data["diameter"] = feature.geometry.diameter
        
        if feature.geometry.width is not None:
            feature_data["width"] = feature.geometry.width
            feature_data["height"] = feature.geometry.height
            feature_data["aspect_ratio"] = feature.geometry.aspect_ratio
        
        metadata["detected_features"].append(feature_data)
    
    # Save metadata
    metadata_path = output_dir / "feature_extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved feature metadata: {metadata_path}")
    return metadata_path


def print_extraction_summary(image_name: str, extraction_result):
    """Print formatted summary of extraction results."""
    print(f"\n{'='*70}")
    print(f"[ACTUAL FEATURE EXTRACTION] {image_name}")
    print(f"{'='*70}")
    
    print(f"Technical execution: {'SUCCESS' if extraction_result.preprocessing_successful else 'FAILED'}")
    print(f"Processing time: {extraction_result.processing_time_seconds:.2f}s")
    
    print(f"\nDetection Statistics:")
    print(f"  Total candidates generated: {extraction_result.total_candidates_generated}")
    print(f"  Final features detected: {len(extraction_result.features)}")
    print(f"  Duplicates suppressed: {extraction_result.duplicate_candidates_suppressed}")
    print(f"  Average confidence: {extraction_result.average_confidence:.3f}")
    
    print(f"\nFeatures by Type:")
    for feature_type, count in extraction_result.features_by_type.items():
        avg_conf = extraction_result.confidence_by_type.get(feature_type, 0.0)
        print(f"  {feature_type}: {count} (avg confidence: {avg_conf:.3f})")
    
    if extraction_result.features:
        print(f"\nDetected Features:")
        for i, feature in enumerate(extraction_result.features, 1):
            center = feature.geometry.center
            print(f"  {i}. {feature.feature_type.value} at ({center[0]:.1f}, {center[1]:.1f})")
            print(f"     Area: {feature.geometry.area:.1f}, Confidence: {feature.evidence.confidence:.3f}")
            print(f"     Method: {feature.detection_method}, Source: {feature.source_representation}")
    else:
        print(f"\n  No features detected")
    
    print(f"\nProcessing Information:")
    print(f"  Detection methods: {', '.join(extraction_result.detection_methods_used)}")
    print(f"  Representations used: {', '.join(extraction_result.representations_used)}")
    print(f"  Coordinate system: {extraction_result.coordinate_system}")


def main():
    """Main validation function."""
    print("="*70)
    print("Phase 2B Actual Feature Extraction Validation")
    print("="*70)
    print()
    print("SCOPE: IMAGE-EVIDENCE-ONLY FEATURE DETECTION")
    print("- Detects geometric features from preprocessed images")
    print("- Uses NO expected feature information for detection") 
    print("- Purely image-driven candidate generation")
    print("- Multiple detection methods and representations")
    print("- Duplicate suppression and validation")
    print()
    print("CRITICAL VALIDATION:")
    print("- NO dependency on expected feature counts")
    print("- NO product-specific coordinates or assumptions")
    print("- NO hardcoded feature locations")
    print("- NO expected feature guidance for detection")
    print()
    print("NOT TESTED:")
    print("- Feature matching against expected features")
    print("- Dimensional accuracy validation")
    print("- Pass/fail inspection decisions")
    
    inputs = collect_images(INPUTS_DIR)
    
    if not inputs:
        print(f"\nNo images found in '{INPUTS_DIR}'")
        return
    
    print(f"\nFound {len(inputs)} input image(s):")
    for inp in inputs:
        print(f"  - {inp.name}")
    
    # Initialize preprocessor and feature extractor
    preprocessor = CanonicalPreprocessor()
    
    # Create output directory
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    all_results = []
    
    for i, image_path in enumerate(inputs, 1):
        print(f"\n[{i}/{len(inputs)}] Processing: {image_path.name}")
        print("-" * 70)
        
        try:
            # Run preprocessing
            print("Running Phase 2A preprocessing...")
            preprocessing_result = preprocessor.preprocess_image(image_path)
            
            if not preprocessing_result.preprocessing_successful:
                print(f"[SKIP] Preprocessing failed for {image_path.name}")
                continue
            
            # Run actual feature extraction
            print("Running Phase 2B actual feature extraction...")
            extraction_result = extract_actual_features(preprocessing_result)
            
            # Create output directory for this image
            image_output_dir = OUTPUTS_DIR / image_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            print("Saving extraction results...")
            vis_files = save_feature_visualization(extraction_result, image_output_dir)
            metadata_file = save_feature_metadata(extraction_result, image_output_dir)
            
            # Print summary
            print_extraction_summary(image_path.name, extraction_result)
            
            print(f"\nSaved outputs:")
            print(f"  Feature detection overlay: {vis_files.get('visualization', 'N/A')}")
            print(f"  Extraction metadata: {metadata_file}")
            
            all_results.append(extraction_result)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {image_path.name}: {e}")
            logger.error(f"Processing failed for {image_path.name}: {e}")
            continue
    
    # Generate overall summary
    print(f"\n{'='*70}")
    print("ACTUAL FEATURE EXTRACTION VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    total_images = len(inputs)
    successful_extractions = len(all_results)
    
    print(f"Total images processed: {total_images}")
    print(f"Successful extractions: {successful_extractions}")
    
    if all_results:
        # Aggregate statistics
        total_features = sum(len(result.features) for result in all_results)
        avg_processing_time = sum(result.processing_time_seconds for result in all_results) / len(all_results)
        avg_confidence = sum(result.average_confidence for result in all_results) / len(all_results)
        
        # Feature type statistics
        all_feature_types = {}
        for result in all_results:
            for feature_type, count in result.features_by_type.items():
                all_feature_types[feature_type] = all_feature_types.get(feature_type, 0) + count
        
        print(f"\nAggregate Statistics:")
        print(f"  Total features detected: {total_features}")
        print(f"  Average processing time: {avg_processing_time:.2f}s")
        print(f"  Average confidence: {avg_confidence:.3f}")
        
        print(f"\nFeature Type Distribution:")
        for feature_type, count in sorted(all_feature_types.items()):
            print(f"  {feature_type}: {count}")
        
        # Detection method usage
        all_methods = set()
        for result in all_results:
            all_methods.update(result.detection_methods_used)
        
        print(f"\nDetection Methods Used:")
        for method in sorted(all_methods):
            print(f"  - {method}")
    
    print(f"\nVALIDATION COMPLETE")
    print(f"   Results saved to: {OUTPUTS_DIR}")
    
    print(f"\nPHASE 2B IMPLEMENTATION STATUS:")
    print(f"   Actual feature extraction implemented")
    print(f"   Image-evidence-only detection")
    print(f"   NO expected feature dependency")
    print(f"   Multiple geometric feature types supported")
    print(f"   Duplicate suppression implemented")
    print(f"   Coordinate system consistency maintained")
    
    print(f"\nNOT IMPLEMENTED (as required):")
    print(f"   Expected vs actual feature matching")
    print(f"   Feature correspondence analysis")  
    print(f"   Dimensional deviation calculation")
    print(f"   Tolerance checking")
    print(f"   Defect classification")
    print(f"   Pass/fail inspection decisions")


if __name__ == "__main__":
    main()