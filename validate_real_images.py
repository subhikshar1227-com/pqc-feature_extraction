#!/usr/bin/env python3
"""
Real Image Validation Script

Validates Phase 2 implementation against all real product input images.
"""

import logging
from pathlib import Path
import traceback
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def validate_real_images():
    """Run Phase 2 pipeline against all real input images."""
    
    # Find input images
    input_dir = Path("data/inputs")
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    image_files = list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} input images")
    
    # Find DXF files for testing
    dxf_dir = Path("data/dxf")
    dxf_files = list(dxf_dir.glob("*.dxf"))
    
    if not dxf_files:
        logger.error(f"No DXF files found in {dxf_dir}")
        return
    
    # Use first DXF file for testing
    test_dxf = dxf_files[0]
    logger.info(f"Using test DXF: {test_dxf}")
    
    results = []
    
    try:
        from feature_inspection import inspect_product_quality
        
        for image_file in image_files:
            logger.info(f"\n--- Processing: {image_file.name} ---")
            
            try:
                # Run Phase 2 pipeline
                result = inspect_product_quality(
                    image_path=image_file,
                    dxf_path=test_dxf,
                    alignment_result=None,  # No alignment for this test
                    save_outputs=True,
                    output_base_dir=Path("outputs")
                )
                
                # Collect results
                image_result = {
                    "input_filename": image_file.name,
                    "detected_actual_feature_count": len(result.actual_feature_set.features),
                    "actual_feature_types": [f.feature_type.value for f in result.actual_feature_set.features],
                    "transformed_feature_count": len(result.actual_feature_set.features),
                    "coordinate_system": result.actual_feature_set.coordinate_system,
                    "transformation_status": "unavailable" if "transform_unavailable" in result.actual_feature_set.coordinate_system else "attempted",
                    "matched_count": len(result.feature_match_set.matches),
                    "missing_count": len(result.feature_match_set.unmatched_expected),
                    "extra_count": len(result.feature_match_set.unmatched_actual),
                    "inspection_status": result.inspection_status.value,
                    "quality_score": float(result.quality_metrics.overall_quality_score),
                    "generated_output_path": f"outputs/{image_file.stem}/phase_2/",
                    "processing_successful": True,
                    "error": None
                }
                
                logger.info(f"✓ Detected: {image_result['detected_actual_feature_count']} features")
                logger.info(f"✓ Types: {image_result['actual_feature_types']}")
                logger.info(f"✓ Coordinate system: {image_result['coordinate_system']}")
                logger.info(f"✓ Matched: {image_result['matched_count']}, Missing: {image_result['missing_count']}, Extra: {image_result['extra_count']}")
                logger.info(f"✓ Status: {image_result['inspection_status']} (score: {image_result['quality_score']:.3f})")
                logger.info(f"✓ Output: {image_result['generated_output_path']}")
                
            except Exception as e:
                error_msg = f"Failed to process {image_file.name}: {e}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                image_result = {
                    "input_filename": image_file.name,
                    "processing_successful": False,
                    "error": error_msg,
                    "detected_actual_feature_count": 0,
                    "actual_feature_types": [],
                    "transformed_feature_count": 0,
                    "coordinate_system": "error",
                    "transformation_status": "error",
                    "matched_count": 0,
                    "missing_count": 0,
                    "extra_count": 0,
                    "inspection_status": "error",
                    "quality_score": 0.0,
                    "generated_output_path": "error"
                }
            
            results.append(image_result)
    
    except ImportError as e:
        logger.error(f"Failed to import Phase 2 components: {e}")
        return
    
    # Generate summary report
    logger.info(f"\n=== REAL IMAGE VALIDATION SUMMARY ===")
    logger.info(f"Total images processed: {len(results)}")
    
    successful = [r for r in results if r["processing_successful"]]
    failed = [r for r in results if not r["processing_successful"]]
    
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        total_features = sum(r["detected_actual_feature_count"] for r in successful)
        avg_features = total_features / len(successful)
        logger.info(f"Average features per image: {avg_features:.1f}")
        
        all_types = set()
        for r in successful:
            all_types.update(r["actual_feature_types"])
        logger.info(f"Feature types detected: {sorted(all_types)}")
        
        coordinate_systems = set(r["coordinate_system"] for r in successful)
        logger.info(f"Coordinate systems: {coordinate_systems}")
        
        statuses = [r["inspection_status"] for r in successful]
        status_counts = {status: statuses.count(status) for status in set(statuses)}
        logger.info(f"Inspection statuses: {status_counts}")
    
    # Save detailed report
    report_path = Path("outputs/real_image_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "validation_summary": {
            "total_images": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0
        },
        "detailed_results": results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Detailed report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    results = validate_real_images()
    
    if results:
        successful_count = sum(1 for r in results if r["processing_successful"])
        print(f"\nValidation complete: {successful_count}/{len(results)} images processed successfully")
        
        if successful_count == len(results):
            print("✅ All real images processed successfully!")
        elif successful_count > 0:
            print("⚠️  Some images processed successfully, check logs for failures")
        else:
            print("❌ No images processed successfully")
    else:
        print("❌ Validation failed to run")