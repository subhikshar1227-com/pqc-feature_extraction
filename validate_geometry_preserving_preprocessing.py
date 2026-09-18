#!/usr/bin/env python3
"""
Geometry-Preserving Phase 2 Preprocessing Validation Script

Tests the corrected canonical preprocessing implementation that prioritizes 
complete physical product silhouette preservation including irregular 
protrusions, tabs, and fine structures.

CRITICAL: Validates geometry preservation, not just technical success.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from feature_inspection.preprocessing.canonical_preprocessor import CanonicalPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_geometry_preserving_preprocessing():
    """
    Validate the geometry-preserving preprocessing implementation.
    
    CRITICAL: Tests complete physical silhouette preservation including 
    irregular protrusions and tabs on circular products.
    """
    print("=" * 80)
    print("GEOMETRY-PRESERVING Phase 2 Preprocessing Validation")  
    print("=" * 80)
    print("\nSCOPE: COMPLETE PHYSICAL GEOMETRY PRESERVATION")
    print("- Complete product silhouette preservation (NO shape assumptions)")
    print("- Irregular protrusions/tabs preservation") 
    print("- Structural edge preservation with texture suppression")
    print("- Corrected texture suppression metrics")
    print("- Comprehensive mask quality diagnostics")
    print("\nCRITICAL VALIDATION:")
    print("- Circular products MUST retain protruding/tab geometry")
    print("- No circular/rectangular/convex assumptions")
    print("- Texture suppression WITHOUT destroying holes/boundaries")
    print("\nNOT TESTED:")
    print("- Feature detection accuracy")
    print("- Feature matching or inspection")
    print()
    
    # Initialize geometry-preserving preprocessor
    preprocessor = CanonicalPreprocessor()
    
    # Find ALL input images (including new circular ones)
    input_dir = Path("data/inputs")
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return False
    
    image_files = list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not image_files:
        print(f"ERROR: No image files found in {input_dir}")
        return False
    
    print(f"Found {len(image_files)} real product images:")
    for img_path in image_files:
        print(f"  - {img_path.name}")
    print()
    
    # Create output directory
    output_base = Path("outputs/geometry_preserving_validation")
    output_base.mkdir(parents=True, exist_ok=True)
    
    validation_results = []
    geometry_failures = []
    
    # Process each image with focus on geometry preservation
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 70)
        
        try:
            # Run geometry-preserving preprocessing
            result = preprocessor.preprocess_image(image_path)
            
            # Create output directory for this image
            image_output_dir = output_base / image_path.stem
            image_output_dir.mkdir(exist_ok=True)
            
            # Save all preprocessing outputs including diagnostic montage
            saved_files = preprocessor.save_preprocessing_outputs(result, image_output_dir)
            
            # Validate geometry preservation and quality
            validation = validate_geometry_preservation(result, image_path)
            validation_results.append(validation)
            
            # Check for geometry failures
            if not validation.get("geometry_preserved", True):
                geometry_failures.append(validation)
            
            # Print detailed results
            print_geometry_preservation_results(result, validation, saved_files)
            
        except Exception as e:
            print(f"ERROR: Preprocessing failed for {image_path.name}: {e}")
            validation_results.append({
                "image": image_path.name,
                "success": False,
                "geometry_preserved": False,
                "error": str(e)
            })
            geometry_failures.append({
                "image": image_path.name,
                "error": str(e)
            })
        
        print()
    
    # Summary reports
    print_validation_summary(validation_results, geometry_failures)
    print_texture_suppression_analysis(validation_results)
    print_mask_quality_analysis(validation_results)
    
    # Honest assessment of validation results
    all_technical_success = all(v.get("technical_success", False) for v in validation_results)
    issues_detected = sum(1 for v in validation_results 
                         if v.get("geometry_status") == "POTENTIAL_GEOMETRY_ISSUES_DETECTED")
    
    print("=" * 80)
    print("VALIDATION CONCLUSION")
    print("=" * 80)
    
    if all_technical_success and issues_detected == 0:
        print("🟡 TECHNICAL SUCCESS - VISUAL REVIEW REQUIRED")
        print("   All images processed successfully with reasonable metrics.")
        print("   However, visual inspection of montages and overlays is REQUIRED")
        print("   to confirm actual geometry preservation.")
        validation_success = True
    elif all_technical_success:
        print("🔴 TECHNICAL SUCCESS - GEOMETRY ISSUES DETECTED")
        print(f"   {issues_detected} images have potential geometry issues.")
        print("   CRITICAL visual inspection required.")
        validation_success = False
    else:
        print("💥 TECHNICAL FAILURES DETECTED")
        print("   Some images failed preprocessing entirely.")
        validation_success = False
    
    print()
    print("NEXT STEPS:")
    print("1. 🔍 MANUALLY INSPECT all preprocessing_montage.png files")
    print("2. 🔍 MANUALLY INSPECT all mask_overlay.png files")  
    print("3. ✅ Verify product silhouettes are completely preserved")
    print("4. ✅ Verify texture suppression is working effectively")
    print("5. ✅ Look for cut-off protrusions, tabs, or mechanical features")
    print()
    
    return validation_success


def validate_geometry_preservation(result, image_path):
    """
    Validate geometry preservation with focus on complete silhouette retention.
    
    CRITICAL: Does NOT declare "PASS" based on scalar metrics alone.
    Requires actual visual inspection for geometry validation.
    """
    
    validation = {
        "image": image_path.name,
        "technical_success": result.preprocessing_successful,
        "isolation_success": result.isolation_successful,
        "visual_review_required": True,  # ALWAYS require visual review
        
        # Basic metrics
        "original_dimensions": result.original_dimensions,
        "processed_dimensions": result.processed_dimensions,
        "scale_factor": result.scale_factor,
        
        # Mask quality diagnostics (informational only)
        "product_area_fraction": result.product_area_fraction,
        "mask_solidity": result.mask_solidity,
        "mask_extent": result.mask_extent,
        "border_touching": result.border_touching_foreground,
        "component_count": result.foreground_component_count,
        "significant_regions": result.significant_foreground_regions,
        
        # CRITICAL: Silhouette validation results
        "silhouette_validation": result.silhouette_validation_result,
        "external_gradient_strength": result.external_gradient_strength,
        "boundary_gradient_consistency": result.boundary_gradient_consistency,
        "suspicious_boundary_fraction": result.suspicious_boundary_fraction,
        
        # CORRECTED edge metrics
        "raw_internal_density": result.raw_internal_edge_density,
        "filtered_internal_density": result.filtered_internal_edge_density,
        "texture_reduction_ratio": result.internal_edge_reduction_ratio,
        "retention_ratio": result.internal_edge_retention_ratio,
        "outer_boundary_density": result.outer_boundary_density,
        "final_edge_density": result.final_edge_density,
    }
    
    # CRITICAL: Geometry preservation assessment based on evidence, not assumptions
    geometry_issues = []
    
    # Check 1: Silhouette validation issues
    if result.silhouette_validation_result != "PASS":
        geometry_issues.append(f"Silhouette validation: {result.silhouette_validation_result}")
    
    # Check 2: High suspicious boundary fraction indicates potential geometry loss
    if result.suspicious_boundary_fraction > 0.3:
        geometry_issues.append(f"High suspicious boundary fraction: {result.suspicious_boundary_fraction:.1%}")
    
    # Check 3: Very low boundary consistency suggests mask doesn't follow real boundaries
    if result.boundary_gradient_consistency < 0.3:
        geometry_issues.append(f"Low boundary consistency: {result.boundary_gradient_consistency:.2f}")
    
    # Check 4: Texture suppression effectiveness - should be substantial
    if result.internal_edge_reduction_ratio < 0.2:  # Less than 20% reduction is suspicious
        geometry_issues.append(f"Minimal texture suppression: {result.internal_edge_reduction_ratio:.1%} reduction")
    
    # Set geometry validation status
    if len(geometry_issues) == 0:
        validation["geometry_status"] = "METRICS_SUGGEST_OK_BUT_VISUAL_REVIEW_REQUIRED"
    else:
        validation["geometry_status"] = "POTENTIAL_GEOMETRY_ISSUES_DETECTED"
        validation["geometry_issues"] = geometry_issues
    
    # Coordinate mapping validation
    try:
        test_x, test_y = 100, 50
        orig_x, orig_y = result.to_original_coordinates(test_x, test_y)
        proc_x, proc_y = result.to_processed_coordinates(orig_x, orig_y)
        
        coord_error = abs(proc_x - test_x) + abs(proc_y - test_y)
        validation["coordinate_mapping_valid"] = coord_error < 1e-10
        validation["coordinate_round_trip_error"] = coord_error
        
    except Exception as e:
        validation["coordinate_mapping_valid"] = False
        validation["coordinate_error"] = str(e)
    
    return validation


def print_geometry_preservation_results(result, validation, saved_files):
    """Print detailed geometry preservation results with honest assessment."""
    
    print(f"Image: {result.source_image_path.name}")
    print(f"Technical execution: {'SUCCESS' if result.preprocessing_successful else 'FAILED'}")
    print(f"Product isolation: {'SUCCESS' if result.isolation_successful else 'FAILED'}")
    
    # CRITICAL: Don't falsely claim geometry preservation
    if validation.get("geometry_status") == "METRICS_SUGGEST_OK_BUT_VISUAL_REVIEW_REQUIRED":
        print(f"Geometry metrics: REASONABLE - VISUAL REVIEW REQUIRED")
    else:
        print(f"Geometry metrics: ISSUES DETECTED - VISUAL REVIEW CRITICAL")
        if "geometry_issues" in validation:
            for issue in validation["geometry_issues"]:
                print(f"  ⚠️ {issue}")
    
    # Silhouette validation results
    print(f"Silhouette validation: {result.silhouette_validation_result}")
    if result.silhouette_validation_result != "PASS":
        print(f"  External gradient strength: {result.external_gradient_strength:.1f}")
        print(f"  Boundary consistency: {result.boundary_gradient_consistency:.2f}")
        print(f"  Suspicious boundary fraction: {result.suspicious_boundary_fraction:.1%}")
    
    print()
    
    print("Image Dimensions:")
    print(f"  Original: {result.original_dimensions[0]}x{result.original_dimensions[1]} (W×H)")
    print(f"  Processed: {result.processed_dimensions[0]}x{result.processed_dimensions[1]} (W×H)")
    print(f"  Scale factor: {result.scale_factor:.3f}")
    print()
    
    if result.isolation_successful:
        print("Product Mask Quality Diagnostics (INFORMATIONAL ONLY):")
        print(f"  Product area: {result.product_area_pixels:,} pixels ({result.product_area_fraction:.1%})")
        print(f"  Bounding box area fraction: {result.mask_bounding_box_area_fraction:.1%}")
        print(f"  Mask solidity: {result.mask_solidity:.3f}")
        print(f"  Mask extent: {result.mask_extent:.3f}")
        print(f"  Foreground components: {result.foreground_component_count}")
        print(f"  Significant regions: {result.significant_foreground_regions}")
        print(f"  Border touching: {'YES' if result.border_touching_foreground else 'NO'}")
        print(f"  Largest component fraction: {result.largest_component_fraction:.1%}")
        print()
        
        print("Edge Extraction with CORRECTED Texture Suppression Metrics:")
        print(f"  Raw internal edges: {result.raw_internal_edge_density:.4f}")
        print(f"  Filtered internal edges: {result.filtered_internal_edge_density:.4f}")
        print(f"  Texture suppression: {result.internal_edge_reduction_ratio:.1%} edges removed")
        print(f"  Edge retention: {result.internal_edge_retention_ratio:.1%} edges preserved")
        print(f"  Outer boundary edges: {result.outer_boundary_density:.4f}")
        print(f"  Final combined edges: {result.final_edge_density:.4f}")
        
        # Texture suppression assessment
        if result.internal_edge_reduction_ratio < 0.2:
            print(f"  ⚠️ WARNING: Minimal texture suppression ({result.internal_edge_reduction_ratio:.1%})")
        elif result.internal_edge_reduction_ratio > 0.7:
            print(f"  ⚠️ WARNING: Aggressive suppression ({result.internal_edge_reduction_ratio:.1%}) - check for over-filtering")
        
        print()
    
    print("Processing Pipeline:")
    for i, step in enumerate(result.processing_steps, 1):
        print(f"  {i:2d}. {step}")
    print()
    
    print("Coordinate Mapping Test:")
    test_x, test_y = 100, 75
    orig_x, orig_y = result.to_original_coordinates(test_x, test_y)
    proc_x, proc_y = result.to_processed_coordinates(orig_x, orig_y)
    print(f"  Test point: ({test_x}, {test_y}) processed")
    print(f"  -> ({orig_x:.1f}, {orig_y:.1f}) original")
    print(f"  -> ({proc_x:.1f}, {proc_y:.1f}) back to processed")
    print()
    
    print("Saved Outputs (🔴 VISUAL INSPECTION REQUIRED 🔴):")
    print(f"  📁 {saved_files['montage'].name} - CRITICAL: Visual geometry validation")
    print(f"  📁 {saved_files['mask_overlay'].name} - Mask contour overlay for inspection")
    print(f"  📁 {saved_files['original'].name} - Original reference")
    print(f"  📁 {saved_files['mask'].name} - Product mask")
    print(f"  📁 {saved_files['raw_internal_edges'].name} - Raw internal edges (before filtering)")
    print(f"  📁 {saved_files['internal_edges'].name} - Internal edges (texture-suppressed)")
    print(f"  📁 {saved_files['outer_boundary'].name} - Outer boundary")
    print(f"  📁 {saved_files['final_edges'].name} - Final combined edges")
    print(f"  📁 {saved_files['metadata'].name} - Complete diagnostics")
    print()


def print_validation_summary(validation_results, geometry_failures):
    """Print overall validation summary with honest geometry assessment."""
    
    print("=" * 80)
    print("PREPROCESSING VALIDATION SUMMARY - HONEST ASSESSMENT")
    print("=" * 80)
    
    total_images = len(validation_results)
    successful_preprocessing = sum(1 for v in validation_results if v.get("technical_success", False))
    successful_isolation = sum(1 for v in validation_results if v.get("isolation_success", False))
    
    # Honest geometry assessment
    metrics_suggest_ok = sum(1 for v in validation_results 
                           if v.get("geometry_status") == "METRICS_SUGGEST_OK_BUT_VISUAL_REVIEW_REQUIRED")
    issues_detected = sum(1 for v in validation_results 
                         if v.get("geometry_status") == "POTENTIAL_GEOMETRY_ISSUES_DETECTED")
    
    print(f"Total images processed: {total_images}")
    print(f"Technical execution success: {successful_preprocessing}/{total_images}")
    print(f"Product isolation success: {successful_isolation}/{total_images}")
    print()
    
    print("GEOMETRY ASSESSMENT:")
    print(f"  Metrics suggest reasonable: {metrics_suggest_ok}/{total_images}")
    print(f"  Potential issues detected: {issues_detected}/{total_images}")
    print(f"  Visual review required: {total_images}/{total_images} (ALL IMAGES)")
    print()
    
    if issues_detected > 0:
        print("🔴 POTENTIAL GEOMETRY ISSUES DETECTED:")
        for validation in validation_results:
            if validation.get("geometry_status") == "POTENTIAL_GEOMETRY_ISSUES_DETECTED":
                print(f"  - {validation['image']}:")
                for issue in validation.get("geometry_issues", []):
                    print(f"    • {issue}")
    else:
        print("🟡 NO OBVIOUS ISSUES IN METRICS - BUT VISUAL REVIEW STILL REQUIRED")
    
    print()
    
    # Statistics for successful isolation results
    successful_results = [v for v in validation_results if v.get("isolation_success", False)]
    if successful_results:
        # Product area statistics
        areas = [v["product_area_fraction"] for v in successful_results]
        silhouette_validations = [v["silhouette_validation"] for v in successful_results]
        
        print("Mask Quality Statistics (INFORMATIONAL ONLY):")
        print(f"  Product area - Min: {min(areas):.1%}, Max: {max(areas):.1%}, Avg: {sum(areas)/len(areas):.1%}")
        
        # Silhouette validation summary
        silhouette_passes = sum(1 for s in silhouette_validations if s == "PASS")
        print(f"  Silhouette validation passes: {silhouette_passes}/{len(successful_results)}")
        
        print()


def print_texture_suppression_analysis(validation_results):
    """Analyze corrected texture suppression effectiveness."""
    
    print("=" * 80)
    print("TEXTURE SUPPRESSION EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    
    successful_results = [v for v in validation_results if v.get("isolation_success", False)]
    
    if successful_results:
        print("Texture Suppression Results (TARGET: 20-60% reduction):")
        
        insufficient_suppression = []
        appropriate_suppression = []
        excessive_suppression = []
        
        for v in successful_results:
            image_name = v["image"]
            reduction_ratio = v["texture_reduction_ratio"]
            
            print(f"  {image_name}:")
            print(f"    Raw internal edges: {v['raw_internal_density']:.4f}")
            print(f"    Filtered internal edges: {v['filtered_internal_density']:.4f}")
            print(f"    Texture reduction: {reduction_ratio:.1%}")
            
            # Categorize suppression effectiveness
            if reduction_ratio < 0.2:
                insufficient_suppression.append(image_name)
                print(f"    ⚠️ INSUFFICIENT texture suppression")
            elif reduction_ratio > 0.7:
                excessive_suppression.append(image_name)
                print(f"    ⚠️ EXCESSIVE suppression - possible over-filtering")
            else:
                appropriate_suppression.append(image_name)
                print(f"    ✓ Appropriate suppression level")
            
            print(f"    Final combined density: {v['final_edge_density']:.4f}")
            print()
        
        # Summary assessment
        print("TEXTURE SUPPRESSION ASSESSMENT:")
        print(f"  Appropriate suppression: {len(appropriate_suppression)}/{len(successful_results)}")
        print(f"  Insufficient suppression: {len(insufficient_suppression)}/{len(successful_results)}")
        print(f"  Excessive suppression: {len(excessive_suppression)}/{len(successful_results)}")
        
        if len(insufficient_suppression) > 0:
            print(f"  🔴 Images with insufficient suppression: {', '.join(insufficient_suppression)}")
        if len(excessive_suppression) > 0:
            print(f"  🔴 Images with excessive suppression: {', '.join(excessive_suppression)}")
        
        print()


def print_mask_quality_analysis(validation_results):
    """Analyze mask quality metrics (informational only)."""
    
    print("=" * 80)
    print("MASK QUALITY ANALYSIS (INFORMATIONAL - NOT PROOF OF GEOMETRY PRESERVATION)")
    print("=" * 80)
    
    successful_results = [v for v in validation_results if v.get("isolation_success", False)]
    
    if successful_results:
        # Collect silhouette validation results
        silhouette_issues = []
        for v in successful_results:
            if v["silhouette_validation"] != "PASS":
                silhouette_issues.append({
                    "image": v["image"],
                    "issue": v["silhouette_validation"],
                    "suspicious_fraction": v["suspicious_boundary_fraction"]
                })
        
        print("Silhouette Validation Results:")
        if len(silhouette_issues) == 0:
            print("  ✓ No obvious silhouette validation issues detected")
        else:
            print(f"  ⚠️ Silhouette issues detected in {len(silhouette_issues)} images:")
            for issue in silhouette_issues:
                print(f"    - {issue['image']}: {issue['issue']} ({issue['suspicious_fraction']:.1%} suspicious)")
        
        print()
        
        print("🔴 CRITICAL REMINDER:")
        print("  These metrics are DIAGNOSTIC ONLY and do NOT prove geometry preservation.")
        print("  Visual inspection of mask overlays and montages is REQUIRED.")
        print("  Look for:")
        print("    • Visible product edges outside the mask")
        print("    • Cut-off protrusions, tabs, or mechanical features")  
        print("    • Mask contours that don't follow actual product boundaries")
        print("    • Missing holes or internal features")
        
        print()


if __name__ == "__main__":
    success = validate_geometry_preserving_preprocessing()
    if success:
        print("✅ PREPROCESSING VALIDATION COMPLETED")
        print("   🔍 VISUAL INSPECTION OF OUTPUTS IS REQUIRED")
        print("   📁 Check outputs/geometry_preserving_validation/*/preprocessing_montage.png")
        sys.exit(0)
    else:
        print("❌ PREPROCESSING VALIDATION DETECTED ISSUES")
        print("   🔍 CRITICAL VISUAL INSPECTION REQUIRED")
        sys.exit(1)