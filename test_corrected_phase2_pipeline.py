#!/usr/bin/env python3
"""
Test script for corrected Phase 2 pipeline that follows the new requirements:

1. Phase 1 used ONLY for blueprint identification (not alignment-based coordinate transformation)  
2. Phase 2 actual detection uses ORIGINAL input image
3. Coordinate transformation currently unavailable (blocks matching)
4. Proper error handling and output persistence
"""

import sys
import logging
from pathlib import Path
import cv2

# Add project to path
sys.path.append('.')

from cad_image_alignment import match_best_template
from dxf_resolver import resolve_dxf
from feature_extraction import extract_expected_features
from feature_inspection.pipeline.orchestrator import inspect_product_quality

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_blueprints():
    """Load available blueprint templates for Phase 1 identification."""
    blueprints_dir = Path("data/blueprints")
    templates = []
    
    for blueprint_file in blueprints_dir.glob("*.png"):
        blueprint_name = blueprint_file.stem
        blueprint_image = cv2.imread(str(blueprint_file), cv2.IMREAD_GRAYSCALE)
        if blueprint_image is not None:
            templates.append((blueprint_name, blueprint_image))
            logger.info(f"Loaded blueprint: {blueprint_name}")
    
    return templates

def identify_blueprint_only(image_path: Path, templates: list):
    """Use Phase 1 ONLY for blueprint identification (not coordinate transformation)."""
    logger.info(f"Phase 1: Identifying blueprint for {image_path.name}...")
    
    # Load input image
    input_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply edge detection (simplified Phase 1 identification)
    edges = cv2.Canny(input_image, 50, 150)
    
    # Match with blueprint templates
    best_matches = match_best_template(templates, edges)
    
    if best_matches:
        best_match = best_matches[0]
        logger.info(f"Blueprint identified: {best_match.name} "
                   f"(identification score: {best_match.result.combined_score:.3f})")
        
        # NOTE: We return the alignment result for interface compatibility but 
        # clearly document that it should NOT be used for coordinate transformation
        logger.warning(f"Phase 1 alignment confidence: {best_match.result.alignment_score:.3f}")
        if best_match.result.alignment_score < 0.8:
            logger.warning("Phase 1 alignment confidence below threshold - not reliable for coordinate transformation")
        
        return best_match.name, best_match.result
    else:
        logger.warning("No blueprint matches found")
        return None, None

def test_corrected_phase2_pipeline(image_path: Path):
    """Test the corrected Phase 2 pipeline per new requirements."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing CORRECTED Phase 2 pipeline: {image_path.name}")
    logger.info(f"{'='*80}")
    
    try:
        # Step 1: Blueprint identification ONLY (Phase 1)
        logger.info("Step 1: Blueprint identification (Phase 1)...")
        templates = load_blueprints()
        identified_blueprint, alignment_result = identify_blueprint_only(image_path, templates)
        
        if identified_blueprint is None:
            logger.error("No blueprint identified - cannot proceed")
            return None
            
        logger.info(f"✅ Blueprint identified: {identified_blueprint}")
        
        # Step 2: DXF resolution using identified blueprint
        logger.info("Step 2: DXF resolution...")
        try:
            dxf_path = resolve_dxf(identified_blueprint)
            logger.info(f"✅ Resolved DXF: {dxf_path.name}")
        except Exception as e:
            logger.error(f"❌ DXF resolution failed: {e}")
            return None
        
        # Step 3: Expected feature extraction from DXF
        logger.info("Step 3: Expected feature extraction...")
        try:
            expected_features = extract_expected_features(dxf_path)
            logger.info(f"✅ Expected features: {len(expected_features.features)} features "
                       f"(coordinate system: dxf_mm)")
            
            # Log expected feature summary
            circles = len([f for f in expected_features.features if f.feature_type.value == "circle"])
            holes = len([f for f in expected_features.features if "hole" in f.feature_type.value])
            logger.info(f"   Expected: {circles} circles, {holes} holes")
            
        except Exception as e:
            logger.error(f"❌ Expected feature extraction failed: {e}")
            return None
        
        # Step 4: Phase 2 inspection with corrected pipeline
        logger.info("Step 4: Phase 2 inspection (corrected pipeline)...")
        logger.info("   - Uses ORIGINAL input image for actual detection")
        logger.info("   - Does NOT use Phase 1 alignment for coordinate transformation")
        logger.info("   - Properly handles coordinate transformation unavailability")
        
        try:
            inspection_result = inspect_product_quality(
                image_path=image_path,              # ORIGINAL input image
                dxf_path=dxf_path,                 # Resolved DXF
                alignment_result=None,             # DELIBERATELY set to None per requirements
                blueprint_name=identified_blueprint, # Blueprint name for reference
                save_outputs=True,                 # Save outputs to outputs/<image>/ 
                output_base_dir=Path("outputs")
            )
            
            # Log results
            logger.info(f"✅ Inspection completed: {inspection_result.inspection_status.value}")
            
            # Handle None quality score safely
            quality_score = inspection_result.quality_metrics.overall_quality_score
            if quality_score is not None:
                logger.info(f"   Quality score: {quality_score:.3f}")
            else:
                logger.info(f"   Quality score: None (geometric comparison unavailable)")
            
            # Log actual features detected
            actual_count = len(inspection_result.actual_feature_set.features)
            logger.info(f"   Actual features detected: {actual_count}")
            logger.info(f"   Actual coordinate system: {inspection_result.actual_feature_set.coordinate_system}")
            
            # Log matching results
            matches = len(inspection_result.feature_match_set.matches)
            missing = len(inspection_result.feature_match_set.unmatched_expected) 
            extra = len(inspection_result.feature_match_set.unmatched_actual)
            logger.info(f"   Matching: {matches} matches, {missing} missing, {extra} extra")
            
            # Check if matching was blocked
            if hasattr(inspection_result.feature_match_set, 'matching_statistics'):
                match_stats = inspection_result.feature_match_set.matching_statistics
                if isinstance(match_stats, dict) and match_stats.get('matching_blocked', False):
                    logger.warning(f"   ⚠️ Matching BLOCKED: {match_stats.get('blocking_reason', 'Unknown reason')}")
                    logger.warning(f"   Expected coord system: {match_stats.get('expected_coordinate_system', 'unknown')}")
                    logger.warning(f"   Actual coord system: {match_stats.get('actual_coordinate_system', 'unknown')}")
            
            # Log transformation status
            if hasattr(inspection_result.quality_metrics, 'transformation_available'):
                if inspection_result.quality_metrics.transformation_available:
                    logger.info("   ✅ Coordinate transformation successful")
                else:
                    logger.warning("   ⚠️ Coordinate transformation unavailable")
            
            return inspection_result
            
        except Exception as e:
            logger.error(f"❌ Phase 2 inspection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test the corrected pipeline with all available input images."""
    inputs_dir = Path("data/inputs")
    
    if not inputs_dir.exists():
        logger.error(f"Inputs directory not found: {inputs_dir}")
        return
    
    logger.info("Testing CORRECTED Phase 2 Pipeline")
    logger.info("Key corrections:")
    logger.info("- Phase 1 alignment result NOT used for coordinate transformation")
    logger.info("- Phase 2 actual detection uses ORIGINAL input image")
    logger.info("- Coordinate transformation unavailability properly handled")
    logger.info("- Matching blocked when coordinate systems incompatible")
    logger.info("- Output persistence errors fixed")
    print()
    
    results = {}
    
    # Test with all available input images  
    for image_file in sorted(inputs_dir.glob("*.jpeg")):
        result = test_corrected_phase2_pipeline(image_file)
        
        if result:
            status = result.inspection_status.value
            score = result.quality_metrics.overall_quality_score
            results[image_file.name] = (status, score)
            
            # Handle None score safely in logging
            if score is not None:
                logger.info(f"✅ {image_file.name}: {status} (score: {score:.3f})")
            else:
                logger.info(f"✅ {image_file.name}: {status} (score: None)")
        else:
            results[image_file.name] = ("ERROR", 0.0)
            logger.error(f"❌ {image_file.name}: ERROR")
        
        print("-" * 80)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for image_name, (status, score) in results.items():
        icon = "✅" if status != "ERROR" else "❌"
        # Handle None score safely in final summary
        if score is not None:
            score_str = f"{score:.3f}"
        else:
            score_str = "None"
        print(f"{icon} {image_name:<40} {status:<10} {score_str}")
    
    # Check outputs directory
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        print(f"\nOutput directories created:")
        for output_subdir in outputs_dir.iterdir():
            if output_subdir.is_dir():
                phase2_dir = output_subdir / "phase_2"
                if phase2_dir.exists():
                    file_count = len(list(phase2_dir.glob("*")))
                    print(f"  {output_subdir.name}/phase_2/ ({file_count} files)")

if __name__ == "__main__":
    main()