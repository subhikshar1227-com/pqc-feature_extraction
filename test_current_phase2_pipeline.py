#!/usr/bin/env python3
"""
Test script to understand current Phase 2 pipeline behavior before making corrections.
"""

import sys
import logging
from pathlib import Path
import cv2

# Add project to path
sys.path.append('.')

from cad_image_alignment import match_best_template, AlignmentResult
from dxf_resolver import resolve_dxf
from feature_extraction import extract_expected_features
from feature_inspection.pipeline.orchestrator import inspect_product_quality

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_blueprints():
    """Load available blueprint templates."""
    blueprints_dir = Path("data/blueprints")
    templates = []
    
    for blueprint_file in blueprints_dir.glob("*.png"):
        blueprint_name = blueprint_file.stem
        blueprint_image = cv2.imread(str(blueprint_file), cv2.IMREAD_GRAYSCALE)
        if blueprint_image is not None:
            templates.append((blueprint_name, blueprint_image))
            logger.info(f"Loaded blueprint: {blueprint_name}")
    
    return templates

def identify_blueprint(image_path: Path, templates: list):
    """Use Phase 1 to identify which blueprint matches the input image."""
    logger.info(f"Identifying blueprint for: {image_path.name}")
    
    # Load input image
    input_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply edge detection (simplified Phase 1)
    edges = cv2.Canny(input_image, 50, 150)
    
    # Try to match with templates
    best_matches = match_best_template(templates, edges)
    
    if best_matches:
        best_match = best_matches[0]
        logger.info(f"Best blueprint match: {best_match.name} (score: {best_match.result.combined_score:.3f})")
        return best_match.name, best_match.result
    else:
        logger.warning("No blueprint matches found")
        return None, None

def test_current_pipeline(image_path: Path):
    """Test the current Phase 2 pipeline to understand its behavior."""
    logger.info(f"\n=== Testing current pipeline for: {image_path.name} ===")
    
    try:
        # Step 1: Blueprint identification (Phase 1)
        logger.info("Step 1: Blueprint identification...")
        templates = load_blueprints()
        identified_blueprint, alignment_result = identify_blueprint(image_path, templates)
        
        if identified_blueprint is None:
            logger.error("No blueprint identified - cannot proceed")
            return None
            
        logger.info(f"Identified blueprint: {identified_blueprint}")
        
        # Step 2: DXF resolution
        logger.info("Step 2: DXF resolution...")
        try:
            dxf_path = resolve_dxf(identified_blueprint)
            logger.info(f"Resolved DXF: {dxf_path.name}")
        except Exception as e:
            logger.error(f"DXF resolution failed: {e}")
            return None
        
        # Step 3: Expected feature extraction
        logger.info("Step 3: Expected feature extraction...")
        try:
            expected_features = extract_expected_features(dxf_path)
            logger.info(f"Expected features extracted: {len(expected_features.features)} features")
        except Exception as e:
            logger.error(f"Expected feature extraction failed: {e}")
            return None
        
        # Step 4: Phase 2 inspection (this is where we need to fix things)
        logger.info("Step 4: Phase 2 inspection...")
        try:
            inspection_result = inspect_product_quality(
                image_path=image_path,
                dxf_path=dxf_path,
                alignment_result=alignment_result,  # This should be removed per requirements
                blueprint_name=identified_blueprint,
                save_outputs=True,
                output_base_dir=Path("outputs")
            )
            
            logger.info(f"Inspection completed: {inspection_result.inspection_status.value}")
            logger.info(f"Quality score: {inspection_result.quality_metrics.overall_quality_score:.3f}")
            return inspection_result
            
        except Exception as e:
            logger.error(f"Phase 2 inspection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test the current pipeline with available input images."""
    inputs_dir = Path("data/inputs")
    
    if not inputs_dir.exists():
        logger.error(f"Inputs directory not found: {inputs_dir}")
        return
    
    # Test with all available input images
    for image_file in inputs_dir.glob("*.jpeg"):
        result = test_current_pipeline(image_file)
        if result:
            logger.info(f"✅ {image_file.name} - {result.inspection_status.value}")
        else:
            logger.error(f"❌ {image_file.name} - Failed")
        print("-" * 80)

if __name__ == "__main__":
    main()