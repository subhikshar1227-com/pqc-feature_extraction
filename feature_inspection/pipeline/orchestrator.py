"""
Phase 2 Pipeline Orchestrator

Coordinates the complete Phase 2 product quality inspection pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import copy
import time

from feature_extraction import extract_expected_features
from feature_extraction.expected.feature_types import ExpectedFeatureSet
from cad_image_alignment import AlignmentResult

from ..models.coordinate_transform import CoordinateTransform
from ..models.inspection_result import InspectionResult
from ..models.actual_feature import ActualFeatureSet
from ..actual.detector import ActualFeatureDetector
from ..matching.matcher import FeatureMatcher
from ..inspection.inspector import QualityInspector
from ..pipeline.coordinate_pipeline import CoordinateTransformationPipeline, TransformationSource
from ..output.persistence import OutputPersistence
from ..output.visualizer import ActualFeatureVisualizer

logger = logging.getLogger(__name__)


def inspect_product_quality(image_path: Path,
                          dxf_path: Path, 
                          alignment_result: Optional[AlignmentResult] = None,
                          blueprint_name: Optional[str] = None,
                          save_outputs: bool = True,
                          output_base_dir: Optional[Path] = None) -> InspectionResult:
    """
    Perform complete Phase 2 product quality inspection.
    
    IMPORTANT: This version does NOT use alignment_result for coordinate transformation.
    Phase 1 alignment is used ONLY for blueprint identification. The alignment_result
    parameter is kept for interface compatibility but NOT used for actual detection
    or coordinate transformation.
    
    Pipeline Flow:
    1. Extract authoritative ExpectedFeatureSet from DXF
    2. Detect actual features from ORIGINAL product image (no alignment dependency)
    3. Store actual features in image pixels
    4. Attempt coordinate transformation (currently unavailable without valid calibration)
    5. If transformation available: Transform actual features to DXF/mm
    6. If transformation unavailable: Block matching, report transformation unavailable
    7. Match Expected(DXF/mm) against Actual(DXF/mm) if both in same coordinate system
    8. Perform geometric comparison (if matching possible)
    9. Perform quality inspection
    10. Save all Phase 2 outputs
    11. Return InspectionResult
    
    Args:
        image_path: Path to the ORIGINAL product image to inspect
        dxf_path: Path to the DXF file with expected geometry  
        alignment_result: UNUSED - kept for interface compatibility
        blueprint_name: Optional name of identified blueprint
        save_outputs: Whether to save output files (default: True)
        output_base_dir: Base directory for outputs (default: 'outputs/')
        
    Returns:
        Complete inspection result with pass/fail determination
    """
    logger.info(f"Starting Phase 2 product quality inspection")
    logger.info(f"Original Image: {image_path.name}")
    logger.info(f"DXF: {dxf_path.name}")
    logger.info(f"NOTE: Phase 1 alignment result NOT used for coordinate transformation")
    
    # Validate inputs
    if not image_path.exists():
        raise FileNotFoundError(f"Product image not found: {image_path}")
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")
    
    try:
        # Initialize pipeline components
        coordinate_pipeline = CoordinateTransformationPipeline()
        output_persistence = OutputPersistence(output_base_dir) if save_outputs else None
        visualizer = ActualFeatureVisualizer() if save_outputs else None
        
        # Step 1: Extract expected features from DXF (authoritative source)
        logger.info("Step 1: Extracting expected features from DXF...")
        expected_feature_set = extract_expected_features(dxf_path)
        logger.info(f"Expected features extracted: {len(expected_feature_set.features)} features")
        
        # Step 2: Detect actual features from ORIGINAL product image
        logger.info("Step 2: Detecting actual features from ORIGINAL product image...")
        detector = ActualFeatureDetector()
        actual_feature_set = detector.detect_features(image_path)
        logger.info(f"Actual features detected: {len(actual_feature_set.features)} features")
        logger.info(f"Actual features coordinate system: {actual_feature_set.coordinate_system}")
        
        # Step 3: Attempt coordinate transformation (currently not available)
        logger.info("Step 3: Attempting coordinate transformation...")
        
        # Get preprocessing metadata for resize scale factor
        preprocessing_metadata = getattr(detector, '_last_preprocessing_metadata', None)
        
        # Step 3: Attempt coordinate transformation using available sources
        logger.info("Step 3: Attempting coordinate transformation...")
        
        # Current implementation: no valid transformation sources available
        # Future: plug in validated calibration sources here
        transformation_source = None  # No valid sources in current repository
        
        # Transform actual features to DXF coordinates if transformation source available
        transformed_actual_set, transformation_metadata = coordinate_pipeline.transform_actual_features(
            actual_feature_set, transformation_source, preprocessing_metadata
        )
        
        logger.info(f"Transformation status: {transformation_metadata.get('transformation_successful', False)}")
        logger.info(f"Transformed features coordinate system: {transformed_actual_set.coordinate_system}")
        
        # Step 4: Validate coordinate system compatibility before matching
        logger.info("Step 4: Validating coordinate system compatibility...")
        coord_validation = coordinate_pipeline.validate_coordinate_consistency(
            expected_feature_set, transformed_actual_set
        )
        
        matching_blocked = False
        if not coord_validation["coordinate_systems_compatible"]:
            logger.error("Coordinate system mismatch detected - matching BLOCKED")
            for error in coord_validation["validation_errors"]:
                logger.error(f"  - {error}")
            logger.error(f"Recommendation: {coord_validation['recommendation']}")
            matching_blocked = True
        
        # Step 5: Perform matching only if coordinate systems are compatible
        if not matching_blocked:
            logger.info("Step 5: Matching expected and actual features...")
            matcher = FeatureMatcher(coordinate_transform=None)  # No coordinate transform needed for same-system matching
            match_set = matcher.match_features(expected_feature_set, transformed_actual_set)
            logger.info(f"Feature matching complete: {len(match_set.matches)} matches, "
                       f"{len(match_set.unmatched_expected)} missing, "
                       f"{len(match_set.unmatched_actual)} extra")
        else:
            logger.warning("Step 5: Matching BLOCKED due to coordinate system incompatibility")
            # Create empty match set indicating matching was not performed
            from ..models.feature_match import FeatureMatchSet, MatchingStatistics
            
            # Create statistics that clearly indicate matching was not performed
            blocked_stats = MatchingStatistics(
                total_expected_features=len(expected_feature_set.features),
                total_actual_features=len(transformed_actual_set.features),
                successful_matches=0,
                exact_matches=0,
                good_matches=0,
                acceptable_matches=0,
                poor_matches=0,
                unmatched_expected=0,  # Not unmatched - matching not performed
                unmatched_actual=0,    # Not unmatched - matching not performed
                average_match_confidence=0.0,
                average_geometric_accuracy=0.0,
                matching_time_seconds=0.0
            )
            
            match_set = FeatureMatchSet(
                matches=[],
                unmatched_expected=[],  # Empty - not "unmatched", but "not compared"
                unmatched_actual=[],   # Empty - not "unmatched", but "not compared"
                matching_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                matching_statistics=blocked_stats,
                matching_algorithm="blocked_coordinate_mismatch",
                configuration_snapshot={
                    "matching_performed": False,
                    "blocking_reason": "Coordinate system incompatibility",
                    "expected_coordinate_system": getattr(expected_feature_set, 'coordinate_system', 'dxf_mm'),
                    "actual_coordinate_system": transformed_actual_set.coordinate_system,
                    "note": "Matching was not performed due to coordinate system mismatch"
                }
            )
        
        # Step 6: Perform quality inspection
        logger.info("Step 6: Performing quality inspection...")
        inspector = QualityInspector()
        inspection_result = inspector.inspect_product(
            expected_feature_set, transformed_actual_set, match_set, blueprint_name
        )
        
        # Add transformation and matching status to inspection result
        if hasattr(inspection_result, 'processing_notes'):
            if matching_blocked:
                inspection_result.processing_notes.append(
                    "Geometric matching not performed due to coordinate system incompatibility"
                )
                inspection_result.processing_notes.append(
                    f"Expected: {getattr(expected_feature_set, 'coordinate_system', 'dxf_mm')}, "
                    f"Actual: {transformed_actual_set.coordinate_system}"
                )
        
        logger.info(f"Quality inspection complete: {inspection_result.inspection_status.value}")
        if hasattr(inspection_result, 'quality_metrics') and inspection_result.quality_metrics.overall_quality_score is not None:
            logger.info(f"Quality score: {inspection_result.quality_metrics.overall_quality_score:.3f}")
        else:
            logger.info("Quality score: not determined (geometric comparison unavailable)")
        
        # Step 7: Save all Phase 2 outputs
        if save_outputs:
            logger.info("Step 7: Saving Phase 2 outputs...")
            
            # Create visualization using ORIGINAL input image as base
            visualization_path = None
            if visualizer:
                try:
                    # Get matched feature IDs for visualization (empty if matching was blocked)
                    matched_ids = [match.actual_feature.feature_id for match in match_set.matches 
                                 if match.actual_feature is not None]
                    
                    # Use ORIGINAL image as base for actual feature visualization
                    temp_viz_path = image_path.parent / f"{image_path.stem}_temp_viz.png"
                    visualization_path = visualizer.visualize_actual_features(
                        image_path, actual_feature_set, temp_viz_path, matched_ids
                    )
                except Exception as e:
                    logger.error(f"Failed to create visualization: {e}")
            
            # Save all outputs
            try:
                # Only pass transformed features if transformation actually succeeded
                transformed_to_save = None
                if transformation_metadata.get('transformation_successful', False) and transformed_actual_set.coordinate_system == "dxf_mm":
                    transformed_to_save = transformed_actual_set
                
                saved_paths = output_persistence.save_all_phase2_outputs(
                    image_path=image_path,
                    actual_features=actual_feature_set,
                    transformed_features=transformed_to_save,  # None if transformation failed
                    match_set=match_set,
                    inspection_result=inspection_result,
                    transformation_metadata=transformation_metadata,
                    visualization_path=visualization_path
                )
                
                logger.info("Phase 2 outputs saved successfully")
                logger.info(f"Output directory: {saved_paths.get('summary', Path('unknown')).parent}")
                
            except Exception as e:
                logger.error(f"Failed to save Phase 2 outputs: {e}")
                import traceback
                traceback.print_exc()
            
            # Clean up temporary visualization
            if visualization_path and visualization_path.exists() and "temp_viz" in str(visualization_path):
                try:
                    visualization_path.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
        
        # Step 9: Return InspectionResult
        return inspection_result
        
    except Exception as e:
        logger.error(f"Phase 2 inspection pipeline failed: {e}")
        raise


def inspect_product_with_phase1_integration(image_path: Path,
                                          expected_feature_set: ExpectedFeatureSet,
                                          alignment_result: Optional[AlignmentResult] = None,
                                          blueprint_name: Optional[str] = None,
                                          save_outputs: bool = True,
                                          output_base_dir: Optional[Path] = None) -> InspectionResult:
    """
    Perform Phase 2 inspection with pre-computed Phase 1 results.
    
    This variant accepts pre-computed expected features from Phase 1,
    avoiding re-computation when Phase 1 has already been executed.
    
    Args:
        image_path: Path to the ORIGINAL product image to inspect
        expected_feature_set: Pre-computed expected features from Phase 1
        alignment_result: Optional alignment result with transformation matrix
        blueprint_name: Optional name of identified blueprint
        save_outputs: Whether to save output files (default: True)
        output_base_dir: Base directory for outputs (default: 'outputs/')
        
    Returns:
        Complete inspection result with pass/fail determination
    """
    logger.info(f"Starting Phase 2 inspection with pre-computed Phase 1 results")
    logger.info(f"Original Image: {image_path.name}")
    logger.info(f"Expected features: {len(expected_feature_set.features)} features")
    
    # Validate inputs
    if not image_path.exists():
        raise FileNotFoundError(f"Product image not found: {image_path}")
    
    try:
        # Initialize pipeline components
        coordinate_pipeline = CoordinateTransformationPipeline()
        output_persistence = OutputPersistence(output_base_dir) if save_outputs else None
        visualizer = ActualFeatureVisualizer() if save_outputs else None
        
        # Step 1: Detect actual features from ORIGINAL product image
        logger.info("Step 1: Detecting actual features from ORIGINAL product image...")
        detector = ActualFeatureDetector()
        actual_feature_set = detector.detect_features(image_path)
        logger.info(f"Actual features detected: {len(actual_feature_set.features)} features")
        
        # Step 2: Setup coordinate transformation
        logger.info("Step 2: Setting up coordinate transformation...")
        
        # Get preprocessing metadata for resize scale factor  
        preprocessing_metadata = getattr(detector, '_last_preprocessing_metadata', None)
        
        # Transform actual features to DXF coordinates
        transformed_actual_set, transformation_metadata = coordinate_pipeline.transform_actual_features(
            actual_feature_set, alignment_result, preprocessing_metadata
        )
        
        logger.info(f"Coordinate transformation complete")
        logger.info(f"Transformation successful: {transformation_metadata['transformation_successful']}")
        
        # Step 3: Validate coordinate system compatibility
        coord_validation = coordinate_pipeline.validate_coordinate_consistency(
            expected_feature_set, transformed_actual_set
        )
        
        if not coord_validation["coordinate_systems_compatible"]:
            logger.error("Coordinate system mismatch detected!")
            for error in coord_validation["validation_errors"]:
                logger.error(f"  - {error}")
            logger.warning("Proceeding with inspection despite coordinate system mismatch")
        
        # Step 4: Match expected and actual features
        logger.info("Step 3: Matching expected and actual features...")
        
        # Use coordinate transform for matching if available
        coordinate_transform = None
        if transformation_metadata['transformation_successful']:
            try:
                coordinate_transform = CoordinateTransform(alignment_result.transform_matrix) if alignment_result else None
            except Exception as e:
                logger.warning(f"Could not create coordinate transform for matching: {e}")
        
        matcher = FeatureMatcher(coordinate_transform)
        match_set = matcher.match_features(expected_feature_set, transformed_actual_set)
        logger.info(f"Feature matching complete: {len(match_set.matches)} matches")
        
        # Step 5: Perform quality inspection
        logger.info("Step 4: Performing quality inspection...")
        inspector = QualityInspector()
        inspection_result = inspector.inspect_product(
            expected_feature_set, transformed_actual_set, match_set, blueprint_name
        )
        
        logger.info(f"Quality inspection complete: {inspection_result.inspection_status.value}")
        
        # Step 6: Save outputs if requested
        if save_outputs:
            logger.info("Step 5: Saving Phase 2 outputs...")
            
            # Create visualization
            visualization_path = None
            if visualizer:
                try:
                    matched_ids = [match.actual_feature.feature_id for match in match_set.matches 
                                 if match.actual_feature is not None]
                    
                    temp_viz_path = image_path.parent / f"{image_path.stem}_temp_viz.png"
                    visualization_path = visualizer.visualize_actual_features(
                        image_path, actual_feature_set, temp_viz_path, matched_ids
                    )
                except Exception as e:
                    logger.error(f"Failed to create visualization: {e}")
            
            # Save all outputs
            try:
                saved_paths = output_persistence.save_all_phase2_outputs(
                    image_path=image_path,
                    actual_features=actual_feature_set,
                    transformed_features=transformed_actual_set if transformation_metadata['transformation_successful'] else None,
                    match_set=match_set,
                    inspection_result=inspection_result,
                    transformation_metadata=transformation_metadata,
                    visualization_path=visualization_path
                )
                
                logger.info("Phase 2 outputs saved successfully")
                
            except Exception as e:
                logger.error(f"Failed to save Phase 2 outputs: {e}")
            
            # Clean up temporary visualization
            if visualization_path and visualization_path.exists() and "temp_viz" in str(visualization_path):
                try:
                    visualization_path.unlink()
                except Exception:
                    pass
        
        return inspection_result
        
    except Exception as e:
        logger.error(f"Phase 2 inspection pipeline failed: {e}")
        raise