"""
Main Expected Feature Extractor

Orchestrates the complete DXF-to-expected-features pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from .dxf import (
    parse_dxf, normalize_geometry, analyze_relationships, reconstruct_geometry
)
from .dxf.parser import get_dxf_units
from .expected import (
    CircleDetector, ThroughHoleDetector, SquareHoleDetector, SignificanceFilter, build_expected_feature_set
)
from .expected.semantic_grouping import SemanticFeatureGrouper
from .expected.feature_types import ExpectedFeatureSet
from .expected.expected_feature_builder import add_feature_relationships

logger = logging.getLogger(__name__)


def extract_expected_features(dxf_path: Path) -> ExpectedFeatureSet:
    """
    Extract expected features from a DXF file.
    
    This is the main entry point for Phase 1: DXF → Expected Features.
    
    Pipeline:
    1. Parse DXF entities
    2. Normalize geometry  
    3. Analyze relationships
    4. Reconstruct geometry
    5. Detect circle features
    6. Detect through hole features
    7. Filter significant features
    8. Build final feature set
    
    Args:
        dxf_path: Path to the DXF file
        
    Returns:
        ExpectedFeatureSet containing all detected expected features
        
    Raises:
        Various exceptions if processing fails at any stage
    """
    logger.info(f"=== Extracting Expected Features from {dxf_path.name} ===")
    
    # Validate input
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")
    
    processing_metadata: Dict[str, Any] = {}
    
    try:
        # Step 1: Parse DXF entities
        logger.info("Step 1: Parsing DXF entities...")
        raw_entities = parse_dxf(dxf_path)
        raw_entity_count = len(raw_entities)
        
        # Get DXF units
        dxf_units = get_dxf_units(dxf_path)
        
        # Count entity types for metadata
        entity_type_counts = {}
        for entity in raw_entities:
            entity_type = entity.entity_type.value
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        processing_metadata["entity_type_counts"] = entity_type_counts
        
        logger.info(f"  Parsed {raw_entity_count} entities: {entity_type_counts}")
        
        # Step 2: Normalize geometry
        logger.info("Step 2: Normalizing geometry...")
        normalized_entities = normalize_geometry(raw_entities)
        normalized_entity_count = len(normalized_entities)
        
        logger.info(f"  Normalized {normalized_entity_count} entities")
        
        # Step 3: Analyze relationships
        logger.info("Step 3: Analyzing geometric relationships...")
        relationships = analyze_relationships(normalized_entities)
        
        logger.info(f"  Found {len(relationships.relationships)} relationships "
                   f"in {len(relationships.entity_groups)} groups")
        
        # Step 4: Reconstruct geometry
        logger.info("Step 4: Reconstructing geometry...")
        reconstructed = reconstruct_geometry(normalized_entities, relationships)
        reconstructed_geometry_count = len(reconstructed.reconstructed_circles)
        
        logger.info(f"  Reconstructed {reconstructed_geometry_count} geometric structures")
        
        # Step 5: Detect circle features
        logger.info("Step 5: Detecting circle features...")
        circle_detector = CircleDetector()
        circle_features = circle_detector.detect_circles(normalized_entities, reconstructed)
        
        logger.info(f"  Detected {len(circle_features)} circle features")
        
        # Step 6: Detect through hole features  
        logger.info("Step 6: Detecting through hole features...")
        hole_detector = ThroughHoleDetector()
        hole_features = hole_detector.detect_through_holes(
            normalized_entities, reconstructed, relationships
        )
        
        logger.info(f"  Detected {len(hole_features)} through hole features")
        
        # Step 7: Detect square/rectangular hole features
        logger.info("Step 7: Detecting square hole features...")
        square_hole_detector = SquareHoleDetector()
        square_hole_features = square_hole_detector.detect_square_holes(
            normalized_entities, relationships
        )
        
        logger.info(f"  Detected {len(square_hole_features)} square hole features")
        
        # Combine all detected features
        all_features = circle_features + hole_features + square_hole_features
        processing_metadata["total_candidates"] = len(all_features)
        
        # Step 8: Apply semantic feature grouping
        logger.info("Step 8: Applying semantic feature grouping...")
        semantic_grouper = SemanticFeatureGrouper()
        grouped_features = semantic_grouper.group_features_semantically(all_features, relationships)
        
        logger.info(f"  Grouped {len(all_features)} raw features into {len(grouped_features)} semantic features")
        
        # Step 9: Filter significant features
        logger.info("Step 9: Filtering significant features...")
        significance_filter = SignificanceFilter()
        significant_features = significance_filter.filter_significant_features(grouped_features)
        
        logger.info(f"  Filtered to {len(significant_features)} significant features")
        
        # Step 10: Build final feature set
        logger.info("Step 10: Building expected feature set...")
        feature_set = build_expected_feature_set(
            features=significant_features,
            dxf_path=dxf_path,
            dxf_units=dxf_units,
            raw_entity_count=raw_entity_count,
            normalized_entity_count=normalized_entity_count,
            reconstructed_geometry_count=reconstructed_geometry_count,
            processing_metadata=processing_metadata
        )
        
        # Add reference geometry for visualization
        feature_set.reference_geometry = normalized_entities
        feature_set.reconstructed_geometry = reconstructed
        
        # Add feature relationships
        add_feature_relationships(feature_set)
        
        logger.info(f"=== Completed Expected Feature Extraction ===")
        logger.info(f"Final result: {feature_set.total_feature_count} features "
                   f"({feature_set.circle_count} circles, {feature_set.through_hole_count} holes)")
        
        return feature_set
        
    except Exception as e:
        logger.error(f"Failed to extract expected features from {dxf_path}: {e}")
        raise


def extract_expected_features_batch(dxf_paths: list[Path]) -> Dict[str, ExpectedFeatureSet]:
    """
    Extract expected features from multiple DXF files.
    
    Args:
        dxf_paths: List of DXF file paths
        
    Returns:
        Dictionary mapping DXF filename to ExpectedFeatureSet
    """
    logger.info(f"=== Batch Expected Feature Extraction for {len(dxf_paths)} files ===")
    
    results = {}
    
    for i, dxf_path in enumerate(dxf_paths, 1):
        logger.info(f"\n[{i}/{len(dxf_paths)}] Processing {dxf_path.name}...")
        
        try:
            feature_set = extract_expected_features(dxf_path)
            results[dxf_path.name] = feature_set
            
            logger.info(f"  ✓ Success: {feature_set.total_feature_count} features")
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            results[dxf_path.name] = None
    
    # Log batch summary
    successful = sum(1 for result in results.values() if result is not None)
    failed = len(results) - successful
    
    logger.info(f"\n=== Batch Summary ===")
    logger.info(f"Total files: {len(dxf_paths)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if successful > 0:
        # Summary statistics
        total_features = sum(
            result.total_feature_count 
            for result in results.values() 
            if result is not None
        )
        logger.info(f"Total features extracted: {total_features}")
    
    return results