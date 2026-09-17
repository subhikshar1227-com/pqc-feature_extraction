"""
Expected Feature Set Builder

Builds the final ExpectedFeatureSet from detected and filtered features.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .feature_types import (
    ExpectedFeature, ExpectedFeatureSet, FeatureDetectionStatistics, FeatureType
)
from ..config import (
    get_config,
    # Confidence binning thresholds
    CONFIDENCE_BIN_VERY_LOW_MAX, CONFIDENCE_BIN_LOW_MAX, CONFIDENCE_BIN_MEDIUM_MAX, CONFIDENCE_BIN_HIGH_MAX,
    # Proximity threshold
    EXPECTED_FEATURE_PROXIMITY_THRESHOLD,
    # Forensic audit parameter
    EXPECTED_FEATURE_CANDIDATE_MULTIPLIER
)

logger = logging.getLogger(__name__)


def build_expected_feature_set(
    features: List[ExpectedFeature],
    dxf_path: Path,
    dxf_units: str,
    raw_entity_count: int,
    normalized_entity_count: int,
    reconstructed_geometry_count: int,
    processing_metadata: Dict[str, Any]
) -> ExpectedFeatureSet:
    """
    Build the final ExpectedFeatureSet from processed features and metadata.
    
    Args:
        features: List of final expected features
        dxf_path: Source DXF file path
        dxf_units: Units from DXF file
        raw_entity_count: Number of raw DXF entities
        normalized_entity_count: Number of successfully normalized entities  
        reconstructed_geometry_count: Number of reconstructed geometric structures
        processing_metadata: Additional processing information
        
    Returns:
        Complete ExpectedFeatureSet with all features and metadata
    """
    logger.debug(f"Building ExpectedFeatureSet for {len(features)} features from {dxf_path.name}")
    
    # Calculate detection statistics
    statistics = _calculate_detection_statistics(
        features, raw_entity_count, normalized_entity_count, processing_metadata
    )
    
    # Get current configuration snapshot
    config_snapshot = get_config()
    
    # Create timestamp
    timestamp = datetime.now().isoformat()
    
    # Build the feature set
    feature_set = ExpectedFeatureSet(
        source_dxf_path=dxf_path,
        dxf_units=dxf_units,
        features=features,
        extraction_timestamp=timestamp,
        processing_statistics=statistics,
        configuration_snapshot=config_snapshot,
        raw_entity_count=raw_entity_count,
        normalized_entity_count=normalized_entity_count,
        reconstructed_geometry_count=reconstructed_geometry_count
    )
    
    # Log summary
    _log_feature_set_summary(feature_set)
    
    return feature_set


def _calculate_detection_statistics(
    features: List[ExpectedFeature],
    raw_entity_count: int,
    normalized_entity_count: int,
    processing_metadata: Dict[str, Any]
) -> FeatureDetectionStatistics:
    """Calculate comprehensive detection statistics."""
    
    # Count features by type
    circles_detected = len([f for f in features if f.feature_type == FeatureType.CIRCLE])
    through_holes_detected = len([f for f in features if f.feature_type == FeatureType.THROUGH_HOLE])
    
    # Count reconstructed vs direct features
    reconstructed_features = len([f for f in features if f.source_type == "reconstructed"])
    
    # Get entity type counts from processing metadata
    entities_by_type = processing_metadata.get("entity_type_counts", {})
    
    # Calculate confidence distribution (binned)
    confidence_distribution = _calculate_confidence_distribution(features)
    
    # Estimate filtered features count (approximate)
    # This would ideally be passed from the filtering stage
    estimated_candidates = processing_metadata.get("total_candidates", len(features) * EXPECTED_FEATURE_CANDIDATE_MULTIPLIER)
    filtered_features = max(0, estimated_candidates - len(features))
    
    return FeatureDetectionStatistics(
        total_entities_processed=raw_entity_count,
        entities_by_type=entities_by_type,
        circles_detected=circles_detected,
        through_holes_detected=through_holes_detected,
        reconstructed_features=reconstructed_features,
        filtered_features=filtered_features,
        final_feature_count=len(features),
        confidence_distribution=confidence_distribution
    )


def _calculate_confidence_distribution(features: List[ExpectedFeature]) -> Dict[str, int]:
    """Calculate binned confidence distribution."""
    
    if not features:
        return {}
    
    # Define confidence bins
    bins = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0
    }
    
    for feature in features:
        confidence = feature.confidence
        
        if confidence < CONFIDENCE_BIN_VERY_LOW_MAX:
            bins["0.0-0.2"] += 1
        elif confidence < CONFIDENCE_BIN_LOW_MAX:
            bins["0.2-0.4"] += 1
        elif confidence < CONFIDENCE_BIN_MEDIUM_MAX:
            bins["0.4-0.6"] += 1
        elif confidence < CONFIDENCE_BIN_HIGH_MAX:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    
    # Remove empty bins
    return {bin_name: count for bin_name, count in bins.items() if count > 0}


def _log_feature_set_summary(feature_set: ExpectedFeatureSet) -> None:
    """Log a comprehensive summary of the feature set."""
    
    logger.info(f"=== Expected Feature Set Summary ===")
    logger.info(f"Source: {feature_set.source_dxf_path.name}")
    logger.info(f"Units: {feature_set.dxf_units}")
    logger.info(f"Extraction time: {feature_set.extraction_timestamp}")
    
    logger.info(f"Entity processing:")
    logger.info(f"  Raw entities: {feature_set.raw_entity_count}")
    logger.info(f"  Normalized entities: {feature_set.normalized_entity_count}")
    logger.info(f"  Reconstructed geometry: {feature_set.reconstructed_geometry_count}")
    
    logger.info(f"Feature detection:")
    logger.info(f"  Total features: {feature_set.total_feature_count}")
    logger.info(f"  Circles: {feature_set.circle_count}")
    logger.info(f"  Through holes: {feature_set.through_hole_count}")
    logger.info(f"  Average confidence: {feature_set.average_confidence:.3f}")
    logger.info(f"  High confidence features: {len(feature_set.high_confidence_features)}")
    
    stats = feature_set.processing_statistics
    logger.info(f"Statistics:")
    logger.info(f"  Entities by type: {stats.entities_by_type}")
    logger.info(f"  Reconstructed features: {stats.reconstructed_features}")
    logger.info(f"  Filtered features: {stats.filtered_features}")
    logger.info(f"  Confidence distribution: {stats.confidence_distribution}")
    
    # Log individual features at debug level
    logger.debug(f"Individual features:")
    for feature in feature_set.features:
        radius_str = f"{feature.radius:.2f}" if feature.radius is not None else "N/A"
        logger.debug(f"  {feature.feature_id}: {feature.feature_type.value} "
                    f"center=({feature.center.x:.2f}, {feature.center.y:.2f}) "
                    f"radius={radius_str} "
                    f"confidence={feature.confidence:.3f} "
                    f"source={feature.source_type}")
    
    logger.info(f"=== End Summary ===")


def add_feature_relationships(feature_set: ExpectedFeatureSet) -> None:
    """
    Add relationship information between features in the set.
    
    This is a post-processing step that identifies spatial and geometric
    relationships between the final features.
    """
    
    if len(feature_set.features) < 2:
        return
    
    logger.debug(f"Adding feature relationships for {len(feature_set.features)} features")
    
    # Simple proximity-based relationships
    proximity_threshold = EXPECTED_FEATURE_PROXIMITY_THRESHOLD  # mm
    
    for i, feature1 in enumerate(feature_set.features):
        for feature2 in feature_set.features[i+1:]:
            
            # Calculate center-to-center distance
            distance = feature1.center.distance_to(feature2.center)
            
            if distance <= proximity_threshold:
                # Add mutual proximity relationships
                relationship_info = f"proximity_{feature2.feature_id}_dist_{distance:.1f}mm"
                if relationship_info not in feature1.relationships:
                    feature1.relationships.append(relationship_info)
                
                relationship_info = f"proximity_{feature1.feature_id}_dist_{distance:.1f}mm"
                if relationship_info not in feature2.relationships:
                    feature2.relationships.append(relationship_info)
    
    # Count relationships added
    total_relationships = sum(len(f.relationships) for f in feature_set.features)
    logger.debug(f"Added {total_relationships} feature relationships")