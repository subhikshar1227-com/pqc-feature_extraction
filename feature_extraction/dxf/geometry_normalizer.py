"""
Geometry Normalization

Converts raw DXF entities into normalized geometric representations
with consistent coordinate systems and computed properties.
"""

import logging
from typing import List

from .entity_models import DxfEntity, NormalizedEntity

logger = logging.getLogger(__name__)


def normalize_geometry(entities: List[DxfEntity]) -> List[NormalizedEntity]:
    """
    Normalize geometric entities into consistent representations.
    
    Args:
        entities: List of raw DXF entities
        
    Returns:
        List of normalized entities with computed geometric properties
    """
    logger.debug(f"Normalizing {len(entities)} entities")
    
    normalized = []
    
    for entity in entities:
        try:
            # Create normalized entity - the __post_init__ method handles
            # coordinate normalization and property computation
            normalized_entity = NormalizedEntity(source_entity=entity)
            normalized.append(normalized_entity)
            
        except Exception as e:
            logger.warning(f"Failed to normalize entity {entity.entity_id}: {e}")
            continue
    
    logger.info(f"Successfully normalized {len(normalized)} entities")
    
    # Log normalization summary
    _log_normalization_summary(entities, normalized)
    
    return normalized


def _log_normalization_summary(raw_entities: List[DxfEntity], 
                              normalized_entities: List[NormalizedEntity]) -> None:
    """Log summary of normalization results."""
    
    # Count by entity type
    raw_counts = {}
    normalized_counts = {}
    
    for entity in raw_entities:
        entity_type = entity.entity_type.value
        raw_counts[entity_type] = raw_counts.get(entity_type, 0) + 1
    
    for norm_entity in normalized_entities:
        entity_type = norm_entity.source_entity.entity_type.value
        normalized_counts[entity_type] = normalized_counts.get(entity_type, 0) + 1
    
    logger.debug(f"Normalization summary:")
    logger.debug(f"  Raw entities: {raw_counts}")
    logger.debug(f"  Normalized: {normalized_counts}")
    
    # Check for any entities lost during normalization
    for entity_type, raw_count in raw_counts.items():
        norm_count = normalized_counts.get(entity_type, 0)
        if norm_count < raw_count:
            logger.warning(f"Lost {raw_count - norm_count} {entity_type} entities during normalization")
    
    # Log geometric property computation statistics
    entities_with_areas = sum(1 for e in normalized_entities if e.enclosed_area is not None)
    entities_with_perimeters = sum(1 for e in normalized_entities if e.perimeter_length is not None)
    entities_with_signatures = sum(1 for e in normalized_entities if e.geometric_signature is not None)
    
    logger.debug(f"Computed properties:")
    logger.debug(f"  Areas calculated: {entities_with_areas}")
    logger.debug(f"  Perimeters calculated: {entities_with_perimeters}")
    logger.debug(f"  Geometric signatures: {entities_with_signatures}")