"""
Geometry Reconstruction and Relationship Analysis - LEGACY VERSION

!!!!! THIS FILE IS NOT USED IN PRODUCTION !!!!!
This is a legacy implementation preserved for historical reference only.
The production implementation is in geometry_reconstruction.py

This file uses older API patterns and should NOT be imported or used.
"""

# LEGACY FILE - DO NOT USE IN PRODUCTION
# See geometry_reconstruction.py for current implementation

import logging
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .entity_models import (
    NormalizedEntity, GeometricRelationship, GeometricRelationships, 
    EntityType, Point2D
)
from ..config import (
    GEOMETRIC_TOLERANCE, ANGULAR_TOLERANCE, RADIUS_TOLERANCE_ABSOLUTE,
    RADIUS_TOLERANCE_RELATIVE, CONCENTRICITY_TOLERANCE,
    CIRCLE_RECONSTRUCTION_MIN_COVERAGE, CIRCLE_RECONSTRUCTION_MAX_GAP,
    RELATIONSHIP_CONFIG
)

logger = logging.getLogger(__name__)


@dataclass
class ReconstructedCircle:
    """A circle reconstructed from multiple arc entities."""
    circle_id: str
    center: Point2D
    radius: float
    source_entities: List[str]
    coverage_fraction: float  # What fraction of the circle is covered by arcs
    confidence: float
    gap_angles: List[float]   # Angular gaps between arcs (degrees)


@dataclass
class ReconstructedGeometry:
    """Container for all reconstructed geometric structures."""
    reconstructed_circles: List[ReconstructedCircle]
    entity_groups: List[List[str]]  # Groups of entities that form structures
    reconstruction_metadata: Dict


def analyze_relationships(entities: List[NormalizedEntity]) -> GeometricRelationships:
    """
    Analyze spatial and geometric relationships between entities.
    
    Args:
        entities: List of normalized entities
        
    Returns:
        GeometricRelationships containing all detected relationships
    """
    logger.debug(f"Analyzing relationships between {len(entities)} entities")
    
    relationships = []
    proximity_threshold = RELATIONSHIP_CONFIG["proximity_threshold"]
    
    # Analyze pairwise relationships
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            
            # Check concentricity for circular entities
            concentric_rel = _check_concentricity(entity1, entity2)
            if concentric_rel:
                relationships.append(concentric_rel)
            
            # Check proximity
            proximity_rel = _check_proximity(entity1, entity2, proximity_threshold)
            if proximity_rel:
                relationships.append(proximity_rel)
            
            # Check radius similarity for circular entities
            radius_sim_rel = _check_radius_similarity(entity1, entity2)
            if radius_sim_rel:
                relationships.append(radius_sim_rel)
            
            # Check arc continuity 
            continuity_rel = _check_arc_continuity(entity1, entity2)
            if continuity_rel:
                relationships.append(continuity_rel)
    
    # Group related entities
    entity_groups = _find_entity_groups(relationships, [e.source_entity.entity_id for e in entities])
    
    # Analysis metadata
    metadata = {
        "total_entities": len(entities),
        "total_relationships": len(relationships),
        "relationship_types": _count_relationship_types(relationships),
        "entity_groups_count": len(entity_groups),
        "proximity_threshold": proximity_threshold
    }
    
    logger.info(f"Found {len(relationships)} relationships in {len(entity_groups)} groups")
    
    return GeometricRelationships(
        relationships=relationships,
        entity_groups=entity_groups,
        analysis_metadata=metadata
    )


def reconstruct_geometry(entities: List[NormalizedEntity], 
                        relationships: GeometricRelationships) -> ReconstructedGeometry:
    """
    Reconstruct meaningful geometric structures from primitive entities.
    
    Args:
        entities: List of normalized entities
        relationships: Analyzed relationships between entities
        
    Returns:
        ReconstructedGeometry containing reconstructed structures
    """
    logger.debug(f"Reconstructing geometry from {len(entities)} entities")
    
    # Create entity lookup for quick access
    entity_lookup = {e.source_entity.entity_id: e for e in entities}
    
    # Reconstruct circles from arc groups
    reconstructed_circles = _reconstruct_circles_from_arcs(
        entities, relationships, entity_lookup
    )
    
    metadata = {
        "circles_reconstructed": len(reconstructed_circles),
        "min_coverage_threshold": CIRCLE_RECONSTRUCTION_MIN_COVERAGE,
        "max_gap_threshold": CIRCLE_RECONSTRUCTION_MAX_GAP
    }
    
    logger.info(f"Reconstructed {len(reconstructed_circles)} circles from arc combinations")
    
    return ReconstructedGeometry(
        reconstructed_circles=reconstructed_circles,
        entity_groups=relationships.entity_groups,
        reconstruction_metadata=metadata
    )


def _check_concentricity(entity1: NormalizedEntity, entity2: NormalizedEntity) -> Optional[GeometricRelationship]:
    """Check if two circular entities are concentric."""
    
    # Both must be circular (CIRCLE or ARC) 
    if (entity1.source_entity.entity_type not in [EntityType.CIRCLE, EntityType.ARC] or
        entity2.source_entity.entity_type not in [EntityType.CIRCLE, EntityType.ARC]):
        return None
    
    if not (entity1.normalized_center and entity2.normalized_center):
        return None
    
    # Calculate distance between centers
    center_distance = entity1.normalized_center.distance_to(entity2.normalized_center)
    
    if center_distance <= CONCENTRICITY_TOLERANCE:
        confidence = max(0.0, 1.0 - (center_distance / CONCENTRICITY_TOLERANCE))
        
        return GeometricRelationship(
            entity1_id=entity1.source_entity.entity_id,
            entity2_id=entity2.source_entity.entity_id,
            relationship_type="concentric",
            confidence=confidence,
            distance=center_distance,
            parameters={"center_tolerance": CONCENTRICITY_TOLERANCE}
        )
    
    return None


def _check_proximity(entity1: NormalizedEntity, entity2: NormalizedEntity, 
                    threshold: float) -> Optional[GeometricRelationship]:
    """Check if two entities are spatially proximate."""
    
    # Get representative points for each entity
    point1 = _get_representative_point(entity1)
    point2 = _get_representative_point(entity2)
    
    if not (point1 and point2):
        return None
    
    distance = point1.distance_to(point2)
    
    if distance <= threshold:
        confidence = max(0.0, 1.0 - (distance / threshold))
        
        return GeometricRelationship(
            entity1_id=entity1.source_entity.entity_id,
            entity2_id=entity2.source_entity.entity_id,
            relationship_type="proximity",
            confidence=confidence,
            distance=distance,
            parameters={"proximity_threshold": threshold}
        )
    
    return None


def _check_radius_similarity(entity1: NormalizedEntity, entity2: NormalizedEntity) -> Optional[GeometricRelationship]:
    """Check if two circular entities have similar radii."""
    
    if (entity1.source_entity.entity_type not in [EntityType.CIRCLE, EntityType.ARC] or
        entity2.source_entity.entity_type not in [EntityType.CIRCLE, EntityType.ARC]):
        return None
    
    r1 = entity1.normalized_radius
    r2 = entity2.normalized_radius
    
    if not (r1 and r2):
        return None
    
    # Check both absolute and relative tolerance
    abs_diff = abs(r1 - r2)
    rel_diff = abs_diff / max(r1, r2)
    
    if (abs_diff <= RADIUS_TOLERANCE_ABSOLUTE and 
        rel_diff <= RADIUS_TOLERANCE_RELATIVE):
        
        confidence = max(0.0, 1.0 - (rel_diff / RADIUS_TOLERANCE_RELATIVE))
        
        return GeometricRelationship(
            entity1_id=entity1.source_entity.entity_id,
            entity2_id=entity2.source_entity.entity_id,
            relationship_type="similar_radius",
            confidence=confidence,
            parameters={
                "radius1": r1,
                "radius2": r2,
                "absolute_diff": abs_diff,
                "relative_diff": rel_diff
            }
        )
    
    return None


def _check_arc_continuity(entity1: NormalizedEntity, entity2: NormalizedEntity) -> Optional[GeometricRelationship]:
    """Check if two arcs could be part of the same circle."""
    
    if (entity1.source_entity.entity_type != EntityType.ARC or
        entity2.source_entity.entity_type != EntityType.ARC):
        return None
    
    # Must have same center and radius (within tolerance)
    if not _check_concentricity(entity1, entity2):
        return None
    
    if not _check_radius_similarity(entity1, entity2):
        return None
    
    # Check angular continuity
    arc1 = entity1.source_entity
    arc2 = entity2.source_entity
    
    if not (arc1.start_angle is not None and arc1.end_angle is not None and
            arc2.start_angle is not None and arc2.end_angle is not None):
        return None
    
    # Calculate minimum angular gap between the arcs
    gap = _calculate_arc_gap(arc1.start_angle, arc1.end_angle, 
                            arc2.start_angle, arc2.end_angle)
    
    if gap <= CIRCLE_RECONSTRUCTION_MAX_GAP:
        confidence = max(0.0, 1.0 - (gap / CIRCLE_RECONSTRUCTION_MAX_GAP))
        
        return GeometricRelationship(
            entity1_id=entity1.source_entity.entity_id,
            entity2_id=entity2.source_entity.entity_id,
            relationship_type="arc_continuity",
            confidence=confidence,
            angle=gap,
            parameters={"angular_gap": gap}
        )
    
    return None


def _get_representative_point(entity: NormalizedEntity) -> Optional[Point2D]:
    """Get a representative point for an entity (center, midpoint, etc.)."""
    
    if entity.normalized_center:
        return entity.normalized_center
    
    if entity.contains_point:
        return entity.contains_point
    
    # For lines, use midpoint
    if entity.source_entity.entity_type == EntityType.LINE:
        start = entity.source_entity.start_point
        end = entity.source_entity.end_point
        if start and end:
            return Point2D((start.x + end.x) / 2, (start.y + end.y) / 2)
    
    # For polylines, use first point or centroid
    if (entity.source_entity.entity_type == EntityType.LWPOLYLINE and 
        entity.normalized_points):
        if len(entity.normalized_points) == 1:
            return entity.normalized_points[0]
        else:
            # Return centroid
            x_sum = sum(p.x for p in entity.normalized_points)
            y_sum = sum(p.y for p in entity.normalized_points)
            n = len(entity.normalized_points)
            return Point2D(x_sum / n, y_sum / n)
    
    return None


def _calculate_arc_gap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate the minimum angular gap between two arcs (degrees)."""
    
    # Normalize angles to [0, 360)
    def normalize_angle(angle):
        return angle % 360
    
    start1, end1 = normalize_angle(start1), normalize_angle(end1)
    start2, end2 = normalize_angle(start2), normalize_angle(end2)
    
    # Handle wrap-around cases
    if end1 < start1:
        end1 += 360
    if end2 < start2:
        end2 += 360
    
    # Calculate gaps in both directions
    gaps = []
    
    # Gap from end of arc1 to start of arc2
    gap1 = (start2 - end1) % 360
    gaps.append(gap1)
    
    # Gap from end of arc2 to start of arc1  
    gap2 = (start1 - end2) % 360
    gaps.append(gap2)
    
    return min(gaps)


def _find_entity_groups(relationships: List[GeometricRelationship], 
                       all_entity_ids: List[str]) -> List[List[str]]:
    """Find groups of related entities using relationship connectivity."""
    
    # Build adjacency graph
    adjacency = {entity_id: set() for entity_id in all_entity_ids}
    
    for rel in relationships:
        # Only use high-confidence relationships for grouping
        if rel.confidence >= 0.5:
            adjacency[rel.entity1_id].add(rel.entity2_id)
            adjacency[rel.entity2_id].add(rel.entity1_id)
    
    # Find connected components
    visited = set()
    groups = []
    
    for entity_id in all_entity_ids:
        if entity_id not in visited:
            group = []
            stack = [entity_id]
            
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    group.append(current)
                    
                    # Add unvisited neighbors to stack
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            if group:
                groups.append(group)
    
    return groups


def _reconstruct_circles_from_arcs(entities: List[NormalizedEntity],
                                  relationships: GeometricRelationships,
                                  entity_lookup: Dict[str, NormalizedEntity]) -> List[ReconstructedCircle]:
    """Reconstruct complete circles from arc combinations."""
    
    reconstructed = []
    
    # Find groups that contain multiple arcs  
    for group in relationships.entity_groups:
        arc_entities = []
        for entity_id in group:
            entity = entity_lookup.get(entity_id)
            if entity and entity.source_entity.entity_type == EntityType.ARC:
                arc_entities.append(entity)
        
        if len(arc_entities) >= 2:
            # Try to reconstruct a circle from this arc group
            circle = _try_reconstruct_circle_from_arc_group(arc_entities)
            if circle:
                reconstructed.append(circle)
    
    return reconstructed


def _try_reconstruct_circle_from_arc_group(arc_entities: List[NormalizedEntity]) -> Optional[ReconstructedCircle]:
    """Attempt to reconstruct a circle from a group of arcs."""
    
    if not arc_entities:
        return None
    
    # All arcs should have the same center and radius (verified by relationships)
    first_arc = arc_entities[0].source_entity
    center = first_arc.center
    radius = first_arc.radius
    
    if not (center and radius):
        return None
    
    # Calculate total angular coverage
    total_coverage = 0.0
    gaps = []
    source_entity_ids = []
    
    # Collect all arc spans
    arc_spans = []
    for arc_entity in arc_entities:
        arc = arc_entity.source_entity
        if arc.start_angle is not None and arc.end_angle is not None:
            span = arc.arc_span_degrees
            if span:
                arc_spans.append((arc.start_angle, arc.end_angle, span))
                total_coverage += span
                source_entity_ids.append(arc.entity_id)
    
    if not arc_spans:
        return None
    
    # Calculate coverage fraction
    coverage_fraction = total_coverage / 360.0
    
    # Check if coverage meets minimum threshold
    if coverage_fraction < CIRCLE_RECONSTRUCTION_MIN_COVERAGE:
        return None
    
    # Calculate gaps (simplified - could be more sophisticated)
    # For now, assume average gap size based on uncovered area
    uncovered_angle = 360.0 - total_coverage
    estimated_gap_count = max(1, len(arc_spans) - 1)
    average_gap = uncovered_angle / estimated_gap_count if estimated_gap_count > 0 else 0
    gaps = [average_gap] * estimated_gap_count
    
    # Calculate confidence based on coverage and number of arcs
    confidence = min(1.0, coverage_fraction * (len(arc_spans) / 4.0))  # Higher confidence with more arcs
    
    circle_id = f"reconstructed_circle_{hash(tuple(source_entity_ids)) % 10000}"
    
    return ReconstructedCircle(
        circle_id=circle_id,
        center=center,
        radius=radius,
        source_entities=source_entity_ids,
        coverage_fraction=coverage_fraction,
        confidence=confidence,
        gap_angles=gaps
    )


def _count_relationship_types(relationships: List[GeometricRelationship]) -> Dict[str, int]:
    """Count relationships by type for metadata."""
    counts = {}
    for rel in relationships:
        counts[rel.relationship_type] = counts.get(rel.relationship_type, 0) + 1
    return counts