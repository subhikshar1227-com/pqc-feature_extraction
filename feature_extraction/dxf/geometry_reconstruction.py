"""
Geometry Reconstruction and Relationship Analysis

Analyzes spatial relationships between DXF entities and reconstructs
meaningful geometric structures from primitive entities with strict geometric validation.
"""

import logging
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from .entity_models import (
    NormalizedEntity, GeometricRelationship, GeometricRelationships, 
    EntityType, Point2D
)
from ..config import (
    ARC_CENTER_TOLERANCE, ARC_RADIUS_TOLERANCE, ARC_ANGULAR_CONTINUITY_TOLERANCE,
    CIRCLE_RECONSTRUCTION_MIN_COVERAGE, CIRCLE_RECONSTRUCTION_SEMICIRCLE_MIN_COVERAGE,
    CIRCLE_RECONSTRUCTION_MAX_GAP, CIRCLE_RECONSTRUCTION_MAX_OVERLAP, 
    CIRCLE_RECONSTRUCTION_MIN_ARCS, CIRCLE_RECONSTRUCTION_MAX_ARCS, 
    CONCENTRICITY_TOLERANCE, RADIUS_TOLERANCE_ABSOLUTE, RADIUS_TOLERANCE_RELATIVE,
    RELATIONSHIP_CONFIG, SEMICIRCLE_ARC_SPAN_MIN, SEMICIRCLE_ARC_SPAN_MAX,
    SEMICIRCLE_ARC_SPAN_TOLERANCE, COVERAGE_MERGE_TOLERANCE,
    COVERAGE_WRAPAROUND_THRESHOLD, COVERAGE_TINY_GAP_THRESHOLD,
    RECONSTRUCTION_COVERAGE_WEIGHT, RECONSTRUCTION_GAP_WEIGHT,
    RECONSTRUCTION_OVERLAP_WEIGHT, RECONSTRUCTION_ARC_COUNT_WEIGHT,
    RECONSTRUCTION_MIN_CONFIDENCE,
    # Additional geometric constants
    ARC_ANGLE_HALF_CIRCLE, ARC_ANGLE_FULL_CIRCLE, ARC_GAP_OVERLAP_TOLERANCE,
    GEOMETRIC_OVERLAP_CONSERVATIVE_ESTIMATE, GEOMETRIC_OVERLAP_MERGE_TOLERANCE_FACTOR,
    HIGH_CONFIDENCE_RELATIONSHIP_THRESHOLD,
    # Arc compatibility parameters from forensic audit
    ARC_COMPATIBILITY_BASE_CONFIDENCE, ARC_COMPATIBILITY_GAP_PENALTY_FACTOR,
    ARC_GROUP_DEFAULT_TOLERANCE
)

logger = logging.getLogger(__name__)


@dataclass
class ArcInterval:
    """Represents an angular interval covered by an arc."""
    start_angle: float  # degrees, normalized to [0, 360)
    end_angle: float    # degrees, normalized to [0, 360)  
    entity_id: str
    
    @property
    def span(self) -> float:
        """Angular span in degrees, handling wraparound."""
        if self.end_angle >= self.start_angle:
            return self.end_angle - self.start_angle
        else:
            return (360.0 - self.start_angle) + self.end_angle
    
    def overlaps_with(self, other: 'ArcInterval', tolerance: float = None) -> bool:
        """Check if this interval overlaps with another within tolerance."""
        if tolerance is None:
            tolerance = ARC_GROUP_DEFAULT_TOLERANCE
            
        # Convert to normalized intervals
        self_intervals = self._to_normalized_intervals()
        other_intervals = other._to_normalized_intervals()
        
        for s_start, s_end in self_intervals:
            for o_start, o_end in other_intervals:
                # Check overlap with tolerance
                if (s_start <= o_end + tolerance and o_start <= s_end + tolerance):
                    return True
        return False
    
    def _to_normalized_intervals(self) -> List[Tuple[float, float]]:
        """Convert to list of intervals handling wraparound."""
        if self.end_angle >= self.start_angle:
            return [(self.start_angle, self.end_angle)]
        else:
            return [(self.start_angle, ARC_ANGLE_FULL_CIRCLE), (0.0, self.end_angle)]


@dataclass
class ArcGroup:
    """Group of arcs that may form a reconstructed circle."""
    center: Point2D
    radius: float
    arcs: List[NormalizedEntity]
    intervals: List[ArcInterval]
    total_coverage: float
    gaps: List[float]  # Gap sizes in degrees
    overlaps: List[float]  # Overlap sizes in degrees

@dataclass
class ReconstructedCircle:
    """A circle reconstructed from multiple arc entities with strict validation."""
    circle_id: str
    center: Point2D
    radius: float
    source_entities: List[str]
    coverage_fraction: float  # Actual coverage after removing overlaps
    confidence: float
    gaps: List[float]   # Gap angles between arcs (degrees)
    overlaps: List[float]  # Overlap angles between arcs (degrees)
    validation_evidence: Dict[str, any]  # Evidence supporting reconstruction


@dataclass
class ReconstructedGeometry:
    """Container for all reconstructed geometric structures."""
    reconstructed_circles: List[ReconstructedCircle]
    arc_groups_analyzed: int
    arc_groups_rejected: int
    rejection_reasons: Dict[str, int]  # Reason -> count


def analyze_relationships(entities: List[NormalizedEntity]) -> GeometricRelationships:
    """
    Analyze spatial and geometric relationships between entities with strict validation.
    """
    logger.debug(f"Analyzing relationships between {len(entities)} entities")
    
    relationships = []
    proximity_threshold = RELATIONSHIP_CONFIG["proximity_threshold"]
    
    # Analyze pairwise relationships with strict criteria
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            
            # Check concentricity for circular entities (strict)
            concentric_rel = _check_strict_concentricity(entity1, entity2)
            if concentric_rel:
                relationships.append(concentric_rel)
            
            # Check arc compatibility for potential circle reconstruction
            arc_compat_rel = _check_arc_compatibility(entity1, entity2)
            if arc_compat_rel:
                relationships.append(arc_compat_rel)
            
            # Check proximity (less restrictive, used for context only)
            proximity_rel = _check_proximity(entity1, entity2, proximity_threshold)
            if proximity_rel:
                relationships.append(proximity_rel)
    
    # Group related entities based on strong relationships only
    entity_groups = _find_entity_groups_strict(
        relationships, 
        [e.source_entity.entity_id for e in entities]
    )
    
    # Analysis metadata
    metadata = {
        "total_entities": len(entities),
        "total_relationships": len(relationships),
        "relationship_types": _count_relationship_types(relationships),
        "entity_groups_count": len(entity_groups),
        "strict_validation": True
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
    Reconstruct circles from arcs with strict geometric validation.
    """
    logger.debug(f"Reconstructing geometry with strict validation")
    
    # Create entity lookup
    entity_lookup = {e.source_entity.entity_id: e for e in entities}
    
    # Find arc entities only
    arc_entities = [e for e in entities if e.source_entity.entity_type == EntityType.ARC]
    
    if len(arc_entities) < CIRCLE_RECONSTRUCTION_MIN_ARCS:
        logger.debug(f"Insufficient arcs for reconstruction: {len(arc_entities)} < {CIRCLE_RECONSTRUCTION_MIN_ARCS}")
        return ReconstructedGeometry(
            reconstructed_circles=[],
            arc_groups_analyzed=0,
            arc_groups_rejected=0,
            rejection_reasons={}
        )
    
    # Find strictly compatible arc groups
    arc_groups = _find_strict_arc_groups(arc_entities, relationships)
    
    # Attempt to reconstruct circles from each group
    reconstructed_circles = []
    rejection_reasons = {}
    
    for group in arc_groups:
        result = _attempt_strict_circle_reconstruction(group, entity_lookup)
        if result is not None:
            reconstructed_circles.append(result)
        else:
            # Track rejection reason (simplified)
            reason = "insufficient_coverage_or_gaps"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    logger.info(f"Reconstructed {len(reconstructed_circles)} circles from {len(arc_groups)} arc groups")
    
    return ReconstructedGeometry(
        reconstructed_circles=reconstructed_circles,
        arc_groups_analyzed=len(arc_groups),
        arc_groups_rejected=len(arc_groups) - len(reconstructed_circles),
        rejection_reasons=rejection_reasons
    )


def _check_strict_concentricity(entity1: NormalizedEntity, entity2: NormalizedEntity) -> Optional[GeometricRelationship]:
    """Check concentricity with strict tolerance requirements."""
    
    # Both must be circular (CIRCLE or ARC) 
    if (entity1.source_entity.entity_type not in [EntityType.CIRCLE, EntityType.ARC] or
        entity2.source_entity.entity_type not in [EntityType.CIRCLE, EntityType.ARC]):
        return None
    
    if not (entity1.normalized_center and entity2.normalized_center):
        return None
    
    # Calculate distance between centers
    center_distance = entity1.normalized_center.distance_to(entity2.normalized_center)
    
    # Strict concentricity requirement
    if center_distance <= CONCENTRICITY_TOLERANCE:
        confidence = max(0.0, 1.0 - (center_distance / CONCENTRICITY_TOLERANCE))
        
        return GeometricRelationship(
            entity1_id=entity1.source_entity.entity_id,
            entity2_id=entity2.source_entity.entity_id,
            relationship_type="strict_concentric",
            confidence=confidence,
            distance=center_distance,
            parameters={"center_tolerance": CONCENTRICITY_TOLERANCE}
        )
    
    return None
def _check_arc_compatibility(entity1: NormalizedEntity, entity2: NormalizedEntity) -> Optional[GeometricRelationship]:
    """Check if two arcs are compatible for circle reconstruction."""
    
    if (entity1.source_entity.entity_type != EntityType.ARC or
        entity2.source_entity.entity_type != EntityType.ARC):
        return None
    
    arc1 = entity1.source_entity
    arc2 = entity2.source_entity
    
    # Must have valid geometric parameters
    if not all([arc1.center, arc1.radius, arc1.start_angle is not None, arc1.end_angle is not None,
                arc2.center, arc2.radius, arc2.start_angle is not None, arc2.end_angle is not None]):
        return None
    
    # Check center compatibility (strict)
    center_distance = arc1.center.distance_to(arc2.center)
    if center_distance > ARC_CENTER_TOLERANCE:
        return None
    
    # Check radius compatibility (strict)
    radius_diff = abs(arc1.radius - arc2.radius)
    if radius_diff > ARC_RADIUS_TOLERANCE:
        return None
    
    # Check for angular continuity or appropriate separation
    gap = _calculate_arc_gap(arc1.start_angle, arc1.end_angle, 
                            arc2.start_angle, arc2.end_angle)
    
    if gap <= ARC_ANGULAR_CONTINUITY_TOLERANCE:
        confidence = ARC_COMPATIBILITY_BASE_CONFIDENCE - (gap / ARC_ANGULAR_CONTINUITY_TOLERANCE) * ARC_COMPATIBILITY_GAP_PENALTY_FACTOR
        
        return GeometricRelationship(
            entity1_id=entity1.source_entity.entity_id,
            entity2_id=entity2.source_entity.entity_id,
            relationship_type="arc_compatible",
            confidence=confidence,
            angle=gap,
            parameters={
                "center_distance": center_distance,
                "radius_difference": radius_diff,
                "angular_gap": gap
            }
        )
    
    return None


def _check_proximity(entity1: NormalizedEntity, entity2: NormalizedEntity, 
                    threshold: float) -> Optional[GeometricRelationship]:
    """Check proximity for context only (not used for reconstruction)."""
    
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

def _find_strict_arc_groups(arc_entities: List[NormalizedEntity], 
                           relationships: GeometricRelationships) -> List[ArcGroup]:
    """Find groups of arcs that are strictly compatible for circle reconstruction."""
    
    # Build compatibility graph based only on arc_compatible relationships
    compatibility_graph = {}
    for entity in arc_entities:
        compatibility_graph[entity.source_entity.entity_id] = set()
    
    for rel in relationships.relationships:
        if (rel.relationship_type == "arc_compatible" and 
            rel.confidence >= HIGH_CONFIDENCE_RELATIONSHIP_THRESHOLD and  # Configurable high confidence required
            rel.entity1_id in compatibility_graph and 
            rel.entity2_id in compatibility_graph):
            
            compatibility_graph[rel.entity1_id].add(rel.entity2_id)
            compatibility_graph[rel.entity2_id].add(rel.entity1_id)
    
    # Find connected components (potential arc groups)
    visited = set()
    arc_groups = []
    entity_lookup = {e.source_entity.entity_id: e for e in arc_entities}
    
    for entity in arc_entities:
        entity_id = entity.source_entity.entity_id
        if entity_id not in visited:
            group_entities = []
            stack = [entity_id]
            
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    group_entities.append(entity_lookup[current])
                    
                    # Add unvisited compatible neighbors
                    for neighbor in compatibility_graph[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            # Only process groups with sufficient arcs
            if len(group_entities) >= CIRCLE_RECONSTRUCTION_MIN_ARCS:
                arc_group = _analyze_arc_group(group_entities)
                if arc_group is not None:
                    arc_groups.append(arc_group)
    
    logger.debug(f"Found {len(arc_groups)} potential arc groups for reconstruction")
    return arc_groups


def _analyze_arc_group(arc_entities: List[NormalizedEntity]) -> Optional[ArcGroup]:
    """Analyze a group of arcs to determine if they can form a circle."""
    
    if len(arc_entities) < CIRCLE_RECONSTRUCTION_MIN_ARCS:
        return None
    
    if len(arc_entities) > CIRCLE_RECONSTRUCTION_MAX_ARCS:
        logger.debug(f"Arc group too large: {len(arc_entities)} > {CIRCLE_RECONSTRUCTION_MAX_ARCS}")
        return None
    
    # Calculate average center and radius
    centers = []
    radii = []
    
    for entity in arc_entities:
        arc = entity.source_entity
        if arc.center and arc.radius is not None:
            centers.append(arc.center)
            radii.append(arc.radius)
    
    if len(centers) != len(arc_entities):
        return None  # Missing geometric data
    
    # Check center consistency
    avg_center_x = sum(c.x for c in centers) / len(centers)
    avg_center_y = sum(c.y for c in centers) / len(centers)
    avg_center = Point2D(avg_center_x, avg_center_y)
    
    max_center_deviation = max(c.distance_to(avg_center) for c in centers)
    if max_center_deviation > ARC_CENTER_TOLERANCE:
        logger.debug(f"Arc group center deviation too large: {max_center_deviation} > {ARC_CENTER_TOLERANCE}")
        return None
    
    # Check radius consistency
    avg_radius = sum(radii) / len(radii)
    max_radius_deviation = max(abs(r - avg_radius) for r in radii)
    if max_radius_deviation > ARC_RADIUS_TOLERANCE:
        logger.debug(f"Arc group radius deviation too large: {max_radius_deviation} > {ARC_RADIUS_TOLERANCE}")
        return None
    # Analyze angular coverage
    intervals = []
    for entity in arc_entities:
        arc = entity.source_entity
        if arc.start_angle is not None and arc.end_angle is not None:
            interval = ArcInterval(
                start_angle=arc.start_angle % 360.0,
                end_angle=arc.end_angle % 360.0,
                entity_id=arc.entity_id
            )
            intervals.append(interval)
    
    if len(intervals) != len(arc_entities):
        return None
    
    # Calculate total coverage and detect overlaps/gaps
    total_coverage, gaps, overlaps = _calculate_coverage_and_gaps(intervals)
    
    return ArcGroup(
        center=avg_center,
        radius=avg_radius,
        arcs=arc_entities,
        intervals=intervals,
        total_coverage=total_coverage,
        gaps=gaps,
        overlaps=overlaps
    )


def _calculate_coverage_and_gaps(intervals: List[ArcInterval]) -> Tuple[float, List[float], List[float]]:
    """Calculate actual angular coverage, gaps, and overlaps using configurable parameters."""
    
    if not intervals:
        return 0.0, [], []
    
    # Special case: semi-circle pair detection using configurable parameters
    if len(intervals) == 2:
        span1, span2 = intervals[0].span, intervals[1].span
        
        # Check if both arcs are approximately semi-circles
        is_semicircle_1 = (SEMICIRCLE_ARC_SPAN_MIN <= span1 <= SEMICIRCLE_ARC_SPAN_MAX or
                          abs(span1 - ARC_ANGLE_HALF_CIRCLE) < SEMICIRCLE_ARC_SPAN_TOLERANCE)
        is_semicircle_2 = (SEMICIRCLE_ARC_SPAN_MIN <= span2 <= SEMICIRCLE_ARC_SPAN_MAX or
                          abs(span2 - ARC_ANGLE_HALF_CIRCLE) < SEMICIRCLE_ARC_SPAN_TOLERANCE)
        
        if is_semicircle_1 and is_semicircle_2:
            # Check if they complement each other to form near-complete circle
            total_span = span1 + span2
            if total_span >= COVERAGE_WRAPAROUND_THRESHOLD:  # Configurable threshold
                coverage_fraction = min(1.0, total_span / ARC_ANGLE_FULL_CIRCLE)
                gaps = [ARC_ANGLE_FULL_CIRCLE - total_span] if total_span < ARC_ANGLE_FULL_CIRCLE else []
                return coverage_fraction, gaps, []
    
    # Convert all intervals to normalized form and merge overlaps
    normalized_intervals = []
    for interval in intervals:
        norm_intervals = interval._to_normalized_intervals()
        normalized_intervals.extend(norm_intervals)
    
    # Sort intervals by start angle
    normalized_intervals.sort(key=lambda x: x[0])
    
    # Merge overlapping intervals using configurable tolerance
    merged_intervals = []
    for start, end in normalized_intervals:
        if not merged_intervals:
            merged_intervals.append((start, end))
        else:
            last_start, last_end = merged_intervals[-1]
            if start <= last_end + COVERAGE_MERGE_TOLERANCE:  # Configurable merging tolerance
                # Merge intervals
                merged_intervals[-1] = (last_start, max(last_end, end))
            else:
                merged_intervals.append((start, end))
    
    # Calculate total coverage
    total_coverage = sum(end - start for start, end in merged_intervals)
    
    # Handle wraparound case using configurable threshold
    if len(merged_intervals) > 1:
        first_start = merged_intervals[0][0]
        last_end = merged_intervals[-1][1]
        if first_start == 0.0 and last_end >= COVERAGE_WRAPAROUND_THRESHOLD:
            # Near full circle case
            total_coverage = min(360.0, total_coverage)
    
    # Calculate gaps using configurable threshold
    gaps = []
    for i in range(len(merged_intervals)):
        next_i = (i + 1) % len(merged_intervals)
        current_end = merged_intervals[i][1]
        next_start = merged_intervals[next_i][0]
        
        if next_i == 0:  # Wraparound case
            gap = (360.0 - current_end) + next_start
        else:
            gap = next_start - current_end
        
        if gap > COVERAGE_TINY_GAP_THRESHOLD:  # Configurable threshold for ignoring tiny gaps
            gaps.append(gap)
    
    # Calculate overlaps (simplified - overlaps within original intervals)
    overlaps = []
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            if intervals[i].overlaps_with(intervals[j], tolerance=COVERAGE_MERGE_TOLERANCE * GEOMETRIC_OVERLAP_MERGE_TOLERANCE_FACTOR):
                # Estimate overlap size using geometric analysis
                overlap = min(intervals[i].span, intervals[j].span) * GEOMETRIC_OVERLAP_CONSERVATIVE_ESTIMATE  # Conservative estimate
                overlaps.append(overlap)
    
    coverage_fraction = total_coverage / 360.0
    
    return coverage_fraction, gaps, overlaps
def _attempt_strict_circle_reconstruction(arc_group: ArcGroup, 
                                        entity_lookup: Dict[str, NormalizedEntity]) -> Optional[ReconstructedCircle]:
    """Attempt to reconstruct a circle from an arc group with strict validation using configurable parameters."""
    
    # Check coverage requirement with special handling for semi-circles
    min_required_coverage = CIRCLE_RECONSTRUCTION_MIN_COVERAGE
    
    # For pairs of semi-circle arcs, use special threshold
    if (len(arc_group.arcs) == 2 and 
        all(SEMICIRCLE_ARC_SPAN_MIN <= interval.span <= SEMICIRCLE_ARC_SPAN_MAX 
            for interval in arc_group.intervals)):
        min_required_coverage = CIRCLE_RECONSTRUCTION_SEMICIRCLE_MIN_COVERAGE
        logger.debug(f"Detected semi-circle pair, using specialized coverage threshold: {min_required_coverage}")
    
    if arc_group.total_coverage < min_required_coverage:
        logger.debug(f"Arc group coverage insufficient: {arc_group.total_coverage:.2f} < {min_required_coverage}")
        return None
    
    # Check gap requirements using configurable threshold
    max_gap = max(arc_group.gaps) if arc_group.gaps else 0.0
    if max_gap > CIRCLE_RECONSTRUCTION_MAX_GAP:
        logger.debug(f"Arc group has gap too large: {max_gap} > {CIRCLE_RECONSTRUCTION_MAX_GAP}")
        return None
    
    # Check overlap requirements using configurable threshold
    total_overlap = sum(arc_group.overlaps)
    if total_overlap > CIRCLE_RECONSTRUCTION_MAX_OVERLAP:
        logger.debug(f"Arc group has excessive overlap: {total_overlap} > {CIRCLE_RECONSTRUCTION_MAX_OVERLAP}")
        return None
    
    # Calculate confidence based on configurable weights
    coverage_factor = arc_group.total_coverage  # Already 0-1
    gap_factor = 1.0 - (max_gap / CIRCLE_RECONSTRUCTION_MAX_GAP) if arc_group.gaps else 1.0
    overlap_penalty = max(0.0, 1.0 - (total_overlap / CIRCLE_RECONSTRUCTION_MAX_OVERLAP))
    
    # Arc count factor: prefer more arcs up to optimal number
    optimal_arc_count = 4
    arc_count_factor = min(1.0, len(arc_group.arcs) / optimal_arc_count)
    
    # Calculate weighted confidence using configurable weights
    confidence = (coverage_factor * RECONSTRUCTION_COVERAGE_WEIGHT + 
                 gap_factor * RECONSTRUCTION_GAP_WEIGHT + 
                 overlap_penalty * RECONSTRUCTION_OVERLAP_WEIGHT + 
                 arc_count_factor * RECONSTRUCTION_ARC_COUNT_WEIGHT)
    
    # Apply configurable minimum confidence threshold
    if confidence < RECONSTRUCTION_MIN_CONFIDENCE:
        logger.debug(f"Arc group confidence too low: {confidence:.2f} < {RECONSTRUCTION_MIN_CONFIDENCE}")
        return None
    
    # Generate unique ID based on geometry
    source_entity_ids = [arc.source_entity.entity_id for arc in arc_group.arcs]
    circle_id = f"reconstructed_circle_{hash(tuple(sorted(source_entity_ids))) % 10000}"
    
    # Validation evidence with configurable parameters
    validation_evidence = {
        "arc_count": len(arc_group.arcs),
        "coverage_fraction": arc_group.total_coverage,
        "max_gap": max_gap,
        "total_overlap": total_overlap,
        "center_consistency": True,  # Already validated in _analyze_arc_group
        "radius_consistency": True,  # Already validated in _analyze_arc_group
        "strict_validation": True,
        "confidence_weights": {
            "coverage": RECONSTRUCTION_COVERAGE_WEIGHT,
            "gap": RECONSTRUCTION_GAP_WEIGHT,
            "overlap": RECONSTRUCTION_OVERLAP_WEIGHT,
            "arc_count": RECONSTRUCTION_ARC_COUNT_WEIGHT
        },
        "thresholds_used": {
            "min_coverage": min_required_coverage,
            "max_gap": CIRCLE_RECONSTRUCTION_MAX_GAP,
            "max_overlap": CIRCLE_RECONSTRUCTION_MAX_OVERLAP,
            "min_confidence": RECONSTRUCTION_MIN_CONFIDENCE
        }
    }
    
    logger.debug(f"Successfully reconstructed circle {circle_id}: "
                f"coverage={arc_group.total_coverage:.2f}, confidence={confidence:.2f}")
    
    return ReconstructedCircle(
        circle_id=circle_id,
        center=arc_group.center,
        radius=arc_group.radius,
        source_entities=source_entity_ids,
        coverage_fraction=arc_group.total_coverage,
        confidence=confidence,
        gaps=arc_group.gaps,
        overlaps=arc_group.overlaps,
        validation_evidence=validation_evidence
    )


def _get_representative_point(entity: NormalizedEntity) -> Optional[Point2D]:
    """Get a representative point for an entity."""
    
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
    
    return None
def _calculate_arc_gap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate the minimum angular gap between two arcs (degrees)."""
    
    # Normalize angles to [0, 360)
    def normalize_angle(angle):
        return angle % 360.0
    
    start1, end1 = normalize_angle(start1), normalize_angle(end1)
    start2, end2 = normalize_angle(start2), normalize_angle(end2)
    
    # Create intervals
    interval1 = ArcInterval(start1, end1, "arc1")
    interval2 = ArcInterval(start2, end2, "arc2")
    
    # Check for overlap first
    if interval1.overlaps_with(interval2, tolerance=ARC_GAP_OVERLAP_TOLERANCE):
        return 0.0  # No gap if overlapping
    
    # Calculate minimum gap
    gaps = []
    
    # Gap from end of arc1 to start of arc2
    gap1 = (start2 - end1) % 360.0
    gaps.append(gap1)
    
    # Gap from end of arc2 to start of arc1  
    gap2 = (start1 - end2) % 360.0
    gaps.append(gap2)
    
    return min(gaps)


def _find_entity_groups_strict(relationships: List[GeometricRelationship], 
                              all_entity_ids: List[str]) -> List[List[str]]:
    """Find groups of entities using only strong relationships."""
    
    # Build adjacency graph using only strong relationships
    adjacency = {entity_id: set() for entity_id in all_entity_ids}
    
    for rel in relationships:
        # Only use high-confidence structural relationships
        if (rel.relationship_type in ["strict_concentric", "arc_compatible"] and 
            rel.confidence >= HIGH_CONFIDENCE_RELATIONSHIP_THRESHOLD):
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


def _count_relationship_types(relationships: List[GeometricRelationship]) -> Dict[str, int]:
    """Count relationships by type for metadata."""
    counts = {}
    for rel in relationships:
        counts[rel.relationship_type] = counts.get(rel.relationship_type, 0) + 1
    return counts