"""
Square/Rectangular Hole Feature Detector

Detects square and rectangular hole features from DXF LINE entities
using strict geometric evidence and closure analysis.
"""

import logging
import math
from typing import List, Dict, Any, Tuple, Optional, Set

from .feature_types import ExpectedFeature, FeatureType
from ..dxf.entity_models import (
    NormalizedEntity, EntityType, GeometricRelationships, Point2D
)
from ..config import (
    GEOMETRIC_TOLERANCE, ANGULAR_TOLERANCE,
    MIN_FEATURE_CONFIDENCE, SIGNIFICANCE_MIN_RADIUS,
    # Square hole detection configuration
    SQUARE_HOLE_SIDE_TOLERANCE_BASE, SQUARE_HOLE_SIDE_TOLERANCE_PERCENTAGE,
    SQUARE_HOLE_MIN_DIMENSION, SQUARE_HOLE_MAX_MIN_DIMENSION, SQUARE_HOLE_MAX_MAX_DIMENSION,
    SQUARE_HOLE_GOOD_SIZE_THRESHOLD, SQUARE_HOLE_BASE_CONFIDENCE_WEIGHT, SQUARE_HOLE_EVIDENCE_WEIGHT,
    # Rectangle detection criteria
    SQUARE_HOLE_ANGLE_TOLERANCE_MULTIPLIER, SQUARE_HOLE_MIN_RIGHT_ANGLES, SQUARE_HOLE_MIN_MATCHING_PAIRS,
    # Additional square hole parameters
    SQUARE_HOLE_MIN_CONFIDENCE_THRESHOLD, SQUARE_HOLE_BASE_CONFIDENCE_RECTANGULAR,
    SQUARE_HOLE_BASE_CONFIDENCE_NON_RECTANGULAR, SQUARE_HOLE_SQUARE_BONUS,
    SQUARE_HOLE_EVIDENCE_GEOMETRIC_CLOSURE, SQUARE_HOLE_EVIDENCE_SIZE_APPROPRIATENESS, 
    SQUARE_HOLE_EVIDENCE_CONTEXT, SQUARE_HOLE_EVIDENCE_LAYER,
    SQUARE_HOLE_EVIDENCE_CONTEXT_MODERATE, SQUARE_HOLE_EVIDENCE_SIZE_GOOD, 
    SQUARE_HOLE_EVIDENCE_SIZE_SMALLER, SQUARE_HOLE_EVIDENCE_SIZE_POOR, 
    SQUARE_HOLE_EVIDENCE_LAYER_POOR, SQUARE_HOLE_EVIDENCE_LAYER_NEUTRAL,
    SQUARE_HOLE_EVIDENCE_LAYER_GOOD,
    # Square hole specific constants
    SQUARE_HOLE_MIN_LINE_ENTITIES, SQUARE_HOLE_MIN_SIDES_RECTANGLE, SQUARE_HOLE_MIN_SIDES_POLYGON,
    SQUARE_HOLE_MAX_CHAIN_LENGTH, SQUARE_HOLE_MIN_VERTICES
)

logger = logging.getLogger(__name__)


class SquareHoleDetector:
    """Detects square and rectangular holes from LINE geometry with geometric evidence."""
    
    def __init__(self, 
                 min_confidence: float = MIN_FEATURE_CONFIDENCE,
                 min_size: float = SIGNIFICANCE_MIN_RADIUS):
        """
        Initialize square hole detector.
        
        Args:
            min_confidence: Minimum confidence required for hole classification
            min_size: Minimum size (side length) for significant holes (mm)
        """
        self.min_confidence = min_confidence
        self.min_size = min_size
        self._feature_counter = 0
    
    def detect_square_holes(self, entities: List[NormalizedEntity],
                          relationships: GeometricRelationships) -> List[ExpectedFeature]:
        """
        Detect square and rectangular hole features from LINE entities.
        
        Uses geometric evidence to identify closed rectangular geometry that
        represents holes rather than general structural features.
        """
        logger.debug(f"Detecting square holes from {len(entities)} entities")
        
        square_holes = []
        rejection_log = {
            "not_closed": 0,
            "insufficient_sides": 0, 
            "non_rectangular": 0,
            "too_small": 0,
            "low_confidence": 0
        }
        
        # Extract LINE entities for analysis
        line_entities = [e for e in entities if e.source_entity.entity_type == EntityType.LINE]
        
        if len(line_entities) < SQUARE_HOLE_MIN_LINE_ENTITIES:
            logger.debug(f"Insufficient LINE entities for square detection: {len(line_entities)} < {SQUARE_HOLE_MIN_LINE_ENTITIES}")
            return []
        
        # Find closed polygonal chains
        closed_polygons = self._find_closed_line_polygons(line_entities, rejection_log)
        
        # Analyze each polygon for rectangular hole evidence
        for polygon in closed_polygons:
            hole_candidate = self._analyze_polygon_for_hole(polygon, entities, relationships)
            
            if hole_candidate is None:
                rejection_log["non_rectangular"] += 1
                continue
            
            # Apply size constraints
            if not self._meets_size_constraints(hole_candidate):
                rejection_log["too_small"] += 1
                continue
            
            # Apply confidence threshold
            if hole_candidate.confidence < self.min_confidence:
                rejection_log["low_confidence"] += 1
                continue
            
            square_holes.append(hole_candidate)
        
        logger.info(f"Square hole detection: {len(square_holes)} holes detected, "
                   f"{sum(rejection_log.values())} rejected "
                   f"(not_closed: {rejection_log['not_closed']}, "
                   f"insufficient_sides: {rejection_log['insufficient_sides']}, "
                   f"non_rectangular: {rejection_log['non_rectangular']}, "
                   f"too_small: {rejection_log['too_small']}, "
                   f"low_confidence: {rejection_log['low_confidence']})")
        
        return square_holes
    
    def _find_closed_line_polygons(self, line_entities: List[NormalizedEntity], 
                                  rejection_log: Dict[str, int]) -> List[List[NormalizedEntity]]:
        """Find closed polygonal chains from LINE entities."""
        
        closed_polygons = []
        used_lines = set()
        
        for start_line in line_entities:
            if start_line.source_entity.entity_id in used_lines:
                continue
            
            # Attempt to build a closed chain starting from this line
            chain = self._build_line_chain(start_line, line_entities, used_lines)
            
            if chain is None:
                continue
            
            # Check if chain forms a closed polygon
            if self._is_closed_polygon(chain):
                # Check minimum sides for rectangle (4 sides)
                if len(chain) >= SQUARE_HOLE_MIN_SIDES_RECTANGLE:
                    closed_polygons.append(chain)
                    # Mark all lines in this chain as used
                    for line in chain:
                        used_lines.add(line.source_entity.entity_id)
                else:
                    rejection_log["insufficient_sides"] += len(chain)
            else:
                rejection_log["not_closed"] += len(chain)
        
        logger.debug(f"Found {len(closed_polygons)} closed polygons from {len(line_entities)} lines")
        return closed_polygons
    
    def _build_line_chain(self, start_line: NormalizedEntity, 
                         all_lines: List[NormalizedEntity],
                         used_lines: Set[str]) -> Optional[List[NormalizedEntity]]:
        """Build a connected chain of lines starting from start_line."""
        
        chain = [start_line]
        current_end = start_line.source_entity.end_point
        
        max_chain_length = SQUARE_HOLE_MAX_CHAIN_LENGTH  # Prevent infinite loops
        
        while len(chain) < max_chain_length:
            # Find the next connected line
            next_line = self._find_connecting_line(current_end, all_lines, chain, used_lines)
            
            if next_line is None:
                break
            
            chain.append(next_line)
            
            # Update current end point
            # The connecting line might need to be traversed forward or backward
            if next_line.source_entity.start_point.distance_to(current_end) <= GEOMETRIC_TOLERANCE:
                current_end = next_line.source_entity.end_point
            else:
                current_end = next_line.source_entity.start_point
            
            # Check if we've closed the loop
            start_point = start_line.source_entity.start_point
            if current_end.distance_to(start_point) <= GEOMETRIC_TOLERANCE:
                break
        
        return chain if len(chain) >= SQUARE_HOLE_MIN_SIDES_POLYGON else None
    
    def _find_connecting_line(self, point: Point2D, 
                            all_lines: List[NormalizedEntity],
                            current_chain: List[NormalizedEntity],
                            used_lines: Set[str]) -> Optional[NormalizedEntity]:
        """Find a line that connects to the given point."""
        
        current_chain_ids = {line.source_entity.entity_id for line in current_chain}
        
        for line in all_lines:
            # Skip if already in current chain or used in another polygon
            if (line.source_entity.entity_id in current_chain_ids or 
                line.source_entity.entity_id in used_lines):
                continue
            
            # Check if either endpoint connects to the target point
            start_dist = line.source_entity.start_point.distance_to(point)
            end_dist = line.source_entity.end_point.distance_to(point)
            
            if start_dist <= GEOMETRIC_TOLERANCE or end_dist <= GEOMETRIC_TOLERANCE:
                return line
        
        return None
    
    def _is_closed_polygon(self, chain: List[NormalizedEntity]) -> bool:
        """Check if the line chain forms a closed polygon."""
        
        if len(chain) < 3:
            return False
        
        # Get start point of first line and end point of last line
        first_line = chain[0]
        last_line = chain[-1]
        
        # Try both orientations for the last line
        start_point = first_line.source_entity.start_point
        
        # Check if last line's end connects to first line's start
        end1 = last_line.source_entity.end_point
        end2 = last_line.source_entity.start_point
        
        dist1 = end1.distance_to(start_point)
        dist2 = end2.distance_to(start_point)
        
        return min(dist1, dist2) <= GEOMETRIC_TOLERANCE
    
    def _analyze_polygon_for_hole(self, polygon: List[NormalizedEntity],
                                all_entities: List[NormalizedEntity],
                                relationships: GeometricRelationships) -> Optional[ExpectedFeature]:
        """Analyze a closed polygon to determine if it represents a hole."""
        
        # Get polygon vertices in order
        vertices = self._extract_ordered_vertices(polygon)
        
        if len(vertices) < SQUARE_HOLE_MIN_VERTICES:
            return None
        
        # Analyze geometric properties
        geometric_analysis = self._analyze_polygon_geometry(vertices)
        
        if not geometric_analysis["is_rectangular"]:
            return None
        
        # Analyze hole evidence
        hole_evidence = self._analyze_hole_evidence(
            geometric_analysis, polygon, all_entities, relationships
        )
        
        # Calculate overall confidence
        confidence = self._calculate_hole_confidence(geometric_analysis, hole_evidence)
        
        if confidence < SQUARE_HOLE_MIN_CONFIDENCE_THRESHOLD:  # Minimum threshold for consideration
            return None
        
        # Determine feature type
        if geometric_analysis["is_square"]:
            feature_type = FeatureType.THROUGH_HOLE  # Use THROUGH_HOLE for square holes
        else:
            feature_type = FeatureType.THROUGH_HOLE  # Use THROUGH_HOLE for rectangular holes
        
        # Create feature
        feature_id = f"square_hole_{self._get_next_id()}"
        
        source_entity_ids = [line.source_entity.entity_id for line in polygon]
        
        return ExpectedFeature(
            feature_id=feature_id,
            feature_type=feature_type,
            confidence=confidence,
            center=geometric_analysis["center"],
            radius=None,  # Square holes don't have radius
            source_entity_ids=source_entity_ids,
            source_type="square_hole_from_lines",
            detection_evidence={
                "hole_evidence": hole_evidence,
                "geometric_analysis": geometric_analysis,
                "validation_method": "closed_polygon_analysis",
                "polygon_sides": len(vertices),
                "is_square": geometric_analysis["is_square"],
                "width": geometric_analysis["width"],
                "height": geometric_analysis["height"]
            },
            geometric_properties={
                "shape": "square" if geometric_analysis["is_square"] else "rectangle",
                "width": geometric_analysis["width"],
                "height": geometric_analysis["height"],
                "area": geometric_analysis["area"],
                "perimeter": geometric_analysis["perimeter"]
            }
        )
    
    def _extract_ordered_vertices(self, polygon: List[NormalizedEntity]) -> List[Point2D]:
        """Extract vertices from polygon in correct order."""
        
        if not polygon:
            return []
        
        vertices = []
        
        # Start with first line's start point
        current_point = polygon[0].source_entity.start_point
        vertices.append(current_point)
        
        for i, line in enumerate(polygon):
            start_point = line.source_entity.start_point
            end_point = line.source_entity.end_point
            
            # Determine which end connects to current point
            start_dist = start_point.distance_to(current_point)
            end_dist = end_point.distance_to(current_point)
            
            if start_dist <= GEOMETRIC_TOLERANCE:
                # Line goes from start to end
                vertices.append(end_point)
                current_point = end_point
            else:
                # Line goes from end to start 
                vertices.append(start_point)
                current_point = start_point
        
        # Remove the last vertex if it's the same as the first (closed polygon)
        if len(vertices) > 1:
            if vertices[-1].distance_to(vertices[0]) <= GEOMETRIC_TOLERANCE:
                vertices.pop()
        
        return vertices
    
    def _analyze_polygon_geometry(self, vertices: List[Point2D]) -> Dict[str, Any]:
        """Analyze geometric properties of the polygon."""
        
        analysis = {
            "is_rectangular": False,
            "is_square": False,
            "center": None,
            "width": 0,
            "height": 0,
            "area": 0,
            "perimeter": 0,
            "side_lengths": [],
            "angles": []
        }
        
        if len(vertices) != 4:
            return analysis
        
        # Calculate side lengths
        side_lengths = []
        for i in range(len(vertices)):
            next_i = (i + 1) % len(vertices)
            length = vertices[i].distance_to(vertices[next_i])
            side_lengths.append(length)
        
        analysis["side_lengths"] = side_lengths
        analysis["perimeter"] = sum(side_lengths)
        
        # Calculate angles
        angles = []
        for i in range(len(vertices)):
            prev_i = (i - 1) % len(vertices)
            next_i = (i + 1) % len(vertices)
            
            # Vectors from current vertex
            v1 = Point2D(vertices[prev_i].x - vertices[i].x, vertices[prev_i].y - vertices[i].y)
            v2 = Point2D(vertices[next_i].x - vertices[i].x, vertices[next_i].y - vertices[i].y)
            
            # Calculate angle
            angle = self._calculate_angle_between_vectors(v1, v2)
            angles.append(angle)
        
        analysis["angles"] = angles
        
        # Check rectangularity
        # 1. Four sides
        # 2. Opposite sides approximately equal
        # 3. All angles approximately 90 degrees
        
        if len(side_lengths) == 4:
            # Check angles (should be close to 90 degrees)
            angle_tolerance = ANGULAR_TOLERANCE * SQUARE_HOLE_ANGLE_TOLERANCE_MULTIPLIER  # Configurable multiplier for rectangles
            right_angles = sum(1 for angle in angles if abs(angle - 90) <= angle_tolerance)
            
            if right_angles >= SQUARE_HOLE_MIN_RIGHT_ANGLES:  # Configurable minimum right angles (4th will be close due to sum constraint)
                # Check opposite sides
                opposite_pairs = [(0, 2), (1, 3)]  # Opposite side indices
                
                side_tolerance = max(SQUARE_HOLE_SIDE_TOLERANCE_BASE, 
                                   min(side_lengths) * SQUARE_HOLE_SIDE_TOLERANCE_PERCENTAGE)  # 10% or 0.5mm, whichever is larger
                
                pairs_match = 0
                for i1, i2 in opposite_pairs:
                    if abs(side_lengths[i1] - side_lengths[i2]) <= side_tolerance:
                        pairs_match += 1
                
                if pairs_match >= SQUARE_HOLE_MIN_MATCHING_PAIRS:  # Configurable minimum matching pairs
                    analysis["is_rectangular"] = True
                    
                    # Calculate dimensions
                    analysis["width"] = max(side_lengths[0], side_lengths[1])
                    analysis["height"] = min(side_lengths[0], side_lengths[1])
                    analysis["area"] = analysis["width"] * analysis["height"]
                    
                    # Check if square (all sides approximately equal)
                    max_side = max(side_lengths)
                    min_side = min(side_lengths)
                    if (max_side - min_side) <= side_tolerance:
                        analysis["is_square"] = True
                        analysis["width"] = analysis["height"] = (max_side + min_side) / 2
        
        # Calculate center (centroid)
        if vertices:
            center_x = sum(v.x for v in vertices) / len(vertices)
            center_y = sum(v.y for v in vertices) / len(vertices)
            analysis["center"] = Point2D(center_x, center_y)
        
        return analysis
    
    def _calculate_angle_between_vectors(self, v1: Point2D, v2: Point2D) -> float:
        """Calculate angle between two vectors in degrees."""
        
        # Calculate dot product
        dot_product = v1.x * v2.x + v1.y * v2.y
        
        # Calculate magnitudes
        mag1 = math.sqrt(v1.x ** 2 + v1.y ** 2)
        mag2 = math.sqrt(v2.x ** 2 + v2.y ** 2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        # Calculate angle in radians
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to avoid numerical errors
        
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def _analyze_hole_evidence(self, geometric_analysis: Dict[str, Any],
                             polygon: List[NormalizedEntity],
                             all_entities: List[NormalizedEntity],
                             relationships: GeometricRelationships) -> Dict[str, Any]:
        """Analyze evidence that this polygon represents a hole."""
        
        evidence = {
            "geometric_closure": 0.0,
            "size_appropriateness": 0.0,
            "context_evidence": 0.0,
            "layer_evidence": 0.0
        }
        
        # Geometric closure evidence (always high for closed polygons)
        evidence["geometric_closure"] = 1.0
        
        # Size appropriateness (reasonable hole size)
        width = geometric_analysis.get("width", 0)
        height = geometric_analysis.get("height", 0)
        
        if width > 0 and height > 0:
            # Reasonable hole sizes: 3mm to 50mm
            min_dim = min(width, height)
            max_dim = max(width, height)
            
            if SQUARE_HOLE_MIN_DIMENSION <= min_dim <= SQUARE_HOLE_MAX_MIN_DIMENSION and max_dim <= SQUARE_HOLE_MAX_MAX_DIMENSION:
                # Good size for holes
                if min_dim >= SQUARE_HOLE_GOOD_SIZE_THRESHOLD:
                    evidence["size_appropriateness"] = SQUARE_HOLE_EVIDENCE_SIZE_GOOD
                else:
                    evidence["size_appropriateness"] = SQUARE_HOLE_EVIDENCE_SIZE_SMALLER  # Smaller holes
            else:
                evidence["size_appropriateness"] = SQUARE_HOLE_EVIDENCE_SIZE_POOR  # Too small or too large
        
        # Context evidence (simplified)
        evidence["context_evidence"] = SQUARE_HOLE_EVIDENCE_CONTEXT_MODERATE  # Moderate evidence for rectangular geometry
        
        # Layer evidence (check if polygon lines are on feature-related layers)
        if polygon:
            layer_names = [line.source_entity.layer.lower() for line in polygon]
            unique_layers = set(layer_names)
            
            hole_keywords = ["hole", "cut", "feature", "mach"]
            construction_keywords = ["construction", "text", "dim"]
            
            has_hole_keywords = any(any(kw in layer for kw in hole_keywords) for layer in unique_layers)
            has_construction_keywords = any(any(kw in layer for kw in construction_keywords) for layer in unique_layers)
            
            if has_hole_keywords:
                evidence["layer_evidence"] = SQUARE_HOLE_EVIDENCE_LAYER_GOOD
            elif has_construction_keywords:
                evidence["layer_evidence"] = SQUARE_HOLE_EVIDENCE_LAYER_POOR
            else:
                evidence["layer_evidence"] = SQUARE_HOLE_EVIDENCE_LAYER_NEUTRAL  # Neutral
        
        return evidence
    
    def _calculate_hole_confidence(self, geometric_analysis: Dict[str, Any],
                                 hole_evidence: Dict[str, Any]) -> float:
        """Calculate overall confidence that this polygon represents a hole."""
        
        # Base confidence from rectangularity
        base_confidence = (SQUARE_HOLE_BASE_CONFIDENCE_RECTANGULAR if geometric_analysis["is_rectangular"] 
                          else SQUARE_HOLE_BASE_CONFIDENCE_NON_RECTANGULAR)
        
        # Square bonus
        if geometric_analysis.get("is_square", False):
            base_confidence += SQUARE_HOLE_SQUARE_BONUS
        
        # Evidence weights
        weights = {
            "geometric_closure": SQUARE_HOLE_EVIDENCE_GEOMETRIC_CLOSURE,
            "size_appropriateness": SQUARE_HOLE_EVIDENCE_SIZE_APPROPRIATENESS,
            "context_evidence": SQUARE_HOLE_EVIDENCE_CONTEXT,
            "layer_evidence": SQUARE_HOLE_EVIDENCE_LAYER
        }
        
        evidence_score = sum(
            weights[key] * hole_evidence[key] 
            for key in weights.keys()
        )
        
        # Combine base confidence with evidence
        final_confidence = base_confidence * SQUARE_HOLE_BASE_CONFIDENCE_WEIGHT + evidence_score * SQUARE_HOLE_EVIDENCE_WEIGHT
        
        return min(1.0, final_confidence)
    
    def _meets_size_constraints(self, feature: ExpectedFeature) -> bool:
        """Check if feature meets size constraints for significance."""
        
        width = feature.geometric_properties.get("width", 0)
        height = feature.geometric_properties.get("height", 0)
        
        min_dimension = min(width, height) if width and height else 0
        
        return min_dimension >= self.min_size
    
    def _get_next_id(self) -> int:
        """Get next sequential feature ID."""
        self._feature_counter += 1
        return self._feature_counter