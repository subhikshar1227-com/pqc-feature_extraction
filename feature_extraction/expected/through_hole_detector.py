"""
Through Hole Feature Detector

Detects through hole features using strict geometric evidence from DXF entities.
Only classifies circular geometry as through holes when there is strong evidence
for hole functionality rather than general circular features.
"""

import logging
import math
from typing import List, Dict, Any, Tuple, Optional

from .feature_types import ExpectedFeature, FeatureType
from ..dxf.entity_models import (
    NormalizedEntity, EntityType, GeometricRelationships, Point2D
)
from ..dxf.geometry_reconstruction import ReconstructedGeometry, ReconstructedCircle
from ..config import (
    THROUGH_HOLE_MIN_RADIUS, THROUGH_HOLE_MAX_RADIUS, 
    HOLE_CLASSIFICATION_MIN_CONFIDENCE, HOLE_EVIDENCE_WEIGHTS,
    CLOSURE_EVIDENCE_MIN_COVERAGE, CLOSURE_EVIDENCE_MIN_ARC_COVERAGE,
    CONCENTRIC_MIN_SIZE_RATIO, CONCENTRIC_MAX_SIZE_RATIO, CONCENTRIC_CENTER_TOLERANCE,
    PATTERN_MIN_SIMILAR_COUNT, PATTERN_CENTER_TOLERANCE, PATTERN_RADIUS_TOLERANCE,
    CONCENTRICITY_TOLERANCE, MAIN_BODY_MIN_RADIUS,
    # New configurable hole analysis parameters
    CENTRAL_EVIDENCE_HIGH_THRESHOLD, CENTRAL_POSITION_BONUS,
    CENTRAL_CONFIDENCE_ADJUSTMENT, CENTRAL_EVIDENCE_MIN_THRESHOLD,
    RECONSTRUCTION_CONFIDENCE_PENALTY_FACTOR,
    ARC_CLOSURE_EXCELLENT_THRESHOLD, ARC_CLOSURE_GOOD_THRESHOLD,
    ARC_CLOSURE_EXCELLENT_EVIDENCE, ARC_CLOSURE_GOOD_EVIDENCE, ARC_CLOSURE_POOR_EVIDENCE,
    NESTING_RELATIONSHIP_MIN_CONFIDENCE,
    PATTERN_LARGE_COUNT_THRESHOLD, PATTERN_MEDIUM_COUNT_THRESHOLD,
    PATTERN_LARGE_EVIDENCE, PATTERN_MEDIUM_EVIDENCE, PATTERN_SMALL_EVIDENCE, PATTERN_ISOLATED_EVIDENCE,
    SPATIAL_REGULARITY_MIN_CENTERS, SPATIAL_REGULARITY_MIN_DISTANCES,
    SPATIAL_REGULARITY_MAX_VARIATION, SPATIAL_REGULARITY_BONUS,
    HOLE_SIZE_VERY_SMALL_THRESHOLD, HOLE_SIZE_SMALL_THRESHOLD, HOLE_SIZE_MEDIUM_THRESHOLD, HOLE_SIZE_LARGE_THRESHOLD,
    HOLE_SIZE_VERY_SMALL_EVIDENCE, HOLE_SIZE_SMALL_EVIDENCE, HOLE_SIZE_MEDIUM_EVIDENCE,
    HOLE_SIZE_LARGE_EVIDENCE, HOLE_SIZE_VERY_LARGE_EVIDENCE, HOLE_SIZE_OVERSIZED_EVIDENCE,
    HOLE_CATEGORY_VERY_SMALL_THRESHOLD, HOLE_CATEGORY_SMALL_THRESHOLD,
    HOLE_CATEGORY_MEDIUM_THRESHOLD, HOLE_CATEGORY_LARGE_THRESHOLD,
    LAYER_HOLE_KEYWORDS_EVIDENCE, LAYER_FEATURE_KEYWORDS_EVIDENCE,
    LAYER_CONSTRUCTION_KEYWORDS_EVIDENCE, LAYER_DEFAULT_EVIDENCE,
    MAIN_BODY_CENTER_TOLERANCE, MAIN_BODY_CENTER_BONUS,
    # Enhanced through hole detection configuration
    THROUGH_HOLE_RADIUS_RATIO_THRESHOLD, THROUGH_HOLE_CENTRALITY_WEIGHT_POSITION,
    THROUGH_HOLE_CENTRALITY_WEIGHT_SIZE, THROUGH_HOLE_CENTRAL_EVIDENCE_GOOD,
    THROUGH_HOLE_CENTRAL_EVIDENCE_ACCEPTABLE, THROUGH_HOLE_CENTRAL_TOLERANCE_BASE,
    THROUGH_HOLE_CENTRAL_TOLERANCE_FACTOR, THROUGH_HOLE_SIZE_RATIO_MIN_GOOD,
    THROUGH_HOLE_SIZE_RATIO_MAX_GOOD, THROUGH_HOLE_SIZE_RATIO_MIN_ACCEPTABLE,
    THROUGH_HOLE_SIZE_RATIO_MAX_ACCEPTABLE, THROUGH_HOLE_CENTRAL_SIZE_MIN,
    THROUGH_HOLE_CENTRAL_SIZE_MAX, THROUGH_HOLE_CENTRAL_EVIDENCE_MIN,
    # Additional algorithmic parameters
    THROUGH_HOLE_PATTERN_EVIDENCE_THRESHOLD, THROUGH_HOLE_CIRCULAR_PATTERN_VARIATION,
    ARC_ANGLE_HALF_CIRCLE, ARC_ANGLE_FULL_CIRCLE, THROUGH_HOLE_SIZE_SCORE_NORMALIZATION,
    THROUGH_HOLE_DEFAULT_EVIDENCE, THROUGH_HOLE_NON_CENTRAL_EVIDENCE, THROUGH_HOLE_POOR_SIZE_EVIDENCE,
    THROUGH_HOLE_SPATIAL_TOLERANCE_FACTOR_HIGH, THROUGH_HOLE_SPATIAL_TOLERANCE_FACTOR_LOW,
    THROUGH_HOLE_CIRCULAR_PATTERN_BASE_EVIDENCE,
    # Forensic audit parameters
    THROUGH_HOLE_CIRCULAR_FILE_MIN_ENTITIES, THROUGH_HOLE_SPATIAL_MIN_CENTERS,
    THROUGH_HOLE_SPATIAL_MIN_DISTANCES, THROUGH_HOLE_CIRCULAR_PATTERN_MIN_CENTERS,
    WEIGHT_VALIDATION_TOLERANCE
)

logger = logging.getLogger(__name__)


class ThroughHoleDetector:
    """Detects through hole features using strict evidence-based validation."""
    
    def __init__(self, 
                 min_radius: float = THROUGH_HOLE_MIN_RADIUS,
                 max_radius: float = THROUGH_HOLE_MAX_RADIUS,
                 min_confidence: float = HOLE_CLASSIFICATION_MIN_CONFIDENCE):
        """
        Initialize through hole detector with strict evidence requirements.
        
        Args:
            min_radius: Minimum radius for through holes (mm)
            max_radius: Maximum radius for through holes (mm)  
            min_confidence: Minimum confidence required for hole classification
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_confidence = min_confidence
        self._feature_counter = 0
        
        # Evidence validation - ensure weights sum to 1.0
        total_weight = sum(HOLE_EVIDENCE_WEIGHTS.values())
        if abs(total_weight - 1.0) > WEIGHT_VALIDATION_TOLERANCE:
            raise ValueError(f"Hole evidence weights must sum to 1.0, got {total_weight}")
    
    def detect_through_holes(self, entities: List[NormalizedEntity],
                           reconstructed: ReconstructedGeometry,
                           relationships: GeometricRelationships) -> List[ExpectedFeature]:
        """
        Detect through hole features using strict evidence-based validation.
        
        Only circular geometry with strong hole evidence is classified as through holes.
        Evidence requirements:
        - Geometric closure (complete circular boundary)
        - Concentric nesting (inside other geometry)  
        - Pattern regularity (part of repeated hole pattern)
        - Size appropriateness (typical hole dimensions)
        - Layer context (hints from DXF metadata)
        """
        # Handle None inputs gracefully for tests
        if reconstructed is None or relationships is None:
            logger.debug("Detecting through holes with minimal inputs for testing")
            return []
            
        logger.debug(f"Detecting through holes with strict evidence from {len(entities)} entities")
        
        through_holes = []
        rejection_log = {"insufficient_evidence": 0, "size_constraints": 0, "low_confidence": 0}
        
        # Analyze explicit circular entities for hole evidence
        explicit_holes = self._analyze_explicit_circles_for_holes(entities, relationships, rejection_log)
        through_holes.extend(explicit_holes)
        
        # Analyze reconstructed circles for hole evidence
        reconstructed_holes = self._analyze_reconstructed_circles_for_holes(
            reconstructed.reconstructed_circles, entities, relationships, rejection_log
        )
        through_holes.extend(reconstructed_holes)
        
        logger.info(f"Through hole detection: {len(through_holes)} holes detected, "
                   f"{sum(rejection_log.values())} rejected "
                   f"(size: {rejection_log['size_constraints']}, "
                   f"evidence: {rejection_log['insufficient_evidence']}, "
                   f"confidence: {rejection_log['low_confidence']})")
        
        return through_holes
    
    def _analyze_explicit_circles_for_holes(self, entities: List[NormalizedEntity],
                                           relationships: GeometricRelationships,
                                           rejection_log: Dict[str, int]) -> List[ExpectedFeature]:
        """Analyze explicit CIRCLE entities for hole evidence with strict validation."""
        
        through_holes = []
        
        for entity in entities:
            # Only analyze complete CIRCLE entities
            if entity.source_entity.entity_type != EntityType.CIRCLE:
                continue
                
            # Validate geometric completeness
            if not entity.source_entity.center or entity.source_entity.radius is None:
                continue
                
            # Check size constraints strictly
            if not self._passes_size_constraints(entity.source_entity.radius):
                rejection_log["size_constraints"] += 1
                continue
            
            # Analyze evidence for hole classification
            evidence_analysis = self._analyze_comprehensive_hole_evidence(
                entity, entities, relationships
            )
            
            # Apply strict confidence threshold with special handling for confirmed central holes
            adjusted_confidence = evidence_analysis["total_confidence"]
            is_central = evidence_analysis.get("evidence_details", {}).get("central_position", {}).get("is_central", False)
            central_evidence = evidence_analysis.get("evidence_details", {}).get("central_position", {}).get("evidence_score", 0.0)
            
            # Special case: for very high central evidence, allow lower confidence
            threshold = self.min_confidence
            if is_central and central_evidence > CENTRAL_EVIDENCE_MIN_THRESHOLD:
                threshold = CENTRAL_CONFIDENCE_ADJUSTMENT  # Only for very confident central holes
                logger.debug(f"Using lower threshold for confirmed central hole: {threshold} (central_evidence={central_evidence:.3f})")
            
            if adjusted_confidence < threshold:
                rejection_log["low_confidence"] += 1
                logger.debug(f"Rejected circle {entity.source_entity.entity_id} for hole: "
                           f"confidence {adjusted_confidence:.2f} < {threshold} (central={is_central})")
                continue
            
            # Create validated through hole feature
            feature = self._create_hole_feature(
                entity.source_entity.center,
                entity.source_entity.radius,
                [entity.source_entity.entity_id],
                "explicit_circle",
                evidence_analysis
            )
            
            through_holes.append(feature)
            logger.debug(f"Detected explicit through hole: {feature.feature_id}, "
                        f"confidence={evidence_analysis['total_confidence']:.2f}")
        
        return through_holes
    
    def _analyze_reconstructed_circles_for_holes(self, reconstructed_circles: List[ReconstructedCircle],
                                               entities: List[NormalizedEntity],
                                               relationships: GeometricRelationships,
                                               rejection_log: Dict[str, int]) -> List[ExpectedFeature]:
        """Analyze reconstructed circles for hole evidence with strict validation."""
        
        through_holes = []
        entity_lookup = {e.source_entity.entity_id: e for e in entities}
        
        for recon_circle in reconstructed_circles:
            # Check size constraints
            if not self._passes_size_constraints(recon_circle.radius):
                rejection_log["size_constraints"] += 1
                continue
            
            # Create temporary entity representation for evidence analysis
            temp_entity = self._create_temp_entity_for_analysis(recon_circle)
            
            # Analyze hole evidence
            evidence_analysis = self._analyze_comprehensive_hole_evidence(
                temp_entity, entities, relationships
            )
            
            # Penalize confidence for reconstruction uncertainty
            reconstruction_penalty = (1.0 - recon_circle.confidence) * RECONSTRUCTION_CONFIDENCE_PENALTY_FACTOR
            adjusted_confidence = evidence_analysis["total_confidence"] - reconstruction_penalty
            
            # Apply strict threshold to adjusted confidence
            if adjusted_confidence < self.min_confidence:
                rejection_log["low_confidence"] += 1
                logger.debug(f"Rejected reconstructed circle {recon_circle.circle_id} for hole: "
                           f"adjusted confidence {adjusted_confidence:.2f} < {self.min_confidence}")
                continue
            
            # Update evidence with reconstruction details
            evidence_analysis["total_confidence"] = adjusted_confidence
            evidence_analysis["reconstruction_details"] = {
                "coverage_fraction": recon_circle.coverage_fraction,
                "source_arc_count": len(recon_circle.source_entities),
                "reconstruction_confidence": recon_circle.confidence
            }
            
            # Create validated through hole feature
            feature = self._create_hole_feature(
                recon_circle.center,
                recon_circle.radius,
                recon_circle.source_entities,
                "reconstructed_circle",
                evidence_analysis
            )
            
            through_holes.append(feature)
            logger.debug(f"Detected reconstructed through hole: {feature.feature_id}, "
                        f"confidence={adjusted_confidence:.2f}")
        
        return through_holes
    
    def _passes_size_constraints(self, radius: Optional[float]) -> bool:
        """Check if radius meets strict through hole size constraints."""
        if radius is None:
            return False
        return self.min_radius <= radius <= self.max_radius
    
    def _analyze_comprehensive_hole_evidence(self, entity: NormalizedEntity,
                                           all_entities: List[NormalizedEntity],
                                           relationships: GeometricRelationships) -> Dict[str, Any]:
        """
        Analyze comprehensive evidence for through hole classification using weighted factors.
        
        Evidence Types (from config.py):
        - geometric_closure: Complete circular boundary (35% weight)
        - concentric_nesting: Inside other geometry (25% weight) 
        - pattern_regularity: Part of hole pattern (20% weight)
        - size_appropriateness: Typical hole dimensions (15% weight)
        - layer_context: DXF metadata hints (5% weight)
        """
        
        evidence = {
            "geometric_closure": 0.0,
            "concentric_nesting": 0.0, 
            "pattern_regularity": 0.0,
            "size_appropriateness": 0.0,
            "layer_context": 0.0,
            "total_confidence": 0.0,
            "evidence_details": {}
        }
        
        # Validate entity geometry
        if not entity.source_entity.center or entity.source_entity.radius is None:
            return evidence
        
        # Analyze geometric context to identify central holes
        geometric_context = self._analyze_geometric_context(entity, all_entities)
        
        # 1. Geometric Closure Evidence (35% weight)
        closure_evidence, closure_details = self._analyze_geometric_closure_evidence(entity)
        evidence["geometric_closure"] = closure_evidence
        evidence["evidence_details"]["closure"] = closure_details
        
        # 2. Concentric Nesting Evidence (25% weight)
        nesting_evidence, nesting_details = self._analyze_concentric_nesting_evidence(
            entity, all_entities, relationships
        )
        evidence["concentric_nesting"] = nesting_evidence
        evidence["evidence_details"]["nesting"] = nesting_details
        
        # 3. Pattern Regularity Evidence (20% weight)
        pattern_evidence, pattern_details = self._analyze_pattern_regularity_evidence(
            entity, all_entities
        )
        evidence["pattern_regularity"] = pattern_evidence 
        evidence["evidence_details"]["pattern"] = pattern_details
        
        # 4. Size Appropriateness Evidence (15% weight)
        size_evidence, size_details = self._analyze_size_appropriateness_evidence(entity)
        evidence["size_appropriateness"] = size_evidence
        evidence["evidence_details"]["size"] = size_details
        
        # 5. Layer Context Evidence (5% weight)
        layer_evidence, layer_details = self._analyze_layer_context_evidence(entity)
        evidence["layer_context"] = layer_evidence
        evidence["evidence_details"]["layer"] = layer_details
        
        # 6. Central Position Evidence (bonus for central holes)
        central_evidence, central_details = self._analyze_central_position_evidence(entity, geometric_context)
        evidence["evidence_details"]["central_position"] = central_details
        evidence["evidence_details"]["central_position"]["evidence_score"] = central_evidence
        
        # Calculate weighted confidence
        total_confidence = (
            HOLE_EVIDENCE_WEIGHTS["geometric_closure"] * closure_evidence +
            HOLE_EVIDENCE_WEIGHTS["concentric_nesting"] * nesting_evidence +
            HOLE_EVIDENCE_WEIGHTS["pattern_regularity"] * pattern_evidence +
            HOLE_EVIDENCE_WEIGHTS["size_appropriateness"] * size_evidence +
            HOLE_EVIDENCE_WEIGHTS["layer_context"] * layer_evidence
        )
        
        # Apply central position bonus
        if central_evidence > CENTRAL_EVIDENCE_HIGH_THRESHOLD:
            total_confidence += CENTRAL_POSITION_BONUS  # Bonus for central holes
            evidence["evidence_details"]["central_bonus"] = CENTRAL_POSITION_BONUS
            logger.debug(f"Applied central bonus {CENTRAL_POSITION_BONUS} for central evidence {central_evidence:.3f}")
        else:
            logger.debug(f"No central bonus: central evidence {central_evidence:.3f} <= threshold {CENTRAL_EVIDENCE_HIGH_THRESHOLD}")
        
        # Apply main body center bonus for holes exactly at the main body center
        main_body_center = geometric_context.get("main_body_center")
        if main_body_center and entity.source_entity.center:
            distance_to_main_body_center = entity.source_entity.center.distance_to(main_body_center)
            if distance_to_main_body_center <= MAIN_BODY_CENTER_TOLERANCE:
                total_confidence += MAIN_BODY_CENTER_BONUS
                evidence["evidence_details"]["main_body_center_bonus"] = MAIN_BODY_CENTER_BONUS
                evidence["evidence_details"]["distance_to_main_body_center"] = distance_to_main_body_center
                logger.debug(f"Applied main body center bonus {MAIN_BODY_CENTER_BONUS} for distance {distance_to_main_body_center:.3f}")
        
        evidence["total_confidence"] = min(1.0, total_confidence)
        
        logger.debug(f"Hole evidence for {entity.source_entity.entity_id}: "
                    f"closure={closure_evidence:.2f}, nesting={nesting_evidence:.2f}, "
                    f"pattern={pattern_evidence:.2f}, size={size_evidence:.2f}, "
                    f"layer={layer_evidence:.2f}, central={central_evidence:.2f} "
                    f"→ total={total_confidence:.2f}")
        
        return evidence
    
    def _analyze_geometric_closure_evidence(self, entity: NormalizedEntity) -> Tuple[float, Dict[str, Any]]:
        """Analyze geometric closure evidence - complete circular boundary."""
        
        details = {"entity_type": entity.source_entity.entity_type.value}
        
        if entity.source_entity.entity_type == EntityType.CIRCLE:
            # Complete circles have maximum closure evidence
            details["closure_type"] = "complete_circle"
            details["coverage"] = 1.0
            return 1.0, details
            
        elif entity.source_entity.entity_type == EntityType.ARC:
            # Single arcs have poor closure evidence for holes
            arc = entity.source_entity
            if arc.start_angle is not None and arc.end_angle is not None:
                # Calculate arc span
                span = abs(arc.end_angle - arc.start_angle)
                if span > ARC_ANGLE_HALF_CIRCLE:  # Normalize for wraparound
                    span = ARC_ANGLE_FULL_CIRCLE - span
                coverage = span / ARC_ANGLE_FULL_CIRCLE
                
                details["closure_type"] = "single_arc"
                details["coverage"] = coverage
                details["arc_span_degrees"] = span
                
                # Only very large arcs get significant closure evidence
                if span >= ARC_CLOSURE_EXCELLENT_THRESHOLD:
                    return ARC_CLOSURE_EXCELLENT_EVIDENCE, details
                elif span >= ARC_CLOSURE_GOOD_THRESHOLD:
                    return ARC_CLOSURE_GOOD_EVIDENCE, details
                else:
                    return ARC_CLOSURE_POOR_EVIDENCE, details  # Poor evidence for partial arcs
            
        details["closure_type"] = "no_closure"
        return 0.0, details
    
    def _analyze_concentric_nesting_evidence(self, entity: NormalizedEntity,
                                           all_entities: List[NormalizedEntity],
                                           relationships: GeometricRelationships) -> Tuple[float, Dict[str, Any]]:
        """Analyze concentric nesting evidence - holes are often inside other geometry."""
        
        details = {"concentric_relationships": [], "best_nesting_ratio": 0.0}
        
        # Find concentric relationships where this entity is the inner one
        entity_id = entity.source_entity.entity_id
        max_evidence = 0.0
        
        for rel in relationships.relationships:
            if (rel.relationship_type in ["strict_concentric", "concentric"] and 
                rel.confidence >= NESTING_RELATIONSHIP_MIN_CONFIDENCE and
                (rel.entity1_id == entity_id or rel.entity2_id == entity_id)):
                
                # Find the other entity
                other_entity_id = rel.entity2_id if rel.entity1_id == entity_id else rel.entity1_id
                other_entity = next((e for e in all_entities 
                                   if e.source_entity.entity_id == other_entity_id), None)
                
                if (other_entity and 
                    other_entity.source_entity.radius is not None and
                    entity.source_entity.radius is not None):
                    
                    this_radius = entity.source_entity.radius
                    other_radius = other_entity.source_entity.radius
                    
                    # Check if this entity is smaller (inner)
                    if this_radius < other_radius:
                        size_ratio = this_radius / other_radius
                        
                        # Validate size ratio constraints
                        if (CONCENTRIC_MIN_SIZE_RATIO <= size_ratio <= CONCENTRIC_MAX_SIZE_RATIO):
                            # Calculate evidence based on size ratio and relationship confidence
                            # Smaller relative size = better hole evidence
                            ratio_factor = 1.0 - size_ratio  # Smaller holes get higher score
                            evidence_score = ratio_factor * rel.confidence
                            
                            details["concentric_relationships"].append({
                                "outer_entity": other_entity_id,
                                "size_ratio": size_ratio,
                                "relationship_confidence": rel.confidence,
                                "evidence_contribution": evidence_score
                            })
                            
                            max_evidence = max(max_evidence, evidence_score)
        
        details["best_nesting_ratio"] = max_evidence
        details["relationship_count"] = len(details["concentric_relationships"])
        
        return min(1.0, max_evidence), details
    
    def _analyze_pattern_regularity_evidence(self, entity: NormalizedEntity,
                                           all_entities: List[NormalizedEntity]) -> Tuple[float, Dict[str, Any]]:
        """Analyze pattern regularity evidence - holes often appear in regular patterns."""
        
        details = {"similar_entities": [], "pattern_score": 0.0}
        
        if entity.source_entity.radius is None:
            return 0.0, details
        
        # Find similar-sized circular entities 
        similar_entities = []
        target_radius = entity.source_entity.radius
        
        for other_entity in all_entities:
            if (other_entity.source_entity.entity_id == entity.source_entity.entity_id or
                other_entity.source_entity.entity_type not in [EntityType.CIRCLE, EntityType.ARC] or
                other_entity.source_entity.radius is None):
                continue
            
            # Check radius similarity
            radius_diff = abs(target_radius - other_entity.source_entity.radius)
            if radius_diff <= PATTERN_RADIUS_TOLERANCE:
                similar_entities.append(other_entity)
                details["similar_entities"].append({
                    "entity_id": other_entity.source_entity.entity_id,
                    "radius": other_entity.source_entity.radius,
                    "radius_diff": radius_diff
                })
        
        # Calculate pattern evidence based on count of similar entities
        similar_count = len(similar_entities)
        details["similar_count"] = similar_count
        
        if similar_count >= PATTERN_MIN_SIMILAR_COUNT:
            # Strong pattern evidence
            if similar_count >= PATTERN_LARGE_COUNT_THRESHOLD:  # 5+ total including this one
                pattern_evidence = PATTERN_LARGE_EVIDENCE
            elif similar_count >= PATTERN_MEDIUM_COUNT_THRESHOLD:  # 3-4 total  
                pattern_evidence = PATTERN_MEDIUM_EVIDENCE
            else:  # 2 total (minimum pattern)
                pattern_evidence = PATTERN_SMALL_EVIDENCE
            
            # Check for spatial regularity (optional enhancement)
            if similar_count >= PATTERN_MEDIUM_COUNT_THRESHOLD and entity.source_entity.center:
                regularity_bonus = self._assess_spatial_regularity(
                    entity, similar_entities
                )
                pattern_evidence = min(1.0, pattern_evidence + regularity_bonus)
        else:
            # Single or insufficient pattern
            pattern_evidence = PATTERN_ISOLATED_EVIDENCE  # Low evidence for isolated features
        
        details["pattern_score"] = pattern_evidence
        return pattern_evidence, details
    
    def _analyze_size_appropriateness_evidence(self, entity: NormalizedEntity) -> Tuple[float, Dict[str, Any]]:
        """Analyze size appropriateness evidence - typical hole dimensions."""
        
        details = {"radius": entity.source_entity.radius}
        
        if entity.source_entity.radius is None:
            return 0.0, details
        
        radius = entity.source_entity.radius
        
        # Size-based evidence for holes (smaller holes more likely)
        if radius <= HOLE_SIZE_VERY_SMALL_THRESHOLD:  # Very small holes (screws, pins)
            size_evidence = HOLE_SIZE_VERY_SMALL_EVIDENCE
        elif radius <= HOLE_SIZE_SMALL_THRESHOLD:  # Small-medium holes (bolts)
            size_evidence = HOLE_SIZE_SMALL_EVIDENCE
        elif radius <= HOLE_SIZE_MEDIUM_THRESHOLD:  # Medium holes
            size_evidence = HOLE_SIZE_MEDIUM_EVIDENCE
        elif radius <= HOLE_SIZE_LARGE_THRESHOLD:  # Large holes
            size_evidence = HOLE_SIZE_LARGE_EVIDENCE
        elif radius <= self.max_radius:  # Very large holes
            size_evidence = HOLE_SIZE_VERY_LARGE_EVIDENCE
        else:
            size_evidence = HOLE_SIZE_OVERSIZED_EVIDENCE  # Too large for typical holes
        
        details["size_category"] = self._categorize_hole_size(radius)
        details["size_evidence"] = size_evidence
        
        return size_evidence, details
    
    def _analyze_layer_context_evidence(self, entity: NormalizedEntity) -> Tuple[float, Dict[str, Any]]:
        """Analyze layer context evidence - DXF layer metadata hints."""
        
        layer_name = entity.source_entity.layer.lower() if entity.source_entity.layer else ""
        details = {"layer_name": entity.source_entity.layer, "layer_analysis": {}}
        
        # Layer name analysis for hole indicators
        hole_keywords = ["hole", "drill", "cut", "bore", "perforation"]
        feature_keywords = ["feature", "mach", "process"]
        construction_keywords = ["construction", "hidden", "aux", "text", "dim", "note"]
        
        keyword_matches = []
        
        # Check for explicit hole keywords
        for keyword in hole_keywords:
            if keyword in layer_name:
                keyword_matches.append(keyword)
                details["layer_analysis"]["hole_keywords"] = keyword_matches
                return LAYER_HOLE_KEYWORDS_EVIDENCE, details
        
        # Check for feature keywords
        for keyword in feature_keywords:
            if keyword in layer_name:
                keyword_matches.append(keyword)
                details["layer_analysis"]["feature_keywords"] = keyword_matches
                return LAYER_FEATURE_KEYWORDS_EVIDENCE, details
        
        # Check for construction keywords (negative evidence)
        for keyword in construction_keywords:
            if keyword in layer_name:
                keyword_matches.append(keyword)
                details["layer_analysis"]["construction_keywords"] = keyword_matches
                return LAYER_CONSTRUCTION_KEYWORDS_EVIDENCE, details
        
        # Default/main layers
        if layer_name in ["0", "default", "main", ""]:
            details["layer_analysis"]["layer_type"] = "default"
            return LAYER_DEFAULT_EVIDENCE, details
        
        # Unknown layer
        details["layer_analysis"]["layer_type"] = "unknown"
        return LAYER_DEFAULT_EVIDENCE, details
    
    def _assess_spatial_regularity(self, entity: NormalizedEntity,
                                 similar_entities: List[NormalizedEntity]) -> float:
        """Assess spatial regularity bonus for pattern evidence."""
        
        if not entity.source_entity.center or len(similar_entities) < SPATIAL_REGULARITY_MIN_CENTERS - 1:
            return 0.0
        
        centers = [entity.source_entity.center]
        for other in similar_entities:
            if other.source_entity.center:
                centers.append(other.source_entity.center)
        
        if len(centers) < SPATIAL_REGULARITY_MIN_CENTERS:  # Need at least 3 points for regularity
            return 0.0
        
        # Simple regularity check - calculate distances between consecutive centers
        distances = []
        for i in range(len(centers) - 1):
            dist = centers[i].distance_to(centers[i + 1])
            distances.append(dist)
        
        if len(distances) < SPATIAL_REGULARITY_MIN_DISTANCES:
            return 0.0
        
        # Check if distances are similar (regular spacing)
        avg_distance = sum(distances) / len(distances)
        max_deviation = max(abs(d - avg_distance) for d in distances)
        
        if avg_distance > 0 and max_deviation / avg_distance < SPATIAL_REGULARITY_MAX_VARIATION:  # Within 30% variation
            return SPATIAL_REGULARITY_BONUS  # Small bonus for spatial regularity
        
        return 0.0
    
    def _categorize_hole_size(self, radius: float) -> str:
        """Categorize hole size for documentation using configurable thresholds."""
        if radius <= HOLE_CATEGORY_VERY_SMALL_THRESHOLD:
            return "very_small"  # Screws, pins
        elif radius <= HOLE_CATEGORY_SMALL_THRESHOLD:
            return "small"       # Small bolts
        elif radius <= HOLE_CATEGORY_MEDIUM_THRESHOLD:
            return "medium"      # Standard bolts
        elif radius <= HOLE_CATEGORY_LARGE_THRESHOLD:
            return "large"       # Large bolts, shafts
        else:
            return "very_large"  # Specialty holes
    
    def _create_hole_feature(self, center: Point2D, radius: float,
                           source_entity_ids: List[str], source_type: str,
                           evidence_analysis: Dict[str, Any]) -> ExpectedFeature:
        """Create a validated through hole feature with comprehensive evidence."""
        
        feature_id = f"hole_{self._get_next_id()}"
        
        return ExpectedFeature(
            feature_id=feature_id,
            feature_type=FeatureType.THROUGH_HOLE,
            confidence=evidence_analysis["total_confidence"],
            center=center,
            radius=radius,
            source_entity_ids=source_entity_ids,
            source_type=source_type,
            detection_evidence={
                **evidence_analysis,
                "validation_method": "comprehensive_evidence_analysis",
                "evidence_weights": HOLE_EVIDENCE_WEIGHTS,
                "confidence_threshold": self.min_confidence
            },
            geometric_properties={
                "size_category": self._categorize_hole_size(radius),
                "evidence_based_classification": True,
                "strict_validation": True
            }
        )
    
    def _create_temp_entity_for_analysis(self, recon_circle: ReconstructedCircle) -> NormalizedEntity:
        """Create temporary NormalizedEntity for reconstructed circle analysis."""
        from ..dxf.entity_models import DxfEntity
        
        temp_dxf_entity = DxfEntity(
            entity_id=recon_circle.circle_id,
            entity_type=EntityType.CIRCLE,  # Treat as complete circle for analysis
            layer="reconstructed", 
            handle=recon_circle.circle_id,
            center=recon_circle.center,
            radius=recon_circle.radius
        )
        
        return NormalizedEntity(source_entity=temp_dxf_entity)
    
    def _analyze_geometric_context(self, entity: NormalizedEntity, 
                                 all_entities: List[NormalizedEntity]) -> Dict[str, Any]:
        """
        Analyze the geometric context using pure spatial distribution analysis.
        
        Identifies main body geometry and spatial relationships using only geometric
        properties, without any product-specific knowledge or hardcoded coordinates.
        """
        
        context = {
            "main_body_center": None,
            "main_body_radius": None,
            "entity_count": len(all_entities),
            "circular_entities": 0,
            "large_circles": [],
            "feature_circles": [],
            "geometric_bounds": None,
            "all_entities": all_entities
        }
        
        # Analyze all circular geometry using configurable thresholds
        large_circles = []
        feature_circles = []
        all_radii = []
        
        for other_entity in all_entities:
            if (other_entity.source_entity.entity_type in [EntityType.CIRCLE, EntityType.ARC] and
                other_entity.source_entity.radius is not None):
                
                radius = other_entity.source_entity.radius
                all_radii.append(radius)
                context["circular_entities"] += 1
                
                # Use configurable threshold for main body vs feature classification
                if radius >= MAIN_BODY_MIN_RADIUS:
                    large_circles.append(other_entity)
                elif self.min_radius <= radius <= self.max_radius:
                    feature_circles.append(other_entity)
        
        context["large_circles"] = large_circles
        context["feature_circles"] = feature_circles
        
        # Calculate geometric bounds of all entities
        context["geometric_bounds"] = self._calculate_geometric_bounds(all_entities)
        
        # Identify main body geometry using geometric analysis
        if large_circles:
            # Find the most central and largest geometry as main body
            main_body = self._identify_main_body_geometry(large_circles, context["geometric_bounds"])
            if main_body:
                context["main_body_center"] = main_body.source_entity.center
                context["main_body_radius"] = main_body.source_entity.radius
        elif all_radii:
            # No explicitly large circles - use statistical analysis
            avg_radius = sum(all_radii) / len(all_radii)
            max_radius = max(all_radii)
            
            # If max radius is significantly larger than average, treat as main body
            if max_radius > avg_radius * THROUGH_HOLE_RADIUS_RATIO_THRESHOLD:
                largest_entity = max(
                    [e for e in all_entities if e.source_entity.radius == max_radius],
                    key=lambda e: e.source_entity.radius or 0
                )
                context["main_body_center"] = largest_entity.source_entity.center
                context["main_body_radius"] = largest_entity.source_entity.radius
        
        return context
    
    def _calculate_geometric_bounds(self, all_entities: List[NormalizedEntity]) -> Dict[str, float]:
        """Calculate bounding box of all geometry."""
        
        if not all_entities:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "center_x": 0, "center_y": 0}
        
        x_coords = []
        y_coords = []
        
        for entity in all_entities:
            if entity.source_entity.center:
                x_coords.append(entity.source_entity.center.x)
                y_coords.append(entity.source_entity.center.y)
                
                # For circles/arcs, include bounds
                if entity.source_entity.radius:
                    radius = entity.source_entity.radius
                    x_coords.extend([
                        entity.source_entity.center.x - radius,
                        entity.source_entity.center.x + radius
                    ])
                    y_coords.extend([
                        entity.source_entity.center.y - radius,
                        entity.source_entity.center.y + radius
                    ])
            
            # For lines, include endpoints
            if hasattr(entity.source_entity, 'start_point') and entity.source_entity.start_point:
                x_coords.append(entity.source_entity.start_point.x)
                y_coords.append(entity.source_entity.start_point.y)
            if hasattr(entity.source_entity, 'end_point') and entity.source_entity.end_point:
                x_coords.append(entity.source_entity.end_point.x)
                y_coords.append(entity.source_entity.end_point.y)
        
        if not x_coords or not y_coords:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "center_x": 0, "center_y": 0}
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return {
            "min_x": min_x, "max_x": max_x,
            "min_y": min_y, "max_y": max_y,
            "center_x": (min_x + max_x) / 2,
            "center_y": (min_y + max_y) / 2,
            "width": max_x - min_x,
            "height": max_y - min_y
        }
    
    def _identify_main_body_geometry(self, large_circles: List[NormalizedEntity], 
                                   geometric_bounds: Dict[str, float]) -> Optional[NormalizedEntity]:
        """
        Identify main body geometry using spatial analysis.
        
        Uses geometric centrality and size to identify the most likely main body element.
        """
        
        if not large_circles:
            return None
        
        if len(large_circles) == 1:
            return large_circles[0]
        
        # Calculate centrality scores for each large circle
        bounds_center_x = geometric_bounds.get("center_x", 0)
        bounds_center_y = geometric_bounds.get("center_y", 0)
        
        best_candidate = None
        best_score = -1
        
        for circle in large_circles:
            if not circle.source_entity.center or not circle.source_entity.radius:
                continue
            
            # Centrality score: distance from geometric center (lower is better)
            center_distance = math.sqrt(
                (circle.source_entity.center.x - bounds_center_x) ** 2 +
                (circle.source_entity.center.y - bounds_center_y) ** 2
            )
            
            # Size score: larger is more likely to be main body
            size_score = circle.source_entity.radius
            
            # Combined score (normalize and weight)
            bounds_diagonal = math.sqrt(
                geometric_bounds.get("width", 1) ** 2 + 
                geometric_bounds.get("height", 1) ** 2
            )
            
            centrality_score = 1.0 - (center_distance / max(bounds_diagonal, 1))
            size_normalized = size_score / THROUGH_HOLE_SIZE_SCORE_NORMALIZATION  # Normalize size
            
            combined_score = (centrality_score * THROUGH_HOLE_CENTRALITY_WEIGHT_POSITION + 
                            min(1.0, size_normalized) * THROUGH_HOLE_CENTRALITY_WEIGHT_SIZE)
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = circle
        
        return best_candidate
    
    def _analyze_central_position_evidence(self, entity: NormalizedEntity,
                                         geometric_context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Analyze evidence for this being a central hole."""
        
        details = {"is_central": False, "distance_from_center": None}
        
        main_body_center = geometric_context.get("main_body_center")
        main_body_radius = geometric_context.get("main_body_radius")
        
        # For circular files with many circular entities, use special central hole detection
        if geometric_context.get("circular_entities", 0) >= THROUGH_HOLE_CIRCULAR_FILE_MIN_ENTITIES:
            return self._analyze_central_position_for_circular_file(entity, geometric_context)
        
        if not main_body_center or not entity.source_entity.center:
            return THROUGH_HOLE_DEFAULT_EVIDENCE, details  # No central position information
        
        # Calculate distance from main body center
        distance = entity.source_entity.center.distance_to(main_body_center)
        details["distance_from_center"] = distance
        
        # Central holes should be very close to the main body center
        central_tolerance = (min(THROUGH_HOLE_CENTRAL_TOLERANCE_BASE, 
                                main_body_radius * THROUGH_HOLE_CENTRAL_TOLERANCE_FACTOR) 
                           if main_body_radius else THROUGH_HOLE_CENTRAL_TOLERANCE_BASE)
        
        if distance <= central_tolerance:
            details["is_central"] = True
            # Central position gives strong hole evidence
            central_evidence = 1.0 - (distance / central_tolerance)
            return central_evidence, details
        else:
            # Not central
            return THROUGH_HOLE_NON_CENTRAL_EVIDENCE, details
    
    def _analyze_central_position_for_circular_file(self, entity: NormalizedEntity,
                                                  geometric_context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze central position evidence based on pure geometric relationships.
        
        Uses spatial distribution analysis within the DXF geometry to identify potential 
        central features without relying on any hardcoded coordinates or product knowledge.
        """
        
        details = {"is_central": False, "distance_from_center": None, "analysis_type": "geometric_centroid"}
        
        if not entity.source_entity.center or entity.source_entity.radius is None:
            return THROUGH_HOLE_DEFAULT_EVIDENCE, details
        
        center = entity.source_entity.center
        radius = entity.source_entity.radius
        
        # Calculate geometric distribution of all feature-sized circular entities
        feature_centers = []
        feature_radii = []
        
        for other_entity in geometric_context.get("all_entities", []):
            if (hasattr(other_entity, 'source_entity') and 
                other_entity.source_entity.entity_type.name in ['CIRCLE', 'ARC'] and
                other_entity.source_entity.center and
                other_entity.source_entity.radius):
                
                # Include only feature-sized entities using configurable thresholds
                other_radius = other_entity.source_entity.radius
                if (self.min_radius <= other_radius <= self.max_radius and 
                    other_radius < MAIN_BODY_MIN_RADIUS):  # Use config-based filtering
                    feature_centers.append(other_entity.source_entity.center)
                    feature_radii.append(other_radius)
        
        if len(feature_centers) < THROUGH_HOLE_SPATIAL_MIN_CENTERS:  # Not enough geometry for spatial analysis
            return THROUGH_HOLE_DEFAULT_EVIDENCE, details
        
        # Calculate geometric centroid of feature distribution
        centroid_x = sum(c.x for c in feature_centers) / len(feature_centers)
        centroid_y = sum(c.y for c in feature_centers) / len(feature_centers)
        
        # Distance from this entity to the geometric centroid
        distance_to_centroid = math.sqrt((center.x - centroid_x) ** 2 + (center.y - centroid_y) ** 2)
        details["distance_from_center"] = distance_to_centroid
        details["geometric_centroid"] = (centroid_x, centroid_y)
        details["feature_count_in_analysis"] = len(feature_centers)
        
        # Calculate spatial distribution statistics for context
        centroid_distances = [
            math.sqrt((c.x - centroid_x) ** 2 + (c.y - centroid_y) ** 2) 
            for c in feature_centers
        ]
        
        if centroid_distances:
            avg_distance_from_centroid = sum(centroid_distances) / len(centroid_distances)
            max_distance_from_centroid = max(centroid_distances)
            
            details["avg_distance_from_centroid"] = avg_distance_from_centroid
            details["max_distance_from_centroid"] = max_distance_from_centroid
            
            # Central tolerance based on spatial distribution
            central_tolerance = min(
                avg_distance_from_centroid * THROUGH_HOLE_SPATIAL_TOLERANCE_FACTOR_HIGH,  # 30% of average distance
                max_distance_from_centroid * THROUGH_HOLE_SPATIAL_TOLERANCE_FACTOR_LOW   # 15% of maximum distance
            )
            
            details["central_tolerance_calculated"] = central_tolerance
            
            if distance_to_centroid <= central_tolerance:
                details["is_central"] = True
                # Calculate evidence based on relative position in distribution
                if central_tolerance > 0:
                    centrality_evidence = 1.0 - (distance_to_centroid / central_tolerance)
                else:
                    centrality_evidence = 1.0
                
                return max(THROUGH_HOLE_CENTRAL_EVIDENCE_MIN, centrality_evidence), details
        
        # Secondary analysis: check if this is at the center of a circular pattern
        pattern_evidence, pattern_details = self._analyze_circular_pattern_centrality(
            entity, feature_centers, feature_radii
        )
        details.update(pattern_details)
        
        if pattern_evidence > THROUGH_HOLE_PATTERN_EVIDENCE_THRESHOLD:
            details["is_central"] = True
            return pattern_evidence, details
        
        # Tertiary analysis: size-based central hole evidence
        size_evidence = self._analyze_central_size_evidence(radius, feature_radii)
        details["size_based_evidence"] = size_evidence
        
        return size_evidence, details
    
    def _analyze_circular_pattern_centrality(self, entity: NormalizedEntity,
                                           feature_centers: List[Point2D],
                                           feature_radii: List[float]) -> Tuple[float, Dict[str, Any]]:
        """Analyze if this entity is at the center of a circular pattern."""
        
        details = {"circular_pattern": False, "pattern_radius": None}
        
        if not entity.source_entity.center or len(feature_centers) < THROUGH_HOLE_CIRCULAR_PATTERN_MIN_CENTERS:
            return 0.0, details
        
        center = entity.source_entity.center
        
        # Calculate distances from this entity to all other features
        distances = [
            math.sqrt((c.x - center.x) ** 2 + (c.y - center.y) ** 2)
            for c in feature_centers if c != center
        ]
        
        if len(distances) < THROUGH_HOLE_SPATIAL_MIN_DISTANCES:
            return 0.0, details
        
        # Check for circular arrangement (similar distances)
        avg_distance = sum(distances) / len(distances)
        distance_std = math.sqrt(sum((d - avg_distance) ** 2 for d in distances) / len(distances))
        
        details["avg_distance_to_features"] = avg_distance
        details["distance_std"] = distance_std
        
        # If features are arranged in a circle around this entity
        if avg_distance > 0 and distance_std / avg_distance < THROUGH_HOLE_CIRCULAR_PATTERN_VARIATION:  # Low variation = circular pattern
            details["circular_pattern"] = True
            details["pattern_radius"] = avg_distance
            
            # Strong evidence if arranged in a circle around this point
            pattern_evidence = THROUGH_HOLE_CIRCULAR_PATTERN_BASE_EVIDENCE - (distance_std / avg_distance)
            return pattern_evidence, details
        
        return 0.0, details
    
    def _analyze_central_size_evidence(self, radius: float, 
                                     other_radii: List[float]) -> float:
        """Analyze size evidence for central hole (should be appropriately sized for function)."""
        
        if not other_radii:
            return THROUGH_HOLE_DEFAULT_EVIDENCE
        
        # Central holes are often moderately sized relative to surrounding features
        avg_radius = sum(other_radii) / len(other_radii)
        
        # Good central hole size: similar to or slightly smaller than average
        if avg_radius > 0:
            size_ratio = radius / avg_radius
            
            if THROUGH_HOLE_SIZE_RATIO_MIN_GOOD <= size_ratio <= THROUGH_HOLE_SIZE_RATIO_MAX_GOOD:
                return THROUGH_HOLE_CENTRAL_EVIDENCE_GOOD  # Good size for central hole
            elif THROUGH_HOLE_SIZE_RATIO_MIN_ACCEPTABLE <= size_ratio <= THROUGH_HOLE_SIZE_RATIO_MAX_ACCEPTABLE:
                return THROUGH_HOLE_CENTRAL_EVIDENCE_ACCEPTABLE
            else:
                return THROUGH_HOLE_POOR_SIZE_EVIDENCE  # Poor size for central hole
        
        # Fallback: absolute size appropriateness
        if THROUGH_HOLE_CENTRAL_SIZE_MIN <= radius <= THROUGH_HOLE_CENTRAL_SIZE_MAX:
            return THROUGH_HOLE_CENTRAL_EVIDENCE_ACCEPTABLE
        else:
            return THROUGH_HOLE_POOR_SIZE_EVIDENCE
    def _get_next_id(self) -> int:
        """Get next sequential feature ID."""
        self._feature_counter += 1
        return self._feature_counter