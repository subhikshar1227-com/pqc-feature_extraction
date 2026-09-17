"""
Circle Feature Detector

Detects prominent circle features from DXF entities and reconstructed geometry
with strict evidence-based validation.
"""

import logging
import math
from typing import List, Dict, Any

from .feature_types import ExpectedFeature, FeatureType
from ..dxf.entity_models import NormalizedEntity, EntityType
from ..dxf.geometry_reconstruction import ReconstructedGeometry, ReconstructedCircle
from ..config import (
    CIRCLE_MIN_RADIUS, CIRCLE_MAX_RADIUS,
    # Circle confidence configuration
    CIRCLE_BASE_CONFIDENCE, CIRCLE_SIZE_FACTOR_IDEAL, CIRCLE_SIZE_FACTOR_ACCEPTABLE,
    CIRCLE_SIZE_FACTOR_SMALL, CIRCLE_SIZE_FACTOR_LARGE, CIRCLE_SIZE_FACTOR_NO_RADIUS,
    CIRCLE_LAYER_FACTOR_AUXILIARY, CIRCLE_LAYER_FACTOR_GEOMETRIC, CIRCLE_LAYER_FACTOR_DEFAULT,
    CIRCLE_CONTEXT_FACTOR_NEUTRAL, CIRCLE_CONTEXT_FACTOR_COMPLEX, CIRCLE_SIZE_DIVERSITY_THRESHOLD,
    # Main body detection configuration
    CIRCLE_MAIN_BODY_CONSERVATIVE_THRESHOLD, CIRCLE_ADAPTIVE_THRESHOLD_MULTIPLIER,
    CIRCLE_RADIUS_FALLBACK_MULTIPLIER, CIRCLE_MIN_SIZE_THRESHOLD, CIRCLE_VERY_LARGE_THRESHOLD,
    CIRCLE_CONCENTRIC_MIN_COUNT,
    # Existing configuration
    FEATURE_IDEAL_MIN_RADIUS, FEATURE_IDEAL_MAX_RADIUS,
    FEATURE_ACCEPTABLE_MIN_RADIUS, FEATURE_ACCEPTABLE_MAX_RADIUS,
    RECONSTRUCTED_MAIN_BODY_MIN_RADIUS, RECONSTRUCTED_MAIN_BODY_MIN_ARC_COUNT,
    MAIN_BODY_MIN_RADIUS, CONCENTRICITY_TOLERANCE
)

logger = logging.getLogger(__name__)


class CircleDetector:
    """Detects circle features with strict geometric validation."""
    
    def __init__(self, 
                 min_radius: float = CIRCLE_MIN_RADIUS,
                 max_radius: float = CIRCLE_MAX_RADIUS):
        """
        Initialize circle detector.
        
        Args:
            min_radius: Minimum radius for circle features (mm)
            max_radius: Maximum radius for circle features (mm)
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self._feature_counter = 0
    
    def detect_circles(self, entities: List[NormalizedEntity], 
                      reconstructed: ReconstructedGeometry) -> List[ExpectedFeature]:
        """
        Detect circle features from entities and reconstructed geometry with strict validation.
        
        Only accepts:
        1. Explicit CIRCLE entities with appropriate size
        2. Reconstructed circles with strong geometric evidence
        
        Rejects:
        - Partial arcs (even if they have radius)
        - Rounded corners represented by single arcs
        - Unrelated arc collections
        """
        # Handle None input gracefully for tests
        if reconstructed is None:
            reconstructed_circles = []
        else:
            reconstructed_circles = reconstructed.reconstructed_circles
            
        logger.debug(f"Detecting circles with strict validation from {len(entities)} entities "
                    f"and {len(reconstructed_circles)} reconstructed circles")
        
        circles = []
        
        # Detect circles from explicit CIRCLE entities only
        direct_circles = self._detect_explicit_circles(entities)
        circles.extend(direct_circles)
        
        # Detect circles from strictly validated reconstructed arc combinations
        reconstructed_circles = self._detect_validated_reconstructed_circles(
            reconstructed_circles
        )
        circles.extend(reconstructed_circles)
        
        logger.info(f"Detected {len(circles)} circle features "
                   f"({len(direct_circles)} explicit, {len(reconstructed_circles)} reconstructed)")
        
        return circles
    
    def _detect_explicit_circles(self, entities: List[NormalizedEntity]) -> List[ExpectedFeature]:
        """Detect circles from explicit CIRCLE entities with intelligent filtering."""
        
        circles = []
        
        # First pass: collect all circles and analyze the geometry context
        all_circles = []
        for entity in entities:
            if (entity.source_entity.entity_type == EntityType.CIRCLE and 
                entity.source_entity.center and 
                entity.source_entity.radius is not None):
                all_circles.append(entity)
        
        if not all_circles:
            return circles
        
        # Analyze circles to distinguish main body from features
        circle_analysis = self._analyze_circle_context(all_circles, entities)
        
        for entity in all_circles:
            # Check size constraints
            if not (self.min_radius <= entity.source_entity.radius <= self.max_radius):
                logger.debug(f"Skipping circle {entity.source_entity.entity_id}: "
                           f"radius {entity.source_entity.radius} outside [{self.min_radius}, {self.max_radius}]")
                continue
            
            # Check if this circle represents a main body vs. a feature
            if self._is_main_body_circle(entity, circle_analysis):
                logger.debug(f"Skipping main body circle {entity.source_entity.entity_id}: "
                           f"radius {entity.source_entity.radius}")
                continue
            
            # Calculate confidence based on geometric completeness and context
            confidence = self._calculate_explicit_circle_confidence(entity, circle_analysis)
            
            # Create feature
            feature_id = f"circle_{self._get_next_id()}"
            
            feature = ExpectedFeature(
                feature_id=feature_id,
                feature_type=FeatureType.CIRCLE,
                confidence=confidence,
                center=entity.source_entity.center,
                radius=entity.source_entity.radius,
                source_entity_ids=[entity.source_entity.entity_id],
                source_type="explicit_circle",
                detection_evidence={
                    "entity_type": "CIRCLE",
                    "radius": entity.source_entity.radius,
                    "geometric_completeness": "complete_circle_entity",
                    "validation_method": "explicit_entity",
                    "area": entity.enclosed_area,
                    "perimeter": entity.perimeter_length,
                    "circle_context": circle_analysis.get("context", {}),
                    "is_main_body": False
                },
                geometric_properties={
                    "layer": entity.source_entity.layer,
                    "handle": entity.source_entity.handle,
                    "entity_complete": True
                }
            )
            
            circles.append(feature)
            logger.debug(f"Detected explicit circle feature: {feature_id}, "
                        f"center=({entity.source_entity.center.x:.2f}, {entity.source_entity.center.y:.2f}), "
                        f"radius={entity.source_entity.radius:.2f}")
        
        return circles
    
    def _detect_validated_reconstructed_circles(self, reconstructed_circles: List[ReconstructedCircle]) -> List[ExpectedFeature]:
        """Detect circles from strictly validated reconstructed arc combinations with main body filtering."""
        
        circles = []
        
        for recon_circle in reconstructed_circles:
            # Only accept circles that passed strict reconstruction validation
            if not recon_circle.validation_evidence.get("strict_validation", False):
                logger.debug(f"Skipping non-validated reconstructed circle {recon_circle.circle_id}")
                continue
            
            # Check size constraints
            if not (self.min_radius <= recon_circle.radius <= self.max_radius):
                logger.debug(f"Skipping reconstructed circle {recon_circle.circle_id}: "
                           f"radius {recon_circle.radius} outside [{self.min_radius}, {self.max_radius}]")
                continue
            
            # Filter out main body circles (large concentric reconstructed circles)
            if self._is_main_body_reconstructed_circle(recon_circle):
                logger.debug(f"Skipping main body reconstructed circle {recon_circle.circle_id}: "
                           f"radius {recon_circle.radius}")
                continue
            
            # Use reconstruction confidence (already calculated with strict criteria)
            confidence = recon_circle.confidence
            
            # Create feature
            feature_id = f"circle_{self._get_next_id()}"
            
            feature = ExpectedFeature(
                feature_id=feature_id,
                feature_type=FeatureType.CIRCLE,
                confidence=confidence,
                center=recon_circle.center,
                radius=recon_circle.radius,
                source_entity_ids=recon_circle.source_entities,
                source_type="reconstructed_circle",
                detection_evidence={
                    "reconstruction_id": recon_circle.circle_id,
                    "coverage_fraction": recon_circle.coverage_fraction,
                    "source_arc_count": len(recon_circle.source_entities),
                    "max_gap": max(recon_circle.gaps) if recon_circle.gaps else 0.0,
                    "total_overlap": sum(recon_circle.overlaps) if recon_circle.overlaps else 0.0,
                    "reconstruction_confidence": recon_circle.confidence,
                    "validation_evidence": recon_circle.validation_evidence,
                    "validation_method": "strict_arc_reconstruction",
                    "is_main_body": False
                },
                geometric_properties={
                    "reconstructed_from": "validated_arc_group",
                    "arc_coverage": f"{recon_circle.coverage_fraction:.1%}",
                    "geometric_validation": "strict"
                }
            )
            
            circles.append(feature)
            logger.debug(f"Detected reconstructed circle feature: {feature_id}, "
                        f"center=({recon_circle.center.x:.2f}, {recon_circle.center.y:.2f}), "
                        f"radius={recon_circle.radius:.2f}, coverage={recon_circle.coverage_fraction:.1%}")
        
        return circles
    
    def _is_main_body_reconstructed_circle(self, recon_circle: ReconstructedCircle) -> bool:
        """
        Determine if a reconstructed circle represents the main body using configurable thresholds.
        
        TASK 3: CONSERVATIVE filtering for reconstructed circles.
        Only filter truly massive reconstructed geometry that's clearly structural.
        """
        
        radius = recon_circle.radius
        
        # Import configuration
        from ..config import RECONSTRUCTED_MAIN_BODY_MIN_RADIUS, RECONSTRUCTED_MAIN_BODY_MIN_ARC_COUNT
        
        # MUCH MORE CONSERVATIVE: Only filter very large reconstructed circles
        # Increased threshold to avoid filtering feature-scale reconstructed circles
        conservative_threshold = max(RECONSTRUCTED_MAIN_BODY_MIN_RADIUS, CIRCLE_MAIN_BODY_CONSERVATIVE_THRESHOLD)
        
        if radius > conservative_threshold:
            logger.debug(f"Reconstructed circle {recon_circle.circle_id} is very large (r={radius:.1f}), "
                        f"likely main body")
            return True
        
        # REMOVED: The arc count + size logic was too aggressive
        # Multiple large concentric arcs suggest main body boundary
        # if radius > (conservative_threshold * 0.8) and len(recon_circle.source_entities) >= RECONSTRUCTED_MAIN_BODY_MIN_ARC_COUNT:
        
        return False
    
    def _calculate_explicit_circle_confidence(self, entity: NormalizedEntity, 
                                            circle_analysis: Dict[str, Any] = None) -> float:
        """
        Calculate confidence score for an explicit circle entity using 
        geometry-derived and configurable thresholds.
        """
        
        # Import configuration  
        from ..config import (FEATURE_IDEAL_MIN_RADIUS, FEATURE_IDEAL_MAX_RADIUS,
                             FEATURE_ACCEPTABLE_MIN_RADIUS, FEATURE_ACCEPTABLE_MAX_RADIUS)
        
        # Base confidence for complete circle entities is high
        base_confidence = CIRCLE_BASE_CONFIDENCE
        
        # Adjust based on size appropriateness using geometric analysis
        radius = entity.source_entity.radius
        size_factor = CIRCLE_SIZE_FACTOR_IDEAL
        
        if radius is not None:
            # Geometric size classification based on feature likelihood
            if FEATURE_IDEAL_MIN_RADIUS <= radius <= FEATURE_IDEAL_MAX_RADIUS:
                # Well-sized circles - optimal for typical features
                size_factor = CIRCLE_SIZE_FACTOR_IDEAL
            elif (FEATURE_ACCEPTABLE_MIN_RADIUS <= radius < FEATURE_IDEAL_MIN_RADIUS or 
                  FEATURE_IDEAL_MAX_RADIUS < radius <= FEATURE_ACCEPTABLE_MAX_RADIUS):
                # Acceptable size range - reasonable features
                size_factor = CIRCLE_SIZE_FACTOR_ACCEPTABLE
            elif radius < FEATURE_ACCEPTABLE_MIN_RADIUS:
                # Very small - might be detail features, lower confidence
                size_factor = CIRCLE_SIZE_FACTOR_SMALL
            else:
                # Very large - likely structural, lowest confidence as feature
                size_factor = CIRCLE_SIZE_FACTOR_LARGE
        else:
            size_factor = CIRCLE_SIZE_FACTOR_NO_RADIUS  # No radius data
        
        # Adjust based on layer context (geometric metadata analysis)
        layer_factor = self._analyze_layer_geometric_context(entity.source_entity.layer)
        
        # Context factor from geometric analysis
        context_factor = self._analyze_geometric_feature_context(entity, circle_analysis)
        
        confidence = base_confidence * size_factor * layer_factor * context_factor
        return min(CIRCLE_SIZE_FACTOR_IDEAL, confidence)
    
    def _analyze_layer_geometric_context(self, layer_name: str) -> float:
        """Analyze layer name for geometric feature context (no product-specific logic)."""
        if not layer_name:
            return CIRCLE_LAYER_FACTOR_GEOMETRIC
        
        layer_lower = layer_name.lower()
        
        # Generic geometric layer analysis
        if any(keyword in layer_lower for keyword in ['construction', 'hidden', 'text', 'dim', 'note']):
            return CIRCLE_LAYER_FACTOR_AUXILIARY  # Lower confidence for auxiliary geometry
        elif any(keyword in layer_lower for keyword in ['feature', 'part', 'main', 'geometry']):
            return CIRCLE_LAYER_FACTOR_GEOMETRIC  # High confidence for geometric feature layers
        else:
            return CIRCLE_LAYER_FACTOR_DEFAULT  # Default high confidence for unlabeled geometry
    
    def _analyze_geometric_feature_context(self, entity: NormalizedEntity, 
                                         circle_analysis: Dict[str, Any]) -> float:
        """
        Analyze geometric context to determine feature likelihood.
        
        TASK 4: REMOVED BIASED ASSUMPTIONS about size vs feature likelihood.
        Large circles are NOT automatically less likely to be features.
        This method now focuses on OBJECTIVE geometric evidence only.
        """
        if not circle_analysis or not entity.source_entity.radius:
            return CIRCLE_CONTEXT_FACTOR_NEUTRAL  # Default neutral confidence
        
        context = circle_analysis.get("context", {})
        
        # REMOVED BIASED LOGIC: 
        # - No "hierarchical geometry" assumptions
        # - No "small in hierarchy = likely feature" bias
        # - No "large in hierarchy = likely structural" bias
        
        # Use OBJECTIVE geometric properties only:
        
        # 1. Size diversity analysis (objective geometric distribution)
        size_diversity = context.get("size_diversity", CIRCLE_CONTEXT_FACTOR_NEUTRAL)
        if size_diversity < CIRCLE_SIZE_DIVERSITY_THRESHOLD:
            # Low diversity - all similar sized, neutral confidence
            return CIRCLE_CONTEXT_FACTOR_NEUTRAL
        else:
            # High diversity - mixed sizes, neutral confidence for all
            return CIRCLE_CONTEXT_FACTOR_COMPLEX  # Slight penalty for complex geometry contexts
        
        # Default: preserve original geometric confidence
        return CIRCLE_CONTEXT_FACTOR_NEUTRAL
    
    def _analyze_circle_context(self, circles: List[NormalizedEntity], 
                              all_entities: List[NormalizedEntity]) -> Dict[str, Any]:
        """
        Analyze the geometric context of circles to understand main body vs. features
        using purely geometric distribution analysis.
        """
        
        analysis = {
            "total_circles": len(circles),
            "radii": [],
            "centers": [],
            "large_circles": [],
            "small_circles": [],
            "context": {}
        }
        
        # Collect basic geometric info
        for circle in circles:
            analysis["radii"].append(circle.source_entity.radius)
            analysis["centers"].append(circle.source_entity.center)
        
        if not analysis["radii"]:
            return analysis
        
        # Geometric distribution analysis
        max_radius = max(analysis["radii"])
        min_radius = min(analysis["radii"])
        avg_radius = sum(analysis["radii"]) / len(analysis["radii"])
        
        # Calculate statistical properties for geometric classification
        radius_std = 0.0
        if len(analysis["radii"]) > 1:
            radius_variance = sum((r - avg_radius)**2 for r in analysis["radii"]) / len(analysis["radii"])
            radius_std = math.sqrt(radius_variance)
        
        # Adaptive threshold based on geometric distribution
        # Use statistical analysis to separate structural from feature geometry
        if radius_std > 0:
            # Use statistical separation: mean + 1.5*std as threshold for large circles
            adaptive_threshold = avg_radius + CIRCLE_ADAPTIVE_THRESHOLD_MULTIPLIER * radius_std
        else:
            adaptive_threshold = avg_radius * CIRCLE_RADIUS_FALLBACK_MULTIPLIER  # Fallback when no variation
        
        # Ensure minimum absolute threshold
        size_threshold = max(CIRCLE_MIN_SIZE_THRESHOLD, adaptive_threshold)
        
        # Classify circles based on geometric analysis
        small_circles = []
        large_circles = []
        
        for circle in circles:
            radius = circle.source_entity.radius
            if radius > size_threshold:
                large_circles.append(circle)
            else:
                small_circles.append(circle)
        
        analysis["large_circles"] = large_circles
        analysis["small_circles"] = small_circles
        
        # Comprehensive geometric context
        analysis["context"] = {
            "max_radius": max_radius,
            "min_radius": min_radius,
            "avg_radius": avg_radius,
            "radius_std": radius_std,
            "size_threshold": size_threshold,
            "adaptive_threshold": adaptive_threshold,
            "has_large_circles": len(large_circles) > 0,
            "has_small_circles": len(small_circles) > 0,
            "small_circles_count": len(small_circles),
            "large_circles_count": len(large_circles),
            "size_diversity": max_radius / min_radius if min_radius > 0 else 1.0,
            "size_separation_factor": adaptive_threshold / avg_radius if avg_radius > 0 else 1.0,
            "geometric_hierarchy": len(large_circles) > 0 and len(small_circles) > 0
        }
        
        logger.debug(f"Geometric circle analysis: {len(circles)} circles, "
                    f"large: {len(large_circles)} (r>{size_threshold:.1f}), "
                    f"small: {len(small_circles)}, "
                    f"size range: {min_radius:.1f} - {max_radius:.1f}, "
                    f"std: {radius_std:.1f}, adaptive_threshold: {adaptive_threshold:.1f}")
        
        return analysis
    
    def _is_main_body_circle(self, circle: NormalizedEntity, 
                           circle_analysis: Dict[str, Any]) -> bool:
        """
        Determine if a circle represents the main body rather than a feature using 
        purely geometric analysis and configurable thresholds.
        
        TASK 3: SEPARATE GEOMETRIC FACT FROM ENGINEERING INTERPRETATION
        - DXF geometry establishes circles, arcs, lines, containment, relative sizes
        - Engineering interpretation infers "main body" vs "feature" from geometry
        - This method must be based on OBJECTIVE geometric evidence only
        
        Uses geometric context analysis to identify structural vs feature geometry:
        - Size-based detection using geometric distribution analysis  
        - Concentric relationship analysis for boundary detection
        - Spatial context relative to other geometry
        
        FIXED: More conservative filtering to preserve feature-scale circles.
        """
        
        # Import configuration
        from ..config import MAIN_BODY_MIN_RADIUS, CONCENTRICITY_TOLERANCE
        
        radius = circle.source_entity.radius
        context = circle_analysis.get("context", {})
        
        # 1. VERY CONSERVATIVE Absolute size-based main body detection
        # Only filter circles that are objectively massive (true structural boundaries)
        if radius > MAIN_BODY_MIN_RADIUS:
            logger.debug(f"Circle {circle.source_entity.entity_id} identified as main body: "
                        f"radius {radius:.1f} > threshold {MAIN_BODY_MIN_RADIUS}")
            return True
        
        # 2. REMOVED AGGRESSIVE RELATIVE SIZE FILTERING
        # The previous logic was too aggressive - removed:
        # - "largest circle in context" filtering
        # - "significantly larger than average" filtering  
        # These caused feature-scale circles (27.5, 29.8, 29.0mm) to be incorrectly filtered
        
        # 3. MUCH MORE CONSERVATIVE concentric boundary detection
        # Only filter if we have STRONG evidence of nested structural boundaries
        large_circles = circle_analysis.get("large_circles", [])
        if len(large_circles) > CIRCLE_CONCENTRIC_MIN_COUNT and circle in large_circles:  # Need multiple large circles
            # Check for concentric large circles indicating nested boundaries
            concentric_count = 0
            for large_circle in large_circles:
                if (large_circle.source_entity.entity_id != circle.source_entity.entity_id and
                    large_circle.source_entity.center and circle.source_entity.center):
                    
                    center_distance = large_circle.source_entity.center.distance_to(
                        circle.source_entity.center
                    )
                    
                    # Must be truly concentric AND both circles must be very large
                    if (center_distance < CONCENTRICITY_TOLERANCE and 
                        large_circle.source_entity.radius > CIRCLE_VERY_LARGE_THRESHOLD and  # Both very large
                        radius > CIRCLE_VERY_LARGE_THRESHOLD):
                        concentric_count += 1
            
            # Only filter if we have multiple concentric very large circles
            if concentric_count >= CIRCLE_CONCENTRIC_MIN_COUNT:
                logger.debug(f"Circle {circle.source_entity.entity_id} identified as main body: "
                           f"multiple concentric large circles (count={concentric_count})")
                return True
        
        # 4. REMOVED CONTAINMENT LOGIC
        # The containment logic was too speculative and filtered legitimate features
        
        # DEFAULT: Preserve circles as features unless we have STRONG structural evidence
        return False
    
    def _get_next_id(self) -> int:
        """Get next sequential feature ID."""
        self._feature_counter += 1
        return self._feature_counter