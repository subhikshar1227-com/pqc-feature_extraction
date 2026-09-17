"""
Semantic Feature Grouping

Groups geometric entities into semantic physical inspection features using general 
geometric relationships. Addresses cases where multiple geometric entities represent 
the same physical feature (e.g., concentric circles, nested boundaries).

This layer sits between raw geometry detection and significance filtering:
Raw Geometry → Expected Features → SEMANTIC GROUPING → Physical Features

Uses only geometric relationships:
- Concentricity and nesting
- Shared centers and containment
- Radial relationships and spacing
- Closed-loop topology
- Connectivity patterns
- Geometric adjacency

No product-specific rules, coordinates, or hardcoded counts.
"""

import logging
import math
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from .feature_types import ExpectedFeature, FeatureType
from ..dxf.entity_models import Point2D, GeometricRelationships
from ..config import (
    FEATURE_DUPLICATE_CENTER_TOLERANCE,
    CONCENTRICITY_TOLERANCE,
    MAIN_BODY_MIN_RADIUS,
    RECONSTRUCTION_MIN_CONFIDENCE,
    # Semantic grouping configuration
    SEMANTIC_NESTING_RATIO_THRESHOLD,
    SEMANTIC_ADJACENCY_TOLERANCE,
    SEMANTIC_CONTAINMENT_TOLERANCE,
    SEMANTIC_HOLE_SIZE_IDEAL_MIN,
    SEMANTIC_HOLE_SIZE_IDEAL_MAX,
    SEMANTIC_HOLE_SIZE_ACCEPTABLE_MIN,
    SEMANTIC_HOLE_SIZE_ACCEPTABLE_MAX,
    SEMANTIC_HOLE_SIZE_BODY_THRESHOLD,
    SEMANTIC_CIRCLE_SIZE_SMALL_MAX,
    SEMANTIC_CIRCLE_SIZE_MEDIUM_MAX,
    SEMANTIC_CIRCLE_SIZE_LARGE_MAX,
    SEMANTIC_CIRCLE_SIZE_BODY_THRESHOLD,
    SEMANTIC_CIRCLE_SIZE_MIN_INSPECTION,
    # Algorithmic scoring values
    SEMANTIC_HOLE_SIZE_SCORE_IDEAL,
    SEMANTIC_HOLE_SIZE_SCORE_ACCEPTABLE,
    SEMANTIC_HOLE_SIZE_SCORE_SMALL,
    SEMANTIC_HOLE_SIZE_SCORE_LARGE,
    SEMANTIC_CIRCLE_SIZE_SCORE_SMALL,
    SEMANTIC_CIRCLE_SIZE_SCORE_MEDIUM,
    SEMANTIC_CIRCLE_SIZE_SCORE_LARGE,
    SEMANTIC_CIRCLE_SIZE_SCORE_VERY_LARGE,
    SEMANTIC_CIRCLE_SOURCE_SCORE_EXPLICIT,
    SEMANTIC_CIRCLE_SOURCE_SCORE_RECONSTRUCTED,
    SEMANTIC_CIRCLE_SOURCE_SCORE_OTHER,
    # Analysis thresholds
    SEMANTIC_CONCENTRIC_MIN_COUNT_FOR_POSITION,
    SEMANTIC_GEOMETRIC_CONTEXT_MIN_RADII,
    SEMANTIC_PERCENTILE_MIN_COUNT_75,
    SEMANTIC_PERCENTILE_MIN_COUNT_90,
    SEMANTIC_SIZE_WEIGHT,
    SEMANTIC_EVIDENCE_WEIGHT,
    SEMANTIC_POSITION_WEIGHT,
    SEMANTIC_CIRCLE_SIZE_WEIGHT,
    SEMANTIC_CIRCLE_CONFIDENCE_WEIGHT,
    SEMANTIC_CIRCLE_SOURCE_WEIGHT,
    SEMANTIC_BODY_ARC_COUNT_THRESHOLD,
    SEMANTIC_BODY_RELATIVE_SIZE_CONFIDENCE,
    SEMANTIC_BODY_ABSOLUTE_SIZE_CONFIDENCE,
    SEMANTIC_BODY_RECONSTRUCTION_CONFIDENCE,
    SEMANTIC_INSPECTION_FEATURE_CONFIDENCE,
    # Percentile calculation factors
    PERCENTILE_75_FACTOR, PERCENTILE_90_FACTOR,
    # Semantic grouping confidence boost parameters
    SEMANTIC_GROUPING_CONFIDENCE_BOOST_MAX, SEMANTIC_GROUPING_CONFIDENCE_BOOST_PER_FEATURE,
    # Hardcoded comparison thresholds
    SEMANTIC_GROUP_MIN_RADII_FOR_SPAN, SEMANTIC_GROUP_MIN_FEATURES_FOR_BOOST,
    # Position scoring defaults
    SEMANTIC_DEFAULT_POSITION_SCORE
)

logger = logging.getLogger(__name__)


@dataclass
class GeometricGroup:
    """Represents a group of geometric features that may belong to the same physical feature."""
    
    group_id: str
    center: Point2D
    features: List[ExpectedFeature] = field(default_factory=list)
    grouping_reason: str = ""
    geometric_evidence: Dict[str, Any] = field(default_factory=dict)
    is_concentric: bool = False
    is_nested: bool = False
    radius_range: Tuple[float, float] = (0.0, 0.0)
    
    def add_feature(self, feature: ExpectedFeature, reason: str = ""):
        """Add a feature to this group."""
        self.features.append(feature)
        if reason and self.grouping_reason:
            self.grouping_reason += f"; {reason}"
        elif reason:
            self.grouping_reason = reason
        
        # Update radius range for circular features
        if feature.radius is not None:
            if not self.features or len(self.features) == 1:
                self.radius_range = (feature.radius, feature.radius)
            else:
                min_r, max_r = self.radius_range
                self.radius_range = (min(min_r, feature.radius), max(max_r, feature.radius))


class SemanticFeatureGrouper:
    """Groups geometric features into semantic physical inspection features."""
    
    def __init__(self):
        """Initialize semantic grouper with geometry-driven parameters."""
        self._group_counter = 0
        
    def group_features_semantically(self, features: List[ExpectedFeature],
                                  relationships: Optional[GeometricRelationships] = None) -> List[ExpectedFeature]:
        """
        Group features semantically using geometric relationships.
        
        Pipeline:
        1. Identify geometric groups (concentric, nested, connected)
        2. Analyze each group for semantic unity
        3. Create representative features for each physical feature
        4. Preserve traceability to original geometry
        
        Args:
            features: Raw detected features
            relationships: Geometric relationships from DXF analysis
            
        Returns:
            List of semantic features (may be fewer than input due to grouping)
        """
        logger.info(f"Semantic grouping: analyzing {len(features)} raw features")
        
        if not features:
            return []
        
        # Step 1: Identify geometric groups based on spatial relationships
        geometric_groups = self._identify_geometric_groups(features)
        
        logger.info(f"Found {len(geometric_groups)} geometric groups")
        
        # Step 2: Analyze each group for semantic unity
        semantic_features = []
        grouping_stats = {
            "single_feature_groups": 0,
            "multi_feature_groups": 0,
            "concentric_groups": 0,
            "nested_groups": 0,
            "features_grouped": 0
        }
        
        for group in geometric_groups:
            if len(group.features) == 1:
                # Single feature - pass through unchanged
                semantic_features.append(group.features[0])
                grouping_stats["single_feature_groups"] += 1
                
            else:
                # Multiple features - create semantic representation
                semantic_feature = self._create_semantic_feature(group, relationships)
                semantic_features.append(semantic_feature)
                
                grouping_stats["multi_feature_groups"] += 1
                grouping_stats["features_grouped"] += len(group.features)
                
                if group.is_concentric:
                    grouping_stats["concentric_groups"] += 1
                if group.is_nested:
                    grouping_stats["nested_groups"] += 1
                
                logger.debug(f"Grouped {len(group.features)} features into semantic feature: "
                           f"{semantic_feature.feature_id} ({group.grouping_reason})")
        
        # Log grouping summary
        logger.info(f"Semantic grouping results:")
        logger.info(f"  Input features: {len(features)}")
        logger.info(f"  Output features: {len(semantic_features)}")
        logger.info(f"  Single-feature groups: {grouping_stats['single_feature_groups']}")
        logger.info(f"  Multi-feature groups: {grouping_stats['multi_feature_groups']}")
        logger.info(f"  Concentric groups: {grouping_stats['concentric_groups']}")
        logger.info(f"  Nested groups: {grouping_stats['nested_groups']}")
        logger.info(f"  Features combined: {grouping_stats['features_grouped']}")
        
        return semantic_features
    def _identify_geometric_groups(self, features: List[ExpectedFeature]) -> List[GeometricGroup]:
        """
        Identify geometric groups based on spatial and geometric relationships.
        
        Groups features that share:
        - Same center location (concentric)
        - Nested/containment relationships
        - Connected topology
        - Geometric adjacency
        """
        groups = []
        assigned_features = set()
        
        for i, feature in enumerate(features):
            if i in assigned_features:
                continue
                
            # Create new group starting with this feature
            group = self._create_geometric_group(feature)
            assigned_features.add(i)
            
            # Find all features that belong to the same geometric group
            for j, other_feature in enumerate(features[i + 1:], i + 1):
                if j in assigned_features:
                    continue
                
                # Check if features should be grouped together
                grouping_analysis = self._analyze_grouping_relationship(feature, other_feature)
                
                if grouping_analysis["should_group"]:
                    group.add_feature(other_feature, grouping_analysis["reason"])
                    assigned_features.add(j)
                    
                    # Update group properties based on relationships
                    if grouping_analysis["relationship_type"] == "concentric":
                        group.is_concentric = True
                    elif grouping_analysis["relationship_type"] == "nested":
                        group.is_nested = True
            
            # Finalize group analysis
            self._finalize_group_analysis(group)
            groups.append(group)
        
        return groups
    
    def _create_geometric_group(self, initial_feature: ExpectedFeature) -> GeometricGroup:
        """Create a new geometric group starting with the given feature."""
        self._group_counter += 1
        
        group = GeometricGroup(
            group_id=f"group_{self._group_counter}",
            center=initial_feature.center
        )
        group.add_feature(initial_feature, "initial_feature")
        
        return group
    
    def _analyze_grouping_relationship(self, feature1: ExpectedFeature, 
                                    feature2: ExpectedFeature) -> Dict[str, Any]:
        """
        Analyze if two features should be grouped together based on geometric relationships.
        
        Returns analysis with should_group flag and relationship details.
        """
        analysis = {
            "should_group": False,
            "reason": "",
            "relationship_type": "none",
            "geometric_evidence": {}
        }
        
        # Calculate center distance
        center_distance = feature1.center.distance_to(feature2.center)
        analysis["geometric_evidence"]["center_distance"] = center_distance
        
        # Check for concentric relationship (same center)
        if center_distance <= CONCENTRICITY_TOLERANCE:
            analysis["should_group"] = True
            analysis["relationship_type"] = "concentric"
            analysis["reason"] = f"concentric_centers_dist_{center_distance:.2f}"
            
            # Additional analysis for concentric features
            if feature1.radius is not None and feature2.radius is not None:
                radius_ratio = max(feature1.radius, feature2.radius) / min(feature1.radius, feature2.radius)
                analysis["geometric_evidence"]["radius_ratio"] = radius_ratio
                
                # Check if they represent nested boundaries (different scales)
                if radius_ratio > SEMANTIC_NESTING_RATIO_THRESHOLD:  # Significant size difference
                    analysis["relationship_type"] = "nested"
                    analysis["reason"] += f"_nested_ratio_{radius_ratio:.1f}"
            
            return analysis
        
        # Check for geometric adjacency (touching or very close boundaries)
        if self._are_geometrically_adjacent(feature1, feature2):
            analysis["should_group"] = True
            analysis["relationship_type"] = "adjacent"
            analysis["reason"] = f"adjacent_boundaries"
            return analysis
        
        # Check for containment relationship
        containment_analysis = self._analyze_containment(feature1, feature2)
        if containment_analysis["is_contained"]:
            analysis["should_group"] = True
            analysis["relationship_type"] = "containment"
            analysis["reason"] = containment_analysis["reason"]
            analysis["geometric_evidence"].update(containment_analysis["evidence"])
            return analysis
        
        # No grouping relationship found
        return analysis
    def _are_geometrically_adjacent(self, feature1: ExpectedFeature, 
                                   feature2: ExpectedFeature) -> bool:
        """Check if two features are geometrically adjacent (touching or very close)."""
        
        if feature1.radius is None or feature2.radius is None:
            return False
        
        center_distance = feature1.center.distance_to(feature2.center)
        
        # Check if circles are touching or overlapping
        radius_sum = feature1.radius + feature2.radius
        radius_diff = abs(feature1.radius - feature2.radius)
        
        # Adjacent if distance is approximately equal to radius sum (touching externally)
        # or approximately equal to radius difference (one inside the other)
        tolerance = SEMANTIC_ADJACENCY_TOLERANCE  # mm tolerance for geometric adjacency
        
        externally_touching = abs(center_distance - radius_sum) <= tolerance
        internally_touching = abs(center_distance - radius_diff) <= tolerance
        
        return externally_touching or internally_touching
    
    def _analyze_containment(self, feature1: ExpectedFeature, 
                           feature2: ExpectedFeature) -> Dict[str, Any]:
        """Analyze if features have a containment relationship."""
        
        analysis = {
            "is_contained": False,
            "reason": "",
            "evidence": {}
        }
        
        if feature1.radius is None or feature2.radius is None:
            return analysis
        
        center_distance = feature1.center.distance_to(feature2.center)
        analysis["evidence"]["center_distance"] = center_distance
        
        # Check if one circle is contained within the other
        larger_feature = feature1 if feature1.radius > feature2.radius else feature2
        smaller_feature = feature2 if feature1.radius > feature2.radius else feature1
        
        # Smaller circle is contained if center distance + smaller radius < larger radius
        containment_distance = center_distance + smaller_feature.radius
        containment_margin = larger_feature.radius - containment_distance
        
        analysis["evidence"]["containment_margin"] = containment_margin
        
        # Allow small tolerance for floating point and measurement precision
        tolerance = SEMANTIC_CONTAINMENT_TOLERANCE  # mm
        
        if containment_margin >= -tolerance:
            analysis["is_contained"] = True
            analysis["reason"] = f"contained_margin_{containment_margin:.2f}"
        
        return analysis
    
    def _finalize_group_analysis(self, group: GeometricGroup) -> None:
        """Finalize analysis of a geometric group after all features are added."""
        
        if len(group.features) <= 1:
            return
        
        # Analyze the complete group properties
        radii = [f.radius for f in group.features if f.radius is not None]
        
        if radii:
            group.radius_range = (min(radii), max(radii))
            
            # Update geometric evidence
            group.geometric_evidence = {
                "feature_count": len(group.features),
                "radius_range": group.radius_range,
                "radius_span": max(radii) - min(radii) if len(radii) > SEMANTIC_GROUP_MIN_RADII_FOR_SPAN else 0,
                "center_variations": self._calculate_center_variations(group.features)
            }
            
            # Determine if this is a concentric/nested arrangement
            if len(radii) > SEMANTIC_GEOMETRIC_CONTEXT_MIN_RADII:
                radius_ratios = []
                sorted_radii = sorted(radii)
                for i in range(len(sorted_radii) - 1):
                    ratio = sorted_radii[i + 1] / sorted_radii[i]
                    radius_ratios.append(ratio)
                
                group.geometric_evidence["radius_ratios"] = radius_ratios
                
                # Concentric with nested scales if significant ratio differences
                if any(ratio > SEMANTIC_NESTING_RATIO_THRESHOLD for ratio in radius_ratios):
                    group.is_nested = True
    
    def _calculate_center_variations(self, features: List[ExpectedFeature]) -> Dict[str, float]:
        """Calculate variations in center positions within a group."""
        
        if len(features) <= 1:
            return {"max_deviation": 0.0, "average_deviation": 0.0}
        
        # Calculate average center
        avg_x = sum(f.center.x for f in features) / len(features)
        avg_y = sum(f.center.y for f in features) / len(features)
        avg_center = Point2D(avg_x, avg_y)
        
        # Calculate deviations
        deviations = [f.center.distance_to(avg_center) for f in features]
        
        return {
            "max_deviation": max(deviations),
            "average_deviation": sum(deviations) / len(deviations)
        }
    def _create_semantic_feature(self, group: GeometricGroup,
                               relationships: Optional[GeometricRelationships] = None) -> ExpectedFeature:
        """
        Create a semantic feature that represents the physical inspection feature
        from a group of geometric features.
        
        Selection strategy:
        1. For concentric circles: select the most inspection-relevant size
        2. For nested features: prefer the primary functional boundary
        3. For mixed types: prefer through_holes over circles
        4. Preserve full traceability to source geometry
        """
        
        if len(group.features) == 1:
            return group.features[0]
        
        # Analyze group composition
        circles = [f for f in group.features if f.feature_type == FeatureType.CIRCLE]
        holes = [f for f in group.features if f.feature_type == FeatureType.THROUGH_HOLE]
        
        # Selection logic based on group analysis
        representative_feature = self._select_group_representative(group, circles, holes)
        
        # Create enhanced semantic feature with full traceability
        semantic_feature = self._create_enhanced_feature(representative_feature, group)
        
        return semantic_feature
    
    def _select_group_representative(self, group: GeometricGroup,
                                   circles: List[ExpectedFeature],
                                   holes: List[ExpectedFeature]) -> ExpectedFeature:
        """Select the best representative feature from a group."""
        
        # Preference 1: Through holes over circles (functional significance)
        if holes:
            if len(holes) == 1:
                return holes[0]
            else:
                # Multiple holes - select most appropriate size
                return self._select_best_hole_size(holes, group)
        
        # Preference 2: Circles - select most inspection-relevant
        if circles:
            if len(circles) == 1:
                return circles[0]
            else:
                # Multiple circles - select most appropriate size
                return self._select_best_circle_size(circles, group)
        
        # Fallback: highest confidence
        return max(group.features, key=lambda f: f.confidence)
    
    def _select_best_hole_size(self, holes: List[ExpectedFeature], 
                              group: GeometricGroup) -> ExpectedFeature:
        """Select the most appropriate hole size from concentric holes."""
        
        # For concentric holes, prefer intermediate sizes that are most likely
        # to represent the actual inspection feature rather than construction geometry
        
        if not holes or not all(h.radius for h in holes):
            return max(holes, key=lambda h: h.confidence)
        
        # Sort by radius
        sorted_holes = sorted(holes, key=lambda h: h.radius)
        
        # Geometric analysis: prefer holes that are not too small or too large
        def hole_appropriateness_score(hole):
            radius = hole.radius
            confidence = hole.confidence
            
            # Size appropriateness (prefer medium-sized holes for inspection)
            if SEMANTIC_HOLE_SIZE_IDEAL_MIN <= radius <= SEMANTIC_HOLE_SIZE_IDEAL_MAX:  # Typical inspection hole range
                size_score = SEMANTIC_HOLE_SIZE_SCORE_IDEAL
            elif SEMANTIC_HOLE_SIZE_ACCEPTABLE_MIN <= radius <= SEMANTIC_HOLE_SIZE_ACCEPTABLE_MAX:  # Acceptable range
                size_score = SEMANTIC_HOLE_SIZE_SCORE_ACCEPTABLE
            elif radius > SEMANTIC_HOLE_SIZE_BODY_THRESHOLD:  # Very large (possibly body geometry)
                size_score = SEMANTIC_HOLE_SIZE_SCORE_LARGE
            else:  # Very small (possibly construction detail)
                size_score = SEMANTIC_HOLE_SIZE_SCORE_SMALL
            
            # Evidence quality
            evidence_score = confidence
            
            # Position in concentric series (prefer middle sizes)
            if len(sorted_holes) > SEMANTIC_CONCENTRIC_MIN_COUNT_FOR_POSITION:
                hole_index = sorted_holes.index(hole)
                position_score = 1.0 - abs(hole_index - len(sorted_holes) / 2) / len(sorted_holes)
            else:
                position_score = SEMANTIC_DEFAULT_POSITION_SCORE
            
            return size_score * SEMANTIC_SIZE_WEIGHT + evidence_score * SEMANTIC_EVIDENCE_WEIGHT + position_score * SEMANTIC_POSITION_WEIGHT
        
        best_hole = max(holes, key=hole_appropriateness_score)
        
        logger.debug(f"Selected hole radius {best_hole.radius:.2f} from {len(holes)} concentric holes "
                    f"in group {group.group_id}")
        
        return best_hole
    
    def _select_best_circle_size(self, circles: List[ExpectedFeature], 
                                group: GeometricGroup) -> ExpectedFeature:
        """Select the most appropriate circle size from concentric circles."""
        
        if not circles or not all(c.radius for c in circles):
            return max(circles, key=lambda c: c.confidence)
        
        # For concentric circles, apply similar logic as holes but with different criteria
        def circle_appropriateness_score(circle):
            radius = circle.radius
            confidence = circle.confidence
            
            # Size appropriateness for inspection features
            if SEMANTIC_CIRCLE_SIZE_MIN_INSPECTION <= radius <= SEMANTIC_CIRCLE_SIZE_SMALL_MAX:  # Small-medium circles (likely inspection features)
                size_score = SEMANTIC_CIRCLE_SIZE_SCORE_SMALL
            elif SEMANTIC_CIRCLE_SIZE_SMALL_MAX < radius <= SEMANTIC_CIRCLE_SIZE_MEDIUM_MAX:  # Medium-large circles
                size_score = SEMANTIC_CIRCLE_SIZE_SCORE_MEDIUM
            elif SEMANTIC_CIRCLE_SIZE_MEDIUM_MAX < radius <= SEMANTIC_CIRCLE_SIZE_LARGE_MAX:  # Large circles (possibly body boundaries)
                size_score = SEMANTIC_CIRCLE_SIZE_SCORE_LARGE
            else:  # Very large or very small
                size_score = SEMANTIC_CIRCLE_SIZE_SCORE_VERY_LARGE
            
            # Prefer explicit circles over reconstructed ones for inspection features
            if circle.source_type == "explicit_circle":
                source_score = SEMANTIC_CIRCLE_SOURCE_SCORE_EXPLICIT
            elif circle.source_type == "reconstructed_circle":
                source_score = SEMANTIC_CIRCLE_SOURCE_SCORE_RECONSTRUCTED
            else:
                source_score = SEMANTIC_CIRCLE_SOURCE_SCORE_OTHER
            
            return size_score * SEMANTIC_CIRCLE_SIZE_WEIGHT + confidence * SEMANTIC_CIRCLE_CONFIDENCE_WEIGHT + source_score * SEMANTIC_CIRCLE_SOURCE_WEIGHT
        
        best_circle = max(circles, key=circle_appropriateness_score)
        
        logger.debug(f"Selected circle radius {best_circle.radius:.2f} from {len(circles)} concentric circles "
                    f"in group {group.group_id}")
        
        return best_circle
    def _create_enhanced_feature(self, representative: ExpectedFeature, 
                               group: GeometricGroup) -> ExpectedFeature:
        """
        Create an enhanced semantic feature with full traceability to the group.
        
        Preserves all source information while indicating semantic grouping was applied.
        """
        
        # Collect all source entity IDs from the group
        all_source_entities = []
        for feature in group.features:
            all_source_entities.extend(feature.source_entity_ids)
        
        # Remove duplicates while preserving order
        unique_source_entities = []
        seen = set()
        for entity_id in all_source_entities:
            if entity_id not in seen:
                unique_source_entities.append(entity_id)
                seen.add(entity_id)
        
        # Create enhanced feature based on representative
        enhanced_feature = ExpectedFeature(
            feature_id=f"semantic_{group.group_id}_{representative.feature_id}",
            feature_type=representative.feature_type,
            confidence=representative.confidence,
            center=representative.center,
            radius=representative.radius,
            diameter=representative.diameter,
            source_entity_ids=unique_source_entities,
            source_type=f"semantic_group_{representative.source_type}",
            detection_evidence=representative.detection_evidence.copy(),
            geometric_properties=representative.geometric_properties.copy()
        )
        
        # Add semantic grouping information to detection evidence
        enhanced_feature.detection_evidence.update({
            "semantic_grouping": {
                "group_id": group.group_id,
                "grouping_reason": group.grouping_reason,
                "original_feature_count": len(group.features),
                "grouped_features": [f.feature_id for f in group.features],
                "is_concentric": group.is_concentric,
                "is_nested": group.is_nested,
                "radius_range": group.radius_range,
                "geometric_evidence": group.geometric_evidence,
                "representative_selection": {
                    "selected_feature": representative.feature_id,
                    "selection_reason": "most_inspection_appropriate"
                }
            }
        })
        
        # Update geometric properties to include group information
        enhanced_feature.geometric_properties.update({
            "semantic_grouping_applied": True,
            "original_geometry_count": len(group.features),
            "concentric_arrangement": group.is_concentric,
            "nested_arrangement": group.is_nested
        })
        
        # Adjust confidence based on grouping evidence
        # Higher confidence when multiple geometric elements confirm the same feature
        if len(group.features) > SEMANTIC_GROUP_MIN_FEATURES_FOR_BOOST:
            # Boost confidence for features confirmed by multiple geometric elements
            grouping_confidence_boost = min(SEMANTIC_GROUPING_CONFIDENCE_BOOST_MAX, 
                                           SEMANTIC_GROUPING_CONFIDENCE_BOOST_PER_FEATURE * len(group.features))
            enhanced_feature.confidence = min(1.0, enhanced_feature.confidence + grouping_confidence_boost)
            
            enhanced_feature.detection_evidence["semantic_grouping"]["confidence_adjustment"] = grouping_confidence_boost
        
        logger.debug(f"Created semantic feature {enhanced_feature.feature_id} from {len(group.features)} "
                    f"geometric features ({group.grouping_reason})")
        
        return enhanced_feature
    
    def _distinguish_body_from_features(self, features: List[ExpectedFeature]) -> Tuple[List[ExpectedFeature], List[ExpectedFeature]]:
        """
        Distinguish body/outline geometry from internal inspection features using geometric analysis.
        
        Body geometry characteristics:
        - Very large circles (likely outer boundaries)
        - Reconstructed from many connected arcs
        - Located at geometric extremes
        - Low feature-appropriateness scores
        
        Inspection features:
        - Moderate sizes (typical hole/boss dimensions)
        - Clear geometric closure
        - Internal positions
        - High feature-appropriateness scores
        """
        
        if not features:
            return [], []
        
        inspection_features = []
        body_geometry = []
        
        # Calculate geometric context for analysis
        geometric_context = self._calculate_geometric_context(features)
        
        for feature in features:
            classification = self._classify_feature_context(feature, geometric_context)
            
            if classification["is_body_geometry"]:
                body_geometry.append(feature)
                logger.debug(f"Classified {feature.feature_id} as body geometry: {classification['reason']}")
            else:
                inspection_features.append(feature)
                logger.debug(f"Classified {feature.feature_id} as inspection feature: {classification['reason']}")
        
        logger.info(f"Body/feature distinction: {len(inspection_features)} inspection features, "
                   f"{len(body_geometry)} body geometry elements")
        
        return inspection_features, body_geometry
    
    def _calculate_geometric_context(self, features: List[ExpectedFeature]) -> Dict[str, Any]:
        """Calculate geometric context for body vs feature classification."""
        
        radii = [f.radius for f in features if f.radius is not None]
        centers = [f.center for f in features]
        
        if not radii or not centers:
            return {}
        
        # Calculate bounds and statistics
        xs = [c.x for c in centers]
        ys = [c.y for c in centers]
        
        context = {
            "radius_stats": {
                "min": min(radii),
                "max": max(radii),
                "median": sorted(radii)[len(radii) // 2],
                "q75": sorted(radii)[int(len(radii) * PERCENTILE_75_FACTOR)] if len(radii) > SEMANTIC_PERCENTILE_MIN_COUNT_75 else max(radii),
                "q90": sorted(radii)[int(len(radii) * PERCENTILE_90_FACTOR)] if len(radii) > SEMANTIC_PERCENTILE_MIN_COUNT_90 else max(radii)
            },
            "spatial_bounds": {
                "x_range": (min(xs), max(xs)),
                "y_range": (min(ys), max(ys)),
                "center": Point2D(sum(xs) / len(xs), sum(ys) / len(ys))
            },
            "feature_count": len(features)
        }
        
        return context
    
    def _classify_feature_context(self, feature: ExpectedFeature, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify feature as body geometry vs inspection feature based on geometric context."""
        
        classification = {
            "is_body_geometry": False,
            "reason": "",
            "confidence": 0.0
        }
        
        if feature.radius is None or not context:
            classification["reason"] = "insufficient_geometric_data"
            return classification
        
        radius = feature.radius
        radius_stats = context.get("radius_stats", {})
        
        # Size-based classification
        if "q90" in radius_stats and radius >= radius_stats["q90"]:
            # Very large relative to other features - likely body geometry
            classification["is_body_geometry"] = True
            classification["reason"] = f"very_large_relative_size_{radius:.1f}_vs_q90_{radius_stats['q90']:.1f}"
            classification["confidence"] = SEMANTIC_BODY_RELATIVE_SIZE_CONFIDENCE
            return classification
        
        # Absolute size thresholds
        if radius > SEMANTIC_CIRCLE_SIZE_BODY_THRESHOLD:  # Very large absolute size - likely body boundary
            classification["is_body_geometry"] = True
            classification["reason"] = f"very_large_absolute_size_{radius:.1f}"
            classification["confidence"] = SEMANTIC_BODY_ABSOLUTE_SIZE_CONFIDENCE
            return classification
        
        # Source type analysis
        if (feature.source_type == "reconstructed_circle" and 
            len(feature.source_entity_ids) > SEMANTIC_BODY_ARC_COUNT_THRESHOLD):  # Reconstructed from many arcs
            classification["is_body_geometry"] = True
            classification["reason"] = f"reconstructed_from_many_arcs_{len(feature.source_entity_ids)}"
            classification["confidence"] = SEMANTIC_BODY_RECONSTRUCTION_CONFIDENCE
            return classification
        
        # Default: inspection feature
        classification["is_body_geometry"] = False
        classification["reason"] = f"inspection_feature_size_{radius:.1f}"
        classification["confidence"] = SEMANTIC_INSPECTION_FEATURE_CONFIDENCE
        
        return classification