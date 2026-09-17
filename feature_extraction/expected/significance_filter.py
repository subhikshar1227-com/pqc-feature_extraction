"""
Significance Filter

Filters detected features using strict geometry-grounded rules to identify
only prominent, significant features. Prevents double-counting and ensures
feature uniqueness through comprehensive deduplication.
"""

import logging
import math
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

from .feature_types import ExpectedFeature, FeatureType
from ..config import (
    MIN_FEATURE_CONFIDENCE, SIGNIFICANCE_MIN_RADIUS,
    FEATURE_DUPLICATE_CENTER_TOLERANCE, FEATURE_DUPLICATE_RADIUS_TOLERANCE,
    FEATURE_DUPLICATE_RADIUS_RELATIVE, PATTERN_MIN_SIMILAR_COUNT,
    # Additional significance filter parameters
    THROUGH_HOLE_PREFERENCE_CONFIDENCE,
    SIGNIFICANCE_LOCATION_TOLERANCE,
    SIGNIFICANCE_HOLE_SIZE_SCORE_VERY_SMALL, SIGNIFICANCE_HOLE_SIZE_SCORE_SMALL,
    SIGNIFICANCE_HOLE_SIZE_SCORE_MEDIUM, SIGNIFICANCE_HOLE_SIZE_SCORE_LARGE,
    SIGNIFICANCE_HOLE_SIZE_SCORE_VERY_LARGE,
    SIGNIFICANCE_CIRCLE_SIZE_SCORE_IDEAL, SIGNIFICANCE_CIRCLE_SIZE_SCORE_ACCEPTABLE,
    SIGNIFICANCE_CIRCLE_SIZE_SCORE_SMALL, SIGNIFICANCE_CIRCLE_SIZE_SCORE_POOR,
    SIGNIFICANCE_CONFIDENCE_WEIGHT_FACTOR,
    # Reference thresholds for scoring
    HOLE_SIZE_VERY_SMALL_THRESHOLD, HOLE_SIZE_SMALL_THRESHOLD, HOLE_SIZE_MEDIUM_THRESHOLD, HOLE_SIZE_LARGE_THRESHOLD,
    FEATURE_IDEAL_MIN_RADIUS, FEATURE_IDEAL_MAX_RADIUS, FEATURE_ACCEPTABLE_MIN_RADIUS, FEATURE_ACCEPTABLE_MAX_RADIUS,
    # Significance filter specific thresholds
    SIGNIFICANCE_FILTER_MIN_FEATURES_FOR_DEDUP, SIGNIFICANCE_FILTER_PERCENTAGE_MULTIPLIER
)

logger = logging.getLogger(__name__)


class SignificanceFilter:
    """Filters features using strict geometry-grounded rules and comprehensive deduplication."""
    
    def __init__(self, 
                 min_confidence: float = MIN_FEATURE_CONFIDENCE,
                 min_radius: float = SIGNIFICANCE_MIN_RADIUS):
        """
        Initialize significance filter with strict geometric criteria.
        
        Args:
            min_confidence: Minimum confidence for feature inclusion
            min_radius: Minimum radius for geometric significance (mm)
        """
        self.min_confidence = min_confidence
        self.min_radius = min_radius
    
    def filter_significant_features(self, features: List[ExpectedFeature]) -> List[ExpectedFeature]:
        """
        Filter features using strict significance criteria and comprehensive deduplication.
        
        Filtering Pipeline:
        1. Confidence threshold (strict minimum)
        2. Geometric significance (size-based)  
        3. Comprehensive deduplication (prevent double-counting)
        4. Final validation (geometric completeness)
        """
        logger.debug(f"Filtering {len(features)} features with strict significance criteria")
        
        if not features:
            return []
        
        # Track filtering stages for diagnostics
        filtering_stages = {
            "input": len(features),
            "confidence_filtered": 0,
            "size_filtered": 0,
            "deduplicated": 0,
            "final": 0
        }
        
        rejection_log = {
            "low_confidence": 0,
            "insufficient_size": 0,
            "duplicate": 0,
            "incomplete_geometry": 0
        }
        
        # Stage 1: Apply strict confidence threshold
        confidence_filtered = self._apply_confidence_filter(features, rejection_log)
        filtering_stages["confidence_filtered"] = len(confidence_filtered)
        
        # Stage 2: Apply geometric significance filter
        size_filtered = self._apply_geometric_significance_filter(confidence_filtered, rejection_log)
        filtering_stages["size_filtered"] = len(size_filtered)
        
        # Stage 3: Comprehensive deduplication to prevent double-counting
        deduplicated = self._apply_comprehensive_deduplication(size_filtered, rejection_log)
        filtering_stages["deduplicated"] = len(deduplicated)
        
        # Stage 4: Final validation
        final_features = self._apply_final_validation(deduplicated, rejection_log)
        filtering_stages["final"] = len(final_features)
        
        self._log_filtering_summary(filtering_stages, rejection_log)
        
        return final_features
    
    def _apply_confidence_filter(self, features: List[ExpectedFeature],
                               rejection_log: Dict[str, int]) -> List[ExpectedFeature]:
        """Apply strict confidence threshold filter."""
        
        filtered = []
        
        for feature in features:
            if feature.confidence >= self.min_confidence:
                filtered.append(feature)
            else:
                rejection_log["low_confidence"] += 1
                logger.debug(f"Rejected {feature.feature_id} for low confidence: "
                           f"{feature.confidence:.3f} < {self.min_confidence}")
        
        return filtered
    
    def _apply_geometric_significance_filter(self, features: List[ExpectedFeature],
                                           rejection_log: Dict[str, int]) -> List[ExpectedFeature]:
        """Apply geometry-based significance filter using absolute size criteria."""
        
        filtered = []
        
        for feature in features:
            # Check size significance based on feature type
            if feature.source_type == "square_hole_from_lines":
                # Square holes - check minimum dimension
                width = feature.geometric_properties.get("width", 0)
                height = feature.geometric_properties.get("height", 0)
                min_dimension = min(width, height) if width and height else 0
                
                if min_dimension < self.min_radius:  # Use same minimum as circles
                    rejection_log["insufficient_size"] += 1
                    logger.debug(f"Rejected {feature.feature_id} for insufficient size: "
                               f"min dimension {min_dimension:.2f} < {self.min_radius}")
                    continue
            else:
                # Circular features - check radius
                if feature.radius is not None and feature.radius < self.min_radius:
                    rejection_log["insufficient_size"] += 1
                    logger.debug(f"Rejected {feature.feature_id} for insufficient size: "
                               f"radius {feature.radius:.2f} < {self.min_radius}")
                    continue
            
            # Check geometric completeness
            if not self._has_complete_geometry(feature):
                rejection_log["incomplete_geometry"] += 1
                logger.debug(f"Rejected {feature.feature_id} for incomplete geometry")
                continue
            
            filtered.append(feature)
        
        return filtered
    
    def _apply_comprehensive_deduplication(self, features: List[ExpectedFeature],
                                         rejection_log: Dict[str, int]) -> List[ExpectedFeature]:
        """
        Apply comprehensive deduplication to prevent double-counting of the same geometry.
        
        Uses strict geometric criteria to identify identical features that may have been
        detected through different methods (e.g., explicit circle AND reconstructed circle).
        """
        
        if len(features) <= SIGNIFICANCE_FILTER_MIN_FEATURES_FOR_DEDUP:
            return features
        
        # Group features that represent the same geometry (strict duplicates)
        duplicate_groups = self._identify_duplicate_groups(features)
        
        deduplicated = []
        duplicates_removed = 0
        
        for group in duplicate_groups:
            # Select the best representative from each duplicate group
            best_feature = self._select_best_representative(group)
            deduplicated.append(best_feature)
            
            duplicates_removed += len(group) - 1
            
            if len(group) > 1:
                logger.debug(f"Deduplicated group of {len(group)} features, "
                           f"selected {best_feature.feature_id} as representative")
        
        # Apply concentric circle filtering for box-type files
        deduplicated = self._filter_concentric_circles(deduplicated, rejection_log)
        
        rejection_log["duplicate"] += duplicates_removed
        return deduplicated
    
    def _apply_final_validation(self, features: List[ExpectedFeature],
                              rejection_log: Dict[str, int]) -> List[ExpectedFeature]:
        """Apply final validation checks before output."""
        
        validated = []
        
        for feature in features:
            # Final sanity checks
            if self._passes_final_validation(feature):
                validated.append(feature)
            else:
                rejection_log["incomplete_geometry"] += 1
                logger.debug(f"Failed final validation: {feature.feature_id}")
        
        return validated
    
    def _identify_duplicate_groups(self, features: List[ExpectedFeature]) -> List[List[ExpectedFeature]]:
        """
        Identify groups of features that represent the same geometric entity.
        
        Uses strict geometric criteria from config.py to determine duplicates.
        """
        
        # Track which features have been assigned to groups
        assigned_features = set()
        duplicate_groups = []
        
        for i, feature1 in enumerate(features):
            if i in assigned_features:
                continue
            
            # Start a new group with this feature
            current_group = [feature1]
            assigned_features.add(i)
            
            # Find all features that are duplicates of feature1
            for j, feature2 in enumerate(features[i + 1:], i + 1):
                if j in assigned_features:
                    continue
                
                if self._are_strict_geometric_duplicates(feature1, feature2):
                    current_group.append(feature2)
                    assigned_features.add(j)
            
            duplicate_groups.append(current_group)
        
        return duplicate_groups
    
    def _are_strict_geometric_duplicates(self, feature1: ExpectedFeature, 
                                       feature2: ExpectedFeature) -> bool:
        """
        Check if two features are strict geometric duplicates using config criteria.
        
        TASK 5: FIXED AGGRESSIVE DEDUPLICATION
        Concentric circles at the same center are NOT duplicates unless they have
        very similar radii or share source entities. Different-sized concentric 
        circles represent different geometric features.
        """
        
        # 1. Check if they share source entities (definitive duplicates)
        if self._share_source_entities(feature1, feature2):
            return True
        
        # 2. Check center proximity using strict tolerance
        center_distance = feature1.center.distance_to(feature2.center)
        if center_distance > FEATURE_DUPLICATE_CENTER_TOLERANCE:
            return False
        
        # 3. For features at the same center, check radius similarity with STRICT criteria
        if feature1.radius is not None and feature2.radius is not None:
            radius_diff = abs(feature1.radius - feature2.radius)
            
            # STRICT absolute tolerance - must be very similar radii to be duplicates
            if radius_diff <= FEATURE_DUPLICATE_RADIUS_TOLERANCE:
                return True
            
            # STRICT relative tolerance - only for very similar sizes
            max_radius = max(feature1.radius, feature2.radius)
            relative_diff = radius_diff / max_radius if max_radius > 0 else 0
            if relative_diff <= FEATURE_DUPLICATE_RADIUS_RELATIVE:
                return True
        
        # 4. Special case: same geometry detected as circle and through_hole with same radius
        if (feature1.feature_type != feature2.feature_type and
            {feature1.feature_type.value, feature2.feature_type.value} == {"circle", "through_hole"}):
            # Only treat as duplicates if they have IDENTICAL radii
            if (feature1.radius is not None and feature2.radius is not None and
                abs(feature1.radius - feature2.radius) <= FEATURE_DUPLICATE_RADIUS_TOLERANCE):
                return True
        
        # DEFAULT: NOT duplicates - preserve concentric circles as separate features
        return False
    
    def _share_source_entities(self, feature1: ExpectedFeature, feature2: ExpectedFeature) -> bool:
        """Check if features share any source entities (indicating same underlying geometry)."""
        
        set1 = set(feature1.source_entity_ids)
        set2 = set(feature2.source_entity_ids)
        
        # If they share any source entities, they're likely duplicates
        return len(set1.intersection(set2)) > 0
    
    def _select_best_representative(self, duplicate_group: List[ExpectedFeature]) -> ExpectedFeature:
        """
        Select the best representative from a group of duplicate features.
        
        Preference order:
        1. Through holes over circles (when there's hole evidence)
        2. Explicit (direct) over reconstructed  
        3. Higher confidence
        4. More complete geometry
        5. Better source type
        """
        
        if len(duplicate_group) == 1:
            return duplicate_group[0]
        
        # Special handling for circle vs through_hole duplicates
        circles = [f for f in duplicate_group if f.feature_type.value == "circle"]
        through_holes = [f for f in duplicate_group if f.feature_type.value == "through_hole"]
        
        if circles and through_holes:
            # Same geometry classified as both circle and through_hole
            # Choose based on evidence strength
            best_hole = max(through_holes, key=lambda f: f.confidence)
            best_circle = max(circles, key=lambda f: f.confidence)
            
            # Prefer through_hole if it has reasonable confidence
            if best_hole.confidence >= THROUGH_HOLE_PREFERENCE_CONFIDENCE:
                logger.debug(f"Selected through_hole over circle due to hole evidence: "
                           f"hole confidence {best_hole.confidence:.3f} vs circle {best_circle.confidence:.3f}")
                return best_hole
            else:
                logger.debug(f"Selected circle over through_hole due to low hole confidence: "
                           f"circle confidence {best_circle.confidence:.3f} vs hole {best_hole.confidence:.3f}")
                return best_circle
        
        # Normal selection logic for same-type duplicates
        def preference_score(feature: ExpectedFeature) -> Tuple[int, float, int]:
            # Source type preference (higher is better)
            source_preference = 2 if feature.source_type == "explicit_circle" else 1
            
            # Confidence score
            confidence = feature.confidence
            
            # Completeness score
            completeness = self._calculate_completeness_score(feature)
            
            return (source_preference, confidence, completeness)
        
        # Select feature with highest preference score
        best_feature = max(duplicate_group, key=preference_score)
        
        logger.debug(f"Selected {best_feature.feature_id} from duplicate group of {len(duplicate_group)} "
                    f"(source: {best_feature.source_type}, confidence: {best_feature.confidence:.3f})")
        
        return best_feature
    
    def _has_complete_geometry(self, feature: ExpectedFeature) -> bool:
        """Check if feature has complete geometric information."""
        
        # Must have center
        if not feature.center:
            return False
        
        # Must have source entities
        if not feature.source_entity_ids:
            return False
        
        # Geometric completeness depends on feature type
        if feature.feature_type == FeatureType.CIRCLE or feature.feature_type == FeatureType.THROUGH_HOLE:
            # Circular features need radius (unless it's a square hole)
            source_type = feature.source_type
            if source_type == "square_hole_from_lines":
                # Square holes don't need radius, check geometric properties instead
                width = feature.geometric_properties.get("width", 0)
                height = feature.geometric_properties.get("height", 0)
                return width > 0 and height > 0
            else:
                # Regular circular features need radius
                if feature.radius is None or feature.radius <= 0:
                    return False
        
        return True
    
    def _passes_final_validation(self, feature: ExpectedFeature) -> bool:
        """Apply final validation checks."""
        
        # Geometric completeness
        if not self._has_complete_geometry(feature):
            return False
        
        # Confidence must still meet minimum
        if feature.confidence < self.min_confidence:
            return False
        
        # Feature type must be supported
        if feature.feature_type not in [FeatureType.CIRCLE, FeatureType.THROUGH_HOLE]:
            return False
        
        return True
    
    def _calculate_completeness_score(self, feature: ExpectedFeature) -> int:
        """Calculate completeness score for feature comparison."""
        
        score = 0
        
        # Has center
        if feature.center:
            score += 1
        
        # Has radius
        if feature.radius is not None:
            score += 1
        
        # Has source entities
        if feature.source_entity_ids:
            score += 1
        
        # Has detection evidence
        if feature.detection_evidence:
            score += 1
        
        # Has geometric properties
        if feature.geometric_properties:
            score += 1
        
        return score
    
    def _filter_concentric_circles(self, features: List[ExpectedFeature], 
                                 rejection_log: Dict[str, int]) -> List[ExpectedFeature]:
        """
        Filter concentric features (circles and holes) to select the most significant one from each location.
        
        TASK 5: DISABLED AGGRESSIVE CONCENTRIC FILTERING
        
        The previous logic was incorrectly treating concentric circles as duplicates
        and selecting only one per location. This removed legitimate concentric features
        that represent different geometric scales (e.g., 14.5, 27.5, 29.8mm circles).
        
        Now: Only filter if features are truly identical (handled in deduplication).
        Concentric circles with different radii are preserved as separate features.
        """
        
        # DISABLED: No longer filter concentric circles with different radii
        # This preserves all legitimate geometric features detected
        
        logger.debug(f"Concentric filtering disabled - preserving {len(features)} features")
        return features
        
        # OLD LOGIC (commented out):
        # This was removing legitimate concentric circles with different radii
        # by treating them as "duplicates" and selecting only the "best" one
        
        """
        if len(features) <= 1:
            return features
        
        # Group features by location (using center proximity)
        location_groups = []
        used_indices = set()
        
        for i, feature1 in enumerate(features):
            if i in used_indices:
                continue
                
            # Start a new location group
            location_group = [feature1]
            used_indices.add(i)
            
            # Find all features at the same location (both circles and holes)
            for j, feature2 in enumerate(features[i + 1:], i + 1):
                if (j not in used_indices and 
                    feature1.center.distance_to(feature2.center) < SIGNIFICANCE_LOCATION_TOLERANCE):  # Configurable location tolerance
                    location_group.append(feature2)
                    used_indices.add(j)
            
            location_groups.append(location_group)
        
        # Process each location group
        filtered_features = []
        
        for group in location_groups:
            if len(group) == 1:
                # Single feature at this location
                filtered_features.append(group[0])
            else:
                # Multiple concentric features - select the most appropriate one
                best_feature = self._select_best_concentric_feature(group)
                filtered_features.append(best_feature)
                
                rejected_count = len(group) - 1
                rejection_log["concentric_filtered"] = rejection_log.get("concentric_filtered", 0) + rejected_count
                
                logger.debug(f"Concentric deduplication: selected {best_feature.feature_type.value} "
                           f"(r={best_feature.radius:.2f}) from {len(group)} features at "
                           f"({best_feature.center.x:.1f}, {best_feature.center.y:.1f})")
        
        return filtered_features
        """
    
    def _select_best_concentric_feature(self, concentric_group: List[ExpectedFeature]) -> ExpectedFeature:
        """
        Select the best feature from a group of concentric features at the same location.
        
        Priority:
        1. Through holes over circles (holes are more functionally significant)
        2. For holes: prefer smaller, more reasonable hole sizes
        3. For circles: prefer ideal feature size range
        4. Higher confidence as tiebreaker
        """
        
        # Separate holes and circles
        holes = [f for f in concentric_group if f.feature_type == FeatureType.THROUGH_HOLE]
        circles = [f for f in concentric_group if f.feature_type == FeatureType.CIRCLE]
        
        # Prefer holes over circles
        if holes:
            # Select best hole based on size appropriateness
            def hole_score(hole):
                radius = hole.radius if hole.radius else 0
                confidence = hole.confidence if hole.confidence else 0
                
                # Import hole size configuration
                from ..config import (
                    HOLE_SIZE_VERY_SMALL_THRESHOLD, HOLE_SIZE_SMALL_THRESHOLD,
                    HOLE_SIZE_MEDIUM_THRESHOLD, HOLE_SIZE_LARGE_THRESHOLD
                )
                
                # Prefer medium-sized holes (most common for mechanical applications)
                if radius <= HOLE_SIZE_VERY_SMALL_THRESHOLD:
                    size_score = SIGNIFICANCE_HOLE_SIZE_SCORE_VERY_SMALL  # Very small holes (pins, screws)
                elif radius <= HOLE_SIZE_SMALL_THRESHOLD:
                    size_score = SIGNIFICANCE_HOLE_SIZE_SCORE_SMALL  # Small holes (small bolts)
                elif radius <= HOLE_SIZE_MEDIUM_THRESHOLD:
                    size_score = SIGNIFICANCE_HOLE_SIZE_SCORE_MEDIUM  # Medium holes (standard bolts) - PREFERRED
                elif radius <= HOLE_SIZE_LARGE_THRESHOLD:
                    size_score = SIGNIFICANCE_HOLE_SIZE_SCORE_LARGE  # Large holes (large bolts)
                else:
                    size_score = SIGNIFICANCE_HOLE_SIZE_SCORE_VERY_LARGE  # Very large holes (less common)
                
                # Combined score: size appropriateness + confidence
                return size_score + confidence * SIGNIFICANCE_CONFIDENCE_WEIGHT_FACTOR
            
            holes.sort(key=hole_score, reverse=True)
            return holes[0]
        
        elif circles:
            # Select best circle using existing logic
            def circle_score(circle):
                radius = circle.radius if circle.radius else 0
                confidence = circle.confidence if circle.confidence else 0
                
                # Import configuration thresholds
                from ..config import (
                    FEATURE_IDEAL_MIN_RADIUS, FEATURE_IDEAL_MAX_RADIUS,
                    FEATURE_ACCEPTABLE_MIN_RADIUS, FEATURE_ACCEPTABLE_MAX_RADIUS,
                    SIGNIFICANCE_MIN_RADIUS
                )
                
                # Prefer circles in the ideal feature size range
                if FEATURE_IDEAL_MIN_RADIUS <= radius <= FEATURE_IDEAL_MAX_RADIUS:
                    size_score = SIGNIFICANCE_CIRCLE_SIZE_SCORE_IDEAL  # Ideal feature size
                elif FEATURE_ACCEPTABLE_MIN_RADIUS <= radius <= FEATURE_ACCEPTABLE_MAX_RADIUS:
                    size_score = SIGNIFICANCE_CIRCLE_SIZE_SCORE_ACCEPTABLE  # Acceptable feature size  
                elif SIGNIFICANCE_MIN_RADIUS <= radius < FEATURE_ACCEPTABLE_MIN_RADIUS:
                    size_score = SIGNIFICANCE_CIRCLE_SIZE_SCORE_SMALL  # Small but possible
                else:
                    size_score = SIGNIFICANCE_CIRCLE_SIZE_SCORE_POOR  # Too large or too small for features
                
                return size_score + confidence * SIGNIFICANCE_CONFIDENCE_WEIGHT_FACTOR
            
            circles.sort(key=circle_score, reverse=True)
            return circles[0]
        
        # Fallback (shouldn't reach here)
        return concentric_group[0]
        
        return filtered_features
    
    def _log_filtering_summary(self, filtering_stages: Dict[str, int],
                             rejection_log: Dict[str, int]) -> None:
        """Log comprehensive filtering summary."""
        
        logger.info(f"Significance filtering pipeline summary:")
        logger.info(f"  Input features: {filtering_stages['input']}")
        logger.info(f"  After confidence filter: {filtering_stages['confidence_filtered']} "
                   f"(rejected {rejection_log['low_confidence']} low confidence)")
        logger.info(f"  After size filter: {filtering_stages['size_filtered']} "
                   f"(rejected {rejection_log['insufficient_size']} insufficient size)")
        logger.info(f"  After deduplication: {filtering_stages['deduplicated']} "
                   f"(removed {rejection_log['duplicate']} duplicates)")
        logger.info(f"  Final features: {filtering_stages['final']} "
                   f"(rejected {rejection_log['incomplete_geometry']} incomplete)")
        
        total_rejected = sum(rejection_log.values())
        if filtering_stages['input'] > 0:
            retention_rate = filtering_stages['final'] / filtering_stages['input']
            logger.info(f"  Total rejected: {total_rejected}, retention rate: {retention_rate:.1%}")
        
        # Log rejection reasons
        if total_rejected > 0:
            logger.debug("Rejection breakdown:")
            for reason, count in rejection_log.items():
                if count > 0:
                    percentage = (count / total_rejected) * SIGNIFICANCE_FILTER_PERCENTAGE_MULTIPLIER
                    logger.debug(f"  {reason}: {count} ({percentage:.1f}%)")