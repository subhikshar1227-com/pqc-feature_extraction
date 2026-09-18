"""
Actual Feature Extractor for Phase 2B

Main orchestrator for extracting features from preprocessed product images.
Coordinates individual feature extractors and handles candidate consolidation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import logging
from collections import defaultdict

from .feature_models import (
    ActualFeature, ActualFeatureType, ActualFeatureExtractionResult
)
from .circle_extractor import CircleExtractor
from .rectangle_extractor import RectangleExtractor  
from .contour_extractor import ContourExtractor
from ..preprocessing import PreprocessingResult
from ..config import (
    MAX_CANDIDATES_PER_TYPE, CANDIDATE_REFINEMENT_ITERATIONS,
    DUPLICATE_CENTER_DISTANCE_THRESHOLD, DUPLICATE_SIZE_RATIO_THRESHOLD,
    DUPLICATE_OVERLAP_THRESHOLD, DUPLICATE_CONFIDENCE_PREFERENCE,
    CROSS_TYPE_DUPLICATE_ENABLED, CROSS_TYPE_CENTER_THRESHOLD,
    CROSS_TYPE_SIZE_THRESHOLD, CROSS_TYPE_OVERLAP_THRESHOLD,
    ENHANCED_DUPLICATE_SUPPRESSION, SAME_TYPE_CENTER_DISTANCE_STRICT,
    HOUGH_CLUSTER_CONSOLIDATION, HOUGH_CLUSTER_RADIUS_TOLERANCE
)

logger = logging.getLogger(__name__)


class ActualFeatureExtractor:
    """
    Main feature extraction orchestrator for Phase 2B.
    
    Extracts actual features from preprocessed product images using
    multiple specialized extractors and consolidates results.
    
    CRITICAL: Uses only image evidence - NO expected feature information.
    """
    
    def __init__(self):
        """Initialize feature extractor with specialized extractors."""
        self.circle_extractor = CircleExtractor()
        self.rectangle_extractor = RectangleExtractor()
        self.contour_extractor = ContourExtractor()
        
        self.max_candidates_per_type = MAX_CANDIDATES_PER_TYPE
        self.refinement_iterations = CANDIDATE_REFINEMENT_ITERATIONS
        
        # Duplicate suppression thresholds
        self.duplicate_center_threshold = DUPLICATE_CENTER_DISTANCE_THRESHOLD
        self.duplicate_size_threshold = DUPLICATE_SIZE_RATIO_THRESHOLD
        self.duplicate_overlap_threshold = DUPLICATE_OVERLAP_THRESHOLD
        self.prefer_confidence = DUPLICATE_CONFIDENCE_PREFERENCE
        
        # Cross-type duplicate suppression
        self.cross_type_enabled = CROSS_TYPE_DUPLICATE_ENABLED
        self.cross_type_center_threshold = CROSS_TYPE_CENTER_THRESHOLD
        self.cross_type_size_threshold = CROSS_TYPE_SIZE_THRESHOLD
        self.cross_type_overlap_threshold = CROSS_TYPE_OVERLAP_THRESHOLD
        
        # Enhanced duplicate suppression parameters
        self.enhanced_duplicate_suppression = ENHANCED_DUPLICATE_SUPPRESSION
        self.strict_center_distance = SAME_TYPE_CENTER_DISTANCE_STRICT
        self.cluster_consolidation = HOUGH_CLUSTER_CONSOLIDATION
        self.cluster_radius_tolerance = HOUGH_CLUSTER_RADIUS_TOLERANCE
    
    def extract_features(self, preprocessing_result: PreprocessingResult) -> ActualFeatureExtractionResult:
        """
        Extract all features from preprocessing result.
        
        Args:
            preprocessing_result: Complete preprocessing output
            
        Returns:
            Complete feature extraction result with detected features
        """
        start_time = time.time()
        
        logger.info(f"Starting actual feature extraction for {preprocessing_result.source_image_path.name}")
        
        # Validate preprocessing success
        if not preprocessing_result.preprocessing_successful or not preprocessing_result.isolation_successful:
            logger.warning("Preprocessing failed - returning empty feature extraction result")
            return self._create_empty_result(preprocessing_result, time.time() - start_time)
        
        # Extract primary representations
        internal_edges = preprocessing_result.internal_geometry_edges
        isolated_product = preprocessing_result.isolated_product_image
        product_mask = preprocessing_result.product_mask
        raw_internal_edges = preprocessing_result.raw_internal_geometry_edges
        outer_boundary = preprocessing_result.outer_boundary_edges
        
        logger.debug("Using preprocessing representations:")
        logger.debug(f"  - internal_geometry_edges: {internal_edges.shape}")
        logger.debug(f"  - isolated_product: {isolated_product.shape}")
        logger.debug(f"  - product_mask: {product_mask.shape}")
        logger.debug(f"  - raw_internal_geometry_edges: {raw_internal_edges.shape}")
        logger.debug(f"  - outer_boundary_edges: {outer_boundary.shape}")
        
        # Extract features by type
        all_candidates = []
        candidate_stats = defaultdict(int)
        
        # Extract circles
        logger.debug("Extracting circular features...")
        circles = self.circle_extractor.extract_circles(
            internal_edges, isolated_product, product_mask, 
            raw_internal_edges, outer_boundary
        )
        all_candidates.extend(circles)
        candidate_stats[ActualFeatureType.CIRCLE.value] = len(circles)
        
        # Extract rectangles/squares
        logger.debug("Extracting rectangular features...")
        rectangles = self.rectangle_extractor.extract_rectangles(
            internal_edges, isolated_product, product_mask, raw_internal_edges
        )
        all_candidates.extend(rectangles)
        candidate_stats[ActualFeatureType.RECTANGLE.value] += len([r for r in rectangles if r.feature_type == ActualFeatureType.RECTANGLE])
        candidate_stats[ActualFeatureType.SQUARE.value] += len([r for r in rectangles if r.feature_type == ActualFeatureType.SQUARE])
        
        # Extract general contours
        logger.debug("Extracting general contour features...")
        contours = self.contour_extractor.extract_contours(
            internal_edges, isolated_product, product_mask, raw_internal_edges
        )
        all_candidates.extend(contours)
        candidate_stats[ActualFeatureType.GENERAL_CONTOUR.value] = len(contours)
        
        logger.info(f"Initial extraction: {len(all_candidates)} total candidates")
        for feature_type, count in candidate_stats.items():
            logger.info(f"  - {feature_type}: {count}")
        
        # Apply candidate limits per type
        limited_candidates = self._apply_candidate_limits(all_candidates)
        
        # Suppress duplicate candidates
        unique_candidates = self._suppress_duplicates(limited_candidates)
        
        # Calculate final statistics
        final_stats_by_type = defaultdict(int)
        confidence_by_type = defaultdict(list)
        
        for feature in unique_candidates:
            feature_type = feature.feature_type.value
            final_stats_by_type[feature_type] += 1
            confidence_by_type[feature_type].append(feature.evidence.confidence)
        
        # Calculate average confidence by type
        avg_confidence_by_type = {}
        for feature_type, confidences in confidence_by_type.items():
            if confidences:
                avg_confidence_by_type[feature_type] = float(np.mean(confidences))
            else:
                avg_confidence_by_type[feature_type] = 0.0
        
        overall_avg_confidence = float(np.mean([f.evidence.confidence for f in unique_candidates])) if unique_candidates else 0.0
        
        # Determine detection methods and representations used
        detection_methods = list(set(f.detection_method for f in unique_candidates))
        representations_used = list(set(f.source_representation for f in unique_candidates))
        
        processing_time = time.time() - start_time
        
        # Create result
        result = ActualFeatureExtractionResult(
            source_image_path=preprocessing_result.source_image_path,
            preprocessing_successful=preprocessing_result.preprocessing_successful,
            features=unique_candidates,
            total_candidates_generated=len(all_candidates),
            candidates_by_type=dict(candidate_stats),
            features_by_type=dict(final_stats_by_type),
            duplicate_candidates_suppressed=len(limited_candidates) - len(unique_candidates),
            average_confidence=overall_avg_confidence,
            confidence_by_type=avg_confidence_by_type,
            detection_methods_used=detection_methods,
            representations_used=representations_used,
            processing_time_seconds=processing_time,
            coordinate_system="processed",
            scale_factor=preprocessing_result.scale_factor,
            roi_offset=preprocessing_result.roi_offset
        )
        
        logger.info(f"Feature extraction complete: {len(unique_candidates)} features in {processing_time:.2f}s")
        logger.info(f"Final feature counts: {dict(final_stats_by_type)}")
        logger.info(f"Average confidence: {overall_avg_confidence:.3f}")
        
        return result
    
    def _apply_candidate_limits(self, candidates: List[ActualFeature]) -> List[ActualFeature]:
        """Apply per-type candidate limits to prevent excessive candidates."""
        limited_candidates = []
        
        # Group candidates by type
        by_type = defaultdict(list)
        for candidate in candidates:
            by_type[candidate.feature_type].append(candidate)
        
        # Apply limits per type
        for feature_type, type_candidates in by_type.items():
            # Sort by confidence (highest first)
            type_candidates.sort(key=lambda f: f.evidence.confidence, reverse=True)
            
            # Take top candidates up to limit
            limited = type_candidates[:self.max_candidates_per_type]
            limited_candidates.extend(limited)
            
            if len(type_candidates) > self.max_candidates_per_type:
                logger.debug(f"Limited {feature_type.value} candidates: {len(type_candidates)} -> {len(limited)}")
        
        return limited_candidates
    
    def _suppress_duplicates(self, candidates: List[ActualFeature]) -> List[ActualFeature]:
        """Suppress duplicate candidates using geometric similarity."""
        if not candidates:
            return candidates
        
        # Sort candidates by confidence (highest first)
        candidates.sort(key=lambda f: f.evidence.confidence, reverse=True)
        
        unique_candidates = []
        
        for candidate in candidates:
            is_duplicate = False
            
            # Check against already accepted candidates
            for accepted in unique_candidates:
                if self._are_duplicates(candidate, accepted):
                    is_duplicate = True
                    logger.debug(f"Suppressing duplicate: {candidate.feature_id} (similar to {accepted.feature_id})")
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _are_duplicates(self, candidate1: ActualFeature, candidate2: ActualFeature) -> bool:
        """Check if two candidates represent the same physical feature."""
        # For same-type duplicates, use stricter thresholds
        if candidate1.feature_type == candidate2.feature_type:
            return self._are_same_type_duplicates(candidate1, candidate2)
        
        # For cross-type duplicates, use more lenient thresholds if enabled
        if self.cross_type_enabled:
            return self._are_cross_type_duplicates(candidate1, candidate2)
        
        return False
    
    def _are_same_type_duplicates(self, candidate1: ActualFeature, candidate2: ActualFeature) -> bool:
        """Check if two candidates of the same type are duplicates."""
        # Use enhanced stricter thresholds for same-type detection
        center_threshold = self.strict_center_distance if self.enhanced_duplicate_suppression else self.duplicate_center_threshold
        
        # Check center distance
        center1 = candidate1.geometry.center
        center2 = candidate2.geometry.center
        center_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        if center_distance > center_threshold:
            return False
        
        # For circles, also check radius similarity if cluster consolidation is enabled
        if (self.cluster_consolidation and 
            candidate1.feature_type.value == "CIRCLE" and
            candidate2.feature_type.value == "CIRCLE"):
            
            radius1 = getattr(candidate1.geometry, 'radius', 0)
            radius2 = getattr(candidate2.geometry, 'radius', 0)
            
            if radius1 > 0 and radius2 > 0:
                radius_ratio = min(radius1, radius2) / max(radius1, radius2)
                if radius_ratio < (1.0 - self.cluster_radius_tolerance):
                    return False
        
        # Check size similarity
        area1 = candidate1.geometry.area
        area2 = candidate2.geometry.area
        
        if area1 == 0 or area2 == 0:
            return False
        
        size_ratio = min(area1, area2) / max(area1, area2)
        if size_ratio < (1.0 - self.duplicate_size_threshold):
            return False
        
        # Check contour overlap for more precise duplicate detection
        overlap = self._calculate_contour_overlap(candidate1.contour, candidate2.contour)
        if overlap < self.duplicate_overlap_threshold:
            return False
        
        return True
    
    def _are_cross_type_duplicates(self, candidate1: ActualFeature, candidate2: ActualFeature) -> bool:
        """Check if two candidates of different types represent the same physical feature."""
        # Check center distance with more lenient threshold
        center1 = candidate1.geometry.center
        center2 = candidate2.geometry.center
        center_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        if center_distance > self.cross_type_center_threshold:
            return False
        
        # Check size similarity with more lenient threshold
        area1 = candidate1.geometry.area
        area2 = candidate2.geometry.area
        
        if area1 == 0 or area2 == 0:
            return False
        
        size_ratio = min(area1, area2) / max(area1, area2)
        if size_ratio < (1.0 - self.cross_type_size_threshold):
            return False
        
        # Check contour overlap with more lenient threshold
        overlap = self._calculate_contour_overlap(candidate1.contour, candidate2.contour)
        if overlap < self.cross_type_overlap_threshold:
            return False
        
        return True
    
    def _calculate_contour_overlap(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """Calculate overlap ratio between two contours."""
        # Create masks for both contours
        # Estimate image size from contour bounds
        all_points = np.vstack([contour1.reshape(-1, 2), contour2.reshape(-1, 2)])
        max_x = int(np.max(all_points[:, 0])) + 10
        max_y = int(np.max(all_points[:, 1])) + 10
        
        mask1 = np.zeros((max_y, max_x), dtype=np.uint8)
        mask2 = np.zeros((max_y, max_x), dtype=np.uint8)
        
        cv2.fillPoly(mask1, [contour1], 255)
        cv2.fillPoly(mask2, [contour2], 255)
        
        # Calculate intersection and union
        intersection = cv2.bitwise_and(mask1, mask2)
        union = cv2.bitwise_or(mask1, mask2)
        
        intersection_area = np.count_nonzero(intersection)
        union_area = np.count_nonzero(union)
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _create_empty_result(self, preprocessing_result: PreprocessingResult, 
                           processing_time: float) -> ActualFeatureExtractionResult:
        """Create empty result for failed preprocessing."""
        return ActualFeatureExtractionResult(
            source_image_path=preprocessing_result.source_image_path,
            preprocessing_successful=preprocessing_result.preprocessing_successful,
            features=[],
            total_candidates_generated=0,
            candidates_by_type={},
            features_by_type={},
            duplicate_candidates_suppressed=0,
            average_confidence=0.0,
            confidence_by_type={},
            detection_methods_used=[],
            representations_used=[],
            processing_time_seconds=processing_time,
            coordinate_system="processed",
            scale_factor=preprocessing_result.scale_factor,
            roi_offset=preprocessing_result.roi_offset
        )


def extract_actual_features(preprocessing_result: PreprocessingResult) -> ActualFeatureExtractionResult:
    """
    Convenience function for extracting actual features from preprocessing result.
    
    Args:
        preprocessing_result: Complete preprocessing output
        
    Returns:
        Complete feature extraction result
    """
    extractor = ActualFeatureExtractor()
    return extractor.extract_features(preprocessing_result)