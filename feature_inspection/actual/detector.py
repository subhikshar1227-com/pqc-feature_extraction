"""
Actual Feature Detection

Main detector for finding geometric features in preprocessed product images.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import logging
import uuid

from feature_extraction.expected.feature_types import FeatureType, Point2D
from ..models.actual_feature import ActualFeature, ActualFeatureSet, ActualDetectionStatistics, DetectionMethod
from .canonical_preprocessor import CanonicalPreprocessor
from ..config import (
    CONTOUR_MIN_AREA, CONTOUR_MAX_AREA, CONTOUR_APPROX_EPSILON_FACTOR,
    CIRCLE_HOUGH_DP, CIRCLE_HOUGH_MIN_DIST_FACTOR, CIRCLE_HOUGH_PARAM1, CIRCLE_HOUGH_PARAM2,
    CIRCLE_MIN_RADIUS_PIXELS, CIRCLE_MAX_RADIUS_PIXELS,
    CIRCLE_CONTOUR_MATCH_THRESHOLD, CIRCLE_ROUNDNESS_THRESHOLD, CIRCLE_AREA_RATIO_THRESHOLD,
    HOLE_MIN_ASPECT_RATIO, HOLE_MAX_ASPECT_RATIO, HOLE_MIN_SOLIDITY, HOLE_DARKNESS_THRESHOLD,
    HOLE_LOCAL_CONTRAST_THRESHOLD, HOLE_ANNULUS_WIDTH_PIXELS, HOLE_MIN_CONTRAST_RATIO,
    RECT_HOLE_MIN_VERTICES, RECT_HOLE_MAX_VERTICES, RECT_HOLE_CORNER_ANGLE_TOLERANCE,
    RECT_HOLE_MIN_WIDTH_PIXELS, RECT_HOLE_MIN_HEIGHT_PIXELS,
    CONFIDENCE_BASE_SCORE, CONFIDENCE_ROUNDNESS_WEIGHT, CONFIDENCE_AREA_RATIO_WEIGHT,
    CONFIDENCE_EDGE_STRENGTH_WEIGHT, CONFIDENCE_CONTOUR_QUALITY_WEIGHT, CONFIDENCE_LOCAL_CONTRAST_WEIGHT,
    DUPLICATE_REMOVAL_DISTANCE_THRESHOLD, DUPLICATE_REMOVAL_SIZE_TOLERANCE
)

logger = logging.getLogger(__name__)


class ActualFeatureDetector:
    """
    Detects geometric features in actual product images.
    
    Uses computer vision techniques to identify circles, holes, and
    rectangular features from preprocessed product images.
    """
    
    def __init__(self):
        """Initialize the actual feature detector."""
        self.canonical_preprocessor = CanonicalPreprocessor()
        self.detection_methods = {
            FeatureType.CIRCLE: self._detect_circles,
            FeatureType.THROUGH_HOLE: self._detect_through_holes,
            FeatureType.SQUARE_HOLE: self._detect_rectangular_holes,
            FeatureType.RECTANGULAR_HOLE: self._detect_rectangular_holes
        }
        self._last_preprocessing_metadata = None
        self._current_product_mask = None
        self._preprocessing_result = None
    
    def detect_features(self, image_path: Path) -> ActualFeatureSet:
        """
        Detect all features in a product image.
        
        Uses the new Phase 2 preprocessing interface to focus
        detection on the actual product rather than background/table.
        
        Args:
            image_path: Path to the product image file
            
        Returns:
            ActualFeatureSet containing all detected features
        """
        start_time = time.time()
        
        logger.info(f"Starting feature detection for: {image_path.name}")
        
        # Apply Phase 2 preprocessing using the canonical implementation
        preprocess_start = time.time()
        try:
            preprocessing_result = self.canonical_preprocessor.preprocess_image(image_path)
            preprocessing_time = time.time() - preprocess_start
        except Exception as e:
            logger.error(f"Phase 2 preprocessing failed: {e}")
            # Return empty result on preprocessing failure
            empty_stats = ActualDetectionStatistics(
                total_contours_found=0,
                contours_after_filtering=0,
                circles_detected=0,
                through_holes_detected=0,
                rectangular_holes_detected=0,
                total_features_detected=0,
                average_confidence=0.0,
                detection_time_seconds=0.0,
                preprocessing_time_seconds=time.time() - preprocess_start
            )
            
            return ActualFeatureSet(
                source_image_path=image_path,
                features=[],
                detection_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                detection_statistics=empty_stats,
                configuration_snapshot=self._get_configuration_snapshot(),
                image_dimensions=(0, 0),
                preprocessing_applied=["preprocessing_failed"],
                coordinate_system="image_pixels"
            )
        
        # Store preprocessing result for coordinate transformation and feature context
        self._preprocessing_result = preprocessing_result
        self._current_product_mask = preprocessing_result.product_mask
        
        # Check if product isolation was successful
        if not preprocessing_result.isolation_successful:
            logger.warning(f"Product isolation failed for {image_path.name}")
            # Return empty feature set with warning status
            empty_stats = ActualDetectionStatistics(
                total_contours_found=0,
                contours_after_filtering=0,
                circles_detected=0,
                through_holes_detected=0,
                rectangular_holes_detected=0,
                total_features_detected=0,
                average_confidence=0.0,
                detection_time_seconds=0.0,
                preprocessing_time_seconds=preprocessing_time
            )
            
            return ActualFeatureSet(
                source_image_path=image_path,
                features=[],
                detection_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                detection_statistics=empty_stats,
                configuration_snapshot=self._get_configuration_snapshot(),
                image_dimensions=preprocessing_result.original_dimensions,
                preprocessing_applied=["phase2_preprocessing", "isolation_failed"],
                coordinate_system="image_pixels"
            )
        
        logger.info(f"Product isolation successful: {preprocessing_result.product_area_fraction:.1%} of image")
        
        # Detect features using the product-focused representations
        detection_start = time.time()
        
        # Use the processed grayscale image and edge representation for detection
        # This ensures dimensional consistency with the product mask
        processed_bgr = cv2.cvtColor(preprocessing_result.processed_image, cv2.COLOR_GRAY2BGR)
        
        all_features, detection_stats = self._detect_all_features(
            processed_bgr,  # Use processed image for consistent dimensions
            preprocessing_result.edge_representation
        )
        
        # Convert all feature coordinates from processed space to original image space
        for feature in all_features:
            orig_x, orig_y = preprocessing_result.to_original_coordinates(feature.center.x, feature.center.y)
            feature.center = Point2D(float(orig_x), float(orig_y))
            
            # Scale radius/dimensions if present
            if feature.radius is not None:
                feature.radius = float(feature.radius / preprocessing_result.scale_factor)
            if feature.width is not None:
                feature.width = float(feature.width / preprocessing_result.scale_factor)
            if feature.height is not None:
                feature.height = float(feature.height / preprocessing_result.scale_factor)
            
            # Update detection evidence to record coordinate transformation
            if feature.detection_evidence is None:
                feature.detection_evidence = {}
            feature.detection_evidence["coordinate_transformation"] = {
                "processed_coordinates": (feature.center.x * preprocessing_result.scale_factor, 
                                        feature.center.y * preprocessing_result.scale_factor),
                "scale_factor": preprocessing_result.scale_factor,
                "roi_offset": preprocessing_result.roi_offset
            }
        
        detection_time = time.time() - detection_start
        
        # Clean up temporary references
        self._current_product_mask = None
        self._preprocessing_result = None
        
        # Create detection statistics
        stats = ActualDetectionStatistics(
            total_contours_found=detection_stats.get("total_contours", 0),
            contours_after_filtering=detection_stats.get("filtered_contours", 0),
            circles_detected=len([f for f in all_features if f.feature_type == FeatureType.CIRCLE]),
            through_holes_detected=len([f for f in all_features if f.feature_type == FeatureType.THROUGH_HOLE]),
            rectangular_holes_detected=len([f for f in all_features 
                                          if f.feature_type in [FeatureType.SQUARE_HOLE, FeatureType.RECTANGULAR_HOLE]]),
            total_features_detected=len(all_features),
            average_confidence=sum(f.confidence for f in all_features) / max(1, len(all_features)),
            detection_time_seconds=detection_time,
            preprocessing_time_seconds=preprocessing_time
        )
        
        # Create feature set with original image dimensions
        feature_set = ActualFeatureSet(
            source_image_path=image_path,
            features=all_features,
            detection_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            detection_statistics=stats,
            configuration_snapshot=self._get_configuration_snapshot(),
            image_dimensions=preprocessing_result.original_dimensions,
            preprocessing_applied=["phase2_preprocessing", "product_isolation", "edge_extraction", "coordinate_transformation"],
            coordinate_system="image_pixels"
        )
        
        total_time = time.time() - start_time
        logger.info(f"Feature detection complete: {len(all_features)} features detected in {total_time:.2f}s")
        logger.info(f"  - Circles: {stats.circles_detected}")
        logger.info(f"  - Through holes: {stats.through_holes_detected}")
        logger.info(f"  - Rectangular holes: {stats.rectangular_holes_detected}")
        logger.info(f"  - All coordinates in ORIGINAL IMAGE PIXELS")
        
        return feature_set
    
    def _detect_all_features(self, image: np.ndarray, edges: np.ndarray) -> Tuple[List[ActualFeature], Dict[str, Any]]:
        """Detect all types of features in the image."""
        all_features = []
        detection_stats = {}
        
        # Find contours with hierarchy to preserve internal/nested contours
        # Use RETR_TREE to get full hierarchy information for hole detection
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detection_stats["total_contours"] = len(contours)
        
        # Filter contours by area
        filtered_contours = []
        filtered_hierarchy = []
        for i, contour in enumerate(contours):
            if CONTOUR_MIN_AREA <= cv2.contourArea(contour) <= CONTOUR_MAX_AREA:
                filtered_contours.append(contour)
                if hierarchy is not None:
                    filtered_hierarchy.append(hierarchy[0][i] if len(hierarchy) > 0 else [-1, -1, -1, -1])
        
        detection_stats["filtered_contours"] = len(filtered_contours)
        
        logger.debug(f"Found {len(contours)} contours, {len(filtered_contours)} after filtering")
        
        # Detect circles using both Hough transform and contour analysis
        circle_features = self._detect_circles(image, edges, filtered_contours)
        all_features.extend(circle_features)
        
        # Detect through holes with hierarchy information
        hole_features = self._detect_through_holes(image, edges, filtered_contours, filtered_hierarchy)
        all_features.extend(hole_features)
        
        # Detect rectangular holes with hierarchy information
        rect_features = self._detect_rectangular_holes(image, edges, filtered_contours, filtered_hierarchy)
        all_features.extend(rect_features)
        
        # Remove duplicate detections
        all_features = self._remove_duplicate_features(all_features)
        
        return all_features, detection_stats
    
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, 
                       contours: List[np.ndarray]) -> List[ActualFeature]:
        """Detect circular features using Hough transform and contour analysis."""
        circles = []
        
        # Method 1: Hough Circle Transform restricted to product mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply product mask to restrict Hough detection to product area only
        if self._current_product_mask is not None:
            # Create masked grayscale for Hough detection
            masked_gray = gray.copy()
            # Ensure mask dimensions match grayscale dimensions
            if masked_gray.shape[:2] != self._current_product_mask.shape[:2]:
                # Resize mask to match grayscale if needed
                mask_resized = cv2.resize(self._current_product_mask, 
                                        (masked_gray.shape[1], masked_gray.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                masked_gray[mask_resized == 0] = 0
            else:
                masked_gray[self._current_product_mask == 0] = 0
        else:
            masked_gray = gray
        
        # Calculate dynamic minimum distance based on minimum radius
        min_dist = int(CIRCLE_MIN_RADIUS_PIXELS * CIRCLE_HOUGH_MIN_DIST_FACTOR * 2)
        
        hough_circles = cv2.HoughCircles(
            masked_gray,
            cv2.HOUGH_GRADIENT,
            dp=CIRCLE_HOUGH_DP,
            minDist=min_dist,
            param1=CIRCLE_HOUGH_PARAM1,
            param2=CIRCLE_HOUGH_PARAM2,
            minRadius=CIRCLE_MIN_RADIUS_PIXELS,
            maxRadius=CIRCLE_MAX_RADIUS_PIXELS
        )
        
        if hough_circles is not None:
            hough_circles = np.round(hough_circles[0, :]).astype("int")
            
            # Limit the number of candidates to prevent overwhelming detection
            if len(hough_circles) > 50:  # Reasonable limit for circles in a single image
                hough_circles = hough_circles[:50]
            
            for (x, y, r) in hough_circles:
                # Additional validation: ensure circle is within product mask
                if self._current_product_mask is not None:
                    mask_height, mask_width = self._current_product_mask.shape
                    if y < mask_height and x < mask_width and self._current_product_mask[y, x] > 0:
                        pass  # Circle is in product area
                    else:
                        continue  # Skip circles outside product area
                
                # Validate circle with contour matching
                confidence = self._validate_circle_with_contours(
                    Point2D(x, y), r, contours, image
                )
                
                if confidence > 0.7:  # Increased minimum confidence threshold for Hough circles
                    feature = ActualFeature(
                        feature_id=self._generate_feature_id("circle"),
                        feature_type=FeatureType.CIRCLE,
                        confidence=confidence,
                        center=Point2D(float(x), float(y)),
                        radius=float(r),
                        detection_method=DetectionMethod.HOUGH_CIRCLES,
                        detection_evidence={
                            "hough_params": [CIRCLE_HOUGH_PARAM1, CIRCLE_HOUGH_PARAM2],
                            "validation_method": "contour_matching",
                            "masked_detection": self._current_product_mask is not None
                        }
                    )
                    circles.append(feature)
        
        # Method 2: Contour-based circle detection
        for contour in contours:
            circle_feature = self._analyze_contour_for_circle(contour, image)
            if circle_feature is not None:
                circles.append(circle_feature)
        
        logger.debug(f"Detected {len(circles)} circular features")
        return circles
    
    def _detect_through_holes(self, image: np.ndarray, edges: np.ndarray,
                             contours: List[np.ndarray], hierarchy: List[List[int]]) -> List[ActualFeature]:
        """Detect through holes by analyzing dark circular regions with local contrast."""
        holes = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for i, contour in enumerate(contours):
            hole_feature = self._analyze_contour_for_hole(contour, gray, hierarchy[i] if hierarchy else None)
            if hole_feature is not None:
                holes.append(hole_feature)
        
        logger.debug(f"Detected {len(holes)} through hole features")
        return holes
    
    def _detect_rectangular_holes(self, image: np.ndarray, edges: np.ndarray,
                                 contours: List[np.ndarray], hierarchy: List[List[int]]) -> List[ActualFeature]:
        """Detect rectangular and square holes."""
        rect_holes = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for i, contour in enumerate(contours):
            rect_feature = self._analyze_contour_for_rectangle(contour, gray, hierarchy[i] if hierarchy else None)
            if rect_feature is not None:
                rect_holes.append(rect_feature)
        
        logger.debug(f"Detected {len(rect_holes)} rectangular hole features")
        return rect_holes
    
    def _validate_circle_with_contours(self, center: Point2D, radius: float,
                                     contours: List[np.ndarray], image: np.ndarray) -> float:
        """Validate a detected circle by matching with contours."""
        best_match_score = 0.0
        
        for contour in contours:
            # Check if contour could represent this circle
            contour_center, contour_radius = cv2.minEnclosingCircle(contour)
            contour_center_point = Point2D(contour_center[0], contour_center[1])
            
            # Distance between centers
            center_distance = center.distance_to(contour_center_point)
            radius_difference = abs(radius - contour_radius)
            
            # Calculate match score
            position_score = max(0, 1.0 - center_distance / radius)
            size_score = max(0, 1.0 - radius_difference / radius)
            
            # Check roundness
            roundness = self._calculate_roundness(contour)
            
            # Combine scores
            match_score = (position_score * 0.4 + size_score * 0.4 + roundness * 0.2)
            best_match_score = max(best_match_score, match_score)
        
        return min(best_match_score, 1.0)
    
    def _analyze_contour_for_circle(self, contour: np.ndarray, image: np.ndarray) -> Optional[ActualFeature]:
        """Analyze a contour to determine if it represents a circle."""
        # Calculate contour properties
        area = cv2.contourArea(contour)
        if area < CONTOUR_MIN_AREA:
            return None
        
        # Get minimum enclosing circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        
        # Check radius bounds
        if not (CIRCLE_MIN_RADIUS_PIXELS <= radius <= CIRCLE_MAX_RADIUS_PIXELS):
            return None
        
        # Calculate roundness
        roundness = self._calculate_roundness(contour)
        if roundness < CIRCLE_ROUNDNESS_THRESHOLD:
            return None
        
        # Calculate area ratio (contour area vs circle area)
        circle_area = np.pi * radius * radius
        area_ratio = area / circle_area
        if area_ratio < CIRCLE_AREA_RATIO_THRESHOLD:
            return None
        
        # Calculate confidence
        confidence = self._calculate_circle_confidence(contour, roundness, area_ratio, image)
        
        return ActualFeature(
            feature_id=self._generate_feature_id("circle"),
            feature_type=FeatureType.CIRCLE,
            confidence=confidence,
            center=Point2D(float(center_x), float(center_y)),
            radius=float(radius),
            detection_method=DetectionMethod.CONTOUR_ANALYSIS,
            detection_evidence={
                "roundness": roundness,
                "area_ratio": area_ratio,
                "contour_area": area
            },
            contour_area=area
        )
    
    def _analyze_contour_for_hole(self, contour: np.ndarray, gray_image: np.ndarray, 
                                 hierarchy_info: Optional[List[int]] = None) -> Optional[ActualFeature]:
        """Analyze a contour to determine if it represents a through hole using local contrast."""
        # Get contour properties
        area = cv2.contourArea(contour)
        if area < CONTOUR_MIN_AREA:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Check if shape is roughly circular (holes are typically round)
        if not (HOLE_MIN_ASPECT_RATIO <= aspect_ratio <= HOLE_MAX_ASPECT_RATIO):
            return None
        
        # Calculate solidity (convexity)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < HOLE_MIN_SOLIDITY:
            return None
        
        # Local contrast analysis for hole detection
        local_contrast_evidence = self._calculate_local_contrast_evidence(contour, gray_image)
        if not local_contrast_evidence["is_hole_candidate"]:
            return None
        
        # Get circle approximation
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        
        # Calculate confidence based on multiple evidence sources
        roundness = self._calculate_roundness(contour)
        
        # Combine evidence for confidence calculation
        confidence_components = {
            "roundness": roundness * 0.3,
            "solidity": solidity * 0.25,
            "local_contrast": local_contrast_evidence["contrast_score"] * 0.35,
            "geometry": min(aspect_ratio, 1/aspect_ratio) * 0.1  # Prefer circular
        }
        
        confidence = sum(confidence_components.values())
        confidence = max(0.0, min(1.0, confidence))
        
        return ActualFeature(
            feature_id=self._generate_feature_id("hole"),
            feature_type=FeatureType.THROUGH_HOLE,
            confidence=confidence,
            center=Point2D(float(center_x), float(center_y)),
            radius=float(radius),
            detection_method=DetectionMethod.CONTOUR_ANALYSIS,
            detection_evidence={
                "solidity": solidity,
                "interior_brightness": local_contrast_evidence["interior_brightness"],
                "annulus_brightness": local_contrast_evidence["annulus_brightness"],
                "local_contrast": local_contrast_evidence["local_contrast"],
                "contrast_ratio": local_contrast_evidence["contrast_ratio"],
                "aspect_ratio": aspect_ratio,
                "confidence_breakdown": confidence_components
            },
            contour_area=area
        )
    
    def _analyze_contour_for_rectangle(self, contour: np.ndarray, gray_image: np.ndarray,
                                      hierarchy_info: Optional[List[int]] = None) -> Optional[ActualFeature]:
        """Analyze a contour to determine if it represents a rectangular hole."""
        # Approximate contour to polygon
        epsilon = CONTOUR_APPROX_EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check vertex count
        if not (RECT_HOLE_MIN_VERTICES <= len(approx) <= RECT_HOLE_MAX_VERTICES):
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check minimum dimensions
        if w < RECT_HOLE_MIN_WIDTH_PIXELS or h < RECT_HOLE_MIN_HEIGHT_PIXELS:
            return None
        
        # Check if angles are approximately right angles for rectangles
        if len(approx) == 4:
            angles = self._calculate_polygon_angles(approx)
            right_angle_count = sum(1 for angle in angles 
                                  if abs(angle - 90) <= RECT_HOLE_CORNER_ANGLE_TOLERANCE)
            
            if right_angle_count < 3:  # At least 3 corners should be close to 90 degrees
                return None
        
        # CRITICAL FIX: Only create hole features if actual hole evidence exists
        local_contrast_evidence = self._calculate_local_contrast_evidence(contour, gray_image)
        if not local_contrast_evidence["is_hole_candidate"]:
            return None  # Reject rectangles that don't have hole characteristics
        
        area = cv2.contourArea(contour)
        
        # Determine feature type based on aspect ratio
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio <= 1.2:  # Nearly square
            feature_type = FeatureType.SQUARE_HOLE
        else:
            feature_type = FeatureType.RECTANGULAR_HOLE
        
        # Calculate confidence based on shape regularity and hole evidence
        shape_regularity = 1.0 / max(len(approx) - 3, 1) if len(approx) >= 4 else 0.5  # Prefer 4 vertices
        
        # Confidence components
        confidence_components = {
            "shape_regularity": shape_regularity * 0.3,
            "local_contrast": local_contrast_evidence["contrast_score"] * 0.5,
            "geometry": min(aspect_ratio, 2.0) / 2.0 * 0.2  # Moderate aspect ratios preferred
        }
        
        confidence = sum(confidence_components.values())
        confidence = max(0.0, min(1.0, confidence))
        
        return ActualFeature(
            feature_id=self._generate_feature_id("rect"),
            feature_type=feature_type,
            confidence=confidence,
            center=Point2D(float(x + w/2), float(y + h/2)),
            width=float(w),
            height=float(h),
            detection_method=DetectionMethod.CONTOUR_ANALYSIS,
            detection_evidence={
                "vertices": len(approx),
                "aspect_ratio": aspect_ratio,
                "interior_brightness": local_contrast_evidence["interior_brightness"],
                "annulus_brightness": local_contrast_evidence["annulus_brightness"],
                "local_contrast": local_contrast_evidence["local_contrast"],
                "contrast_ratio": local_contrast_evidence["contrast_ratio"],
                "confidence_breakdown": confidence_components,
                "right_angle_count": sum(1 for angle in self._calculate_polygon_angles(approx) 
                                       if abs(angle - 90) <= RECT_HOLE_CORNER_ANGLE_TOLERANCE) if len(approx) == 4 else 0
            },
            contour_area=area
        )
    
    def _calculate_roundness(self, contour: np.ndarray) -> float:
        """Calculate how round a contour is (0 = not round, 1 = perfect circle)."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Roundness = 4π × Area / Perimeter²
        roundness = (4 * np.pi * area) / (perimeter * perimeter)
        return min(roundness, 1.0)
    
    def _calculate_circle_confidence(self, contour: np.ndarray, roundness: float,
                                   area_ratio: float, image: np.ndarray) -> float:
        """Calculate confidence score for a circle detection with proper bounds."""
        # Start with base confidence - ensure all weights sum to create bounded result
        total_weight = (CONFIDENCE_ROUNDNESS_WEIGHT + CONFIDENCE_AREA_RATIO_WEIGHT + 
                       CONFIDENCE_EDGE_STRENGTH_WEIGHT + CONFIDENCE_CONTOUR_QUALITY_WEIGHT)
        
        # Normalize weights to ensure they sum to a reasonable value
        if total_weight > 1.0:
            weight_scale = 0.7 / total_weight  # Leave room for base score
            roundness_weight = CONFIDENCE_ROUNDNESS_WEIGHT * weight_scale
            area_weight = CONFIDENCE_AREA_RATIO_WEIGHT * weight_scale
            edge_weight = CONFIDENCE_EDGE_STRENGTH_WEIGHT * weight_scale
            quality_weight = CONFIDENCE_CONTOUR_QUALITY_WEIGHT * weight_scale
        else:
            roundness_weight = CONFIDENCE_ROUNDNESS_WEIGHT
            area_weight = CONFIDENCE_AREA_RATIO_WEIGHT  
            edge_weight = CONFIDENCE_EDGE_STRENGTH_WEIGHT
            quality_weight = CONFIDENCE_CONTOUR_QUALITY_WEIGHT
        
        # Calculate individual contributions (each 0-1)
        roundness_score = roundness * roundness_weight
        area_score = area_ratio * area_weight
        
        edge_strength = self._calculate_edge_strength(contour, image)
        edge_score = edge_strength * edge_weight
        
        contour_quality = min(1.0, cv2.contourArea(contour) / 1000.0)  # Normalize by typical feature size
        quality_score = contour_quality * quality_weight
        
        # Combine with base score
        confidence = CONFIDENCE_BASE_SCORE + roundness_score + area_score + edge_score + quality_score
        
        # Ensure final confidence is properly bounded
        return max(0.0, min(1.0, confidence))
    
    def _calculate_local_contrast_evidence(self, contour: np.ndarray, gray_image: np.ndarray) -> Dict[str, Any]:
        """
        Calculate local contrast evidence for hole detection.
        
        Compares the brightness inside the contour with a surrounding annulus
        to determine if the region has hole-like characteristics.
        
        Returns:
            Dictionary with contrast analysis results and hole candidacy
        """
        # Create interior mask
        interior_mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.fillPoly(interior_mask, [contour], 255)
        
        # Create annulus mask (surrounding region)
        dilated_mask = cv2.dilate(interior_mask, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (HOLE_ANNULUS_WIDTH_PIXELS * 2 + 1, 
                                                          HOLE_ANNULUS_WIDTH_PIXELS * 2 + 1)))
        annulus_mask = dilated_mask - interior_mask
        
        # Calculate brightness in each region
        interior_pixels = gray_image[interior_mask > 0]
        annulus_pixels = gray_image[annulus_mask > 0]
        
        if len(interior_pixels) == 0 or len(annulus_pixels) == 0:
            return {
                "is_hole_candidate": False,
                "interior_brightness": 0.0,
                "annulus_brightness": 0.0,
                "local_contrast": 0.0,
                "contrast_ratio": 0.0,
                "contrast_score": 0.0
            }
        
        interior_brightness = float(np.mean(interior_pixels))
        annulus_brightness = float(np.mean(annulus_pixels))
        
        # Calculate contrast metrics
        local_contrast = annulus_brightness - interior_brightness  # Positive for holes (darker interior)
        contrast_ratio = interior_brightness / annulus_brightness if annulus_brightness > 0 else 1.0
        
        # Determine if this is a hole candidate based on multiple criteria
        absolute_darkness_ok = interior_brightness <= HOLE_DARKNESS_THRESHOLD
        local_contrast_ok = local_contrast >= HOLE_LOCAL_CONTRAST_THRESHOLD
        contrast_ratio_ok = contrast_ratio <= HOLE_MIN_CONTRAST_RATIO
        
        # A region is a hole candidate if it satisfies contrast requirements
        is_hole_candidate = local_contrast_ok and contrast_ratio_ok
        
        # Calculate normalized contrast score (0-1)
        contrast_score = 0.0
        if is_hole_candidate:
            # Score based on how much darker the interior is
            normalized_contrast = min(local_contrast / 50.0, 1.0)  # Normalize to reasonable contrast range
            normalized_ratio = max(0, (HOLE_MIN_CONTRAST_RATIO - contrast_ratio) / HOLE_MIN_CONTRAST_RATIO)
            contrast_score = (normalized_contrast + normalized_ratio) / 2.0
        
        return {
            "is_hole_candidate": is_hole_candidate,
            "interior_brightness": interior_brightness,
            "annulus_brightness": annulus_brightness,
            "local_contrast": local_contrast,
            "contrast_ratio": contrast_ratio,
            "contrast_score": contrast_score,
            "absolute_darkness_ok": absolute_darkness_ok,
            "local_contrast_ok": local_contrast_ok,
            "contrast_ratio_ok": contrast_ratio_ok
        }
    
    def _calculate_edge_strength(self, contour: np.ndarray, image: np.ndarray) -> float:
        """Calculate edge strength around a contour."""
        # Create mask for contour
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, 2)
        
        # Calculate gradient magnitude along contour
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Average gradient magnitude along contour
        masked_gradient = gradient_magnitude[mask > 0]
        if len(masked_gradient) == 0:
            return 0.0
        
        avg_gradient = masked_gradient.mean()
        return min(avg_gradient / 100.0, 1.0)  # Normalize
    
    def _calculate_polygon_angles(self, polygon: np.ndarray) -> List[float]:
        """Calculate interior angles of a polygon."""
        angles = []
        n = len(polygon)
        
        for i in range(n):
            # Get three consecutive points
            p1 = polygon[i][0]
            p2 = polygon[(i + 1) % n][0]
            p3 = polygon[(i + 2) % n][0]
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)
        
        return angles
    
    def _remove_duplicate_features(self, features: List[ActualFeature]) -> List[ActualFeature]:
        """Remove duplicate feature detections based on proximity and size."""
        if len(features) <= 1:
            return features
        
        # Sort by confidence (highest first)
        sorted_features = sorted(features, key=lambda f: f.confidence, reverse=True)
        unique_features = []
        
        for feature in sorted_features:
            is_duplicate = False
            
            for existing in unique_features:
                # Check if features are too close and similar
                if self._are_features_duplicates(feature, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_features.append(feature)
        
        logger.debug(f"Removed {len(features) - len(unique_features)} duplicate features")
        return unique_features
    
    def _are_features_duplicates(self, feature1: ActualFeature, feature2: ActualFeature) -> bool:
        """Check if two features are likely duplicates using configurable thresholds."""
        # Must be same type
        if feature1.feature_type != feature2.feature_type:
            return False
        
        # Check center distance
        center_distance = feature1.center.distance_to(feature2.center)
        
        # Check size similarity
        if feature1.radius is not None and feature2.radius is not None:
            avg_radius = (feature1.radius + feature2.radius) / 2
            size_threshold = avg_radius * DUPLICATE_REMOVAL_SIZE_TOLERANCE
            size_difference = abs(feature1.radius - feature2.radius)
            
            return (center_distance < DUPLICATE_REMOVAL_DISTANCE_THRESHOLD and 
                   size_difference < size_threshold)
        
        # For non-circular features, use distance threshold only
        return center_distance < DUPLICATE_REMOVAL_DISTANCE_THRESHOLD
    
    def _generate_feature_id(self, prefix: str) -> str:
        """Generate a unique feature ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _get_configuration_snapshot(self) -> Dict[str, Any]:
        """Get current configuration for reproducibility."""
        return {
            "contour_min_area": CONTOUR_MIN_AREA,
            "contour_max_area": CONTOUR_MAX_AREA,
            "circle_min_radius": CIRCLE_MIN_RADIUS_PIXELS,
            "circle_max_radius": CIRCLE_MAX_RADIUS_PIXELS,
            "circle_roundness_threshold": CIRCLE_ROUNDNESS_THRESHOLD,
            "hole_darkness_threshold": HOLE_DARKNESS_THRESHOLD,
            "hole_local_contrast_threshold": HOLE_LOCAL_CONTRAST_THRESHOLD,
            "hole_annulus_width": HOLE_ANNULUS_WIDTH_PIXELS,
            "hole_min_contrast_ratio": HOLE_MIN_CONTRAST_RATIO,
            "rect_hole_min_vertices": RECT_HOLE_MIN_VERTICES,
            "rect_hole_max_vertices": RECT_HOLE_MAX_VERTICES,
            "confidence_base_score": CONFIDENCE_BASE_SCORE,
            "duplicate_removal_distance": DUPLICATE_REMOVAL_DISTANCE_THRESHOLD,
            "duplicate_removal_size_tolerance": DUPLICATE_REMOVAL_SIZE_TOLERANCE
        }