"""
Rectangle/Square Feature Extractor for Phase 2B

Detects rectangular and square features from preprocessed product images
using contour analysis and geometric validation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from .feature_models import ActualFeature, ActualFeatureType, GeometricProperties, EvidenceMetrics
from ..config import (
    RECTANGLE_MIN_AREA, RECTANGLE_MAX_AREA, RECTANGLE_MIN_SIDE_LENGTH,
    RECTANGLE_CONTOUR_APPROX_EPSILON, RECTANGLE_ASPECT_RATIO_TOLERANCE,
    RECTANGLE_ANGLE_TOLERANCE, RECTANGLE_SIDE_RATIO_TOLERANCE,
    FEATURE_MIN_CONFIDENCE, FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD,
    EDGE_EVIDENCE_WEIGHT, CONTOUR_EVIDENCE_WEIGHT, GEOMETRY_EVIDENCE_WEIGHT
)

logger = logging.getLogger(__name__)


class RectangleExtractor:
    """
    Extracts rectangular and square features from preprocessed product images.
    
    Uses contour analysis with geometric validation:
    1. Contour approximation to polygons
    2. Four-sided polygon validation
    3. Right angle validation
    4. Side length relationships
    5. Edge support validation
    """
    
    def __init__(self):
        """Initialize rectangle extractor with configuration parameters."""
        self.min_area = RECTANGLE_MIN_AREA
        self.max_area = RECTANGLE_MAX_AREA
        self.min_side_length = RECTANGLE_MIN_SIDE_LENGTH
        self.approx_epsilon = RECTANGLE_CONTOUR_APPROX_EPSILON
        self.aspect_ratio_tolerance = RECTANGLE_ASPECT_RATIO_TOLERANCE
        self.angle_tolerance = RECTANGLE_ANGLE_TOLERANCE
        self.side_ratio_tolerance = RECTANGLE_SIDE_RATIO_TOLERANCE
        self.min_confidence = FEATURE_MIN_CONFIDENCE
        self.mask_overlap_threshold = FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD
    
    def extract_rectangles(self, 
                          internal_edges: np.ndarray,
                          isolated_product: np.ndarray,
                          product_mask: np.ndarray,
                          raw_internal_edges: Optional[np.ndarray] = None) -> List[ActualFeature]:
        """
        Extract rectangular features from preprocessed representations.
        
        Args:
            internal_edges: Primary geometric edge representation
            isolated_product: Primary visual representation
            product_mask: Spatial constraint mask
            raw_internal_edges: Supporting edge evidence
            
        Returns:
            List of detected rectangular/square features
        """
        logger.debug("Starting rectangle extraction")
        
        candidates = []
        
        # Method 1: Primary edge-based detection
        edge_rectangles = self._extract_contour_rectangles(internal_edges, product_mask)
        candidates.extend(edge_rectangles)
        
        # Method 2: Raw edge fallback for missed features
        if raw_internal_edges is not None:
            raw_rectangles = self._extract_contour_rectangles(raw_internal_edges, product_mask,
                                                            source_rep="raw_internal_edges")
            candidates.extend(raw_rectangles)
        
        # Validate and refine candidates
        validated_rectangles = []
        for candidate in candidates:
            if self._validate_rectangle_candidate(candidate, internal_edges, isolated_product, product_mask):
                validated_rectangles.append(candidate)
        
        logger.info(f"Rectangle extraction: {len(candidates)} candidates -> {len(validated_rectangles)} validated")
        return validated_rectangles
    
    def _extract_contour_rectangles(self, edge_image: np.ndarray,
                                   product_mask: np.ndarray,
                                   source_rep: str = "internal_geometry_edges") -> List[ActualFeature]:
        """Extract rectangles using contour analysis."""
        rectangles = []
        
        # Find contours in edge image
        contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Basic area filtering
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.approx_epsilon * perimeter
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Must be a quadrilateral
            if len(approx_contour) != 4:
                continue
            
            # Extract corner points
            corners = approx_contour.reshape(4, 2)
            
            # Validate rectangle geometry
            if not self._validate_rectangle_geometry(corners):
                continue
            
            # Calculate geometric properties
            geometry_props = self._calculate_rectangle_geometry(corners, area, perimeter)
            if geometry_props is None:
                continue
            
            # Check overlap with product mask
            if not self._check_product_mask_overlap(geometry_props.center, corners, product_mask):
                continue
            
            # Determine if it's a square or rectangle
            aspect_ratio = geometry_props.aspect_ratio
            is_square = abs(aspect_ratio - 1.0) < self.aspect_ratio_tolerance
            feature_type = ActualFeatureType.SQUARE if is_square else ActualFeatureType.RECTANGLE
            
            # Calculate evidence metrics
            edge_support = self._calculate_edge_support(corners, edge_image)
            rectangularity = self._calculate_rectangularity(corners)
            
            evidence = EvidenceMetrics(
                confidence=self._calculate_contour_confidence(geometry_props, edge_support, rectangularity),
                edge_support=edge_support,
                contour_quality=rectangularity,
                intensity_consistency=0.5,  # Will be refined in validation
                geometric_consistency=rectangularity,
                internal_edge_evidence=1.0 if source_rep == "internal_geometry_edges" else 0.0,
                raw_edge_evidence=1.0 if source_rep == "raw_internal_edges" else 0.0
            )
            
            # Create feature
            center_x, center_y = geometry_props.center
            feature_id = f"rectangle_{len(rectangles)}_{int(center_x)}_{int(center_y)}"
            
            rectangle = ActualFeature(
                feature_id=feature_id,
                feature_type=feature_type,
                geometry=geometry_props,
                contour=approx_contour,
                evidence=evidence,
                source_representation=source_rep,
                detection_method="contour_analysis"
            )
            
            rectangles.append(rectangle)
        
        return rectangles
    
    def _validate_rectangle_geometry(self, corners: np.ndarray) -> bool:
        """Validate that corners form a valid rectangle."""
        # Order corners (top-left, top-right, bottom-right, bottom-left)
        corners = self._order_rectangle_corners(corners)
        
        # Calculate side lengths
        sides = []
        for i in range(4):
            next_i = (i + 1) % 4
            side_length = np.linalg.norm(corners[next_i] - corners[i])
            sides.append(side_length)
        
        # Check minimum side length
        if any(side < self.min_side_length for side in sides):
            return False
        
        # Check that opposite sides are approximately equal
        side1, side2, side3, side4 = sides
        
        # Opposite sides should be similar
        ratio1 = min(side1, side3) / max(side1, side3)
        ratio2 = min(side2, side4) / max(side2, side4)
        
        if ratio1 < (1.0 - self.side_ratio_tolerance) or ratio2 < (1.0 - self.side_ratio_tolerance):
            return False
        
        # Check angles are approximately 90 degrees
        angles = self._calculate_corner_angles(corners)
        for angle in angles:
            if abs(angle - 90) > self.angle_tolerance:
                return False
        
        return True
    
    def _order_rectangle_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as top-left, top-right, bottom-right, bottom-left."""
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Calculate angles from centroid to each corner
        angles = []
        for corner in corners:
            dx = corner[0] - centroid[0]
            dy = corner[1] - centroid[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        ordered_corners = corners[sorted_indices]
        
        return ordered_corners
    
    def _calculate_corner_angles(self, corners: np.ndarray) -> List[float]:
        """Calculate interior angles at each corner."""
        angles = []
        
        for i in range(4):
            prev_i = (i - 1) % 4
            next_i = (i + 1) % 4
            
            # Vectors from current corner to neighbors
            vec1 = corners[prev_i] - corners[i]
            vec2 = corners[next_i] - corners[i]
            
            # Calculate angle
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
            angle = np.degrees(np.arccos(cos_angle))
            
            angles.append(angle)
        
        return angles
    
    def _calculate_rectangle_geometry(self, corners: np.ndarray, 
                                    area: float, perimeter: float) -> Optional[GeometricProperties]:
        """Calculate complete geometric properties for rectangle."""
        # Order corners
        corners = self._order_rectangle_corners(corners)
        
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Calculate side lengths
        width = np.linalg.norm(corners[1] - corners[0])  # top side
        height = np.linalg.norm(corners[3] - corners[0])  # left side
        
        # Ensure width >= height for consistency
        if height > width:
            width, height = height, width
        
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Calculate bounding box
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        # Calculate geometric quality metrics
        rectangularity = self._calculate_rectangularity(corners)
        
        # Calculate convex hull for solidity
        hull = cv2.convexHull(corners)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        extent = area / (bbox_w * bbox_h) if (bbox_w * bbox_h) > 0 else 0.0
        
        return GeometricProperties(
            center=(float(center[0]), float(center[1])),
            area=area,
            perimeter=perimeter,
            bounding_box=(x_min, y_min, bbox_w, bbox_h),
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            circularity=0.0,  # Not applicable for rectangles
            solidity=solidity,
            extent=extent,
            convexity=1.0  # Rectangles are convex
        )
    
    def _calculate_rectangularity(self, corners: np.ndarray) -> float:
        """Calculate how well the shape approximates a rectangle."""
        # Check angle consistency (all should be ~90 degrees)
        angles = self._calculate_corner_angles(corners)
        angle_deviations = [abs(angle - 90) for angle in angles]
        max_angle_deviation = max(angle_deviations)
        
        angle_score = max(0.0, 1.0 - max_angle_deviation / 45.0)  # Normalize by 45 degrees
        
        # Check side ratios (opposite sides should be equal)
        sides = []
        for i in range(4):
            next_i = (i + 1) % 4
            side_length = np.linalg.norm(corners[next_i] - corners[i])
            sides.append(side_length)
        
        side1, side2, side3, side4 = sides
        ratio1 = min(side1, side3) / max(side1, side3)
        ratio2 = min(side2, side4) / max(side2, side4)
        
        side_score = (ratio1 + ratio2) / 2.0
        
        # Combined rectangularity score
        rectangularity = (angle_score + side_score) / 2.0
        
        return rectangularity
    
    def _check_product_mask_overlap(self, center: Tuple[float, float],
                                   corners: np.ndarray, product_mask: np.ndarray) -> bool:
        """Check if rectangle overlaps sufficiently with product mask."""
        # Create rectangle mask
        h, w = product_mask.shape
        rect_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(rect_mask, [corners.astype(np.int32)], 255)
        
        # Calculate overlap
        intersection = cv2.bitwise_and(rect_mask, product_mask)
        intersection_area = np.count_nonzero(intersection)
        rect_area = np.count_nonzero(rect_mask)
        
        if rect_area == 0:
            return False
        
        overlap_ratio = intersection_area / rect_area
        return overlap_ratio >= self.mask_overlap_threshold
    
    def _calculate_edge_support(self, corners: np.ndarray, edge_image: np.ndarray) -> float:
        """Calculate fraction of rectangle perimeter supported by edges."""
        total_support = 0.0
        total_length = 0.0
        
        h, w = edge_image.shape
        
        # Check each side of the rectangle
        for i in range(4):
            next_i = (i + 1) % 4
            start_point = corners[i]
            end_point = corners[next_i]
            
            # Sample points along the side
            side_length = np.linalg.norm(end_point - start_point)
            num_samples = max(5, int(side_length / 2))  # Sample every ~2 pixels
            
            supported_samples = 0
            valid_samples = 0
            
            for j in range(num_samples + 1):
                t = j / num_samples if num_samples > 0 else 0
                point = start_point + t * (end_point - start_point)
                x, y = int(point[0]), int(point[1])
                
                # Check bounds
                if 0 <= x < w and 0 <= y < h:
                    valid_samples += 1
                    # Check for edge support in neighborhood
                    if self._check_edge_neighborhood(x, y, edge_image, neighborhood_size=3):
                        supported_samples += 1
            
            if valid_samples > 0:
                side_support = supported_samples / valid_samples
                total_support += side_support * side_length
                total_length += side_length
        
        if total_length == 0:
            return 0.0
        
        return total_support / total_length
    
    def _check_edge_neighborhood(self, x: int, y: int, edge_image: np.ndarray,
                                neighborhood_size: int = 3) -> bool:
        """Check if there's an edge in the neighborhood of a point."""
        h, w = edge_image.shape
        half_size = neighborhood_size // 2
        
        y_start = max(0, y - half_size)
        y_end = min(h, y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(w, x + half_size + 1)
        
        neighborhood = edge_image[y_start:y_end, x_start:x_end]
        return np.any(neighborhood > 0)
    
    def _validate_rectangle_candidate(self, candidate: ActualFeature,
                                    internal_edges: np.ndarray,
                                    isolated_product: np.ndarray,
                                    product_mask: np.ndarray) -> bool:
        """Validate rectangle candidate using multiple evidence sources."""
        
        # Calculate intensity consistency
        intensity_consistency = self._calculate_intensity_consistency(
            candidate.contour, isolated_product
        )
        candidate.evidence.intensity_consistency = intensity_consistency
        
        # Update overall confidence
        confidence = self._calculate_final_confidence(candidate.evidence, intensity_consistency)
        candidate.evidence.confidence = confidence
        
        # Apply confidence threshold
        if confidence < self.min_confidence:
            return False
        
        return True
    
    def _calculate_intensity_consistency(self, contour: np.ndarray,
                                       isolated_product: np.ndarray) -> float:
        """Calculate consistency of intensity within rectangle."""
        # Create rectangle mask
        h, w = isolated_product.shape[:2]
        rect_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(rect_mask, [contour], 255)
        
        # Extract pixel intensities
        if len(isolated_product.shape) == 3:
            gray = cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY)
        else:
            gray = isolated_product
        
        pixels = gray[rect_mask > 0]
        
        if len(pixels) == 0:
            return 0.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_intensity = np.mean(pixels)
        std_intensity = np.std(pixels)
        
        if mean_intensity == 0:
            return 0.0
        
        cv = std_intensity / mean_intensity
        # Convert to consistency score (higher is better)
        consistency = max(0.0, 1.0 - cv)
        
        return consistency
    
    def _calculate_contour_confidence(self, geometry: GeometricProperties,
                                    edge_support: float, rectangularity: float) -> float:
        """Calculate confidence for contour-based rectangle detection."""
        # Combine multiple factors
        rectangularity_score = rectangularity
        solidity_score = geometry.solidity
        edge_score = edge_support
        
        # Weighted combination
        confidence = (
            GEOMETRY_EVIDENCE_WEIGHT * rectangularity_score +
            CONTOUR_EVIDENCE_WEIGHT * solidity_score +
            EDGE_EVIDENCE_WEIGHT * edge_score
        )
        
        return min(1.0, confidence)
    
    def _calculate_final_confidence(self, evidence: EvidenceMetrics,
                                   intensity_consistency: float) -> float:
        """Calculate final confidence combining all evidence sources."""
        return (
            EDGE_EVIDENCE_WEIGHT * evidence.edge_support +
            CONTOUR_EVIDENCE_WEIGHT * evidence.contour_quality +
            0.2 * intensity_consistency +  # Intensity evidence weight
            GEOMETRY_EVIDENCE_WEIGHT * evidence.geometric_consistency
        )