"""
General Contour Feature Extractor for Phase 2B

Extracts meaningful geometric features that don't fit circle/rectangle
categories, preserving them as general contour features.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from .feature_models import ActualFeature, ActualFeatureType, GeometricProperties, EvidenceMetrics
from ..config import (
    CONTOUR_MIN_AREA, CONTOUR_MAX_AREA, CONTOUR_MIN_PERIMETER,
    CONTOUR_APPROX_EPSILON, CONTOUR_CONVEXITY_THRESHOLD, CONTOUR_SOLIDITY_THRESHOLD,
    FEATURE_MIN_CONFIDENCE, FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD,
    EDGE_EVIDENCE_WEIGHT, CONTOUR_EVIDENCE_WEIGHT, GEOMETRY_EVIDENCE_WEIGHT,
    CONTOUR_EDGE_SUPPORT_INTERPOLATION_THRESHOLD, CONTOUR_HIERARCHY_FILTERING_ENABLED,
    CONTOUR_MIN_AREA_RATIO, CONTOUR_MAX_ENCLOSING_RATIO, CONTOUR_MULTIPLE_PROPERTY_VALIDATION,
    CONTOUR_MIN_SOLIDITY_ENHANCED, CONTOUR_MIN_EXTENT, CONTOUR_MAX_ASPECT_RATIO,
    CONTOUR_INTENSITY_EVIDENCE_WEIGHT, CONTOUR_SIGNIFICANCE_VALIDATION,
    CONTOUR_MIN_AREA_ENHANCED, CONTOUR_MAX_AREA_IMAGE_FRACTION,
    CONTOUR_GEOMETRIC_STABILITY_REQUIRED
)

logger = logging.getLogger(__name__)


class ContourExtractor:
    """
    Extracts general contour features from preprocessed product images.
    
    Preserves meaningful geometric features that don't fit standard
    primitive categories (circles, rectangles) but still represent
    valid product geometry.
    """
    
    def __init__(self):
        """Initialize contour extractor with configuration parameters."""
        self.min_area = CONTOUR_MIN_AREA
        self.max_area = CONTOUR_MAX_AREA
        self.min_perimeter = CONTOUR_MIN_PERIMETER
        self.approx_epsilon = CONTOUR_APPROX_EPSILON
        self.convexity_threshold = CONTOUR_CONVEXITY_THRESHOLD
        self.solidity_threshold = CONTOUR_SOLIDITY_THRESHOLD
        self.min_confidence = FEATURE_MIN_CONFIDENCE
        self.mask_overlap_threshold = FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD
        
        # Enhanced validation parameters
        self.hierarchy_filtering = CONTOUR_HIERARCHY_FILTERING_ENABLED
        self.min_area_ratio = CONTOUR_MIN_AREA_RATIO
        self.max_enclosing_ratio = CONTOUR_MAX_ENCLOSING_RATIO
        self.multiple_property_validation = CONTOUR_MULTIPLE_PROPERTY_VALIDATION
        self.enhanced_solidity_threshold = CONTOUR_MIN_SOLIDITY_ENHANCED
        self.min_extent = CONTOUR_MIN_EXTENT
        self.max_aspect_ratio = CONTOUR_MAX_ASPECT_RATIO
        self.interpolation_threshold = CONTOUR_EDGE_SUPPORT_INTERPOLATION_THRESHOLD
        self.intensity_weight = CONTOUR_INTENSITY_EVIDENCE_WEIGHT
        
        # Enhanced significance validation parameters
        self.significance_validation = CONTOUR_SIGNIFICANCE_VALIDATION
        self.enhanced_min_area = CONTOUR_MIN_AREA_ENHANCED
        self.max_area_image_fraction = CONTOUR_MAX_AREA_IMAGE_FRACTION
        self.geometric_stability_required = CONTOUR_GEOMETRIC_STABILITY_REQUIRED
    
    def extract_contours(self,
                        internal_edges: np.ndarray,
                        isolated_product: np.ndarray,
                        product_mask: np.ndarray,
                        raw_internal_edges: Optional[np.ndarray] = None) -> List[ActualFeature]:
        """
        Extract general contour features from preprocessed representations.
        
        Args:
            internal_edges: Primary geometric edge representation
            isolated_product: Primary visual representation
            product_mask: Spatial constraint mask
            raw_internal_edges: Supporting edge evidence
            
        Returns:
            List of detected general contour features
        """
        logger.debug("Starting general contour extraction")
        
        candidates = []
        
        # Method 1: Primary edge-based detection
        edge_contours = self._extract_general_contours(internal_edges, product_mask)
        candidates.extend(edge_contours)
        
        # Method 2: Raw edge fallback for missed features
        if raw_internal_edges is not None:
            raw_contours = self._extract_general_contours(raw_internal_edges, product_mask,
                                                         source_rep="raw_internal_edges")
            candidates.extend(raw_contours)
        
        # Validate and refine candidates
        validated_contours = []
        for candidate in candidates:
            if self._validate_contour_candidate(candidate, internal_edges, isolated_product, product_mask):
                validated_contours.append(candidate)
        
        logger.info(f"Contour extraction: {len(candidates)} candidates -> {len(validated_contours)} validated")
        return validated_contours
    
    def _extract_general_contours(self, edge_image: np.ndarray,
                                 product_mask: np.ndarray,
                                 source_rep: str = "internal_geometry_edges") -> List[ActualFeature]:
        """Extract general contours with improved hierarchy handling."""
        contours_list = []
        
        # Find contours with hierarchy information
        contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate product mask area for enclosing contour detection
        product_mask_area = np.count_nonzero(product_mask)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Basic filtering with enhanced significance validation
            if self.significance_validation:
                # Use enhanced minimum area
                if area < self.enhanced_min_area or area > self.max_area:
                    continue
                
                # Check image-relative maximum area
                image_area = edge_image.shape[0] * edge_image.shape[1]
                max_area_absolute = image_area * self.max_area_image_fraction
                if area > max_area_absolute:
                    continue
            else:
                # Original basic filtering
                if area < self.min_area or area > self.max_area:
                    continue
            
            if perimeter < self.min_perimeter:
                continue
            
            # Hierarchy-based filtering to reduce duplicates
            if self.hierarchy_filtering and hierarchy is not None:
                # Skip contours that are too small compared to their parent
                if self._is_insignificant_nested_contour(i, contour, contours, hierarchy[0], area):
                    continue
                
                # Skip contours that are too large (likely product boundary)
                if self._is_enclosing_product_contour(contour, product_mask_area, area):
                    continue
            
            # Filter out simple primitives that should be handled by specialized extractors
            if self._is_simple_circle(contour, area, perimeter):
                continue
            
            if self._is_simple_rectangle(contour):
                continue
            
            # Calculate geometric properties
            geometry_props = self._calculate_contour_geometry(contour, area, perimeter)
            if geometry_props is None:
                continue
            
            # Geometric stability validation
            if self.geometric_stability_required:
                if not self._validate_geometric_stability(contour, geometry_props):
                    continue
            
            # Enhanced quality thresholds when enabled
            if self.multiple_property_validation:
                if not self._passes_enhanced_validation(geometry_props):
                    continue
            else:
                # Original validation
                if geometry_props.convexity < self.convexity_threshold:
                    continue
                    
                if geometry_props.solidity < self.solidity_threshold:
                    continue
            
            # Check overlap with product mask
            if not self._check_product_mask_overlap(contour, product_mask):
                continue
            
            # Calculate evidence metrics
            edge_support = self._calculate_edge_support(contour, edge_image)
            contour_quality = self._calculate_contour_quality(geometry_props)
            
            evidence = EvidenceMetrics(
                confidence=self._calculate_contour_confidence(geometry_props, edge_support, contour_quality),
                edge_support=edge_support,
                contour_quality=contour_quality,
                intensity_consistency=0.5,  # Will be refined in validation
                geometric_consistency=contour_quality,
                internal_edge_evidence=1.0 if source_rep == "internal_geometry_edges" else 0.0,
                raw_edge_evidence=1.0 if source_rep == "raw_internal_edges" else 0.0
            )
            
            # Create feature
            center_x, center_y = geometry_props.center
            feature_id = f"contour_{len(contours_list)}_{int(center_x)}_{int(center_y)}"
            
            contour_feature = ActualFeature(
                feature_id=feature_id,
                feature_type=ActualFeatureType.GENERAL_CONTOUR,
                geometry=geometry_props,
                contour=contour,
                evidence=evidence,
                source_representation=source_rep,
                detection_method="contour_analysis"
            )
            
            contours_list.append(contour_feature)
        
        return contours_list
    
    def _is_simple_circle(self, contour: np.ndarray, area: float, perimeter: float) -> bool:
        """Check if contour is a simple circle (should be handled by circle extractor)."""
        if perimeter == 0:
            return False
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity > 0.7  # High circularity threshold
    
    def _is_simple_rectangle(self, contour: np.ndarray) -> bool:
        """Check if contour is a simple rectangle (should be handled by rectangle extractor)."""
        # Approximate contour to polygon
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If it approximates to 4 points, it's likely a rectangle
        return len(approx) == 4
    
    def _calculate_contour_geometry(self, contour: np.ndarray,
                                   area: float, perimeter: float) -> Optional[GeometricProperties]:
        """Calculate complete geometric properties for general contour."""
        # Calculate moments
        moments = cv2.moments(contour)
        
        # Calculate center (centroid)
        if moments['m00'] == 0:
            return None
        
        center_x = moments['m10'] / moments['m00']
        center_y = moments['m01'] / moments['m00']
        center = (float(center_x), float(center_y))
        
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # Calculate geometric quality metrics
        solidity = area / hull_area if hull_area > 0 else 0.0
        extent = area / (w * h) if (w * h) > 0 else 0.0
        convexity = hull_perimeter / perimeter if perimeter > 0 else 0.0
        
        # Calculate circularity for reference
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        
        return GeometricProperties(
            center=center,
            area=area,
            perimeter=perimeter,
            bounding_box=(x, y, w, h),
            width=float(w),
            height=float(h),
            aspect_ratio=float(w) / float(h) if h > 0 else 1.0,
            circularity=circularity,
            solidity=solidity,
            extent=extent,
            convexity=convexity
        )
    
    def _calculate_contour_quality(self, geometry: GeometricProperties) -> float:
        """Calculate overall quality score for contour."""
        # Combine multiple geometric quality factors
        solidity_score = geometry.solidity
        convexity_score = geometry.convexity
        extent_score = geometry.extent
        
        # Weighted combination
        quality = (
            0.4 * solidity_score +
            0.3 * convexity_score +
            0.3 * extent_score
        )
        
        return min(1.0, quality)
    
    def _check_product_mask_overlap(self, contour: np.ndarray, product_mask: np.ndarray) -> bool:
        """Check if contour overlaps sufficiently with product mask."""
        # Create contour mask
        h, w = product_mask.shape
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(contour_mask, [contour], 255)
        
        # Calculate overlap
        intersection = cv2.bitwise_and(contour_mask, product_mask)
        intersection_area = np.count_nonzero(intersection)
        contour_area = np.count_nonzero(contour_mask)
        
        if contour_area == 0:
            return False
        
        overlap_ratio = intersection_area / contour_area
        return overlap_ratio >= self.mask_overlap_threshold
    
    def _calculate_edge_support(self, contour: np.ndarray, edge_image: np.ndarray) -> float:
        """Calculate fraction of contour perimeter supported by edges."""
        # Sample points along contour perimeter
        perimeter_points = []
        
        # Get contour points
        if len(contour.shape) == 3:
            contour_points = contour.reshape(-1, 2)
        else:
            contour_points = contour
        
        # Interpolate additional points for better coverage using configurable threshold
        interpolated_points = []
        for i in range(len(contour_points)):
            current_point = contour_points[i]
            next_point = contour_points[(i + 1) % len(contour_points)]
            
            # Add current point
            interpolated_points.append(current_point)
            
            # Add interpolated points between current and next
            dist = np.linalg.norm(next_point - current_point)
            if dist > self.interpolation_threshold:  # Use configurable threshold
                num_interp = int(dist / self.interpolation_threshold)
                for j in range(1, num_interp):
                    t = j / num_interp
                    interp_point = current_point + t * (next_point - current_point)
                    interpolated_points.append(interp_point.astype(int))
        
        if not interpolated_points:
            return 0.0
        
        # Check edge support for each point
        supported_points = 0
        valid_points = 0
        h, w = edge_image.shape
        
        for point in interpolated_points:
            x, y = int(point[0]), int(point[1])
            
            # Check bounds
            if 0 <= x < w and 0 <= y < h:
                valid_points += 1
                # Check for edge support in neighborhood
                if self._check_edge_neighborhood(x, y, edge_image, neighborhood_size=3):
                    supported_points += 1
        
        if valid_points == 0:
            return 0.0
        
        return supported_points / valid_points
    
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
    
    def _validate_contour_candidate(self, candidate: ActualFeature,
                                   internal_edges: np.ndarray,
                                   isolated_product: np.ndarray,
                                   product_mask: np.ndarray) -> bool:
        """Validate contour candidate using multiple evidence sources."""
        
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
        """Calculate consistency of intensity within contour."""
        # Create contour mask
        h, w = isolated_product.shape[:2]
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(contour_mask, [contour], 255)
        
        # Extract pixel intensities
        if len(isolated_product.shape) == 3:
            gray = cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY)
        else:
            gray = isolated_product
        
        pixels = gray[contour_mask > 0]
        
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
    
    def _is_insignificant_nested_contour(self, index: int, contour: np.ndarray, 
                                        all_contours: List[np.ndarray], hierarchy: np.ndarray,
                                        contour_area: float) -> bool:
        """Check if contour is an insignificant nested contour compared to its parent."""
        # Check if this contour has a parent
        parent_index = hierarchy[index][3]  # Parent index
        if parent_index == -1:
            return False  # No parent, not nested
        
        # Calculate parent area
        parent_area = cv2.contourArea(all_contours[parent_index])
        if parent_area == 0:
            return False
        
        # Check if this contour is too small relative to parent
        area_ratio = contour_area / parent_area
        return area_ratio < self.min_area_ratio
    
    def _is_enclosing_product_contour(self, contour: np.ndarray, 
                                     product_mask_area: int, contour_area: float) -> bool:
        """Check if contour is likely the outer product boundary."""
        if product_mask_area == 0:
            return False
        
        # Check if contour area is too large relative to product mask
        area_ratio = contour_area / product_mask_area
        return area_ratio > self.max_enclosing_ratio
    
    def _passes_enhanced_validation(self, geometry: GeometricProperties) -> bool:
        """Enhanced validation using multiple geometric properties."""
        # Check enhanced solidity threshold
        if geometry.solidity < self.enhanced_solidity_threshold:
            return False
        
        # Check minimum extent
        if geometry.extent < self.min_extent:
            return False
        
        # Check aspect ratio bounds
        if geometry.aspect_ratio > self.max_aspect_ratio:
            return False
        
        # Check convexity (original threshold)
        if geometry.convexity < self.convexity_threshold:
            return False
        
        return True
    
    def _validate_geometric_stability(self, contour: np.ndarray, geometry: GeometricProperties) -> bool:
        """Validate geometric stability to reject noise and fragments."""
        # Check perimeter to area ratio - should be reasonable for meaningful geometry
        if geometry.area > 0:
            perimeter_area_ratio = geometry.perimeter / np.sqrt(geometry.area)
            if perimeter_area_ratio > 15.0:  # Very fragmented/noisy contours
                return False
        
        # Check extent - contour should fill reasonable portion of bounding box  
        if geometry.extent < 0.15:  # Less than 15% of bounding box
            return False
        
        # Check contour point density - too few points suggests poor quality
        num_points = len(contour) if len(contour.shape) == 2 else len(contour.reshape(-1, 2))
        if num_points < 5:  # Very few contour points
            return False
        
        # Check for degenerate shapes (very thin or extreme aspect ratios)
        if geometry.aspect_ratio > 20.0:  # Extremely thin shapes
            return False
        
        return True
    
    def _calculate_contour_confidence(self, geometry: GeometricProperties,
                                    edge_support: float, contour_quality: float) -> float:
        """Calculate confidence for contour-based detection."""
        # Combine multiple factors
        quality_score = contour_quality
        solidity_score = geometry.solidity
        edge_score = edge_support
        
        # Weighted combination
        confidence = (
            GEOMETRY_EVIDENCE_WEIGHT * quality_score +
            CONTOUR_EVIDENCE_WEIGHT * solidity_score +
            EDGE_EVIDENCE_WEIGHT * edge_score
        )
        
        return min(1.0, confidence)
    
    def _calculate_final_confidence(self, evidence: EvidenceMetrics,
                                   intensity_consistency: float) -> float:
        """Calculate final confidence combining all evidence sources with configurable weights."""
        return (
            EDGE_EVIDENCE_WEIGHT * evidence.edge_support +
            CONTOUR_EVIDENCE_WEIGHT * evidence.contour_quality +
            self.intensity_weight * intensity_consistency +  # Use configurable intensity evidence weight
            GEOMETRY_EVIDENCE_WEIGHT * evidence.geometric_consistency
        )