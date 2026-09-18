"""
Circle Feature Extractor for Phase 2B

Detects circular features from preprocessed product images using multiple
geometric and edge-based evidence sources.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from .feature_models import ActualFeature, ActualFeatureType, GeometricProperties, EvidenceMetrics
from ..config import (
    CIRCLE_MIN_RADIUS, CIRCLE_MAX_RADIUS, CIRCLE_MIN_AREA, CIRCLE_MAX_AREA,
    CIRCLE_CIRCULARITY_THRESHOLD, CIRCLE_CONTOUR_APPROX_EPSILON,
    CIRCLE_HOUGH_ACCUMULATOR_THRESHOLD, CIRCLE_MIN_CENTER_DISTANCE,
    CIRCLE_EDGE_SUPPORT_THRESHOLD, FEATURE_MIN_CONFIDENCE,
    FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD, EDGE_EVIDENCE_WEIGHT,
    CONTOUR_EVIDENCE_WEIGHT, INTENSITY_EVIDENCE_WEIGHT, GEOMETRY_EVIDENCE_WEIGHT,
    HOUGH_CIRCLE_PARAM1, HOUGH_CIRCLE_PARAM2_MULTIPLIER, HOUGH_CIRCLE_BLUR_KERNEL_SIZE,
    HOUGH_CIRCLE_DP, HOUGH_CIRCLE_MIN_DIST_MULTIPLIER, HOUGH_CIRCLE_CONTOUR_SAMPLING_POINTS,
    EDGE_SUPPORT_SAMPLING_POINTS, EDGE_SUPPORT_NEIGHBORHOOD_SIZE,
    HOUGH_CONFIDENCE_RADIUS_WEIGHT, HOUGH_CONFIDENCE_EDGE_WEIGHT,
    CONTOUR_INTENSITY_EVIDENCE_WEIGHT, HOUGH_GEOMETRIC_EVIDENCE_REQUIRED,
    HOUGH_EDGE_SUPPORT_MULTIPLIER, HOUGH_RADIUS_ANOMALY_DETECTION,
    HOUGH_MAX_RADIUS_PERCENTILE, HOUGH_AREA_POPULATION_THRESHOLD,
    ENHANCED_DUPLICATE_SUPPRESSION, SAME_TYPE_CENTER_DISTANCE_STRICT,
    HOUGH_CLUSTER_CONSOLIDATION, HOUGH_CLUSTER_RADIUS_TOLERANCE,
    CONFIDENCE_GEOMETRIC_VALIDATION, HOUGH_CONFIDENCE_GEOMETRIC_WEIGHT,
    HOUGH_ANGULAR_COVERAGE_REQUIRED, HOUGH_MIN_ANGULAR_COVERAGE,
    HOUGH_ANGULAR_SECTORS, HOUGH_MIN_SECTOR_COVERAGE,
    HOUGH_EDGE_CONTINUITY_REQUIRED, HOUGH_MAX_EDGE_GAP_RATIO,
    HOUGH_RADIAL_CONSISTENCY_REQUIRED, HOUGH_RADIAL_SAMPLES,
    HOUGH_MIN_RADIAL_AGREEMENT, HOUGH_RADIAL_TOLERANCE_PIXELS,
    HOUGH_RADIAL_SEARCH_STEP, HOUGH_CONTOUR_AGREEMENT_BONUS,
    HOUGH_ONLY_PENALTY, HOUGH_TEXTURE_DISCRIMINATION, HOUGH_MIN_LOCAL_CONTRAST,
    HOUGH_CLUSTER_CONSOLIDATION_ENHANCED, HOUGH_SCALE_AWARE_DISTANCE,
    HOUGH_CLUSTER_RADIUS_FACTOR, HOUGH_CONCENTRIC_DETECTION,
    HOUGH_CONCENTRIC_RADIUS_TOLERANCE, HOUGH_EVIDENCE_BASED_CONSOLIDATION
)

logger = logging.getLogger(__name__)


class CircleExtractor:
    """
    Extracts circular features from preprocessed product images.
    
    Uses multiple evidence sources:
    1. Contour-based circle detection
    2. Hough circle transform
    3. Edge support validation
    4. Geometric consistency checks
    """
    
    def __init__(self):
        """Initialize circle extractor with configuration parameters."""
        self.min_radius = CIRCLE_MIN_RADIUS
        self.max_radius = CIRCLE_MAX_RADIUS
        self.min_area = CIRCLE_MIN_AREA
        self.max_area = CIRCLE_MAX_AREA
        self.circularity_threshold = CIRCLE_CIRCULARITY_THRESHOLD
        self.approx_epsilon = CIRCLE_CONTOUR_APPROX_EPSILON
        self.hough_threshold = CIRCLE_HOUGH_ACCUMULATOR_THRESHOLD
        self.min_center_distance = CIRCLE_MIN_CENTER_DISTANCE
        self.edge_support_threshold = CIRCLE_EDGE_SUPPORT_THRESHOLD
        self.min_confidence = FEATURE_MIN_CONFIDENCE
        self.mask_overlap_threshold = FEATURE_PRODUCT_MASK_OVERLAP_THRESHOLD
        
        # Hough-specific parameters
        self.hough_param1 = HOUGH_CIRCLE_PARAM1
        self.hough_param2 = int(CIRCLE_HOUGH_ACCUMULATOR_THRESHOLD * HOUGH_CIRCLE_PARAM2_MULTIPLIER)
        self.hough_blur_kernel = HOUGH_CIRCLE_BLUR_KERNEL_SIZE
        self.hough_dp = HOUGH_CIRCLE_DP
        self.hough_min_dist = int(CIRCLE_MIN_CENTER_DISTANCE * HOUGH_CIRCLE_MIN_DIST_MULTIPLIER)
        self.hough_sampling_points = HOUGH_CIRCLE_CONTOUR_SAMPLING_POINTS
        
        # Edge support parameters
        self.edge_sampling_points = EDGE_SUPPORT_SAMPLING_POINTS
        self.edge_neighborhood_size = EDGE_SUPPORT_NEIGHBORHOOD_SIZE
        
        # Confidence calculation parameters
        self.hough_radius_weight = HOUGH_CONFIDENCE_RADIUS_WEIGHT
        self.hough_edge_weight = HOUGH_CONFIDENCE_EDGE_WEIGHT
        self.intensity_weight = CONTOUR_INTENSITY_EVIDENCE_WEIGHT
        
        # Enhanced validation parameters
        self.geometric_evidence_required = HOUGH_GEOMETRIC_EVIDENCE_REQUIRED
        self.edge_support_multiplier = HOUGH_EDGE_SUPPORT_MULTIPLIER
        self.radius_anomaly_detection = HOUGH_RADIUS_ANOMALY_DETECTION
        self.max_radius_percentile = HOUGH_MAX_RADIUS_PERCENTILE
        self.area_population_threshold = HOUGH_AREA_POPULATION_THRESHOLD
        self.enhanced_duplicate_suppression = ENHANCED_DUPLICATE_SUPPRESSION
        self.strict_center_distance = SAME_TYPE_CENTER_DISTANCE_STRICT
        self.cluster_consolidation = HOUGH_CLUSTER_CONSOLIDATION
        self.cluster_radius_tolerance = HOUGH_CLUSTER_RADIUS_TOLERANCE
        self.confidence_geometric_validation = CONFIDENCE_GEOMETRIC_VALIDATION
        self.confidence_geometric_weight = HOUGH_CONFIDENCE_GEOMETRIC_WEIGHT
        
        # Enhanced Hough image evidence validation parameters
        self.angular_coverage_required = HOUGH_ANGULAR_COVERAGE_REQUIRED
        self.min_angular_coverage = HOUGH_MIN_ANGULAR_COVERAGE
        self.angular_sectors = HOUGH_ANGULAR_SECTORS
        self.min_sector_coverage = HOUGH_MIN_SECTOR_COVERAGE
        self.edge_continuity_required = HOUGH_EDGE_CONTINUITY_REQUIRED
        self.max_edge_gap_ratio = HOUGH_MAX_EDGE_GAP_RATIO
        self.radial_consistency_required = HOUGH_RADIAL_CONSISTENCY_REQUIRED
        self.radial_samples = HOUGH_RADIAL_SAMPLES
        self.min_radial_agreement = HOUGH_MIN_RADIAL_AGREEMENT
        self.radial_tolerance_pixels = HOUGH_RADIAL_TOLERANCE_PIXELS
        self.radial_search_step = HOUGH_RADIAL_SEARCH_STEP
        
        # Multi-method agreement parameters
        self.contour_agreement_bonus = HOUGH_CONTOUR_AGREEMENT_BONUS
        self.hough_only_penalty = HOUGH_ONLY_PENALTY
        self.texture_discrimination = HOUGH_TEXTURE_DISCRIMINATION
        self.min_local_contrast = HOUGH_MIN_LOCAL_CONTRAST
        
        # Enhanced clustering parameters
        self.enhanced_clustering = HOUGH_CLUSTER_CONSOLIDATION_ENHANCED
        self.scale_aware_distance = HOUGH_SCALE_AWARE_DISTANCE
        self.cluster_radius_factor = HOUGH_CLUSTER_RADIUS_FACTOR
        self.concentric_detection = HOUGH_CONCENTRIC_DETECTION
        self.concentric_radius_tolerance = HOUGH_CONCENTRIC_RADIUS_TOLERANCE
        self.evidence_based_consolidation = HOUGH_EVIDENCE_BASED_CONSOLIDATION
    
    def extract_circles(self, 
                       internal_edges: np.ndarray,
                       isolated_product: np.ndarray, 
                       product_mask: np.ndarray,
                       raw_internal_edges: Optional[np.ndarray] = None,
                       outer_boundary: Optional[np.ndarray] = None) -> List[ActualFeature]:
        """
        Extract circular features from preprocessed representations.
        
        Args:
            internal_edges: Primary geometric edge representation
            isolated_product: Primary visual representation
            product_mask: Spatial constraint mask
            raw_internal_edges: Supporting edge evidence
            outer_boundary: Outer boundary edges
            
        Returns:
            List of detected circular features
        """
        logger.debug("Starting circle extraction")
        
        candidates = []
        
        # Method 1: Contour-based circle detection
        logger.debug(f"Extracting contour circles from internal_edges: {internal_edges.shape}")
        contour_circles = self._extract_contour_circles(internal_edges, product_mask)
        logger.debug(f"Found {len(contour_circles)} contour circles")
        candidates.extend(contour_circles)
        
        # Method 2: Hough circle detection
        logger.debug(f"Extracting Hough circles from isolated_product: {isolated_product.shape}")
        hough_circles = self._extract_hough_circles(isolated_product, product_mask, internal_edges)
        logger.debug(f"Found {len(hough_circles)} Hough circles")
        candidates.extend(hough_circles)
        
        # Method 3: Raw edge fallback for missed features
        if raw_internal_edges is not None:
            logger.debug(f"Extracting raw edge circles from raw_internal_edges: {raw_internal_edges.shape}")
            raw_circles = self._extract_contour_circles(raw_internal_edges, product_mask, 
                                                       source_rep="raw_internal_edges")
            logger.debug(f"Found {len(raw_circles)} raw edge circles")
            candidates.extend(raw_circles)
        
        # Validate and refine candidates
        validated_circles = []
        for candidate in candidates:
            if self._validate_circle_candidate(candidate, internal_edges, isolated_product, product_mask):
                validated_circles.append(candidate)
        
        # Apply population-based area validation if enabled
        if self.radius_anomaly_detection and len(validated_circles) > 2:
            validated_circles = self._apply_population_area_validation(validated_circles)
        
        # Apply enhanced cluster consolidation if enabled
        if self.enhanced_clustering and len(validated_circles) > 1:
            validated_circles = self._apply_enhanced_cluster_consolidation(validated_circles)
        
        logger.info(f"Circle extraction: {len(candidates)} candidates -> {len(validated_circles)} validated")
        return validated_circles
    
    def _extract_contour_circles(self, edge_image: np.ndarray, 
                                product_mask: np.ndarray,
                                source_rep: str = "internal_geometry_edges") -> List[ActualFeature]:
        """Extract circles using contour analysis."""
        circles = []
        
        # Find contours in edge image
        contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Basic area filtering
            if area < self.min_area or area > self.max_area:
                continue
            
            # Calculate geometric properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Check circularity threshold
            if circularity < self.circularity_threshold:
                continue
            
            # Calculate center and radius
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            center = (float(center_x), float(center_y))
            
            # Validate radius bounds
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # Check overlap with product mask
            if not self._check_product_mask_overlap(center, radius, product_mask):
                continue
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create geometric properties
            geometry = GeometricProperties(
                center=center,
                area=area,
                perimeter=perimeter,
                bounding_box=(x, y, w, h),
                radius=radius,
                diameter=2 * radius,
                circularity=circularity,
                solidity=self._calculate_solidity(contour),
                extent=area / (w * h) if w > 0 and h > 0 else 0,
                convexity=self._calculate_convexity(contour)
            )
            
            # Calculate evidence metrics
            edge_support = self._calculate_edge_support(center, radius, edge_image)
            evidence = EvidenceMetrics(
                confidence=self._calculate_contour_confidence(geometry, edge_support),
                edge_support=edge_support,
                contour_quality=circularity,
                intensity_consistency=0.5,  # Will be refined in validation
                geometric_consistency=circularity,
                internal_edge_evidence=1.0 if source_rep == "internal_geometry_edges" else 0.0,
                raw_edge_evidence=1.0 if source_rep == "raw_internal_edges" else 0.0
            )
            
            # Create feature
            feature_id = f"circle_contour_{len(circles)}_{int(center_x)}_{int(center_y)}"
            circle = ActualFeature(
                feature_id=feature_id,
                feature_type=ActualFeatureType.CIRCLE,
                geometry=geometry,
                contour=contour,
                evidence=evidence,
                source_representation=source_rep,
                detection_method="contour_analysis"
            )
            
            circles.append(circle)
        
        return circles
    
    def _extract_hough_circles(self, isolated_product: np.ndarray,
                              product_mask: np.ndarray,
                              edge_reference: np.ndarray) -> List[ActualFeature]:
        """Extract circles using Hough circle transform with configurable parameters."""
        circles = []
        
        # Apply Hough circle detection
        gray = cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY) if len(isolated_product.shape) == 3 else isolated_product
        
        # Use configurable Gaussian blur for Hough
        kernel_size = self.hough_blur_kernel
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        hough_circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,  # Configurable upper Canny threshold
            param2=self.hough_param2,  # Configurable accumulator threshold
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        if hough_circles is not None:
            hough_circles = np.round(hough_circles[0, :]).astype("int")
            
            for i, (center_x, center_y, radius) in enumerate(hough_circles):
                center = (float(center_x), float(center_y))
                radius = float(radius)
                
                # Check overlap with product mask
                if not self._check_product_mask_overlap(center, radius, product_mask):
                    continue
                
                # Calculate area and perimeter  
                area = np.pi * radius * radius
                perimeter = 2 * np.pi * radius
                
                # Validate area bounds
                if area < self.min_area or area > self.max_area:
                    continue
                
                # Radius anomaly detection
                if self.radius_anomaly_detection:
                    if not self._validate_radius_bounds(radius, product_mask):
                        continue
                
                # Enhanced Hough validation - treat as candidate only
                if (self.angular_coverage_required or self.edge_continuity_required or 
                    self.radial_consistency_required or self.texture_discrimination):
                    validation_result = self._validate_hough_candidate(
                        center, radius, edge_reference, isolated_product, product_mask
                    )
                    
                    if not validation_result["valid"]:
                        logger.debug(f"Hough circle rejected at ({center[0]:.1f},{center[1]:.1f}) r={radius:.1f}: {validation_result['rejection_reason']}")
                        continue
                else:
                    # Simplified validation when enhanced features are disabled
                    edge_support = self._calculate_edge_support(center, radius, edge_reference)
                    
                    # Use stricter edge support threshold for Hough circles
                    if self.geometric_evidence_required:
                        required_edge_support = self.edge_support_threshold * self.edge_support_multiplier
                        if edge_support < required_edge_support:
                            continue
                    else:
                        if edge_support < self.edge_support_threshold * 0.8:  # Original lenient threshold
                            continue
                    
                    # Create simplified validation result
                    validation_result = {
                        "valid": True,
                        "edge_support": edge_support,
                        "angular_coverage": 1.0,
                        "radial_consistency": 1.0,
                        "local_contrast": 20.0,
                        "circularity_evidence": 0.8,
                        "solidity_evidence": 0.8,
                        "convexity_evidence": 0.8,
                        "contour_quality": 0.8,
                        "intensity_consistency": 0.5,
                        "geometric_consistency": 0.8,
                        "intensity_evidence": 0.5
                    }
                
                # Create approximate contour for the circle using configurable sampling
                angles = np.linspace(0, 2*np.pi, self.hough_sampling_points)
                contour_x = center_x + radius * np.cos(angles)
                contour_y = center_y + radius * np.sin(angles)
                contour = np.column_stack((contour_x, contour_y)).astype(np.int32)
                
                # Calculate bounding box
                x = int(center_x - radius)
                y = int(center_y - radius)
                w = h = int(2 * radius)
                
                # Create geometric properties based on IMAGE EVIDENCE, not mathematical perfection
                geometry = GeometricProperties(
                    center=center,
                    area=area,
                    perimeter=perimeter,
                    bounding_box=(x, y, w, h),
                    radius=radius,
                    diameter=2 * radius,
                    circularity=validation_result["circularity_evidence"],  # Based on actual evidence
                    solidity=validation_result["solidity_evidence"],       # Based on actual evidence
                    extent=np.pi / 4,  # Circle extent in bounding box
                    convexity=validation_result["convexity_evidence"]      # Based on actual evidence
                )
                
                # Calculate evidence metrics based on actual validation
                evidence = EvidenceMetrics(
                    confidence=self._calculate_hough_evidence_confidence(validation_result),
                    edge_support=validation_result["edge_support"],
                    contour_quality=validation_result["contour_quality"],
                    intensity_consistency=validation_result["intensity_consistency"],
                    geometric_consistency=validation_result["geometric_consistency"],
                    intensity_evidence=validation_result["intensity_evidence"]
                )
                
                # Add rejection reason metadata for diagnostics
                if hasattr(evidence, 'rejection_reason'):
                    evidence.rejection_reason = validation_result.get("rejection_reason", "")
                if hasattr(evidence, 'angular_coverage'):
                    evidence.angular_coverage = validation_result.get("angular_coverage", 0.0)
                if hasattr(evidence, 'radial_consistency'):
                    evidence.radial_consistency = validation_result.get("radial_consistency", 0.0)
                if hasattr(evidence, 'local_contrast'):
                    evidence.local_contrast = validation_result.get("local_contrast", 0.0)
                
                # Create feature
                feature_id = f"circle_hough_{i}_{int(center_x)}_{int(center_y)}"
                circle = ActualFeature(
                    feature_id=feature_id,
                    feature_type=ActualFeatureType.CIRCLE,
                    geometry=geometry,
                    contour=contour,
                    evidence=evidence,
                    source_representation="isolated_product",
                    detection_method="hough_circles"
                )
                
                circles.append(circle)
        
        return circles
    
    def _validate_circle_candidate(self, candidate: ActualFeature,
                                  internal_edges: np.ndarray,
                                  isolated_product: np.ndarray,
                                  product_mask: np.ndarray) -> bool:
        """Validate circle candidate using multiple evidence sources."""
        
        # Refine intensity consistency
        intensity_consistency = self._calculate_intensity_consistency(
            candidate.geometry.center, candidate.geometry.radius, isolated_product
        )
        candidate.evidence.intensity_consistency = intensity_consistency
        candidate.evidence.intensity_evidence = intensity_consistency
        
        # Update overall confidence
        confidence = self._calculate_final_confidence(candidate.evidence)
        candidate.evidence.confidence = confidence
        
        # Apply confidence threshold
        if confidence < self.min_confidence:
            return False
        
        # Check edge support
        if candidate.evidence.edge_support < self.edge_support_threshold:
            return False
        
        return True
    
    def _check_product_mask_overlap(self, center: Tuple[float, float], 
                                   radius: float, product_mask: np.ndarray) -> bool:
        """Check if circle overlaps sufficiently with product mask."""
        # Create circle mask
        h, w = product_mask.shape
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
        
        # Calculate overlap
        intersection = cv2.bitwise_and(circle_mask, product_mask)
        intersection_area = np.count_nonzero(intersection)
        circle_area = np.count_nonzero(circle_mask)
        
        if circle_area == 0:
            return False
        
        overlap_ratio = intersection_area / circle_area
        return overlap_ratio >= self.mask_overlap_threshold
    
    def _calculate_edge_support(self, center: Tuple[float, float], 
                               radius: float, edge_image: np.ndarray) -> float:
        """Calculate fraction of circle perimeter supported by edges."""
        # Sample points around circle perimeter using configurable sampling
        angles = np.linspace(0, 2*np.pi, self.edge_sampling_points, endpoint=False)
        
        supported_points = 0
        valid_points = 0
        
        h, w = edge_image.shape
        
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            
            # Check bounds
            if 0 <= x < w and 0 <= y < h:
                valid_points += 1
                # Check for edge support in configurable neighborhood
                if self._check_edge_neighborhood(x, y, edge_image, neighborhood_size=self.edge_neighborhood_size):
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
    
    def _calculate_intensity_consistency(self, center: Tuple[float, float],
                                       radius: float, isolated_product: np.ndarray) -> float:
        """Calculate consistency of intensity within circle."""
        # Create circle mask
        h, w = isolated_product.shape[:2]
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius * 0.8), 255, -1)
        
        # Extract pixel intensities
        if len(isolated_product.shape) == 3:
            gray = cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY)
        else:
            gray = isolated_product
            
        pixels = gray[circle_mask > 0]
        
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
    
    def _validate_radius_bounds(self, radius: float, product_mask: np.ndarray) -> bool:
        """Validate radius bounds using image-relative constraints."""
        # Get image dimensions
        h, w = product_mask.shape
        image_diagonal = np.sqrt(h*h + w*w)
        
        # Maximum radius as percentile of image diagonal
        max_radius_image_relative = image_diagonal * self.max_radius_percentile
        if radius > max_radius_image_relative:
            return False
        
        # Additional reasonableness check - radius shouldn't be larger than product dimensions
        product_region = np.where(product_mask > 0)
        if len(product_region[0]) > 0:
            product_height = np.max(product_region[0]) - np.min(product_region[0])
            product_width = np.max(product_region[1]) - np.min(product_region[1])
            max_product_dimension = max(product_height, product_width)
            
            # Radius shouldn't exceed product dimensions
            if radius > max_product_dimension:
                return False
        
        return True
    
    def _apply_population_area_validation(self, circles: List[ActualFeature]) -> List[ActualFeature]:
        """Apply population-based area validation to reject anomalously large circles."""
        if len(circles) < 3:
            return circles
        
        # Calculate area statistics
        areas = [circle.geometry.area for circle in circles]
        median_area = np.median(areas)
        
        # Filter out anomalously large circles
        filtered_circles = []
        for circle in circles:
            area_ratio = circle.geometry.area / median_area if median_area > 0 else 1.0
            
            # Reject circles that are much larger than typical population
            if area_ratio <= self.area_population_threshold:
                filtered_circles.append(circle)
            else:
                logger.debug(f"Rejected anomalous circle {circle.feature_id}: area {circle.geometry.area} is {area_ratio:.1f}x median")
        
        return filtered_circles
    
    def _validate_hough_candidate(self, center: Tuple[float, float], radius: float,
                                 edge_image: np.ndarray, intensity_image: np.ndarray,
                                 product_mask: np.ndarray) -> dict:
        """
        Comprehensive validation of Hough circle candidate based on actual image evidence.
        Treats Hough as candidate generator only - validates against real image data.
        """
        result = {
            "valid": True,
            "rejection_reason": "",
            "edge_support": 0.0,
            "angular_coverage": 0.0,
            "sector_coverage": 0.0,
            "radial_consistency": 0.0,
            "local_contrast": 0.0,
            "circularity_evidence": 0.0,
            "solidity_evidence": 0.0,
            "convexity_evidence": 0.0,
            "contour_quality": 0.0,
            "intensity_consistency": 0.0,
            "geometric_consistency": 0.0,
            "intensity_evidence": 0.0,
            "contour_agreement": False,
            "angular_uniformity": 0.0,
            "max_gap_ratio": 1.0
        }
        
        # Calculate basic edge support first to adapt thresholds
        basic_edge_support = self._calculate_edge_support(center, radius, edge_image)
        result["edge_support"] = basic_edge_support
        
        # Adaptive thresholds based on edge quality - more lenient for high-quality edges
        edge_quality_factor = min(1.0, basic_edge_support / 0.7)  # Scale factor [0,1] for adaptation
        
        # 1. Angular coverage analysis
        if self.angular_coverage_required:
            angular_result = self._analyze_angular_coverage(center, radius, edge_image)
            result["angular_coverage"] = angular_result["coverage"]
            result["sector_coverage"] = angular_result["sector_coverage"]
            result["angular_uniformity"] = angular_result["uniformity"]
            result["contour_agreement"] = bool(angular_result["sector_coverage"] > 0.5 and angular_result["uniformity"] > 0.25)
            
            # Reject localized or cluster-biased support.
            min_coverage = max(self.min_angular_coverage * (0.5 + 0.3 * edge_quality_factor), 0.45)
            
            if angular_result["coverage"] < min_coverage:
                result["valid"] = False
                result["rejection_reason"] = f"Insufficient angular coverage: {angular_result['coverage']:.2f} < {min_coverage:.2f}"
                return result
            
            min_sector = max(self.min_sector_coverage * (0.6 + 0.2 * edge_quality_factor), 0.35)
            if angular_result["sector_coverage"] < min_sector:
                result["valid"] = False  
                result["rejection_reason"] = f"Insufficient sector coverage: {angular_result['sector_coverage']:.2f} < {min_sector:.2f}"
                return result

            if angular_result["uniformity"] < 0.2:
                result["valid"] = False
                result["rejection_reason"] = (
                    "Localized angular support: evidence concentrated in a small arc "
                    f"instead of a distributed ring (uniformity={angular_result['uniformity']:.2f})"
                )
                return result
        
        # 2. Edge continuity analysis - keep it strict enough to reject local arcs
        if self.edge_continuity_required:
            continuity_result = self._analyze_edge_continuity(center, radius, edge_image)
            result["max_gap_ratio"] = continuity_result["max_gap_ratio"]
            max_gap_allowed = min(self.max_edge_gap_ratio, 0.35)
            if continuity_result["max_gap_ratio"] > max_gap_allowed:
                result["valid"] = False
                result["rejection_reason"] = f"Excessive edge gaps: {continuity_result['max_gap_ratio']:.2f} > {max_gap_allowed:.2f}"
                return result
        
        # 3. Radial consistency analysis - require evidence in the actual radius band
        if self.radial_consistency_required:
            radial_result = self._analyze_radial_consistency(center, radius, edge_image)
            result["radial_consistency"] = radial_result["consistency"]
            min_radial = max(self.min_radial_agreement, 0.55)
            if radial_result["consistency"] < min_radial:
                result["valid"] = False
                result["rejection_reason"] = f"Poor radial consistency: {radial_result['consistency']:.2f} < {min_radial:.2f}"
                return result
        
        # 4. Texture discrimination - only strict for weak edge cases
        if self.texture_discrimination:
            contrast_result = self._analyze_local_contrast(center, radius, intensity_image)
            result["local_contrast"] = contrast_result["contrast"]
            min_contrast = self.min_local_contrast * (0.4 + 0.4 * (1.0 - edge_quality_factor))
            if contrast_result["contrast"] < min_contrast:
                result["valid"] = False
                result["rejection_reason"] = f"Texture-like evidence: contrast {contrast_result['contrast']:.1f} < {min_contrast:.1f}"
                return result
        
        # 5. Calculate evidence-based geometric properties
        result["intensity_consistency"] = self._calculate_intensity_consistency(center, radius, intensity_image)
        edge_uniformity = angular_result["uniformity"] if self.angular_coverage_required else 0.5
        result["circularity_evidence"] = min(1.0, (result["angular_coverage"] + edge_uniformity + basic_edge_support) / 3.0)
        result["solidity_evidence"] = min(1.0, result["radial_consistency"] * 1.2) if self.radial_consistency_required else min(1.0, basic_edge_support * 1.1)
        result["convexity_evidence"] = min(1.0, result["edge_support"] * 1.1)
        result["contour_quality"] = result["circularity_evidence"]
        result["geometric_consistency"] = (result["circularity_evidence"] + result["solidity_evidence"]) / 2.0
        result["intensity_evidence"] = result["intensity_consistency"]
        
        return result
    
    def _analyze_angular_coverage(self, center: Tuple[float, float], radius: float, 
                                 edge_image: np.ndarray) -> dict:
        """Analyze edge support coverage around the full circumference."""
        sectors = self.angular_sectors
        sector_size = 2 * np.pi / sectors
        
        sector_support = []
        edge_positions = []
        
        for sector in range(sectors):
            angle_start = sector * sector_size
            angle_end = (sector + 1) * sector_size
            
            # Sample points in this sector
            sector_angles = np.linspace(angle_start, angle_end, 8, endpoint=False)
            sector_edges = 0
            sector_total = 0
            
            for angle in sector_angles:
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                
                if 0 <= x < edge_image.shape[1] and 0 <= y < edge_image.shape[0]:
                    sector_total += 1
                    if self._check_edge_neighborhood(x, y, edge_image, self.edge_neighborhood_size):
                        sector_edges += 1
                        edge_positions.append(angle)
            
            if sector_total > 0:
                sector_support.append(sector_edges / sector_total)
            else:
                sector_support.append(0.0)
        
        # Calculate coverage metrics. A dense cluster in a small arc can create
        # a reasonable aggregate score while still lacking genuine circular support.
        supported_sectors = sum(1 for support in sector_support if support > 0.25)
        sector_coverage = supported_sectors / sectors
        overall_coverage = np.mean(sector_support)
        uniformity = 1.0 - np.std(sector_support) if sector_support else 0.0
        concentration = max(sector_support) / (np.mean(sector_support) + 1e-6) if sector_support and np.mean(sector_support) > 0 else 0.0
        
        return {
            "coverage": overall_coverage,
            "sector_coverage": sector_coverage,
            "uniformity": max(0.0, uniformity),
            "concentration": concentration,
            "sector_support": sector_support,
            "edge_positions": edge_positions
        }
    
    def _analyze_edge_continuity(self, center: Tuple[float, float], radius: float,
                                edge_image: np.ndarray) -> dict:
        """Analyze continuity of edge support around circumference."""
        # Sample many points around circumference
        num_samples = 64
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        
        edge_present = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            
            if 0 <= x < edge_image.shape[1] and 0 <= y < edge_image.shape[0]:
                edge_present.append(self._check_edge_neighborhood(x, y, edge_image, self.edge_neighborhood_size))
            else:
                edge_present.append(False)
        
        # Find gaps in edge support. Localized curvature should not pass as a full
        # circle just because the aggregate circumference has a few supported segments.
        gaps = []
        gap_start = None
        
        extended_edges = edge_present + edge_present[:10]
        for i, has_edge in enumerate(extended_edges):
            if i >= len(edge_present):
                break
            if not has_edge and gap_start is None:
                gap_start = i
            elif has_edge and gap_start is not None:
                gap_length = i - gap_start
                gaps.append(gap_length)
                gap_start = None
        
        max_gap = max(gaps) if gaps else 0
        max_gap_ratio = max_gap / num_samples
        
        return {
            "max_gap": max_gap,
            "max_gap_ratio": max_gap_ratio,
            "num_gaps": len(gaps),
            "edge_present": edge_present
        }
    
    def _analyze_radial_consistency(self, center: Tuple[float, float], radius: float,
                                   edge_image: np.ndarray) -> dict:
        """Analyze radial consistency - edges should be at expected radius."""
        angles = np.linspace(0, 2*np.pi, self.radial_samples, endpoint=False)
        radius_deviations = []
        
        for angle in angles:
            # Require support near the proposed circumference, not just any nearby edge.
            # A single nearby boundary should not count as radial agreement.
            candidate_radii = [
                max(1, int(round(radius + offset)))
                for offset in (-self.radial_search_step, 0, self.radial_search_step)
            ]
            local_hits = 0
            for r in candidate_radii:
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                if 0 <= x < edge_image.shape[1] and 0 <= y < edge_image.shape[0]:
                    if self._check_edge_neighborhood(x, y, edge_image, self.edge_neighborhood_size):
                        local_hits += 1
            supported = local_hits >= 2
            radius_deviations.append(0.0 if supported else 1.0)
        
        if not radius_deviations:
            return {"consistency": 0.0, "deviations": []}
        
        avg_deviation = np.mean(radius_deviations)
        consistency = max(0.0, 1.0 - avg_deviation * 3.0)
        
        return {
            "consistency": consistency,
            "deviations": radius_deviations,
            "avg_deviation": avg_deviation
        }
    
    def _analyze_local_contrast(self, center: Tuple[float, float], radius: float,
                               intensity_image: np.ndarray) -> dict:
        """Analyze local contrast to discriminate features from texture."""
        # Convert to grayscale if needed
        if len(intensity_image.shape) == 3:
            gray = cv2.cvtColor(intensity_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = intensity_image
        
        # Sample interior and exterior regions
        interior_samples = []
        exterior_samples = []
        
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        
        for angle in angles:
            # Interior sample (0.6 * radius from center)
            int_x = int(center[0] + 0.6 * radius * np.cos(angle))
            int_y = int(center[1] + 0.6 * radius * np.sin(angle))
            
            # Exterior sample (1.4 * radius from center)  
            ext_x = int(center[0] + 1.4 * radius * np.cos(angle))
            ext_y = int(center[1] + 1.4 * radius * np.sin(angle))
            
            if 0 <= int_x < gray.shape[1] and 0 <= int_y < gray.shape[0]:
                interior_samples.append(gray[int_y, int_x])
            
            if 0 <= ext_x < gray.shape[1] and 0 <= ext_y < gray.shape[0]:
                exterior_samples.append(gray[ext_y, ext_x])
        
        if not interior_samples or not exterior_samples:
            return {"contrast": 0.0}
        
        # Calculate contrast
        interior_mean = np.mean(interior_samples)
        exterior_mean = np.mean(exterior_samples)
        contrast = abs(interior_mean - exterior_mean)
        
        return {
            "contrast": contrast,
            "interior_mean": interior_mean,
            "exterior_mean": exterior_mean
        }
    
    def _calculate_hough_evidence_confidence(self, validation_result: dict) -> float:
        """Calculate confidence based on actual image evidence, not mathematical perfection."""
        if not validation_result["valid"]:
            return 0.0
        
        # Weight actual evidence components
        edge_score = validation_result["edge_support"]
        angular_score = validation_result["angular_coverage"]
        consistency_score = validation_result["geometric_consistency"]
        intensity_score = validation_result["intensity_consistency"]

        # Base confidence from evidence
        base_confidence = (
            0.3 * edge_score +
            0.25 * angular_score + 
            0.25 * consistency_score +
            0.2 * intensity_score
        )

        # Hough-only candidates require additional evidence. Without distributed ring
        # agreement, they should not receive a strong confidence score.
        if not validation_result.get('contour_agreement', False):
            base_confidence *= (1.0 - self.hough_only_penalty)

        if validation_result.get('sector_coverage', 0.0) < 0.35:
            base_confidence *= 0.8

        return min(1.0, max(0.0, base_confidence))
    
    def _apply_enhanced_cluster_consolidation(self, circles: List[ActualFeature]) -> List[ActualFeature]:
        """Apply enhanced cluster consolidation with scale-aware distance and evidence-based selection."""
        if len(circles) <= 1:
            return circles
        
        # Group circles into clusters
        clusters = []
        used_indices = set()
        
        for i, circle1 in enumerate(circles):
            if i in used_indices:
                continue
            
            cluster = [i]
            used_indices.add(i)
            
            for j, circle2 in enumerate(circles[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                if self._should_cluster_circles(circle1, circle2):
                    cluster.append(j)
                    used_indices.add(j)
            
            # Add the completed cluster to the list
            clusters.append(cluster)
        
        # Select best representative from each cluster
        consolidated_circles = []
        
        for cluster_indices in clusters:
            if len(cluster_indices) == 1:
                consolidated_circles.append(circles[cluster_indices[0]])
            else:
                # Select best circle from cluster based on evidence quality
                cluster_circles = [circles[i] for i in cluster_indices]
                best_circle = self._select_best_from_cluster(cluster_circles)
                consolidated_circles.append(best_circle)
                
                # Log consolidation for diagnostics
                logger.debug(f"Consolidated cluster of {len(cluster_circles)} circles into best representative")
        
        return consolidated_circles
    
    def _should_cluster_circles(self, circle1: ActualFeature, circle2: ActualFeature) -> bool:
        """Determine if two circles should be clustered using scale-aware distance."""
        center1 = circle1.geometry.center
        center2 = circle2.geometry.center
        radius1 = getattr(circle1.geometry, 'radius', 0)
        radius2 = getattr(circle2.geometry, 'radius', 0)
        
        # Calculate center distance
        center_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        if self.scale_aware_distance:
            # Use scale-aware clustering distance
            larger_radius = max(radius1, radius2)
            cluster_distance = larger_radius * self.cluster_radius_factor
            
            if center_distance > cluster_distance:
                return False
        else:
            # Use fixed distance
            if center_distance > self.strict_center_distance:
                return False
        
        # Check for concentric circles if enabled
        if self.concentric_detection:
            radius_ratio = min(radius1, radius2) / max(radius1, radius2) if max(radius1, radius2) > 0 else 0
            if (center_distance < min(radius1, radius2) * 0.3 and  # Very close centers
                abs(radius1 - radius2) / max(radius1, radius2) > self.concentric_radius_tolerance):
                return True  # Concentric circles should be clustered
        
        # Check radius compatibility
        if radius1 > 0 and radius2 > 0:
            radius_ratio = min(radius1, radius2) / max(radius1, radius2)
            if radius_ratio < (1.0 - self.cluster_radius_tolerance):
                return False
        
        # Check area compatibility
        area1 = circle1.geometry.area
        area2 = circle2.geometry.area
        if area1 > 0 and area2 > 0:
            area_ratio = min(area1, area2) / max(area1, area2)
            if area_ratio < 0.5:  # Very different sizes
                return False
        
        return True
    
    def _select_best_from_cluster(self, cluster_circles: List[ActualFeature]) -> ActualFeature:
        """Select the best representative circle from a cluster based on evidence quality."""
        if not cluster_circles:
            return None
        
        if len(cluster_circles) == 1:
            return cluster_circles[0]
        
        if self.evidence_based_consolidation:
            # Score based on evidence quality, not just confidence
            best_circle = None
            best_score = -1.0
            
            for circle in cluster_circles:
                # Combine multiple evidence factors
                evidence_score = (
                    0.3 * circle.evidence.edge_support +
                    0.25 * getattr(circle.evidence, 'angular_coverage', 0.5) +
                    0.2 * circle.evidence.intensity_consistency +
                    0.15 * circle.evidence.geometric_consistency +
                    0.1 * (1.0 if circle.detection_method == "contour_analysis" else 0.5)  # Slight preference for contour-based
                )
                
                if evidence_score > best_score:
                    best_score = evidence_score
                    best_circle = circle
            
            return best_circle
        else:
            # Fall back to confidence-based selection
            return max(cluster_circles, key=lambda c: c.evidence.confidence)
    
    def _calculate_solidity(self, contour: np.ndarray) -> float:
        """Calculate solidity (area/convex_hull_area)."""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0.0
        
        return area / hull_area
    
    def _calculate_convexity(self, contour: np.ndarray) -> float:
        """Calculate convexity (convex_hull_perimeter/perimeter)."""
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        
        if perimeter == 0:
            return 0.0
        
        return hull_perimeter / perimeter
    
    def _calculate_contour_confidence(self, geometry: GeometricProperties, 
                                    edge_support: float) -> float:
        """Calculate confidence for contour-based circle detection."""
        # Combine multiple factors
        circularity_score = min(1.0, geometry.circularity / 0.8)  # Normalize to reasonable range
        solidity_score = geometry.solidity
        edge_score = edge_support
        
        # Weighted combination
        confidence = (
            GEOMETRY_EVIDENCE_WEIGHT * circularity_score +
            CONTOUR_EVIDENCE_WEIGHT * solidity_score +
            EDGE_EVIDENCE_WEIGHT * edge_score
        )
        
        return min(1.0, confidence)
    
    def _calculate_hough_confidence(self, radius: float, edge_support: float) -> float:
        """Calculate confidence for Hough-based circle detection using enhanced validation."""
        # This method is deprecated - use _calculate_hough_evidence_confidence instead
        # Kept for backward compatibility with non-enhanced validation
        if self.confidence_geometric_validation:
            # Require actual geometric evidence instead of assuming perfect geometry
            # Validate radius reasonableness (not just that it passed bounds)
            radius_score = min(1.0, max(0.1, 1.0 - (radius / self.max_radius) ** 2))  # Penalize very large radii
            edge_score = edge_support
            geometric_score = min(radius_score, edge_score)  # Both must be good
            
            # Use configurable weights with geometric validation
            confidence = (
                self.confidence_geometric_weight * geometric_score +
                self.hough_edge_weight * edge_score
            )
        else:
            # Original logic - assume perfect geometry
            radius_score = 1.0  # Hough already filtered by radius
            edge_score = edge_support
            
            confidence = (
                self.hough_radius_weight * radius_score +
                self.hough_edge_weight * edge_score
            )
        
        return min(1.0, confidence)
    
    def _calculate_final_confidence(self, evidence: EvidenceMetrics) -> float:
        """Calculate final confidence combining all evidence sources."""
        return (
            EDGE_EVIDENCE_WEIGHT * evidence.edge_support +
            CONTOUR_EVIDENCE_WEIGHT * evidence.contour_quality +
            INTENSITY_EVIDENCE_WEIGHT * evidence.intensity_consistency +
            GEOMETRY_EVIDENCE_WEIGHT * evidence.geometric_consistency
        )