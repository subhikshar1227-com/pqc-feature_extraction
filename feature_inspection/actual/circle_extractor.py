"""
Circle Feature Extractor for Phase 2B

Detects circular features from preprocessed product images using multiple
geometric and edge-based evidence sources.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from collections import defaultdict

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
    ACTUAL_CIRCLE_RADIAL_POLARITY_REQUIRED, ACTUAL_CIRCLE_RADIAL_POLARITY_SAMPLES,
    ACTUAL_CIRCLE_RADIAL_POLARITY_SEARCH_PIXELS, ACTUAL_CIRCLE_RADIAL_POLARITY_MIN_GRADIENT,
    ACTUAL_CIRCLE_MIN_RADIAL_POLARITY_COHERENCE, ACTUAL_CIRCLE_MIN_RADIAL_POLARITY_COVERAGE,
    HOUGH_CONFIDENCE_EDGE_SUPPORT_WEIGHT, HOUGH_CONFIDENCE_ANGULAR_COVERAGE_WEIGHT,
    HOUGH_CONFIDENCE_RADIAL_CONSISTENCY_WEIGHT,
    HOUGH_CONFIDENCE_INTENSITY_CONSISTENCY_WEIGHT,
    HOUGH_CONFIDENCE_POLARITY_COHERENCE_WEIGHT,
    HOUGH_CONFIDENCE_POLARITY_COVERAGE_WEIGHT,
    HOUGH_CIRCLE_NEIGHBOR_DIAGNOSTIC_LIMIT,
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
        self.radial_polarity_required = ACTUAL_CIRCLE_RADIAL_POLARITY_REQUIRED
        self.radial_polarity_samples = ACTUAL_CIRCLE_RADIAL_POLARITY_SAMPLES
        self.radial_polarity_search_pixels = ACTUAL_CIRCLE_RADIAL_POLARITY_SEARCH_PIXELS
        self.radial_polarity_min_gradient = ACTUAL_CIRCLE_RADIAL_POLARITY_MIN_GRADIENT
        self.min_radial_polarity_coherence = ACTUAL_CIRCLE_MIN_RADIAL_POLARITY_COHERENCE
        self.min_radial_polarity_coverage = ACTUAL_CIRCLE_MIN_RADIAL_POLARITY_COVERAGE
        
        # Enhanced clustering parameters
        self.enhanced_clustering = HOUGH_CLUSTER_CONSOLIDATION_ENHANCED
        self.scale_aware_distance = HOUGH_SCALE_AWARE_DISTANCE
        self.cluster_radius_factor = HOUGH_CLUSTER_RADIUS_FACTOR
        self.concentric_detection = HOUGH_CONCENTRIC_DETECTION
        self.concentric_radius_tolerance = HOUGH_CONCENTRIC_RADIUS_TOLERANCE
        self.evidence_based_consolidation = HOUGH_EVIDENCE_BASED_CONSOLIDATION
        self.last_diagnostics = {}
        self._last_hough_candidate_diagnostics = []
        self._radial_gradient_source_id = None
        self._radial_gradient_x = None
        self._radial_gradient_y = None
        self._last_hough_raw_count = 0
        self._last_hough_prevalidation_count = 0
    
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
        self._last_hough_candidate_diagnostics = []
        circle_candidate_evidence = []
        self._last_hough_raw_count = 0
        self._last_hough_prevalidation_count = 0
        
        # Method 1: Contour-based circle detection
        logger.debug(f"Extracting contour circles from internal_edges: {internal_edges.shape}")
        contour_circles = self._extract_contour_circles(internal_edges, product_mask)
        logger.debug(f"Found {len(contour_circles)} contour circles")
        candidates.extend(contour_circles)

        raw_circles = []
        if raw_internal_edges is not None:
            logger.debug(f"Extracting raw edge fallback circles from raw_internal_edges: {raw_internal_edges.shape}")
            raw_circles = self._extract_contour_circles(
                raw_internal_edges, product_mask, source_rep="raw_internal_edges"
            )
            logger.debug(f"Found {len(raw_circles)} raw edge circles")
            candidates.extend(raw_circles)
        
        # Method 2: Hough circle detection
        logger.debug(f"Extracting Hough circles from isolated_product: {isolated_product.shape}")
        hough_circles = self._extract_hough_circles(
            isolated_product, product_mask, internal_edges,
            contour_support_candidates=contour_circles + raw_circles
        )
        logger.debug(f"Found {len(hough_circles)} Hough circles")
        candidates.extend(hough_circles)
        
        # Validate and refine candidates
        validated_circles = []
        validated_by_detector = defaultdict(int)
        for candidate in candidates:
            validated = self._validate_circle_candidate(candidate, internal_edges, isolated_product, product_mask)
            evidence_diagnostic = candidate.processing_parameters.get("circle_validation_evidence")
            if evidence_diagnostic is not None and candidate.detection_method != "hough_circles":
                circle_candidate_evidence.append(evidence_diagnostic)
            if validated:
                validated_circles.append(candidate)
                validated_by_detector[self._detector_key(candidate)] += 1

        raw_by_detector = {
            "circle_contour_analysis:internal_geometry_edges": len(contour_circles),
            "circle_hough_circles:isolated_product": self._last_hough_raw_count,
        }
        if raw_internal_edges is not None:
            raw_by_detector["circle_contour_analysis:raw_internal_edges"] = len(raw_circles)
        prevalidation_by_detector = {
            "circle_contour_analysis:internal_geometry_edges": len(contour_circles),
            "circle_hough_circles:isolated_product": self._last_hough_prevalidation_count,
        }
        if raw_internal_edges is not None:
            prevalidation_by_detector["circle_contour_analysis:raw_internal_edges"] = len(raw_circles)
        candidate_counts = defaultdict(int)
        for candidate in candidates:
            candidate_counts[self._detector_key(candidate)] += 1
        rejected_before_validation = {
            key: max(0, raw_by_detector.get(key, 0) - prevalidation_by_detector.get(key, 0))
            for key in raw_by_detector
        }
        rejected_validation = {
            key: max(0, candidate_counts.get(key, 0) - validated_by_detector.get(key, 0))
            for key in candidate_counts
        }
        for key, count in rejected_before_validation.items():
            rejected_validation[key] = rejected_validation.get(key, 0) + count
        self.last_diagnostics = {
            "raw_candidates_by_detector": raw_by_detector,
            "prevalidation_candidates_by_detector": prevalidation_by_detector,
            "validated_candidates_by_detector": dict(validated_by_detector),
            "rejected_by_validation_by_detector": rejected_validation,
            "hough_candidate_evidence": self._last_hough_candidate_diagnostics,
            "contour_candidate_evidence": circle_candidate_evidence,
        }
        
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
                              edge_reference: np.ndarray,
                              contour_support_candidates: Optional[List[ActualFeature]] = None) -> List[ActualFeature]:
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
        
        self._last_hough_raw_count = int(len(hough_circles[0])) if hough_circles is not None else 0
        if hough_circles is not None:
            hough_circles = np.round(hough_circles[0, :]).astype("int")
            
            for i, (center_x, center_y, radius) in enumerate(hough_circles):
                center = (float(center_x), float(center_y))
                radius = float(radius)
                candidate_diagnostic = {
                    "candidate_index": int(i),
                    "center": center,
                    "radius": radius,
                    "hough_accumulator_score": None,
                    "hough_accumulator_threshold": self.hough_param2,
                    "gate_decisions": {},
                }
                self._last_hough_candidate_diagnostics.append(candidate_diagnostic)

                area = np.pi * radius * radius
                radius_valid = self.min_radius <= radius <= self.max_radius
                if self.radius_anomaly_detection:
                    radius_valid = radius_valid and self._validate_radius_bounds(radius, product_mask)
                candidate_diagnostic.update({
                    "area": area,
                    "area_valid": self.min_area <= area <= self.max_area,
                    "radius_valid": radius_valid,
                    "gate_decisions": {
                        "area": self.min_area <= area <= self.max_area,
                        "radius": radius_valid,
                    },
                })

                if (self.angular_coverage_required or self.edge_continuity_required or
                    self.radial_consistency_required or self.texture_discrimination or
                    self.radial_polarity_required):
                    validation_result = self._validate_hough_candidate(
                        center, radius, edge_reference, isolated_product, product_mask
                    )
                    candidate_diagnostic.update(validation_result)
                    candidate_diagnostic["gate_decisions"]["image_evidence"] = validation_result["valid"]
                    candidate_diagnostic["hough_evidence_confidence"] = (
                        self._calculate_hough_evidence_confidence(validation_result)
                    )
                    candidate_diagnostic["contour_support"] = self._measure_contour_support(
                        center, radius, contour_support_candidates or []
                    )
                else:
                    validation_result = None
                    candidate_diagnostic["hough_evidence_confidence"] = None
                
                # Check overlap with product mask
                mask_overlap = self._calculate_product_mask_overlap(center, radius, product_mask)
                candidate_diagnostic["product_mask_overlap"] = mask_overlap
                candidate_diagnostic["gate_decisions"]["product_mask_overlap"] = mask_overlap >= self.mask_overlap_threshold
                if mask_overlap < self.mask_overlap_threshold:
                    candidate_diagnostic.update({"decision": "rejected", "rejection_reason": "product_mask_overlap"})
                    continue
                
                # Calculate perimeter after the independent radius/area measurements.
                perimeter = 2 * np.pi * radius
                
                # Validate area bounds
                if area < self.min_area or area > self.max_area:
                    candidate_diagnostic.update({"area_valid": False, "decision": "rejected", "rejection_reason": "area_bounds"})
                    candidate_diagnostic["gate_decisions"]["area"] = False
                    continue
                candidate_diagnostic["area_valid"] = True
                candidate_diagnostic["gate_decisions"]["area"] = True
                
                # Radius anomaly detection
                candidate_diagnostic["radius_valid"] = radius_valid
                candidate_diagnostic["gate_decisions"]["radius"] = radius_valid
                if not radius_valid:
                    candidate_diagnostic.update({"decision": "rejected", "rejection_reason": "radius_bounds"})
                    continue
                
                # Enhanced Hough validation - treat as candidate only
                if validation_result is not None:
                    if not validation_result["valid"]:
                        logger.debug(f"Hough circle rejected at ({center[0]:.1f},{center[1]:.1f}) r={radius:.1f}: {validation_result['rejection_reason']}")
                        candidate_diagnostic.update({"decision": "rejected", "rejection_reason": validation_result["rejection_reason"]})
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
                    candidate_diagnostic.update(validation_result)
                
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
                circle.processing_parameters["circle_validation_evidence"] = candidate_diagnostic
                circle.processing_parameters["hough_prevalidation_confidence"] = evidence.confidence
                candidate_diagnostic.update({
                    "decision": "passed_hough_gates",
                    "feature_id": feature_id,
                    "hough_prevalidation_confidence": evidence.confidence,
                })
                
                circles.append(circle)
        self._last_hough_prevalidation_count = len(circles)
        self._add_nearby_boundary_diagnostics(contour_support_candidates or [])
        
        return circles

    def _measure_contour_support(self, center: Tuple[float, float], radius: float,
                                 contour_candidates: List[ActualFeature]) -> dict:
        """Describe the nearest measured contour circle; it shares the primary edge pixels."""
        if not contour_candidates:
            return {"available": False, "source_pixels_independent": False}

        nearest = min(
            contour_candidates,
            key=lambda candidate: (
                np.hypot(center[0] - candidate.geometry.center[0], center[1] - candidate.geometry.center[1])
                / max(radius, 1.0)
                + abs(radius - (candidate.geometry.radius or 0.0)) / max(radius, 1.0)
            ),
        )
        center_residual = float(
            np.hypot(center[0] - nearest.geometry.center[0], center[1] - nearest.geometry.center[1]) / max(radius, 1.0)
        )
        radius_residual = float(abs(radius - (nearest.geometry.radius or 0.0)) / max(radius, 1.0))
        return {
            "available": True,
            "source_pixels_independent": False,
            "source_representation": nearest.source_representation,
            "feature_id": nearest.feature_id,
            "normalized_center_residual": center_residual,
            "normalized_radius_residual": radius_residual,
            "contour_quality": nearest.evidence.contour_quality,
            "edge_support": nearest.evidence.edge_support,
        }

    def _add_nearby_boundary_diagnostics(self, contour_features: List[ActualFeature]) -> None:
        for record in self._last_hough_candidate_diagnostics:
            relationships = []
            center = record["center"]
            radius = record["radius"]
            for other in self._last_hough_candidate_diagnostics:
                if other is record:
                    continue
                other_radius = other["radius"]
                center_distance = float(np.hypot(
                    center[0] - other["center"][0], center[1] - other["center"][1]
                ))
                scale = max((radius + other_radius) / 2.0, 1.0)
                relationships.append({
                    "candidate_index": other["candidate_index"],
                    "detection_method": "hough_circles",
                    "source_representation": "isolated_product",
                    "center_distance": center_distance,
                    "radius_difference": abs(radius - other_radius),
                    "normalized_boundary_distance": (center_distance + abs(radius - other_radius)) / scale,
                    "circle_intersection": center_distance <= radius + other_radius,
                    "nested_or_concentric": center_distance + min(radius, other_radius) <= max(radius, other_radius),
                    "candidate_decision": other.get("decision", "pending"),
                })
            for other in contour_features:
                other_radius = other.geometry.radius or 0.0
                center_distance = float(np.hypot(
                    center[0] - other.geometry.center[0], center[1] - other.geometry.center[1]
                ))
                scale = max((radius + other_radius) / 2.0, 1.0)
                relationships.append({
                    "feature_id": other.feature_id,
                    "detection_method": other.detection_method,
                    "source_representation": other.source_representation,
                    "center_distance": center_distance,
                    "radius_difference": abs(radius - other_radius),
                    "normalized_boundary_distance": (center_distance + abs(radius - other_radius)) / scale,
                    "circle_intersection": center_distance <= radius + other_radius,
                    "nested_or_concentric": center_distance + min(radius, other_radius) <= max(radius, other_radius),
                    "source_evidence_independence": "shares thresholded edge pixels with Hough edge validation",
                })
            relationships.sort(key=lambda item: item["normalized_boundary_distance"])
            record["nearest_detected_circle_boundaries"] = relationships[:HOUGH_CIRCLE_NEIGHBOR_DIAGNOSTIC_LIMIT]

    @staticmethod
    def _detector_key(candidate: ActualFeature) -> str:
        return f"circle_{candidate.detection_method}:{candidate.source_representation}"
    
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

        polarity_result = self._analyze_radial_polarity(
            candidate.geometry.center, candidate.geometry.radius, isolated_product
        )
        diagnostic = candidate.processing_parameters.setdefault("circle_validation_evidence", {})
        diagnostic.update({
            "feature_id": candidate.feature_id,
            "detection_method": candidate.detection_method,
            "source_representation": candidate.source_representation,
            "center": candidate.geometry.center,
            "radius": candidate.geometry.radius,
            "edge_support": candidate.evidence.edge_support,
            "contour_quality": candidate.evidence.contour_quality,
            "geometric_consistency": candidate.evidence.geometric_consistency,
            "intensity_consistency": intensity_consistency,
            "radial_polarity_coherence": polarity_result["coherence"],
            "radial_polarity_coverage": polarity_result["coverage"],
            "radial_polarity_positive_fraction": polarity_result["positive_fraction"],
            "polarity_gate_enabled": self.radial_polarity_required,
            "polarity_gate_passed": (
                polarity_result["coherence"] >= self.min_radial_polarity_coherence
                and polarity_result["coverage"] >= self.min_radial_polarity_coverage
            ),
            "source_evidence_independence": "intensity-gradient signal; independent of thresholded edge support map",
        })

        if self.radial_polarity_required and not diagnostic["polarity_gate_passed"]:
            diagnostic.update({
                "decision": "rejected",
                "rejection_reason": "inconsistent_radial_transition_polarity",
            })
            return False
        
        # Recompute Hough confidence from the original candidate evidence after
        # intensity refinement; never let a synthetic fitted contour imply geometry.
        if candidate.detection_method == "hough_circles":
            validation_evidence = candidate.processing_parameters.get("circle_validation_evidence", {})
            validation_evidence["intensity_consistency"] = intensity_consistency
            confidence = self._calculate_hough_evidence_confidence(validation_evidence)
        else:
            confidence = self._calculate_final_confidence(candidate.evidence)
        candidate.evidence.confidence = confidence
        
        # Apply confidence threshold
        if confidence < self.min_confidence:
            diagnostic.update({
                "decision": "rejected",
                "rejection_reason": "common_confidence_threshold",
                "final_confidence": confidence,
            })
            return False
        
        # Check edge support
        if candidate.evidence.edge_support < self.edge_support_threshold:
            diagnostic.update({
                "decision": "rejected",
                "rejection_reason": "common_edge_support_threshold",
                "final_confidence": confidence,
            })
            return False

        diagnostic.update({
            "decision": "validated",
            "rejection_reason": "",
            "final_confidence": confidence,
        })
        
        return True
    
    def _check_product_mask_overlap(self, center: Tuple[float, float], 
                                   radius: float, product_mask: np.ndarray) -> bool:
        """Check if circle overlaps sufficiently with product mask."""
        return self._calculate_product_mask_overlap(center, radius, product_mask) >= self.mask_overlap_threshold

    @staticmethod
    def _calculate_product_mask_overlap(center: Tuple[float, float], radius: float,
                                        product_mask: np.ndarray) -> float:
        """Return the fraction of the candidate disk contained by the product mask."""
        # Create circle mask
        h, w = product_mask.shape
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
        
        # Calculate overlap
        intersection = cv2.bitwise_and(circle_mask, product_mask)
        intersection_area = np.count_nonzero(intersection)
        circle_area = np.count_nonzero(circle_mask)
        
        if circle_area == 0:
            return 0.0
        
        return intersection_area / circle_area
    
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
            "max_gap_ratio": 1.0,
            "radial_polarity_coherence": 0.0,
            "radial_polarity_coverage": 0.0,
            "radial_polarity_positive_fraction": 0.0,
            "contour_support": None,
            "independent_contour_agreement": None,
        }
        
        # Calculate basic edge support first to adapt thresholds
        basic_edge_support = self._calculate_edge_support(center, radius, edge_image)
        result["edge_support"] = basic_edge_support
        
        # Adaptive thresholds based on edge quality - more lenient for high-quality edges
        edge_quality_factor = min(1.0, basic_edge_support / 0.7)  # Scale factor [0,1] for adaptation
        angular_result = self._analyze_angular_coverage(center, radius, edge_image)
        continuity_result = self._analyze_edge_continuity(center, radius, edge_image)
        radial_result = self._analyze_radial_consistency(center, radius, edge_image)
        contrast_result = self._analyze_local_contrast(center, radius, intensity_image)
        polarity_result = self._analyze_radial_polarity(center, radius, intensity_image)

        result.update({
            "angular_coverage": angular_result["coverage"],
            "sector_coverage": angular_result["sector_coverage"],
            "angular_uniformity": angular_result["uniformity"],
            "max_gap_ratio": continuity_result["max_gap_ratio"],
            "radial_consistency": radial_result["consistency"],
            "local_contrast": contrast_result["contrast"],
            "radial_polarity_coherence": polarity_result["coherence"],
            "radial_polarity_coverage": polarity_result["coverage"],
            "radial_polarity_positive_fraction": polarity_result["positive_fraction"],
            "contour_agreement": False,
        })

        rejection_reasons = []
        min_coverage = max(self.min_angular_coverage * (0.5 + 0.3 * edge_quality_factor), 0.45)
        min_sector = max(self.min_sector_coverage * (0.6 + 0.2 * edge_quality_factor), 0.35)
        max_gap_allowed = min(self.max_edge_gap_ratio, 0.35)
        min_radial = max(self.min_radial_agreement, 0.55)
        min_contrast = self.min_local_contrast * (0.4 + 0.4 * (1.0 - edge_quality_factor))

        if self.angular_coverage_required and angular_result["coverage"] < min_coverage:
            rejection_reasons.append(f"Insufficient angular coverage: {angular_result['coverage']:.2f} < {min_coverage:.2f}")
        if self.angular_coverage_required and angular_result["sector_coverage"] < min_sector:
            rejection_reasons.append(f"Insufficient sector coverage: {angular_result['sector_coverage']:.2f} < {min_sector:.2f}")
        if self.angular_coverage_required and angular_result["uniformity"] < 0.2:
            rejection_reasons.append(f"Localized angular support: uniformity={angular_result['uniformity']:.2f}")
        if self.edge_continuity_required and continuity_result["max_gap_ratio"] > max_gap_allowed:
            rejection_reasons.append(f"Excessive edge gaps: {continuity_result['max_gap_ratio']:.2f} > {max_gap_allowed:.2f}")
        if self.radial_consistency_required and radial_result["consistency"] < min_radial:
            rejection_reasons.append(f"Poor radial consistency: {radial_result['consistency']:.2f} < {min_radial:.2f}")
        if self.texture_discrimination and contrast_result["contrast"] < min_contrast:
            rejection_reasons.append(f"Texture-like evidence: contrast {contrast_result['contrast']:.1f} < {min_contrast:.1f}")
        if self.radial_polarity_required:
            if polarity_result["coherence"] < self.min_radial_polarity_coherence:
                rejection_reasons.append(
                    "Inconsistent radial transition polarity: "
                    f"{polarity_result['coherence']:.2f} < {self.min_radial_polarity_coherence:.2f}"
                )
            if polarity_result["coverage"] < self.min_radial_polarity_coverage:
                rejection_reasons.append(
                    "Insufficient radial transition coverage: "
                    f"{polarity_result['coverage']:.2f} < {self.min_radial_polarity_coverage:.2f}"
                )

        result["valid"] = not rejection_reasons
        result["rejection_reason"] = "; ".join(rejection_reasons)
        result["intensity_consistency"] = self._calculate_intensity_consistency(center, radius, intensity_image)

        # These are derived summaries, not independent measurements of a fitted contour.
        edge_uniformity = angular_result["uniformity"]
        result["circularity_evidence"] = min(1.0, (result["angular_coverage"] + edge_uniformity + basic_edge_support) / 3.0)
        result["solidity_evidence"] = min(1.0, result["radial_consistency"] * 1.2)
        result["convexity_evidence"] = min(1.0, result["edge_support"] * 1.1)
        result["contour_quality"] = result["circularity_evidence"]
        result["geometric_consistency"] = (result["circularity_evidence"] + result["solidity_evidence"]) / 2.0
        result["intensity_evidence"] = result["intensity_consistency"]
        
        return result

    def _analyze_radial_polarity(self, center: Tuple[float, float], radius: float,
                                 intensity_image: np.ndarray) -> dict:
        """Measure whether image transitions around a candidate share one radial polarity."""
        source_id = id(intensity_image)
        if source_id != self._radial_gradient_source_id:
            gray = cv2.cvtColor(intensity_image, cv2.COLOR_BGR2GRAY) if intensity_image.ndim == 3 else intensity_image
            self._radial_gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            self._radial_gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            self._radial_gradient_source_id = source_id
        gradient_x = self._radial_gradient_x
        gradient_y = self._radial_gradient_y
        h, w = gradient_x.shape
        signed_transitions = []

        for angle in np.linspace(0, 2 * np.pi, self.radial_polarity_samples, endpoint=False):
            cosine, sine = np.cos(angle), np.sin(angle)
            strongest_transition = 0.0
            for radial_offset in range(-self.radial_polarity_search_pixels, self.radial_polarity_search_pixels + 1):
                sample_radius = max(1.0, radius + radial_offset)
                x = int(round(center[0] + sample_radius * cosine))
                y = int(round(center[1] + sample_radius * sine))
                if 0 <= x < w and 0 <= y < h:
                    transition = float(gradient_x[y, x] * cosine + gradient_y[y, x] * sine)
                    if abs(transition) > abs(strongest_transition):
                        strongest_transition = transition
            if abs(strongest_transition) >= self.radial_polarity_min_gradient:
                signed_transitions.append(strongest_transition)

        coverage = len(signed_transitions) / self.radial_polarity_samples
        if not signed_transitions:
            return {"coherence": 0.0, "coverage": 0.0, "positive_fraction": 0.0}

        transitions = np.asarray(signed_transitions, dtype=np.float32)
        coherence = abs(float(np.sum(transitions))) / (float(np.sum(np.abs(transitions))) + 1e-6)
        return {
            "coherence": coherence,
            "coverage": coverage,
            "positive_fraction": float(np.mean(transitions > 0)),
        }
    
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
        
        # Edge support, angular coverage, and radial consistency share the edge map;
        # confidence also requires image-intensity polarity evidence.
        base_confidence = (
            HOUGH_CONFIDENCE_EDGE_SUPPORT_WEIGHT * validation_result.get("edge_support", 0.0)
            + HOUGH_CONFIDENCE_ANGULAR_COVERAGE_WEIGHT * validation_result.get("angular_coverage", 0.0)
            + HOUGH_CONFIDENCE_RADIAL_CONSISTENCY_WEIGHT * validation_result.get("radial_consistency", 0.0)
            + HOUGH_CONFIDENCE_INTENSITY_CONSISTENCY_WEIGHT * validation_result.get("intensity_consistency", 0.0)
            + HOUGH_CONFIDENCE_POLARITY_COHERENCE_WEIGHT * validation_result.get("radial_polarity_coherence", 0.0)
            + HOUGH_CONFIDENCE_POLARITY_COVERAGE_WEIGHT * validation_result.get("radial_polarity_coverage", 0.0)
        )

        # Hough-only candidates require additional evidence. Without distributed ring
        # agreement, they should not receive a strong confidence score.
        if not validation_result.get('contour_agreement', False):
            base_confidence *= (1.0 - self.hough_only_penalty)

        return min(1.0, max(0.0, base_confidence))
    
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