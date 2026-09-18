"""
Canonical Phase 2 Preprocessing Module

GEOMETRY-PRESERVATION FOCUSED IMPLEMENTATION

This is the single authoritative Phase 2 preprocessing implementation with 
critical corrections to preserve complete physical product geometry including 
irregular protrusions, tabs, and fine structures while suppressing surface texture.

The preprocessing produces separate representations:
1. Product mask (complete physical silhouette preservation)
2. Internal geometry edges (structural edges with texture suppression)  
3. Outer boundary edges (actual product contour)
4. Final combined edge representation

All processing is configuration-driven with no hardcoded product assumptions.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Import centralized configuration
from ..config import (
    PREPROCESSING_MAX_RESOLUTION, PREPROCESSING_GAUSSIAN_BLUR_KERNEL,
    PREPROCESSING_GRADIENT_THRESHOLD, PREPROCESSING_CANNY_OUTER_LOW,
    PREPROCESSING_CANNY_OUTER_HIGH, PREPROCESSING_MIN_COMPONENT_AREA_PX,
    PREPROCESSING_BORDER_MARGIN_FRACTION, PREPROCESSING_MASK_MORPH_KERNEL_SIZE,
    PREPROCESSING_MASK_MORPH_CLOSE_ITERS, PREPROCESSING_MASK_MORPH_OPEN_ITERS,
    PREPROCESSING_CLAHE_CLIP_LIMIT, PREPROCESSING_CLAHE_TILE_GRID_SIZE,
    PREPROCESSING_INTERNAL_EDGE_MAX_DENSITY, PREPROCESSING_GRADIENT_STEP_MULTIPLIER,
    PREPROCESSING_GRADIENT_MAX_RETRIES, PREPROCESSING_COMPONENT_AREA_WEIGHT,
    PREPROCESSING_COMPONENT_CENTRALITY_WEIGHT, PREPROCESSING_BORDER_MARGIN_MIN_PX,
    PREPROCESSING_GRADIENT_KERNEL_SIZE, PREPROCESSING_EDGE_CLOSE_KERNEL_SIZE,
    PREPROCESSING_INTERNAL_EDGE_COMBINATION_WEIGHT, PREPROCESSING_OUTER_BOUNDARY_COMBINATION_WEIGHT,
    PREPROCESSING_NOISE_REDUCTION_KERNEL_SIZE, PREPROCESSING_MIN_EDGE_STRENGTH,
    PREPROCESSING_TEXTURE_SUPPRESSION_THRESHOLD,
    # New geometry preservation parameters
    PREPROCESSING_MASK_MORPHOLOGY_ENABLED, PREPROCESSING_CONSERVATIVE_MASK_KERNEL_SIZE,
    PREPROCESSING_CONSERVATIVE_MASK_ITERATIONS, PREPROCESSING_CONTOUR_BASED_MASK_REFINEMENT,
    PREPROCESSING_BACKGROUND_ESTIMATION_ENABLED, PREPROCESSING_MULTI_THRESHOLD_FUSION,
    # New anti-hardcoding parameters
    PREPROCESSING_BACKGROUND_BORDER_FRACTION, PREPROCESSING_BACKGROUND_BORDER_MIN_PIXELS,
    PREPROCESSING_ADAPTIVE_THRESHOLD_BLOCK_SIZE, PREPROCESSING_ADAPTIVE_THRESHOLD_C,
    PREPROCESSING_GRADIENT_SOBEL_KERNEL_SIZE, PREPROCESSING_HOLE_AREA_FRACTION_THRESHOLD,
    PREPROCESSING_SIGNIFICANT_REGION_AREA_FRACTION, PREPROCESSING_STRUCTURE_AREA_FRACTION_THRESHOLD,
    PREPROCESSING_CONTOUR_PERIMETER_THRESHOLD, PREPROCESSING_CONTOUR_CIRCULARITY_THRESHOLD,
    # New stabilization parameters for removing hard-coded values
    PREPROCESSING_OTSU_EVIDENCE_WEIGHT, PREPROCESSING_BACKGROUND_EVIDENCE_WEIGHT, 
    PREPROCESSING_GRADIENT_EVIDENCE_WEIGHT, PREPROCESSING_EVIDENCE_NORMALIZATION_THRESHOLD,
    PREPROCESSING_GRADIENT_DILATION_KERNEL_SIZE, PREPROCESSING_GRADIENT_DILATION_ITERATIONS,
    PREPROCESSING_CONTOUR_REFINEMENT_AREA_FACTOR, PREPROCESSING_SILHOUETTE_ANALYSIS_KERNEL_SIZE,
    PREPROCESSING_SILHOUETTE_ANALYSIS_ITERATIONS, PREPROCESSING_SILHOUETTE_GRADIENT_PERCENTILE,
    PREPROCESSING_SUSPICIOUS_BOUNDARY_THRESHOLD, PREPROCESSING_WEAK_BOUNDARY_THRESHOLD,
    PREPROCESSING_EXTERNAL_VS_BOUNDARY_RATIO, PREPROCESSING_TEXTURE_SCALE_EVIDENCE_THRESHOLD,
    PREPROCESSING_TEXTURE_COMPACTNESS_THRESHOLD, PREPROCESSING_TEXTURE_EVIDENCE_COMBINATION_THRESHOLD,
    PREPROCESSING_TEXTURE_CLEANUP_KERNEL_SIZE, PREPROCESSING_TEXTURE_STRONG_GRADIENT_MULTIPLIER,
    PREPROCESSING_FINAL_EDGE_CLEANUP_THRESHOLD,
    
    # Final micro-fix parameters
    PREPROCESSING_TEXTURE_MULTI_SCALE_LEVELS, PREPROCESSING_TEXTURE_FINAL_CLEANUP_ITERATIONS
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """
    Complete result of Phase 2 preprocessing operations with geometry preservation focus.
    
    Contains all information needed to interpret and use the preprocessing output,
    with comprehensive mask quality diagnostics to detect geometry loss.
    """
    # Input information
    source_image_path: Path                  # Path to original image file
    original_image: np.ndarray              # Original input image (BGR)
    
    # Processing outputs - separate representations
    processed_image: np.ndarray             # Processed grayscale image (noise reduced)
    product_mask: np.ndarray                # Binary mask of isolated product (MUST preserve complete silhouette)
    isolated_product_image: np.ndarray      # Product region extracted from processed image
    raw_internal_geometry_edges: np.ndarray # Raw internal edges BEFORE texture suppression
    internal_geometry_edges: np.ndarray     # Internal geometric structures (texture-suppressed)
    outer_boundary_edges: np.ndarray        # Product silhouette edges (from actual mask)
    edge_representation: np.ndarray         # Final combined edge map for detection
    
    # Coordinate mapping
    original_dimensions: Tuple[int, int]    # (width, height) of original image
    processed_dimensions: Tuple[int, int]   # (width, height) of processed image  
    scale_factor: float                     # processed_size / original_size
    roi_offset: Tuple[int, int]            # (x_offset, y_offset) if ROI cropping used
    
    # Processing status
    preprocessing_successful: bool          # Whether preprocessing completed successfully
    isolation_successful: bool             # Whether product isolation succeeded
    
    # Product mask quality diagnostics - CRITICAL for geometry preservation validation
    product_area_pixels: int               # Number of pixels in product mask
    product_area_fraction: float          # Fraction of image occupied by product
    mask_bounding_box: Tuple[int, int, int, int]  # (x, y, width, height) of mask bounding box
    mask_bounding_box_area_fraction: float # Fraction of image occupied by bounding box
    foreground_component_count: int        # Number of connected foreground components
    largest_component_fraction: float      # Fraction of mask area in largest component
    mask_contour_area: float              # Area of mask's outer contour
    mask_contour_perimeter: float         # Perimeter of mask's outer contour
    mask_extent: float                    # Ratio of contour area to bounding box area
    mask_solidity: float                  # Ratio of contour area to convex hull area
    border_touching_foreground: bool       # Whether any foreground touches image border
    significant_foreground_regions: int    # Number of significant connected regions
    
    # CRITICAL: Silhouette preservation validation - checks for geometry loss
    silhouette_validation_result: str      # "PASS", "SUSPICIOUS_EXTERNAL_GRADIENTS", etc.
    external_gradient_strength: float     # Average gradient strength just outside mask
    boundary_gradient_consistency: float  # Ratio of boundary to external gradient strength
    suspicious_boundary_fraction: float   # Fraction of external boundary with strong gradients
    boundary_gradient_strength: float     # Average gradient strength at mask boundary
    
    # Edge extraction quality metrics - CORRECTED texture suppression metrics  
    raw_internal_edge_density: float       # Edge density before texture suppression
    filtered_internal_edge_density: float  # Edge density after texture suppression
    internal_edge_retention_ratio: float   # Fraction of edges retained after filtering
    internal_edge_reduction_ratio: float   # Fraction of edges removed by filtering
    outer_boundary_density: float         # Density of outer boundary pixels
    final_edge_density: float             # Density of final combined edges
    
    # Processing metadata
    processing_steps: list[str]           # List of processing steps applied
    configuration_snapshot: Dict[str, Any] # Configuration used for reproducibility
    
    def to_original_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Convert processed image coordinates to original image coordinates."""
        # Apply inverse scaling and offset correction
        orig_x = (x + self.roi_offset[0]) / self.scale_factor
        orig_y = (y + self.roi_offset[1]) / self.scale_factor
        return orig_x, orig_y
    
    def to_processed_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Convert original image coordinates to processed image coordinates."""
        # Apply scaling and offset
        proc_x = x * self.scale_factor - self.roi_offset[0]
        proc_y = y * self.scale_factor - self.roi_offset[1]
        return proc_x, proc_y


class CanonicalPreprocessor:
    """
    Canonical Phase 2 preprocessor focused on geometry preservation.
    
    This implementation prioritizes complete physical product silhouette preservation
    including irregular protrusions, tabs, and fine structures while suppressing 
    surface texture noise.
    
    CRITICAL: No shape assumptions (circular/rectangular/convex/symmetric/centered)
    """
    
    def __init__(self):
        """Initialize the canonical preprocessor with centralized configuration."""
        # Use centralized configuration values
        self.max_resolution = PREPROCESSING_MAX_RESOLUTION
        self.gradient_threshold = PREPROCESSING_GRADIENT_THRESHOLD
        self.canny_low = PREPROCESSING_CANNY_OUTER_LOW
        self.canny_high = PREPROCESSING_CANNY_OUTER_HIGH
        self.min_component_area = PREPROCESSING_MIN_COMPONENT_AREA_PX
        self.border_margin_fraction = PREPROCESSING_BORDER_MARGIN_FRACTION
        self.blur_kernel = (PREPROCESSING_GAUSSIAN_BLUR_KERNEL, PREPROCESSING_GAUSSIAN_BLUR_KERNEL)
        
        # Store configuration snapshot for reproducibility
        self._config_snapshot = {
            "max_resolution": self.max_resolution,
            "gradient_threshold": self.gradient_threshold,
            "canny_low": self.canny_low,
            "canny_high": self.canny_high,
            "min_component_area": self.min_component_area,
            "border_margin_fraction": self.border_margin_fraction,
            "blur_kernel": self.blur_kernel,
            "mask_morph_kernel_size": PREPROCESSING_MASK_MORPH_KERNEL_SIZE,
            "mask_morph_close_iters": PREPROCESSING_MASK_MORPH_CLOSE_ITERS,
            "mask_morph_open_iters": PREPROCESSING_MASK_MORPH_OPEN_ITERS,
            "clahe_clip_limit": PREPROCESSING_CLAHE_CLIP_LIMIT,
            "clahe_tile_grid": PREPROCESSING_CLAHE_TILE_GRID_SIZE,
            "internal_edge_max_density": PREPROCESSING_INTERNAL_EDGE_MAX_DENSITY,
            "gradient_step_multiplier": PREPROCESSING_GRADIENT_STEP_MULTIPLIER,
            "gradient_max_retries": PREPROCESSING_GRADIENT_MAX_RETRIES,
            "internal_edge_weight": PREPROCESSING_INTERNAL_EDGE_COMBINATION_WEIGHT,
            "outer_boundary_weight": PREPROCESSING_OUTER_BOUNDARY_COMBINATION_WEIGHT,
            "noise_reduction_kernel": PREPROCESSING_NOISE_REDUCTION_KERNEL_SIZE,
            "min_edge_strength": PREPROCESSING_MIN_EDGE_STRENGTH,
            "texture_suppression_threshold": PREPROCESSING_TEXTURE_SUPPRESSION_THRESHOLD,
            # Geometry preservation parameters
            "mask_morphology_enabled": PREPROCESSING_MASK_MORPHOLOGY_ENABLED,
            "conservative_mask_kernel": PREPROCESSING_CONSERVATIVE_MASK_KERNEL_SIZE,
            "conservative_mask_iterations": PREPROCESSING_CONSERVATIVE_MASK_ITERATIONS,
            "contour_based_refinement": PREPROCESSING_CONTOUR_BASED_MASK_REFINEMENT,
            "background_estimation_enabled": PREPROCESSING_BACKGROUND_ESTIMATION_ENABLED,
            "multi_threshold_fusion": PREPROCESSING_MULTI_THRESHOLD_FUSION,
            # Anti-hardcoding parameters
            "background_border_fraction": PREPROCESSING_BACKGROUND_BORDER_FRACTION,
            "background_border_min_pixels": PREPROCESSING_BACKGROUND_BORDER_MIN_PIXELS,
            "adaptive_threshold_block_size": PREPROCESSING_ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            "adaptive_threshold_c": PREPROCESSING_ADAPTIVE_THRESHOLD_C,
            "gradient_sobel_kernel_size": PREPROCESSING_GRADIENT_SOBEL_KERNEL_SIZE,
            "hole_area_fraction_threshold": PREPROCESSING_HOLE_AREA_FRACTION_THRESHOLD,
            "significant_region_area_fraction": PREPROCESSING_SIGNIFICANT_REGION_AREA_FRACTION,
            "structure_area_fraction_threshold": PREPROCESSING_STRUCTURE_AREA_FRACTION_THRESHOLD,
            "contour_perimeter_threshold": PREPROCESSING_CONTOUR_PERIMETER_THRESHOLD,
            "contour_circularity_threshold": PREPROCESSING_CONTOUR_CIRCULARITY_THRESHOLD,
            # New stabilization parameters - all formerly hard-coded values
            "otsu_evidence_weight": PREPROCESSING_OTSU_EVIDENCE_WEIGHT,
            "background_evidence_weight": PREPROCESSING_BACKGROUND_EVIDENCE_WEIGHT,
            "gradient_evidence_weight": PREPROCESSING_GRADIENT_EVIDENCE_WEIGHT,
            "evidence_normalization_threshold": PREPROCESSING_EVIDENCE_NORMALIZATION_THRESHOLD,
            "gradient_dilation_kernel_size": PREPROCESSING_GRADIENT_DILATION_KERNEL_SIZE,
            "gradient_dilation_iterations": PREPROCESSING_GRADIENT_DILATION_ITERATIONS,
            "contour_refinement_area_factor": PREPROCESSING_CONTOUR_REFINEMENT_AREA_FACTOR,
            "silhouette_analysis_kernel_size": PREPROCESSING_SILHOUETTE_ANALYSIS_KERNEL_SIZE,
            "silhouette_analysis_iterations": PREPROCESSING_SILHOUETTE_ANALYSIS_ITERATIONS,
            "silhouette_gradient_percentile": PREPROCESSING_SILHOUETTE_GRADIENT_PERCENTILE,
            "suspicious_boundary_threshold": PREPROCESSING_SUSPICIOUS_BOUNDARY_THRESHOLD,
            "weak_boundary_threshold": PREPROCESSING_WEAK_BOUNDARY_THRESHOLD,
            "external_vs_boundary_ratio": PREPROCESSING_EXTERNAL_VS_BOUNDARY_RATIO,
            "texture_scale_evidence_threshold": PREPROCESSING_TEXTURE_SCALE_EVIDENCE_THRESHOLD,
            "texture_compactness_threshold": PREPROCESSING_TEXTURE_COMPACTNESS_THRESHOLD,
            "texture_evidence_combination_threshold": PREPROCESSING_TEXTURE_EVIDENCE_COMBINATION_THRESHOLD,
            "texture_cleanup_kernel_size": PREPROCESSING_TEXTURE_CLEANUP_KERNEL_SIZE,
            "texture_strong_gradient_multiplier": PREPROCESSING_TEXTURE_STRONG_GRADIENT_MULTIPLIER,
            "final_edge_cleanup_threshold": PREPROCESSING_FINAL_EDGE_CLEANUP_THRESHOLD,
            
            # Final micro-fix parameters
            "texture_multi_scale_levels": PREPROCESSING_TEXTURE_MULTI_SCALE_LEVELS,
            "texture_final_cleanup_iterations": PREPROCESSING_TEXTURE_FINAL_CLEANUP_ITERATIONS,
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load a grayscale image, returning None if the file can't be read."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"cv2.imread returned None for {path}")
            raise ValueError(f"Cannot load image: {path}")
        return img

    def _cap_resolution(self, img: np.ndarray) -> np.ndarray:
        """
        Downsample so the long edge ≤ max_resolution.
        Preserves aspect ratio. Returns the original if already small enough.
        """
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge <= self.max_resolution:
            return img
        scale = self.max_resolution / long_edge
        return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def _estimate_background(self, img: np.ndarray) -> np.ndarray:
        """
        Estimate background using border region statistics.
        
        This helps distinguish foreground product from background without 
        shape assumptions.
        """
        h, w = img.shape
        border_width = max(PREPROCESSING_BACKGROUND_BORDER_MIN_PIXELS, 
                          int(min(h, w) * PREPROCESSING_BACKGROUND_BORDER_FRACTION))
        
        # Extract border pixels
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[:border_width, :] = 255  # Top
        border_mask[-border_width:, :] = 255  # Bottom  
        border_mask[:, :border_width] = 255  # Left
        border_mask[:, -border_width:] = 255  # Right
        
        border_pixels = img[border_mask > 0]
        if len(border_pixels) == 0:
            return np.full_like(img, 128, dtype=np.uint8)
        
        # Calculate background statistics
        bg_mean = np.mean(border_pixels)
        bg_std = np.std(border_pixels)
        
        # Create background model
        background = np.full_like(img, bg_mean, dtype=np.float32)
        
        return background.astype(np.uint8)
    
    def _multi_threshold_segmentation(self, img: np.ndarray) -> np.ndarray:
        """
        Use multiple thresholding approaches and fuse results to get robust 
        foreground segmentation WITHOUT destroying physical geometry.
        
        CRITICAL: Does NOT reject border-touching components - products can
        legitimately extend to image edges with protrusions or tabs.
        """
        h, w = img.shape
        
        # Method 1: Standard Otsu (both polarities)
        _, otsu_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, otsu_fwd = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Background-adaptive thresholding if enabled
        background_mask = None
        if PREPROCESSING_BACKGROUND_ESTIMATION_ENABLED:
            background = self._estimate_background(img)
            diff = cv2.absdiff(img, background)
            _, background_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 3: Gradient-based segmentation for strong object boundaries
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=PREPROCESSING_GRADIENT_SOBEL_KERNEL_SIZE)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=PREPROCESSING_GRADIENT_SOBEL_KERNEL_SIZE) 
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mag = (gradient_mag * 255 / np.max(gradient_mag)).astype(np.uint8)
        _, grad_thresh = cv2.threshold(gradient_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Create evidence accumulator - combine multiple methods
        evidence = np.zeros((h, w), dtype=np.float32)
        
        # Weight each method's contribution
        if np.sum(otsu_inv) > np.sum(otsu_fwd):
            evidence += otsu_inv.astype(np.float32) * PREPROCESSING_OTSU_EVIDENCE_WEIGHT
        else:
            evidence += otsu_fwd.astype(np.float32) * PREPROCESSING_OTSU_EVIDENCE_WEIGHT
            
        if background_mask is not None:
            evidence += background_mask.astype(np.float32) * PREPROCESSING_BACKGROUND_EVIDENCE_WEIGHT
        
        # Add gradient evidence (dilate first to fill object interiors)
        grad_dilated = cv2.dilate(grad_thresh, 
                                  np.ones((PREPROCESSING_GRADIENT_DILATION_KERNEL_SIZE, 
                                          PREPROCESSING_GRADIENT_DILATION_KERNEL_SIZE), np.uint8), 
                                  iterations=PREPROCESSING_GRADIENT_DILATION_ITERATIONS)
        evidence += grad_dilated.astype(np.float32) * PREPROCESSING_GRADIENT_EVIDENCE_WEIGHT
        
        # Convert evidence to final mask
        evidence_normalized = (evidence * 255 / np.max(evidence)).astype(np.uint8)
        _, initial_mask = cv2.threshold(evidence_normalized, PREPROCESSING_EVIDENCE_NORMALIZATION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # CRITICAL IMPROVEMENT: Apply morphological closing to connect nearby fragments
        # This helps form coherent mask while preserving overall geometry  
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        connected_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=2)
        
        # Find ALL significant components, including border-touching
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected_mask, 8)
        
        # Select components based on size, NOT location relative to borders
        combined_mask = np.zeros_like(connected_mask)
        component_areas = []
        
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            component_areas.append((area, lbl))
            
            # Only size-based filtering - NO border rejection
            if area >= self.min_component_area:
                component_mask = np.where(labels == lbl, 255, 0).astype(np.uint8)
                combined_mask = cv2.bitwise_or(combined_mask, component_mask)
        
        # If still fragmented (>3 components), try to merge nearby components
        if np.count_nonzero(combined_mask) > 0:
            num_final_labels, final_labels, final_stats, _ = cv2.connectedComponentsWithStats(combined_mask, 8)
            if num_final_labels > 4:  # Background + 3+ foreground components indicates fragmentation
                # Apply additional closing with larger kernel to merge fragments
                larger_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, larger_kernel, iterations=1)
        
        # If no components found, fall back to largest component from best Otsu
        if np.sum(combined_mask) == 0:
            logger.warning("No significant components found, using largest Otsu component")
            best_otsu = otsu_inv if np.sum(otsu_inv) > np.sum(otsu_fwd) else otsu_fwd
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(best_otsu, 8)
            if num_labels > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                combined_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
            else:
                combined_mask = best_otsu
        
        return combined_mask

    def _refine_mask_contour_based(self, mask: np.ndarray) -> np.ndarray:
        """
        Minimal refinement that preserves ALL significant contours.
        
        CRITICAL: Does NOT throw away smaller components that might be
        legitimate product parts, tabs, or protrusions.
        """
        if not PREPROCESSING_CONTOUR_BASED_MASK_REFINEMENT:
            return mask
            
        # Find ALL external contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        # Create refined mask that preserves ALL significant contours
        refined_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Only filter out truly tiny noise - NOT legitimate small features
            if area >= self.min_component_area * PREPROCESSING_CONTOUR_REFINEMENT_AREA_FACTOR:
                cv2.fillPoly(refined_mask, [contour], 255)
        
        # If refined mask is empty, return original
        if np.sum(refined_mask) == 0:
            logger.warning("Contour refinement removed all content, returning original mask")
            return mask
        
        return refined_mask

    def _isolate_product_geometry_preserving(self, blur: np.ndarray) -> np.ndarray:
        """
        Isolate product using geometry-preserving segmentation.
        
        CRITICAL: Preserves complete physical silhouette including protrusions/tabs.
        NO shape assumptions (circular/rectangular/convex/symmetric).
        """
        logger.debug("Starting geometry-preserving product isolation")
        
        # Use multi-threshold fusion for robust segmentation
        if PREPROCESSING_MULTI_THRESHOLD_FUSION:
            mask = self._multi_threshold_segmentation(blur)
        else:
            # Fallback to simple Otsu if fusion disabled
            _, mask_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, mask_fwd = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = mask_inv if np.sum(mask_inv) > np.sum(mask_fwd) else mask_fwd
        
        # Apply contour-based refinement instead of destructive morphology
        mask = self._refine_mask_contour_based(mask)
        
        return mask

    def _calculate_mask_quality_metrics(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive mask quality metrics to detect geometry loss.
        
        These metrics help identify when the mask has incorrectly removed 
        physical product geometry.
        """
        h, w = mask.shape
        total_pixels = h * w
        
        # Basic area metrics
        product_pixels = int(np.count_nonzero(mask))
        product_area_fraction = float(product_pixels) / float(total_pixels)
        
        # Bounding box analysis
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, bbox_w, bbox_h = cv2.boundingRect(main_contour)
            bbox_area = bbox_w * bbox_h
            bbox_area_fraction = float(bbox_area) / float(total_pixels)
            
            # Contour quality metrics
            contour_area = cv2.contourArea(main_contour)
            contour_perimeter = cv2.arcLength(main_contour, True)
            
            # Extent: ratio of contour area to bounding box area
            extent = contour_area / bbox_area if bbox_area > 0 else 0
            
            # Solidity: ratio of contour area to convex hull area  
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0
            
        else:
            x, y, bbox_w, bbox_h = 0, 0, 0, 0
            bbox_area_fraction = 0.0
            contour_area = 0.0
            contour_perimeter = 0.0
            extent = 0.0
            solidity = 0.0
        
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        foreground_components = num_labels - 1  # Subtract background
        
        if foreground_components > 0:
            component_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            largest_component_area = max(component_areas)
            largest_component_fraction = largest_component_area / product_pixels if product_pixels > 0 else 0
            
            # Count significant regions
            significant_regions = sum(1 for area in component_areas 
                                   if area > self.min_component_area * PREPROCESSING_SIGNIFICANT_REGION_AREA_FRACTION)
        else:
            largest_component_fraction = 0.0
            significant_regions = 0
        
        # Check border touching
        border_margin = max(PREPROCESSING_BORDER_MARGIN_MIN_PX, 
                           int(min(h, w) * self.border_margin_fraction))
        border_region = np.zeros_like(mask)
        border_region[:border_margin, :] = 255
        border_region[-border_margin:, :] = 255
        border_region[:, :border_margin] = 255
        border_region[:, -border_margin:] = 255
        
        border_touching = bool(np.any(cv2.bitwise_and(mask, border_region)))
        
        return {
            "product_area_pixels": product_pixels,
            "product_area_fraction": product_area_fraction,
            "mask_bounding_box": (x, y, bbox_w, bbox_h),
            "mask_bounding_box_area_fraction": bbox_area_fraction,
            "foreground_component_count": foreground_components,
            "largest_component_fraction": largest_component_fraction,
            "mask_contour_area": float(contour_area),
            "mask_contour_perimeter": float(contour_perimeter),
            "mask_extent": float(extent),
            "mask_solidity": float(solidity),
            "border_touching_foreground": border_touching,
            "significant_foreground_regions": significant_regions,
        }

    def _validate_silhouette_preservation(self, mask: np.ndarray, original_img: np.ndarray) -> Dict[str, Any]:
        """
        CRITICAL: Validate that the mask preserves the actual physical silhouette
        by checking for strong gradients just outside the mask boundary.
        
        This detects when the segmentation has cut through visible product edges.
        """
        
        # Extract mask boundary 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {
                "silhouette_validation": "FAILED_NO_CONTOUR",
                "external_gradient_strength": 0.0,
                "boundary_gradient_consistency": 0.0,
                "suspicious_boundary_fraction": 1.0,
                "boundary_gradient_strength": 0.0
            }
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create boundary analysis regions
        boundary_mask = np.zeros_like(mask)
        cv2.drawContours(boundary_mask, [main_contour], -1, 255, 1)
        
        # Create external analysis region (just outside the mask)
        kernel_size = PREPROCESSING_SILHOUETTE_ANALYSIS_KERNEL_SIZE
        external_region = cv2.dilate(mask, 
                                   np.ones((kernel_size, kernel_size), np.uint8), 
                                   iterations=PREPROCESSING_SILHOUETTE_ANALYSIS_ITERATIONS)
        external_region = cv2.bitwise_xor(external_region, mask)
        
        # Calculate image gradients
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze gradients just outside the mask boundary
        external_gradients = gradient_mag[external_region > 0]
        boundary_gradients = gradient_mag[boundary_mask > 0]
        
        # Calculate metrics
        external_gradient_strength = float(np.mean(external_gradients)) if len(external_gradients) > 0 else 0.0
        boundary_gradient_strength = float(np.mean(boundary_gradients)) if len(boundary_gradients) > 0 else 0.0
        
        # Check for suspicious areas where strong gradients exist just outside mask
        strong_external_threshold = np.percentile(gradient_mag.flatten(), PREPROCESSING_SILHOUETTE_GRADIENT_PERCENTILE)
        suspicious_external = external_gradients > strong_external_threshold
        suspicious_fraction = float(np.sum(suspicious_external)) / float(len(external_gradients)) if len(external_gradients) > 0 else 0.0
        
        # Boundary consistency: ratio of boundary strength to external strength
        boundary_consistency = boundary_gradient_strength / max(external_gradient_strength, 1.0)
        
        # Determine validation result
        validation_result = "PASS"
        if suspicious_fraction > PREPROCESSING_SUSPICIOUS_BOUNDARY_THRESHOLD:
            validation_result = "SUSPICIOUS_EXTERNAL_GRADIENTS"
        elif boundary_consistency < PREPROCESSING_WEAK_BOUNDARY_THRESHOLD:
            validation_result = "WEAK_BOUNDARY_EVIDENCE"
        elif external_gradient_strength > boundary_gradient_strength * PREPROCESSING_EXTERNAL_VS_BOUNDARY_RATIO:
            validation_result = "STRONGER_EXTERNAL_THAN_BOUNDARY"
        
        return {
            "silhouette_validation": validation_result,
            "external_gradient_strength": external_gradient_strength,
            "boundary_gradient_consistency": boundary_consistency,
            "suspicious_boundary_fraction": suspicious_fraction,
            "boundary_gradient_strength": boundary_gradient_strength
        }

    def _extract_internal_geometry_structure_preserving(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Extract meaningful internal geometric edges while suppressing surface texture.
        
        CRITICAL: Preserves structural geometry (holes, boundaries, mechanical features)
        while removing photographic texture/noise.
        
        Returns: (raw_internal_edges, filtered_internal_edges, metrics)
        """
        masked = cv2.bitwise_and(img, img, mask=mask)

        # Apply CLAHE for local contrast enhancement
        clahe_clip = PREPROCESSING_CLAHE_CLIP_LIMIT
        clahe_grid = PREPROCESSING_CLAHE_TILE_GRID_SIZE
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        enhanced = clahe.apply(masked)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

        # Extract raw internal edges using morphological gradient
        grad_kernel_size = PREPROCESSING_GRADIENT_KERNEL_SIZE
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (grad_kernel_size, grad_kernel_size))
        gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, k)

        # Raw edge extraction with adaptive thresholding
        mask_area = int(np.count_nonzero(mask))
        max_density = PREPROCESSING_INTERNAL_EDGE_MAX_DENSITY
        step_multiplier = PREPROCESSING_GRADIENT_STEP_MULTIPLIER
        max_retries = PREPROCESSING_GRADIENT_MAX_RETRIES

        thresh = self.gradient_threshold
        for retry in range(max_retries):
            _, raw_edges = cv2.threshold(gradient, thresh, 255, cv2.THRESH_BINARY)
            raw_edges = cv2.bitwise_and(raw_edges, raw_edges, mask=mask)
            
            raw_density = float(np.count_nonzero(raw_edges)) / float(mask_area) if mask_area > 0 else 0.0
            if raw_density <= max_density:
                break
            thresh = int(thresh * step_multiplier)

        # Calculate raw edge metrics BEFORE texture suppression
        raw_edge_pixels = int(np.count_nonzero(raw_edges))
        raw_edge_density = float(raw_edge_pixels) / float(mask.size)

        # TEXTURE SUPPRESSION: Structure-preserving edge filtering
        filtered_edges = self._suppress_texture_preserve_structure(raw_edges, gradient, mask)
        
        # Calculate filtered edge metrics AFTER texture suppression  
        filtered_edge_pixels = int(np.count_nonzero(filtered_edges))
        filtered_edge_density = float(filtered_edge_pixels) / float(mask.size)
        
        # Calculate CORRECT texture suppression metrics
        if raw_edge_pixels > 0:
            retention_ratio = float(filtered_edge_pixels) / float(raw_edge_pixels)
            reduction_ratio = 1.0 - retention_ratio
        else:
            retention_ratio = 1.0
            reduction_ratio = 0.0
        
        metrics = {
            "raw_internal_edge_density": raw_edge_density,
            "filtered_internal_edge_density": filtered_edge_density,
            "internal_edge_retention_ratio": retention_ratio,
            "internal_edge_reduction_ratio": reduction_ratio,
        }

        return raw_edges, filtered_edges, metrics

    def _suppress_texture_preserve_structure(self, raw_edges: np.ndarray, gradient: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Suppress surface texture while preserving structural geometry.
        
        Uses multi-scale structural analysis to distinguish meaningful geometry
        from photographic texture/noise.
        
        CRITICAL: Must significantly reduce texture (target 30-60% reduction)
        while preserving holes, boundaries, and mechanical features.
        """
        
        # Stage 1: Multi-scale consistency check
        # Edges that persist across scales are more likely structural
        scales = PREPROCESSING_TEXTURE_MULTI_SCALE_LEVELS
        scale_evidence = np.zeros_like(raw_edges, dtype=np.float32)
        
        for scale in scales:
            if scale > 1:
                # Apply mild blur and re-extract edges
                blurred = cv2.GaussianBlur(gradient, (scale*2+1, scale*2+1), scale*0.7)
            else:
                blurred = gradient.copy()
                
            # Extract edges at this scale
            _, scale_edges = cv2.threshold(blurred, PREPROCESSING_MIN_EDGE_STRENGTH, 255, cv2.THRESH_BINARY)
            scale_edges = cv2.bitwise_and(scale_edges, scale_edges, mask=mask)
            
            # Add evidence (weight inversely with scale)
            scale_evidence += scale_edges.astype(np.float32) * (1.0 / scale)
        
        # Normalize scale evidence
        if np.max(scale_evidence) > 0:
            scale_evidence = scale_evidence / np.max(scale_evidence)
        
        # Stage 2: Connectivity analysis
        # Connected structures are more likely to be meaningful geometry
        contours, _ = cv2.findContours(raw_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        connectivity_evidence = np.zeros_like(raw_edges, dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Preserve contours that are:
            # 1. Large enough to be meaningful
            # 2. Have reasonable connectivity (not just noise fragments)
            if area > PREPROCESSING_STRUCTURE_AREA_FRACTION_THRESHOLD * self.min_component_area:
                cv2.drawContours(connectivity_evidence, [contour], -1, 255, -1)
            elif perimeter > PREPROCESSING_CONTOUR_PERIMETER_THRESHOLD:
                # Preserve longer thin structures (like grooves, edges)
                if perimeter > 0:
                    compactness = (perimeter * perimeter) / (4 * np.pi * max(area, 1))
                    if compactness < PREPROCESSING_TEXTURE_COMPACTNESS_THRESHOLD:  # Not too fragmented
                        cv2.drawContours(connectivity_evidence, [contour], -1, 255, -1)
        
        # Stage 3: Gradient strength filtering
        # Strong gradients are more likely to be real boundaries
        strong_gradient_mask = gradient > (PREPROCESSING_MIN_EDGE_STRENGTH * PREPROCESSING_TEXTURE_STRONG_GRADIENT_MULTIPLIER)
        
        # Stage 4: Combine evidence sources
        # Only keep edges that have multiple forms of structural evidence
        
        # Convert evidence to weights
        scale_weight = (scale_evidence > PREPROCESSING_TEXTURE_SCALE_EVIDENCE_THRESHOLD).astype(np.float32)
        connectivity_weight = (connectivity_evidence > 0).astype(np.float32) 
        strength_weight = strong_gradient_mask.astype(np.float32)
        
        # Require at least 2 out of 3 evidence types for preservation
        combined_evidence = scale_weight + connectivity_weight + strength_weight
        structure_mask = combined_evidence >= PREPROCESSING_TEXTURE_EVIDENCE_COMBINATION_THRESHOLD
        
        # Apply structural filter to raw edges
        filtered_edges = cv2.bitwise_and(raw_edges, raw_edges, mask=structure_mask.astype(np.uint8) * 255)
        
        # Final cleanup - remove very isolated pixels (single pixel noise)
        kernel = np.ones((PREPROCESSING_TEXTURE_CLEANUP_KERNEL_SIZE, PREPROCESSING_TEXTURE_CLEANUP_KERNEL_SIZE), np.uint8)
        filtered_edges = cv2.morphologyEx(filtered_edges, cv2.MORPH_OPEN, kernel, iterations=PREPROCESSING_TEXTURE_FINAL_CLEANUP_ITERATIONS)
        
        # Ensure we're still within the product mask
        filtered_edges = cv2.bitwise_and(filtered_edges, filtered_edges, mask=mask)
        
        return filtered_edges

    def _extract_outer_boundary_from_actual_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract outer boundary from the actual detected product mask.
        
        CRITICAL: Uses actual mask contour, not approximated shapes.
        """
        # Extract boundary using Canny on the mask itself
        boundary = cv2.Canny(mask, self.canny_low, self.canny_high)
        
        # Alternative: use contour-based boundary for more precise control
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_boundary = np.zeros_like(mask)
        
        if contours:
            # Draw all external contours (preserves protrusions/tabs)
            cv2.drawContours(contour_boundary, contours, -1, 255, 1)
        
        # Combine both methods for robustness
        combined_boundary = cv2.bitwise_or(boundary, contour_boundary)
        
        return combined_boundary

    def _combine_edge_representations_corrected(self, internal_edges: np.ndarray, 
                                              outer_boundary: np.ndarray) -> np.ndarray:
        """
        Combine internal geometry and outer boundary into final edge representation.
        
        Uses configurable weights to balance contributions without corrupting 
        texture suppression metrics.
        """
        internal_weight = PREPROCESSING_INTERNAL_EDGE_COMBINATION_WEIGHT
        outer_weight = PREPROCESSING_OUTER_BOUNDARY_COMBINATION_WEIGHT
        
        # Normalize weights
        total_weight = internal_weight + outer_weight
        internal_weight = internal_weight / total_weight
        outer_weight = outer_weight / total_weight
        
        # Combine using weighted addition
        internal_weighted = (internal_edges.astype(np.float32) * internal_weight)
        outer_weighted = (outer_boundary.astype(np.float32) * outer_weight)
        
        combined = internal_weighted + outer_weighted
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        # Apply final cleanup - remove very weak combined edges
        _, final_edges = cv2.threshold(combined, PREPROCESSING_FINAL_EDGE_CLEANUP_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        return final_edges
    
    def preprocess_image(self, image_path: Path) -> PreprocessingResult:
        """
        Apply complete Phase 2 preprocessing with geometry preservation focus.
        
        CRITICAL: Preserves complete physical product silhouette including 
        irregular protrusions, tabs, and fine structures.
        
        Args:
            image_path: Path to input product image
            
        Returns:
            PreprocessingResult with comprehensive quality diagnostics
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Starting geometry-preserving Phase 2 preprocessing: {image_path.name}")
        
        try:
            # Step 1: Load and prepare image
            original_gray = self._load_image(image_path)
            
            # Load original BGR for reference
            original_bgr = cv2.imread(str(image_path))
            if original_bgr is None:
                raise ValueError(f"Failed to load BGR image: {image_path}")
            
            original_height, original_width = original_gray.shape
            original_dims = (original_width, original_height)
            
            logger.debug(f"Loaded image: {original_width}x{original_height}")
            
            # Step 2: Cap resolution if needed
            processed_gray = self._cap_resolution(original_gray)
            processed_height, processed_width = processed_gray.shape
            processed_dims = (processed_width, processed_height)
            
            # Calculate scale factor
            scale_factor = min(processed_width / original_width, processed_height / original_height)
            
            if scale_factor < 1.0:
                logger.debug(f"Resized image: {original_dims} -> {processed_dims} (scale: {scale_factor:.3f})")
            
            # Step 3: Initial gentle noise reduction (preserve edges)
            blur = cv2.GaussianBlur(processed_gray, self.blur_kernel, 0)
            
            # Step 4: CRITICAL - Geometry-preserving product isolation
            product_mask = self._isolate_product_geometry_preserving(blur)
            
            # Step 5: Calculate comprehensive mask quality diagnostics 
            mask_metrics = self._calculate_mask_quality_metrics(product_mask)
            
            # Step 5.1: CRITICAL - Validate actual silhouette preservation
            silhouette_validation = self._validate_silhouette_preservation(product_mask, original_bgr)
            
            # Combine mask metrics with silhouette validation
            mask_metrics.update(silhouette_validation)
            
            # Step 6: Verify isolation success and quality
            isolation_successful = np.any(product_mask)
            if isolation_successful:
                logger.info(f"Product isolation successful: {mask_metrics['product_area_pixels']} pixels "
                           f"({mask_metrics['product_area_fraction']:.1%})")
                
                # Check for potential geometry loss indicators
                if mask_metrics['border_touching_foreground']:
                    logger.warning("Product mask touches image border - possible geometry truncation")
                    
                if mask_metrics['mask_solidity'] < 0.7:
                    logger.warning(f"Low mask solidity ({mask_metrics['mask_solidity']:.2f}) - possible geometry loss")
                    
                if mask_metrics['significant_foreground_regions'] > 3:
                    logger.warning(f"Multiple significant regions ({mask_metrics['significant_foreground_regions']}) - possible fragmentation")
            else:
                logger.warning("Product isolation failed")
            
            # Step 7: Extract isolated product image
            isolated_product_image = cv2.bitwise_and(processed_gray, processed_gray, mask=product_mask)
            
            # Step 8: Extract internal geometry with texture suppression
            if isolation_successful:
                raw_internal, filtered_internal, edge_metrics = self._extract_internal_geometry_structure_preserving(
                    processed_gray, product_mask)
                
                # Extract outer boundary from actual mask (not approximation)
                outer_boundary_edges = self._extract_outer_boundary_from_actual_mask(product_mask)
                outer_boundary_density = float(np.count_nonzero(outer_boundary_edges)) / float(product_mask.size)
                
                # Combine into final representation with corrected metrics
                final_edges = self._combine_edge_representations_corrected(filtered_internal, outer_boundary_edges)
                final_edge_density = float(np.count_nonzero(final_edges)) / float(product_mask.size)
                
                logger.info(f"Edge extraction complete - raw: {edge_metrics['raw_internal_edge_density']:.3f}, "
                           f"filtered: {edge_metrics['filtered_internal_edge_density']:.3f}, "
                           f"reduction: {edge_metrics['internal_edge_reduction_ratio']:.1%}")
                
            else:
                # Create empty representations for failed isolation
                raw_internal = np.zeros_like(processed_gray)
                filtered_internal = np.zeros_like(processed_gray)
                outer_boundary_edges = np.zeros_like(processed_gray)
                final_edges = np.zeros_like(processed_gray)
                
                edge_metrics = {
                    "raw_internal_edge_density": 0.0,
                    "filtered_internal_edge_density": 0.0,
                    "internal_edge_retention_ratio": 1.0,
                    "internal_edge_reduction_ratio": 0.0,
                }
                outer_boundary_density = 0.0
                final_edge_density = 0.0
            
            # Step 9: Build comprehensive result with proper diagnostics
            result = PreprocessingResult(
                source_image_path=image_path,
                original_image=original_bgr,
                processed_image=processed_gray,
                product_mask=product_mask,
                isolated_product_image=isolated_product_image,
                raw_internal_geometry_edges=raw_internal,  # CRITICAL: Store raw edges directly  
                internal_geometry_edges=filtered_internal,  # Use filtered, not raw
                outer_boundary_edges=outer_boundary_edges,
                edge_representation=final_edges,
                original_dimensions=original_dims,
                processed_dimensions=processed_dims,
                scale_factor=scale_factor,
                roi_offset=(0, 0),  # No ROI cropping in current implementation
                preprocessing_successful=True,
                isolation_successful=isolation_successful,
                
                # Mask quality diagnostics from mask_metrics
                product_area_pixels=mask_metrics['product_area_pixels'],
                product_area_fraction=mask_metrics['product_area_fraction'],
                mask_bounding_box=mask_metrics['mask_bounding_box'],
                mask_bounding_box_area_fraction=mask_metrics['mask_bounding_box_area_fraction'],
                foreground_component_count=mask_metrics['foreground_component_count'],
                largest_component_fraction=mask_metrics['largest_component_fraction'],
                mask_contour_area=mask_metrics['mask_contour_area'],
                mask_contour_perimeter=mask_metrics['mask_contour_perimeter'],
                mask_extent=mask_metrics['mask_extent'],
                mask_solidity=mask_metrics['mask_solidity'],
                border_touching_foreground=mask_metrics['border_touching_foreground'],
                significant_foreground_regions=mask_metrics['significant_foreground_regions'],
                # CRITICAL: Silhouette preservation validation results
                silhouette_validation_result=mask_metrics['silhouette_validation'],
                external_gradient_strength=mask_metrics['external_gradient_strength'],
                boundary_gradient_consistency=mask_metrics['boundary_gradient_consistency'],
                suspicious_boundary_fraction=mask_metrics['suspicious_boundary_fraction'],
                boundary_gradient_strength=mask_metrics['boundary_gradient_strength'],
                
                # CORRECTED edge extraction metrics from edge_metrics
                raw_internal_edge_density=edge_metrics['raw_internal_edge_density'],
                filtered_internal_edge_density=edge_metrics['filtered_internal_edge_density'],
                internal_edge_retention_ratio=edge_metrics['internal_edge_retention_ratio'],
                internal_edge_reduction_ratio=edge_metrics['internal_edge_reduction_ratio'],
                outer_boundary_density=outer_boundary_density,
                final_edge_density=final_edge_density,
                
                processing_steps=[
                    "load_grayscale",
                    "cap_resolution" if scale_factor < 1.0 else "no_resize",
                    "gaussian_blur_noise_reduction",
                    "background_estimation" if PREPROCESSING_BACKGROUND_ESTIMATION_ENABLED else "skip_background_estimation",
                    "multi_threshold_fusion" if PREPROCESSING_MULTI_THRESHOLD_FUSION else "simple_otsu_isolation",
                    "geometry_preserving_segmentation",
                    "contour_based_mask_refinement" if PREPROCESSING_CONTOUR_BASED_MASK_REFINEMENT else "morphological_mask_refinement",
                    "mask_quality_diagnostics",
                    "product_region_extraction",
                    "clahe_contrast_enhancement",
                    "raw_internal_edge_extraction",
                    "structure_preserving_texture_suppression",
                    "actual_mask_outer_boundary_extraction",
                    "corrected_weighted_edge_combination",
                    "final_edge_cleanup"
                ],
                configuration_snapshot=self._config_snapshot.copy()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Return failed result instead of raising
            return self._create_failed_result(image_path, str(e))
    
    def _create_failed_result(self, image_path: Path, error_message: str) -> PreprocessingResult:
        """Create a failed preprocessing result for error handling."""
        # Try to load original image for dimensions, fallback if that fails too
        try:
            original_bgr = cv2.imread(str(image_path))
            if original_bgr is not None:
                original_dims = (original_bgr.shape[1], original_bgr.shape[0])
                empty_gray = np.zeros((original_bgr.shape[0], original_bgr.shape[1]), dtype=np.uint8)
            else:
                original_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
                original_dims = (100, 100)
                empty_gray = np.zeros((100, 100), dtype=np.uint8)
        except:
            original_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
            original_dims = (100, 100)
            empty_gray = np.zeros((100, 100), dtype=np.uint8)
        
        empty_mask = np.zeros_like(empty_gray, dtype=np.uint8)
        
        return PreprocessingResult(
            source_image_path=image_path,
            original_image=original_bgr,
            processed_image=empty_gray,
            product_mask=empty_mask,
            isolated_product_image=empty_mask,
            raw_internal_geometry_edges=empty_mask,
            internal_geometry_edges=empty_mask,
            outer_boundary_edges=empty_mask,
            edge_representation=empty_mask,
            original_dimensions=original_dims,
            processed_dimensions=(empty_gray.shape[1], empty_gray.shape[0]),
            scale_factor=1.0,
            roi_offset=(0, 0),
            preprocessing_successful=False,
            isolation_successful=False,
            
            # Failed mask quality diagnostics
            product_area_pixels=0,
            product_area_fraction=0.0,
            mask_bounding_box=(0, 0, 0, 0),
            mask_bounding_box_area_fraction=0.0,
            foreground_component_count=0,
            largest_component_fraction=0.0,
            mask_contour_area=0.0,
            mask_contour_perimeter=0.0,
            mask_extent=0.0,
            mask_solidity=0.0,
            border_touching_foreground=False,
            significant_foreground_regions=0,
            
            # Failed silhouette validation
            silhouette_validation_result="FAILED_NO_MASK",
            external_gradient_strength=0.0,
            boundary_gradient_consistency=0.0,
            suspicious_boundary_fraction=0.0,
            boundary_gradient_strength=0.0,
            
            # Failed edge extraction metrics
            raw_internal_edge_density=0.0,
            filtered_internal_edge_density=0.0,
            internal_edge_retention_ratio=1.0,
            internal_edge_reduction_ratio=0.0,
            outer_boundary_density=0.0,
            final_edge_density=0.0,
            
            processing_steps=["failed"],
            configuration_snapshot={"error": error_message}
        )
    
    def save_preprocessing_outputs(self, result: PreprocessingResult, output_dir: Path) -> Dict[str, Path]:
        """
        Save all preprocessing outputs with comprehensive diagnostics for geometry validation.
        
        CRITICAL: Includes raw internal edges and enhanced visual inspection outputs.
        
        Args:
            result: Preprocessing result to save
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary mapping output types to saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # Save original image copy for reference
        original_path = output_dir / "original_image.png"
        cv2.imwrite(str(original_path), result.original_image)
        saved_files["original"] = original_path
        
        # Save product mask - CRITICAL for geometry validation
        mask_path = output_dir / "product_mask.png"
        cv2.imwrite(str(mask_path), result.product_mask)
        saved_files["mask"] = mask_path
        
        # Create mask overlay for geometry inspection - ENHANCED
        mask_overlay_path = output_dir / "mask_overlay.png"
        if result.preprocessing_successful:
            mask_overlay = result.original_image.copy()
            # Draw mask contour in bright green for visibility
            contours, _ = cv2.findContours(result.product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_overlay, contours, -1, (0, 255, 0), 3)  # Thicker green line
            
            # Draw bounding box in blue
            x, y, w, h = result.mask_bounding_box
            cv2.rectangle(mask_overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add semi-transparent mask overlay to show filled area
            mask_colored = cv2.applyColorMap(result.product_mask, cv2.COLORMAP_JET)
            mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)
            
            cv2.imwrite(str(mask_overlay_path), mask_overlay)
        else:
            cv2.imwrite(str(mask_overlay_path), result.original_image)
        saved_files["mask_overlay"] = mask_overlay_path
        
        # Save isolated product image
        isolated_path = output_dir / "isolated_product.png"
        cv2.imwrite(str(isolated_path), result.isolated_product_image)
        saved_files["isolated_product"] = isolated_path
        
        # CRITICAL: Save RAW internal edges (stored from preprocessing - no re-extraction)
        raw_internal_path = output_dir / "raw_internal_geometry_edges.png"
        cv2.imwrite(str(raw_internal_path), result.raw_internal_geometry_edges)
        saved_files["raw_internal_edges"] = raw_internal_path
        
        # Save filtered internal geometry edges
        internal_path = output_dir / "internal_geometry_edges.png"
        cv2.imwrite(str(internal_path), result.internal_geometry_edges)
        saved_files["internal_edges"] = internal_path
        
        # Save outer boundary edges
        outer_path = output_dir / "outer_boundary_edges.png"
        cv2.imwrite(str(outer_path), result.outer_boundary_edges)
        saved_files["outer_boundary"] = outer_path
        
        # Save final combined edge representation (main output for feature detection)
        final_path = output_dir / "final_preprocessed_edges.png"
        cv2.imwrite(str(final_path), result.edge_representation)
        saved_files["final_edges"] = final_path
        
        # Save processed grayscale image
        processed_path = output_dir / "processed_image.png"
        cv2.imwrite(str(processed_path), result.processed_image)
        saved_files["processed"] = processed_path
        
        # Create enhanced comparison montage for visual inspection
        montage_path = output_dir / "preprocessing_montage.png"
        self._create_enhanced_visual_montage(result, montage_path, raw_internal_path)
        saved_files["montage"] = montage_path
        
        # Save comprehensive metadata as JSON with enhanced metrics
        metadata = {
            "source_image": str(result.source_image_path),
            "preprocessing_successful": bool(result.preprocessing_successful),
            "isolation_successful": bool(result.isolation_successful),
            "original_dimensions": result.original_dimensions,
            "processed_dimensions": result.processed_dimensions,
            "scale_factor": float(result.scale_factor),
            "roi_offset": result.roi_offset,
            
            # Mask quality diagnostics
            "product_area_pixels": int(result.product_area_pixels),
            "product_area_fraction": float(result.product_area_fraction),
            "mask_bounding_box": result.mask_bounding_box,
            "mask_bounding_box_area_fraction": float(result.mask_bounding_box_area_fraction),
            "foreground_component_count": int(result.foreground_component_count),
            "largest_component_fraction": float(result.largest_component_fraction),
            "mask_contour_area": float(result.mask_contour_area),
            "mask_contour_perimeter": float(result.mask_contour_perimeter),
            "mask_extent": float(result.mask_extent),
            "mask_solidity": float(result.mask_solidity),
            "border_touching_foreground": bool(result.border_touching_foreground),
            "significant_foreground_regions": int(result.significant_foreground_regions),
            
            # CRITICAL: Silhouette validation results
            "silhouette_validation_result": result.silhouette_validation_result,
            "external_gradient_strength": float(result.external_gradient_strength),
            "boundary_gradient_consistency": float(result.boundary_gradient_consistency),
            "suspicious_boundary_fraction": float(result.suspicious_boundary_fraction),
            "boundary_gradient_strength": float(result.boundary_gradient_strength),
            
            # CORRECTED edge extraction metrics
            "raw_internal_edge_density": float(result.raw_internal_edge_density),
            "filtered_internal_edge_density": float(result.filtered_internal_edge_density),
            "internal_edge_retention_ratio": float(result.internal_edge_retention_ratio),
            "internal_edge_reduction_ratio": float(result.internal_edge_reduction_ratio),
            "outer_boundary_density": float(result.outer_boundary_density),
            "final_edge_density": float(result.final_edge_density),
            
            "processing_steps": result.processing_steps,
            "configuration": result.configuration_snapshot
        }
        
        metadata_path = output_dir / "preprocessing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files["metadata"] = metadata_path
        
        logger.info(f"Saved preprocessing outputs to: {output_dir}")
        return saved_files
    
    def _create_enhanced_visual_montage(self, result: PreprocessingResult, output_path: Path, raw_internal_path: Path):
        """Create enhanced visual comparison montage for geometry preservation validation."""
        
        # Load raw internal edges
        raw_internal_edges = cv2.imread(str(raw_internal_path), cv2.IMREAD_GRAYSCALE)
        if raw_internal_edges is None:
            raw_internal_edges = np.zeros_like(result.internal_geometry_edges)
        
        # Convert grayscale images to BGR for consistent montage
        original_bgr = result.original_image
        processed_bgr = cv2.cvtColor(result.processed_image, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.cvtColor(result.product_mask, cv2.COLOR_GRAY2BGR)
        isolated_bgr = cv2.cvtColor(result.isolated_product_image, cv2.COLOR_GRAY2BGR)
        raw_internal_bgr = cv2.cvtColor(raw_internal_edges, cv2.COLOR_GRAY2BGR)
        filtered_internal_bgr = cv2.cvtColor(result.internal_geometry_edges, cv2.COLOR_GRAY2BGR)
        outer_bgr = cv2.cvtColor(result.outer_boundary_edges, cv2.COLOR_GRAY2BGR)
        final_bgr = cv2.cvtColor(result.edge_representation, cv2.COLOR_GRAY2BGR)
        
        # Create enhanced mask overlay
        mask_overlay = original_bgr.copy()
        contours, _ = cv2.findContours(result.product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_overlay, contours, -1, (0, 255, 0), 3)
        
        # Add silhouette validation status
        status_color = (0, 255, 0) if result.silhouette_validation_result == "PASS" else (0, 0, 255)
        cv2.putText(mask_overlay, f"Silhouette: {result.silhouette_validation_result}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Resize all images to same height for montage
        target_height = 200
        images = [original_bgr, mask_overlay, isolated_bgr, raw_internal_bgr, 
                 filtered_internal_bgr, outer_bgr, final_bgr]
        labels = ["Original", "Mask Overlay", "Isolated Product", "Raw Internal Edges", 
                 "Filtered Internal", "Outer Boundary", "Final Edges"]
        
        resized_images = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            aspect_ratio = w / h
            new_width = int(target_height * aspect_ratio)
            resized = cv2.resize(img, (new_width, target_height))
            
            # Add label with texture suppression info for edge images
            labeled = resized.copy()
            label_text = labels[i]
            if i == 3:  # Raw internal edges
                label_text += f" ({result.raw_internal_edge_density:.3f})"
            elif i == 4:  # Filtered internal edges
                reduction = result.internal_edge_reduction_ratio
                label_text += f" ({result.filtered_internal_edge_density:.3f}, -{reduction:.1%})"
            
            cv2.putText(labeled, label_text, (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            resized_images.append(labeled)
        
        # Create montage (2 rows: first 4 images, then 3 images)
        top_row = np.hstack(resized_images[:4])
        bottom_row = np.hstack(resized_images[4:])
        
        # Pad bottom row to match width if needed
        if bottom_row.shape[1] < top_row.shape[1]:
            padding_width = top_row.shape[1] - bottom_row.shape[1]
            padding = np.zeros((bottom_row.shape[0], padding_width, 3), dtype=np.uint8)
            bottom_row = np.hstack([bottom_row, padding])
        
        montage = np.vstack([top_row, bottom_row])
        cv2.imwrite(str(output_path), montage)