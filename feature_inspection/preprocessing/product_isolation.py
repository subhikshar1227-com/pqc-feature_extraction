"""
Product Isolation Preprocessing Module

DEPRECATED: This module has been superseded by the canonical preprocessor.
Use feature_inspection.actual.canonical_preprocessor.CanonicalPreprocessor instead.

This module will be removed in a future version.

Extracts and provides the proven product isolation preprocessing logic
from Phase 1 alignment for reuse in Phase 2 actual feature detection.

This module preserves the exact behavior of the existing Phase 1 preprocessing
while making it available as a modular component.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any
import logging
import warnings

logger = logging.getLogger(__name__)

# Issue deprecation warning when module is imported
warnings.warn(
    "ProductIsolationPreprocessor is deprecated. Use feature_inspection.actual.canonical_preprocessor.CanonicalPreprocessor instead.",
    DeprecationWarning,
    stacklevel=2
)


class ProductIsolationPreprocessor:
    """
    DEPRECATED: Use feature_inspection.actual.canonical_preprocessor.CanonicalPreprocessor instead.
    
    Handles product isolation and edge extraction using the proven
    Phase 1 preprocessing algorithms.
    
    This extracts the good existing preprocessing logic from alignment.py
    and quick_test.py without modifying its behavior.
    """
    
    def __init__(self,
                 gradient_threshold: int = 12,
                 canny_outer_low: int = 50,
                 canny_outer_high: int = 150,
                 min_component_area_px: int = 200,
                 border_margin_fraction: float = 0.01,
                 mask_morph_kernel_size: int = 9,
                 mask_morph_close_iters: int = 3,
                 mask_morph_open_iters: int = 2,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid: Tuple[int, int] = (8, 8),
                 internal_edge_max_density: float = 0.08,
                 gradient_step_multiplier: float = 1.5,
                 gradient_max_retries: int = 4,
                 component_area_weight: float = 0.95,
                 component_centrality_weight: float = 0.05,
                 border_margin_min_px: int = 3,
                 gradient_kernel_size: int = 3,
                 edge_close_kernel_size: int = 2):
        """Initialize product isolation preprocessor with configuration."""
        # Store all configuration parameters
        self.gradient_threshold = gradient_threshold
        self.canny_outer_low = canny_outer_low
        self.canny_outer_high = canny_outer_high
        self.min_component_area_px = min_component_area_px
        self.border_margin_fraction = border_margin_fraction
        self.mask_morph_kernel_size = mask_morph_kernel_size
        self.mask_morph_close_iters = mask_morph_close_iters
        self.mask_morph_open_iters = mask_morph_open_iters
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.internal_edge_max_density = internal_edge_max_density
        self.gradient_step_multiplier = gradient_step_multiplier
        self.gradient_max_retries = gradient_max_retries
        self.component_area_weight = component_area_weight
        self.component_centrality_weight = component_centrality_weight
        self.border_margin_min_px = border_margin_min_px
        self.gradient_kernel_size = gradient_kernel_size
        self.edge_close_kernel_size = edge_close_kernel_size
        
    def isolate_and_extract_edges(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Complete product isolation and edge extraction pipeline.
        
        This is the main entry point that replicates the proven Phase 1 preprocessing:
        1. Blur the input image
        2. Isolate the product using dual-polarity Otsu thresholding
        3. Extract internal edges using CLAHE and morphological gradient
        4. Extract outer boundary using Canny
        5. Combine edges into final edge map
        
        Args:
            image: Input BGR image (numpy array)
            
        Returns:
            Tuple of (edge_map, product_mask, metadata)
            - edge_map: Combined internal and outer edges (uint8 binary)
            - product_mask: Binary mask of isolated product (uint8 binary)
            - metadata: Dictionary with processing information and coordinate mapping
        """
        if image is None:
            raise ValueError("Input image is None")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
            
        logger.debug(f"Starting product isolation preprocessing on image shape {image.shape}")
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Apply Gaussian blur (matching quick_test.py preprocessing)
        blur_kernel = (5, 5)  # Default from quick_test.py
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
        
        # Step 2: Isolate product using dual-polarity Otsu thresholding
        product_mask = self._isolate_part(blurred)
        
        # Check if product isolation was successful
        if not np.any(product_mask):
            logger.warning("Product isolation failed - no product mask generated")
            # Return empty results but don't crash
            empty_edges = np.zeros_like(gray)
            metadata = {
                "isolation_successful": False,
                "isolation_method": "dual_polarity_otsu",
                "product_area_pixels": 0,
                "product_area_fraction": 0.0,
                "coordinate_transform_info": {
                    "processed_to_original_scale": 1.0,
                    "processed_offset": (0, 0),
                    "original_dimensions": image.shape[:2]
                },
                "processing_steps": ["blur", "isolation_failed"]
            }
            return empty_edges, product_mask, metadata
        
        # Step 3: Extract internal edges from the product region
        internal_edges = self._extract_internal_edges(gray, product_mask)
        
        # Step 4: Extract outer boundary of the product
        outer_boundary = self._extract_outer_boundary(product_mask)
        
        # Step 5: Combine internal and outer edges
        combined_edges = cv2.bitwise_or(internal_edges, outer_boundary)
        
        # Calculate metadata
        product_area = int(np.count_nonzero(product_mask))
        total_area = product_mask.size
        
        metadata = {
            "isolation_successful": True,
            "isolation_method": "dual_polarity_otsu",
            "product_area_pixels": product_area,
            "product_area_fraction": float(product_area) / float(total_area),
            "internal_edge_pixels": int(np.count_nonzero(internal_edges)),
            "outer_edge_pixels": int(np.count_nonzero(outer_boundary)),
            "total_edge_pixels": int(np.count_nonzero(combined_edges)),
            "coordinate_transform_info": {
                "processed_to_original_scale": 1.0,
                "processed_offset": (0, 0),
                "original_dimensions": image.shape[:2]
            },
            "processing_steps": ["blur", "isolation", "internal_edges", "outer_boundary", "combine_edges"],
            "configuration": {
                "gradient_threshold": self.gradient_threshold,
                "border_margin_fraction": self.border_margin_fraction,
                "min_component_area_px": self.min_component_area_px
            }
        }
        
        logger.debug(f"Product isolation complete: {product_area} pixels ({metadata['product_area_fraction']:.3f} of image)")
        logger.debug(f"Edge extraction complete: {metadata['total_edge_pixels']} edge pixels")
        
        return combined_edges, product_mask, metadata
    
    def _isolate_part(self, blur: np.ndarray) -> np.ndarray:
        """
        Isolate the part from the background using adaptive Otsu thresholding.
        
        This is an exact copy of the isolate_part function from quick_test.py
        """
        inv_mask, inv_score = self._threshold_and_pick_part(
            blur, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            self.mask_morph_kernel_size, self.mask_morph_close_iters, 
            self.mask_morph_open_iters, self.min_component_area_px, self.border_margin_fraction,
        )
        fwd_mask, fwd_score = self._threshold_and_pick_part(
            blur, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            self.mask_morph_kernel_size, self.mask_morph_close_iters,
            self.mask_morph_open_iters, self.min_component_area_px, self.border_margin_fraction,
        )

        if inv_score >= fwd_score and np.any(inv_mask):
            return inv_mask
        if np.any(fwd_mask):
            return fwd_mask

        # Both polarities found only border-touching components — fall back to
        # largest-component without border rejection (better than nothing).
        logger.warning("isolate_part: no non-border component found, using largest component")
        _, raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.mask_morph_kernel_size, self.mask_morph_kernel_size))
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=self.mask_morph_close_iters)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k, iterations=self.mask_morph_open_iters)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            return np.where(labels == largest, np.uint8(255), np.uint8(0))
        return raw

    def _threshold_and_pick_part(self, blur: np.ndarray, thresh_flags: int, kernel_size: int,
                               close_iters: int, open_iters: int, min_area: int, 
                               border_margin: float) -> Tuple[np.ndarray, float]:
        """
        Apply Otsu threshold with given polarity, morphologically clean the mask,
        then pick the single connected component that best represents the part.
        
        This is an exact copy of the _threshold_and_pick_part function from quick_test.py
        """
        _, raw = cv2.threshold(blur, 0, 255, thresh_flags)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=close_iters)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k, iterations=open_iters)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        h, w = raw.shape
        border_margin_px = max(self.border_margin_min_px, int(min(h, w) * border_margin))

        best_mask = np.zeros_like(raw)
        best_score = -1.0

        for lbl in range(1, num_labels):
            s = stats[lbl]
            area = s[cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x0, y0 = s[cv2.CC_STAT_LEFT], s[cv2.CC_STAT_TOP]
            x1 = x0 + s[cv2.CC_STAT_WIDTH]
            y1 = y0 + s[cv2.CC_STAT_HEIGHT]
            score = self._score_component(x0, y0, x1, y1, area, w, h, border_margin_px)
            if score > best_score:
                best_score = score
                best_mask = np.where(labels == lbl, np.uint8(255), np.uint8(0))

        return best_mask, best_score

    def _score_component(self, x0: int, y0: int, x1: int, y1: int, area: int,
                       img_w: int, img_h: int, border_margin: int) -> float:
        """
        Score a connected component as a part candidate.
        
        This is an exact copy of the _score_component function from quick_test.py
        """
        if x0 <= border_margin or y0 <= border_margin or \
           x1 >= img_w - border_margin or y1 >= img_h - border_margin:
            return -1.0  # touches border → reject

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        dist = np.hypot(cx - img_w / 2, cy - img_h / 2)
        max_dist = np.hypot(img_w / 2, img_h / 2)
        centrality = 1.0 - dist / max_dist
        return float(area) * (self.component_area_weight + self.component_centrality_weight * centrality)

    def _extract_internal_edges(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract edges from inside the part mask.
        
        This is an exact copy of the extract_internal_edges function from quick_test.py
        """
        masked = cv2.bitwise_and(img, img, mask=mask)

        # CLAHE: equalise local contrast so all bolt holes are equally visible
        # regardless of where the lighting hotspot falls on the part surface.
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
        enhanced = clahe.apply(masked)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.gradient_kernel_size, self.gradient_kernel_size))
        grad = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, k)

        mask_area = int(np.count_nonzero(mask))

        thresh = self.gradient_threshold
        for _ in range(self.gradient_max_retries):
            _, edges = cv2.threshold(grad, thresh, 255, cv2.THRESH_BINARY)
            edges = cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (self.edge_close_kernel_size, self.edge_close_kernel_size))
            )
            edges = cv2.bitwise_and(edges, edges, mask=mask)
            density = float(np.count_nonzero(edges)) / float(mask_area) if mask_area > 0 else 0.0
            if density <= self.internal_edge_max_density:
                break
            thresh = int(thresh * self.gradient_step_multiplier)

        return edges

    def _extract_outer_boundary(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract the outer boundary of the part mask using Canny.
        
        This is an exact copy of the extract_outer_boundary function from quick_test.py
        """
        return cv2.Canny(mask, self.canny_outer_low, self.canny_outer_high)