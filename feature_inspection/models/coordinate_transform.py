"""
Coordinate Transformation Utilities

Utilities for converting between different coordinate systems used in Phase 2.
"""

import numpy as np
from typing import Tuple, Optional, Union
import logging

from feature_extraction.expected.feature_types import Point2D

logger = logging.getLogger(__name__)


class CoordinateTransform:
    """
    Handles coordinate transformations between DXF and image coordinate systems.
    
    Uses the transformation matrix from Phase 1 alignment to convert between
    DXF millimeter coordinates and image pixel coordinates.
    """
    
    def __init__(self, transform_matrix: Optional[np.ndarray] = None):
        """
        Initialize coordinate transformer.
        
        Args:
            transform_matrix: 3x3 homogeneous transformation matrix from alignment
                            If None, transformations will not be available
        """
        self.transform_matrix = transform_matrix
        self.inverse_transform_matrix = None
        
        if transform_matrix is not None:
            self._validate_transform_matrix(transform_matrix)
            self._compute_inverse_transform()
    
    def _validate_transform_matrix(self, matrix: np.ndarray) -> None:
        """Validate that the transformation matrix is valid."""
        if matrix.shape != (3, 3):
            raise ValueError(f"Transform matrix must be 3x3, got shape {matrix.shape}")
        
        # Check if matrix is approximately singular
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:
            logger.warning(f"Transform matrix appears singular (det={det})")
        
        # Log matrix properties for debugging
        logger.debug(f"Transform matrix determinant: {det}")
        logger.debug(f"Transform matrix:\n{matrix}")
    
    def _compute_inverse_transform(self) -> None:
        """Compute the inverse transformation matrix."""
        try:
            self.inverse_transform_matrix = np.linalg.inv(self.transform_matrix)
            logger.debug("Successfully computed inverse transform matrix")
        except np.linalg.LinAlgError as e:
            logger.error(f"Failed to compute inverse transform: {e}")
            self.inverse_transform_matrix = None
    
    def is_available(self) -> bool:
        """Check if coordinate transformations are available."""
        return (self.transform_matrix is not None and 
                self.inverse_transform_matrix is not None)
    
    def dxf_to_image(self, point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Transform a point from DXF coordinates to image pixel coordinates.
        
        Args:
            point: Point in DXF coordinate system (mm)
            
        Returns:
            Point in image pixel coordinate system
            
        Raises:
            ValueError: If transformation is not available
        """
        if not self.is_available():
            raise ValueError("Coordinate transformation not available - no valid transform matrix")
        
        # Convert input to homogeneous coordinates
        if isinstance(point, Point2D):
            dxf_point = np.array([point.x, point.y, 1.0])
        else:
            dxf_point = np.array([point[0], point[1], 1.0])
        
        # Apply transformation
        image_homogeneous = self.transform_matrix @ dxf_point
        
        # Convert back to 2D coordinates
        if abs(image_homogeneous[2]) < 1e-10:
            logger.warning("Transformation resulted in point at infinity")
            return Point2D(0.0, 0.0)
        
        image_x = image_homogeneous[0] / image_homogeneous[2]
        image_y = image_homogeneous[1] / image_homogeneous[2]
        
        return Point2D(image_x, image_y)
    
    def image_to_dxf(self, point: Union[Point2D, Tuple[float, float]]) -> Point2D:
        """
        Transform a point from image pixel coordinates to DXF coordinates.
        
        Args:
            point: Point in image pixel coordinate system
            
        Returns:
            Point in DXF coordinate system (mm)
            
        Raises:
            ValueError: If transformation is not available
        """
        if not self.is_available():
            raise ValueError("Coordinate transformation not available - no valid transform matrix")
        
        # Convert input to homogeneous coordinates
        if isinstance(point, Point2D):
            image_point = np.array([point.x, point.y, 1.0])
        else:
            image_point = np.array([point[0], point[1], 1.0])
        
        # Apply inverse transformation
        dxf_homogeneous = self.inverse_transform_matrix @ image_point
        
        # Convert back to 2D coordinates
        if abs(dxf_homogeneous[2]) < 1e-10:
            logger.warning("Inverse transformation resulted in point at infinity")
            return Point2D(0.0, 0.0)
        
        dxf_x = dxf_homogeneous[0] / dxf_homogeneous[2]
        dxf_y = dxf_homogeneous[1] / dxf_homogeneous[2]
        
        return Point2D(dxf_x, dxf_y)
    
    def transform_radius_dxf_to_image(self, radius_mm: float, center_dxf: Point2D) -> float:
        """
        Transform a radius from DXF scale to image scale.
        
        Since radius is a scalar, we need to account for scaling effects
        by transforming a circle and measuring the resulting radius.
        
        Args:
            radius_mm: Radius in DXF coordinates (mm)
            center_dxf: Center point in DXF coordinates (for scale reference)
            
        Returns:
            Radius in image pixel coordinates
        """
        if not self.is_available():
            raise ValueError("Coordinate transformation not available")
        
        # Transform center point
        center_image = self.dxf_to_image(center_dxf)
        
        # Transform a point on the circle boundary
        boundary_dxf = Point2D(center_dxf.x + radius_mm, center_dxf.y)
        boundary_image = self.dxf_to_image(boundary_dxf)
        
        # Calculate radius in image coordinates
        radius_pixels = center_image.distance_to(boundary_image)
        
        return radius_pixels
    
    def transform_radius_image_to_dxf(self, radius_pixels: float, center_image: Point2D) -> float:
        """
        Transform a radius from image scale to DXF scale.
        
        Args:
            radius_pixels: Radius in image pixel coordinates
            center_image: Center point in image coordinates (for scale reference)
            
        Returns:
            Radius in DXF coordinates (mm)
        """
        if not self.is_available():
            raise ValueError("Coordinate transformation not available")
        
        # Transform center point to DXF
        center_dxf = self.image_to_dxf(center_image)
        
        # Transform a point on the circle boundary
        boundary_image = Point2D(center_image.x + radius_pixels, center_image.y)
        boundary_dxf = self.image_to_dxf(boundary_image)
        
        # Calculate radius in DXF coordinates
        radius_mm = center_dxf.distance_to(boundary_dxf)
        
        return radius_mm
    
    def get_scale_factor(self, reference_point: Point2D) -> Optional[float]:
        """
        Get the approximate scale factor (pixels per mm) at a reference point.
        
        Args:
            reference_point: Reference point in DXF coordinates
            
        Returns:
            Scale factor (pixels/mm) or None if transformation not available
        """
        if not self.is_available():
            return None
        
        # Transform a 1mm unit vector and measure its length in image space
        point1_dxf = reference_point
        point2_dxf = Point2D(reference_point.x + 1.0, reference_point.y)  # 1mm offset
        
        point1_image = self.dxf_to_image(point1_dxf)
        point2_image = self.dxf_to_image(point2_dxf)
        
        # Distance in pixels for 1mm in DXF
        pixels_per_mm = point1_image.distance_to(point2_image)
        
        return pixels_per_mm
    
    def validate_transformation(self, test_points: Optional[list] = None) -> dict:
        """
        Validate transformation accuracy by round-trip testing.
        
        Args:
            test_points: List of test points in DXF coordinates. If None, uses default points.
            
        Returns:
            Dictionary with validation results
        """
        if not self.is_available():
            return {"error": "Transformation not available"}
        
        if test_points is None:
            # Default test points
            test_points = [
                Point2D(0.0, 0.0),
                Point2D(10.0, 10.0),
                Point2D(-5.0, 15.0),
                Point2D(20.0, -10.0)
            ]
        
        results = {
            "test_count": len(test_points),
            "max_error": 0.0,
            "avg_error": 0.0,
            "errors": []
        }
        
        total_error = 0.0
        
        for i, original_dxf in enumerate(test_points):
            # Round trip: DXF -> Image -> DXF
            image_point = self.dxf_to_image(original_dxf)
            recovered_dxf = self.image_to_dxf(image_point)
            
            # Calculate error
            error = original_dxf.distance_to(recovered_dxf)
            results["errors"].append(error)
            
            total_error += error
            results["max_error"] = max(results["max_error"], error)
        
        results["avg_error"] = total_error / len(test_points)
        results["validation_passed"] = results["max_error"] < 0.1  # 0.1mm tolerance
        
        logger.debug(f"Transformation validation: max_error={results['max_error']:.4f}mm, "
                    f"avg_error={results['avg_error']:.4f}mm")
        
        return results