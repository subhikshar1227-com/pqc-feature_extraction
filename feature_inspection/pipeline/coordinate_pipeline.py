"""
Coordinate Transformation Pipeline Stage

Handles the explicit transformation of actual features from image pixels to DXF coordinates.
"""

import logging
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import copy

from feature_extraction.expected.feature_types import Point2D, FeatureType
from ..models.actual_feature import ActualFeature, ActualFeatureSet
from ..models.coordinate_transform import CoordinateTransform

logger = logging.getLogger(__name__)


class TransformationSource:
    """
    Abstract interface for coordinate transformation sources.
    
    Future implementations can include:
    - Calibrated pixel-to-mm mappings
    - Machine vision calibration
    - Manual coordinate calibration
    - Validated alignment results
    """
    
    def __init__(self, transform_matrix: Optional[Any] = None, source_type: str = "unknown"):
        self.transform_matrix = transform_matrix
        self.source_type = source_type
        self.is_valid = self._validate()
    
    def _validate(self) -> bool:
        """Validate if this transformation source is reliable."""
        return False  # Default: no valid sources available
    
    def get_coordinate_transform(self) -> Optional[CoordinateTransform]:
        """Get CoordinateTransform instance if valid."""
        if not self.is_valid:
            return None
        try:
            return CoordinateTransform(self.transform_matrix)
        except Exception:
            return None


class CoordinateTransformationPipeline:
    """
    Pipeline stage for transforming actual features from image coordinates to DXF coordinates.
    
    Handles:
    1. Image preprocessing coordinate normalization 
    2. Transformation source discovery and validation
    3. Feature geometry transformation (centers, radii, dimensions)
    4. Coordinate system tracking and validation
    """
    
    def __init__(self):
        """Initialize coordinate transformation pipeline."""
        pass
    
    def transform_actual_features(self, 
                                actual_feature_set: ActualFeatureSet,
                                transformation_source: Optional[TransformationSource] = None,
                                preprocessing_metadata: Optional[dict] = None) -> Tuple[ActualFeatureSet, dict]:
        """
        Transform actual features from image pixels to DXF coordinates.
        
        Args:
            actual_feature_set: Features detected in image pixel coordinates
            transformation_source: OPTIONAL transformation source (e.g., calibrated mapping)
            preprocessing_metadata: Metadata from image preprocessing (resize info)
            
        Returns:
            Tuple of (transformed_feature_set, transformation_metadata)
        """
        logger.info(f"Starting coordinate transformation for {len(actual_feature_set.features)} features")
        
        transformation_metadata = {
            "input_coordinate_system": actual_feature_set.coordinate_system,
            "output_coordinate_system": "dxf_mm",
            "transformation_source_available": transformation_source is not None,
            "transformation_source_valid": False,
            "preprocessing_resize": False,
            "transformation_successful": False,
            "transformation_errors": [],
            "feature_transform_results": []
        }
        
        # Step 1: Check if preprocessing resized the image
        resize_scale_factor = 1.0
        if preprocessing_metadata and "resize_info" in preprocessing_metadata:
            resize_info = preprocessing_metadata["resize_info"]
            if resize_info["resized"]:
                resize_scale_factor = 1.0 / resize_info["scale_factor"]  # Inverse to scale up coordinates
                transformation_metadata["preprocessing_resize"] = True
                transformation_metadata["resize_scale_factor"] = resize_scale_factor
                logger.info(f"Image was resized during preprocessing, scale factor: {resize_scale_factor}")
        
        # Step 2: Validate transformation source
        coordinate_transform = None
        if transformation_source is not None:
            transformation_metadata["transformation_source_valid"] = transformation_source.is_valid
            transformation_metadata["transformation_source_type"] = transformation_source.source_type
            
            if transformation_source.is_valid:
                coordinate_transform = transformation_source.get_coordinate_transform()
                if coordinate_transform and coordinate_transform.is_available():
                    logger.info(f"Valid transformation source available: {transformation_source.source_type}")
                else:
                    transformation_metadata["transformation_errors"].append("Transformation source validation failed")
                    logger.warning("Transformation source failed validation")
            else:
                transformation_metadata["transformation_errors"].append(f"Invalid transformation source: {transformation_source.source_type}")
                logger.warning(f"Invalid transformation source: {transformation_source.source_type}")
        else:
            transformation_metadata["transformation_errors"].append("No transformation source provided")
            logger.info("No transformation source provided")
        
        # Step 3: Transform features or return with transformation unavailable
        if coordinate_transform and coordinate_transform.is_available():
            try:
                transformed_features = []
                
                for feature in actual_feature_set.features:
                    transformed_feature, transform_result = self._transform_single_feature(
                        feature, coordinate_transform, resize_scale_factor, transformation_source.source_type
                    )
                    transformed_features.append(transformed_feature)
                    transformation_metadata["feature_transform_results"].append(transform_result)
                
                # Create transformed feature set
                transformed_feature_set = self._create_transformed_feature_set(
                    actual_feature_set, transformed_features, coordinate_transform
                )
                
                transformation_metadata["transformation_successful"] = True
                transformation_metadata["output_coordinate_system"] = "dxf_mm"
                logger.info(f"Successfully transformed {len(transformed_features)} features to DXF coordinates")
                
                return transformed_feature_set, transformation_metadata
                
            except Exception as e:
                transformation_metadata["transformation_errors"].append(f"Feature transformation failed: {e}")
                logger.error(f"Feature transformation failed: {e}")
        
        # Transformation not available or failed
        logger.warning("Coordinate transformation not available - returning features with transform unavailable marker")
        
        # Return original feature set with coordinate system marked as unavailable
        unavailable_feature_set = copy.deepcopy(actual_feature_set)
        unavailable_feature_set.coordinate_system = "image_pixels_transform_unavailable"
        transformation_metadata["output_coordinate_system"] = "transformation_unavailable"
        
        return unavailable_feature_set, transformation_metadata
    
    def _transform_single_feature(self, 
                                feature: ActualFeature, 
                                coordinate_transform: CoordinateTransform,
                                resize_scale_factor: float,
                                transformation_source_type: str) -> Tuple[ActualFeature, dict]:
        """
        Transform a single actual feature from image pixels to DXF coordinates.
        
        Args:
            feature: Feature in image pixel coordinates
            coordinate_transform: Coordinate transformation instance
            resize_scale_factor: Scale factor to account for preprocessing resize
            transformation_source_type: Type of transformation source used
            
        Returns:
            Tuple of (transformed_feature, transformation_result)
        """
        transform_result = {
            "feature_id": feature.feature_id,
            "original_center": (feature.center.x, feature.center.y),
            "transformed_center": None,
            "original_radius": feature.radius,
            "transformed_radius": None,
            "original_dimensions": None,
            "transformed_dimensions": None,
            "transformation_success": False,
            "errors": []
        }
        
        try:
            # Step 1: Apply resize scale correction to get original image coordinates
            original_center = Point2D(
                feature.center.x * resize_scale_factor,
                feature.center.y * resize_scale_factor
            )
            
            # Step 2: Transform center from image pixels to DXF mm
            transformed_center = coordinate_transform.image_to_dxf(original_center)
            transform_result["transformed_center"] = (transformed_center.x, transformed_center.y)
            
            # Step 3: Transform geometry (radius or dimensions)
            transformed_radius = None
            transformed_width = None 
            transformed_height = None
            
            if feature.radius is not None:
                # Transform circular geometry
                original_radius = feature.radius * resize_scale_factor
                transformed_radius = coordinate_transform.transform_radius_image_to_dxf(
                    original_radius, original_center
                )
                transform_result["transformed_radius"] = transformed_radius
            
            if feature.width is not None and feature.height is not None:
                # Transform rectangular geometry
                # Use center point as reference for scaling
                scale_factor = coordinate_transform.get_scale_factor(transformed_center)
                if scale_factor is not None:
                    # Convert to mm using scale factor
                    transformed_width = (feature.width * resize_scale_factor) / scale_factor
                    transformed_height = (feature.height * resize_scale_factor) / scale_factor
                    transform_result["transformed_dimensions"] = (transformed_width, transformed_height)
                    transform_result["original_dimensions"] = (feature.width, feature.height)
                else:
                    transform_result["errors"].append("Could not determine scale factor for rectangular dimensions")
            
            # Create transformed feature
            transformed_feature = ActualFeature(
                feature_id=feature.feature_id,
                feature_type=feature.feature_type,
                confidence=feature.confidence,
                center=transformed_center,
                radius=transformed_radius,
                width=transformed_width,
                height=transformed_height,
                detection_method=feature.detection_method,
                coordinate_system="dxf_mm",
                detection_evidence=feature.detection_evidence.copy(),
                quality_metrics=feature.quality_metrics.copy(),
                contour_area=feature.contour_area
            )
            
            # Add transformation provenance to evidence
            transformed_feature.detection_evidence["coordinate_transformation"] = {
                "original_coordinate_system": "image_pixels",
                "resize_scale_factor": resize_scale_factor,
                "transformation_source_type": transformation_source_type,
                "transformation_method": f"{transformation_source_type}_inverse"
            }
            
            transform_result["transformation_success"] = True
            
            return transformed_feature, transform_result
            
        except Exception as e:
            transform_result["errors"].append(str(e))
            logger.error(f"Failed to transform feature {feature.feature_id}: {e}")
            
            # Return original feature with error indication
            error_feature = copy.deepcopy(feature)
            error_feature.coordinate_system = "image_pixels_transform_failed"
            error_feature.detection_evidence["transformation_error"] = str(e)
            
            return error_feature, transform_result
    
    def _create_transformed_feature_set(self, 
                                      original_set: ActualFeatureSet,
                                      transformed_features: list,
                                      coordinate_transform: CoordinateTransform) -> ActualFeatureSet:
        """
        Create a new ActualFeatureSet with transformed features.
        
        Args:
            original_set: Original feature set in image coordinates
            transformed_features: List of transformed features
            coordinate_transform: Coordinate transformation used
            
        Returns:
            New ActualFeatureSet with transformed features
        """
        # Create new feature set with transformed features
        transformed_set = ActualFeatureSet(
            source_image_path=original_set.source_image_path,
            features=transformed_features,
            detection_timestamp=original_set.detection_timestamp,
            detection_statistics=original_set.detection_statistics,
            configuration_snapshot=original_set.configuration_snapshot,
            image_dimensions=original_set.image_dimensions,
            preprocessing_applied=original_set.preprocessing_applied,
            coordinate_system="dxf_mm",
            transform_matrix=coordinate_transform.transform_matrix.copy() if coordinate_transform.transform_matrix is not None else None
        )
        
        return transformed_set
    
    def validate_coordinate_consistency(self, 
                                      expected_features, 
                                      actual_features) -> dict:
        """
        Validate that expected and actual features are in compatible coordinate systems.
        
        Args:
            expected_features: Expected feature set
            actual_features: Actual feature set
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            "coordinate_systems_compatible": False,
            "expected_coordinate_system": getattr(expected_features, 'coordinate_system', 'dxf_mm'),
            "actual_coordinate_system": getattr(actual_features, 'coordinate_system', 'unknown'),
            "validation_errors": [],
            "recommendation": None
        }
        
        expected_coord_sys = validation_result["expected_coordinate_system"] 
        actual_coord_sys = validation_result["actual_coordinate_system"]
        
        # Check coordinate system compatibility
        if expected_coord_sys == "dxf_mm" and actual_coord_sys == "dxf_mm":
            validation_result["coordinate_systems_compatible"] = True
            validation_result["recommendation"] = "Proceed with matching - coordinate systems compatible"
        
        elif actual_coord_sys == "image_pixels":
            validation_result["validation_errors"].append(
                "Actual features in image pixels, expected features in DXF mm - coordinate transformation required"
            )
            validation_result["recommendation"] = "Transform actual features to DXF coordinates before matching"
        
        elif actual_coord_sys == "image_pixels_transform_unavailable":
            validation_result["validation_errors"].append(
                "Coordinate transformation not available - cannot convert to compatible coordinate system"
            )
            validation_result["recommendation"] = "Cannot perform geometric matching - alignment required"
        
        elif actual_coord_sys == "image_pixels_transform_failed":
            validation_result["validation_errors"].append(
                "Coordinate transformation failed - actual features remain in image pixels"
            )
            validation_result["recommendation"] = "Cannot perform geometric matching - check transformation matrix"
        
        else:
            validation_result["validation_errors"].append(
                f"Unknown coordinate system combination: expected={expected_coord_sys}, actual={actual_coord_sys}"
            )
            validation_result["recommendation"] = "Check coordinate system specifications"
        
        return validation_result