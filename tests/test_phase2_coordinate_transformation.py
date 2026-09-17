"""
Tests for Phase 2: Coordinate Transformation Pipeline

Tests the coordinate transformation pipeline that converts actual features
from image pixels to DXF coordinates.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from feature_extraction.expected.feature_types import Point2D, FeatureType
from feature_inspection.models.actual_feature import ActualFeature, ActualFeatureSet, ActualDetectionStatistics, DetectionMethod
from feature_inspection.models.coordinate_transform import CoordinateTransform
from feature_inspection.pipeline.coordinate_pipeline import CoordinateTransformationPipeline, TransformationSource
from cad_image_alignment import AlignmentResult


# Helper class for tests
class AlignmentTransformationSource(TransformationSource):
    """Test helper to convert AlignmentResult to TransformationSource."""
    
    def __init__(self, alignment_result: AlignmentResult):
        self.alignment_result = alignment_result
        super().__init__(alignment_result.transform_matrix, "test_alignment")
    
    def _validate(self) -> bool:
        """Validate alignment result for transformation."""
        if self.alignment_result is None:
            return False
        
        # Check if transformation matrix exists and is valid
        if self.transform_matrix is None:
            return False
        
        try:
            # Test if matrix is invertible (determinant != 0)
            import numpy as np
            det = np.linalg.det(self.transform_matrix[:2, :2])  # Check 2x2 portion
            return abs(det) > 1e-10  # Not singular
        except Exception:
            return False


class TestCoordinateTransformationPipeline:
    """Test the coordinate transformation pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = CoordinateTransformationPipeline()
        
        # Create test actual feature set in image pixels
        self.test_features = [
            ActualFeature(
                feature_id="test_circle",
                feature_type=FeatureType.CIRCLE,
                confidence=0.9,
                center=Point2D(100.0, 200.0),
                radius=25.0,
                detection_method=DetectionMethod.HOUGH_CIRCLES
            ),
            ActualFeature(
                feature_id="test_hole",
                feature_type=FeatureType.THROUGH_HOLE,
                confidence=0.8,
                center=Point2D(300.0, 400.0),
                radius=15.0,
                detection_method=DetectionMethod.CONTOUR_ANALYSIS
            )
        ]
        
        self.test_feature_set = ActualFeatureSet(
            source_image_path=Path("test_image.png"),
            features=self.test_features,
            detection_timestamp="2024-01-01 12:00:00",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=10, contours_after_filtering=5,
                circles_detected=1, through_holes_detected=1, rectangular_holes_detected=0,
                total_features_detected=2, average_confidence=0.85,
                detection_time_seconds=1.0, preprocessing_time_seconds=0.5
            ),
            configuration_snapshot={},
            image_dimensions=(800, 600),
            preprocessing_applied=[]
        )
    
    def test_transformation_with_identity_matrix(self):
        """Test coordinate transformation with identity matrix."""
        # Create identity alignment result
        identity_matrix = np.eye(3)
        alignment_result = AlignmentResult(
            aligned_image=np.ones((400, 400), dtype=np.uint8) * 128,
            transform_matrix=identity_matrix,
            alignment_score=1.0,
            coverage=1.0,
            edge_score=1.0,
            hole_diff=0,
            combined_score=1.0,
            strategy="identity_test",
            high_confidence=True,
            identified=True
        )
        
        # No preprocessing resize
        preprocessing_metadata = {}
        
        # Create transformation source from alignment result
        transformation_source = AlignmentTransformationSource(alignment_result)
        
        transformed_set, metadata = self.pipeline.transform_actual_features(
            self.test_feature_set, transformation_source, preprocessing_metadata
        )
        
        # Check transformation succeeded
        assert metadata['transformation_successful'] is True
        assert transformed_set.coordinate_system == "dxf_mm"
        assert len(transformed_set.features) == 2
        
        # With identity matrix, coordinates should be nearly the same
        for original, transformed in zip(self.test_features, transformed_set.features):
            assert abs(transformed.center.x - original.center.x) < 1.0
            assert abs(transformed.center.y - original.center.y) < 1.0
            assert abs(transformed.radius - original.radius) < 1.0 if transformed.radius else True
            
            # Check feature identity preserved
            assert transformed.feature_id == original.feature_id
            assert transformed.feature_type == original.feature_type
            assert transformed.confidence == original.confidence
            assert transformed.detection_method == original.detection_method
    
    def test_transformation_with_scale_matrix(self):
        """Test coordinate transformation with scaling matrix."""
        # Create 2x scale matrix
        scale_matrix = np.array([
            [0.5, 0.0, 0.0],  # Scale down by 2x (pixels to mm)
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        alignment_result = AlignmentResult(
            aligned_image=np.ones((400, 400), dtype=np.uint8) * 128,
            transform_matrix=scale_matrix,
            alignment_score=0.9,
            coverage=0.9,
            edge_score=0.9,
            hole_diff=0,
            combined_score=0.9,
            strategy="scale_test",
            high_confidence=True,
            identified=True
        )
        
        transformation_source = AlignmentTransformationSource(alignment_result)
        
        transformed_set, metadata = self.pipeline.transform_actual_features(
            self.test_feature_set, transformation_source, {}
        )
        
        assert metadata['transformation_successful'] is True
        assert transformed_set.coordinate_system == "dxf_mm"
        
        # Check scaling applied correctly
        original_feature = self.test_features[0]  # Circle at (100, 200), radius 25
        transformed_feature = transformed_set.features[0]
        
        # The transformation is image_to_dxf, so we need to check the inverse transformation
        # With 0.5 scale matrix: image_to_dxf scales by 2x (inverse of 0.5)
        # So (100, 200) -> (200, 400), radius 25 -> 50
        assert abs(transformed_feature.center.x - 200.0) < 1.0
        assert abs(transformed_feature.center.y - 400.0) < 1.0
        assert abs(transformed_feature.radius - 50.0) < 1.0
    
    def test_transformation_with_preprocessing_resize(self):
        """Test transformation accounting for preprocessing resize."""
        # Create preprocessing metadata indicating 0.5 scale resize
        preprocessing_metadata = {
            "resize_info": {
                "resized": True,
                "scale_factor": 0.5,  # Image was scaled down by 2x
                "original_size": (1600, 1200),
                "new_size": (800, 600)
            }
        }
        
        identity_matrix = np.eye(3)
        alignment_result = AlignmentResult(
            aligned_image=np.ones((800, 600), dtype=np.uint8) * 128,
            transform_matrix=identity_matrix,
            alignment_score=1.0,
            coverage=1.0,
            edge_score=1.0,
            hole_diff=0,
            combined_score=1.0,
            strategy="resize_test",
            high_confidence=True,
            identified=True
        )
        
        transformation_source = AlignmentTransformationSource(alignment_result)
        
        transformed_set, metadata = self.pipeline.transform_actual_features(
            self.test_feature_set, transformation_source, preprocessing_metadata
        )
        
        assert metadata['transformation_successful'] is True
        assert metadata['preprocessing_resize'] is True
        assert metadata['resize_scale_factor'] == 2.0  # Inverse of 0.5
        
        # Features should be scaled up to account for preprocessing resize
        original_feature = self.test_features[0]  # Circle at (100, 200), radius 25
        transformed_feature = transformed_set.features[0]
        
        # With 2x scale correction: (100, 200) -> (200, 400), radius 25 -> 50
        assert abs(transformed_feature.center.x - 200.0) < 1.0
        assert abs(transformed_feature.center.y - 400.0) < 1.0
        assert abs(transformed_feature.radius - 50.0) < 1.0
    
    def test_transformation_unavailable_no_alignment(self):
        """Test behavior when no alignment result is provided."""
        transformed_set, metadata = self.pipeline.transform_actual_features(
            self.test_feature_set, None, {}
        )
        
        assert metadata['transformation_successful'] is False
        assert metadata['transformation_source_available'] is False
        assert "No transformation source provided" in metadata['transformation_errors']
        assert transformed_set.coordinate_system == "image_pixels_transform_unavailable"
    
    def test_transformation_unavailable_invalid_matrix(self):
        """Test behavior with invalid transformation matrix."""
        # Create singular matrix (not invertible)
        singular_matrix = np.zeros((3, 3))
        alignment_result = AlignmentResult(
            aligned_image=np.ones((400, 400), dtype=np.uint8) * 128,
            transform_matrix=singular_matrix,
            alignment_score=0.1,
            coverage=0.1,
            edge_score=0.1,
            hole_diff=10,
            combined_score=0.1,
            strategy="invalid_test",
            high_confidence=False,
            identified=False
        )
        
        transformation_source = AlignmentTransformationSource(alignment_result)
        
        transformed_set, metadata = self.pipeline.transform_actual_features(
            self.test_feature_set, transformation_source, {}
        )
        
        assert metadata['transformation_successful'] is False
        assert len(metadata['transformation_errors']) > 0
    
    def test_coordinate_consistency_validation_compatible(self):
        """Test coordinate system compatibility validation - compatible case."""
        # Mock expected features in DXF coordinates
        expected_features = type('ExpectedFeatureSet', (), {
            'coordinate_system': 'dxf_mm'
        })()
        
        # Mock actual features in DXF coordinates
        actual_features = type('ActualFeatureSet', (), {
            'coordinate_system': 'dxf_mm'
        })()
        
        validation = self.pipeline.validate_coordinate_consistency(expected_features, actual_features)
        
        assert validation['coordinate_systems_compatible'] is True
        assert len(validation['validation_errors']) == 0
        assert "compatible" in validation['recommendation'].lower()
    
    def test_coordinate_consistency_validation_incompatible(self):
        """Test coordinate system compatibility validation - incompatible case."""
        expected_features = type('ExpectedFeatureSet', (), {
            'coordinate_system': 'dxf_mm'
        })()
        
        actual_features = type('ActualFeatureSet', (), {
            'coordinate_system': 'image_pixels'
        })()
        
        validation = self.pipeline.validate_coordinate_consistency(expected_features, actual_features)
        
        assert validation['coordinate_systems_compatible'] is False
        assert len(validation['validation_errors']) > 0
        assert "transform" in validation['recommendation'].lower()
    
    def test_coordinate_consistency_validation_transform_failed(self):
        """Test validation when coordinate transformation failed."""
        expected_features = type('ExpectedFeatureSet', (), {
            'coordinate_system': 'dxf_mm'
        })()
        
        actual_features = type('ActualFeatureSet', (), {
            'coordinate_system': 'image_pixels_transform_failed'
        })()
        
        validation = self.pipeline.validate_coordinate_consistency(expected_features, actual_features)
        
        assert validation['coordinate_systems_compatible'] is False
        assert "transformation failed" in validation['validation_errors'][0]
        assert "check transformation matrix" in validation['recommendation'].lower()
    
    def test_rectangular_feature_transformation(self):
        """Test transformation of rectangular features with width/height."""
        rect_feature = ActualFeature(
            feature_id="test_rect",
            feature_type=FeatureType.RECTANGULAR_HOLE,
            confidence=0.85,
            center=Point2D(100.0, 100.0),
            width=40.0,
            height=60.0,
            detection_method=DetectionMethod.CONTOUR_ANALYSIS
        )
        
        rect_feature_set = ActualFeatureSet(
            source_image_path=Path("test.png"),
            features=[rect_feature],
            detection_timestamp="2024-01-01",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=1, contours_after_filtering=1,
                circles_detected=0, through_holes_detected=0, rectangular_holes_detected=1,
                total_features_detected=1, average_confidence=0.85,
                detection_time_seconds=1.0, preprocessing_time_seconds=0.5
            ),
            configuration_snapshot={},
            image_dimensions=(400, 400),
            preprocessing_applied=[]
        )
        
        # Use 0.5 scale matrix (2 pixels per mm)
        scale_matrix = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        alignment_result = AlignmentResult(
            aligned_image=np.ones((400, 400), dtype=np.uint8) * 128,
            transform_matrix=scale_matrix,
            alignment_score=0.9,
            coverage=0.9,
            edge_score=0.9,
            hole_diff=0,
            combined_score=0.9,
            strategy="rect_test",
            high_confidence=True,
            identified=True
        )
        
        transformation_source = AlignmentTransformationSource(alignment_result)
        
        transformed_set, metadata = self.pipeline.transform_actual_features(
            rect_feature_set, transformation_source, {}
        )
        
        assert metadata['transformation_successful'] is True
        
        transformed_rect = transformed_set.features[0]
        assert transformed_rect.feature_type == FeatureType.RECTANGULAR_HOLE
        assert transformed_rect.width is not None
        assert transformed_rect.height is not None
        
        # With 0.5 scale matrix: image_to_dxf scales by 2x (inverse of 0.5)
        # And with get_scale_factor, 40 pixels -> 80mm, 60 pixels -> 120mm
        assert abs(transformed_rect.width - 80.0) < 1.0
        assert abs(transformed_rect.height - 120.0) < 1.0
    
    def test_transformation_preserves_evidence_and_metadata(self):
        """Test that transformation preserves feature evidence and metadata."""
        feature_with_evidence = ActualFeature(
            feature_id="evidence_test",
            feature_type=FeatureType.CIRCLE,
            confidence=0.9,
            center=Point2D(100.0, 100.0),
            radius=20.0,
            detection_method=DetectionMethod.HOUGH_CIRCLES,
            detection_evidence={"hough_params": [50, 30], "validation_score": 0.95},
            quality_metrics={"roundness": 0.92, "edge_strength": 0.88}
        )
        
        evidence_feature_set = ActualFeatureSet(
            source_image_path=Path("test.png"),
            features=[feature_with_evidence],
            detection_timestamp="2024-01-01",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=1, contours_after_filtering=1,
                circles_detected=1, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=1, average_confidence=0.9,
                detection_time_seconds=1.0, preprocessing_time_seconds=0.5
            ),
            configuration_snapshot={},
            image_dimensions=(400, 400),
            preprocessing_applied=[]
        )
        
        identity_matrix = np.eye(3)
        alignment_result = AlignmentResult(
            aligned_image=np.ones((400, 400), dtype=np.uint8) * 128,
            transform_matrix=identity_matrix,
            alignment_score=1.0,
            coverage=1.0,
            edge_score=1.0,
            hole_diff=0,
            combined_score=1.0,
            strategy="evidence_test",
            high_confidence=True,
            identified=True
        )
        
        transformation_source = AlignmentTransformationSource(alignment_result)
        
        transformed_set, metadata = self.pipeline.transform_actual_features(
            evidence_feature_set, transformation_source, {}
        )
        
        transformed_feature = transformed_set.features[0]
        
        # Check original evidence preserved
        assert "hough_params" in transformed_feature.detection_evidence
        assert transformed_feature.detection_evidence["hough_params"] == [50, 30]
        assert transformed_feature.quality_metrics["roundness"] == 0.92
        
        # Check transformation evidence added
        assert "coordinate_transformation" in transformed_feature.detection_evidence
        assert transformed_feature.detection_evidence["coordinate_transformation"]["original_coordinate_system"] == "image_pixels"
        
        # Check feature identity preserved
        assert transformed_feature.feature_id == "evidence_test"
        assert transformed_feature.confidence == 0.9
        assert transformed_feature.detection_method == DetectionMethod.HOUGH_CIRCLES


class TestCoordinateTransformRoundTrip:
    """Test round-trip coordinate transformations."""
    
    def test_pixel_to_dxf_round_trip(self):
        """Test round-trip transformation: pixel -> DXF -> pixel."""
        # Create test transformation matrix
        transform_matrix = np.array([
            [0.1, 0.0, 10.0],   # Scale and translation
            [0.0, 0.1, 20.0],
            [0.0, 0.0, 1.0]
        ])
        
        coord_transform = CoordinateTransform(transform_matrix)
        
        # Test point
        original_pixel = Point2D(100.0, 200.0)
        
        # Round trip
        dxf_point = coord_transform.image_to_dxf(original_pixel)
        recovered_pixel = coord_transform.dxf_to_image(dxf_point)
        
        # Should recover original point within tolerance
        assert abs(recovered_pixel.x - original_pixel.x) < 0.001
        assert abs(recovered_pixel.y - original_pixel.y) < 0.001
    
    def test_dxf_to_pixel_round_trip(self):
        """Test round-trip transformation: DXF -> pixel -> DXF."""
        transform_matrix = np.array([
            [0.1, 0.0, 10.0],
            [0.0, 0.1, 20.0],
            [0.0, 0.0, 1.0]
        ])
        
        coord_transform = CoordinateTransform(transform_matrix)
        
        # Test point in DXF coordinates
        original_dxf = Point2D(50.0, 75.0)
        
        # Round trip
        pixel_point = coord_transform.dxf_to_image(original_dxf)
        recovered_dxf = coord_transform.image_to_dxf(pixel_point)
        
        # Should recover original point within tolerance
        assert abs(recovered_dxf.x - original_dxf.x) < 0.001
        assert abs(recovered_dxf.y - original_dxf.y) < 0.001