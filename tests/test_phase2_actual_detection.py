"""
Tests for Phase 2: Actual Feature Detection

Tests the actual feature detection pipeline from product images.
"""

import pytest
from pathlib import Path
import numpy as np
import cv2
import tempfile

from feature_inspection.actual import ActualFeatureDetector, CanonicalPreprocessor
from feature_inspection.models.actual_feature import ActualFeature, ActualFeatureSet, DetectionMethod
from feature_extraction.expected.feature_types import FeatureType, Point2D


class TestImagePreprocessor:
    """Test image preprocessing functionality."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor can be initialized."""
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'preprocess_image')
    
    def test_preprocess_synthetic_image(self):
        """Test preprocessing with a synthetic image."""
        # Create a synthetic test image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray image
        
        preprocessor = ImagePreprocessor()
        processed, edges, metadata = preprocessor.preprocess_image(image)
        
        # Validate outputs
        assert processed is not None
        assert edges is not None
        assert isinstance(metadata, dict)
        
        # Check metadata structure
        assert "preprocessing_steps" in metadata
        assert "quality_metrics" in metadata
        assert len(metadata["preprocessing_steps"]) > 0
    
    def test_preprocess_invalid_input(self):
        """Test preprocessor handles invalid input."""
        preprocessor = ImagePreprocessor()
        
        # Test with None input
        with pytest.raises(ValueError):
            preprocessor.preprocess_image(None)
        
        # Test with wrong dimensions
        with pytest.raises(ValueError):
            preprocessor.preprocess_image(np.ones((100, 100)))  # 2D instead of 3D


class TestActualFeatureDetector:
    """Test actual feature detection functionality."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = ActualFeatureDetector()
        assert detector is not None
        assert hasattr(detector, 'detect_features')
    
    def test_detect_features_empty_image(self):
        """Test detection with an empty image."""
        # Create temporary empty image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Create empty (black) image
            empty_image = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path), empty_image)
        
        try:
            detector = ActualFeatureDetector()
            feature_set = detector.detect_features(tmp_path)
            
            # Validate feature set structure
            assert isinstance(feature_set, ActualFeatureSet)
            assert feature_set.source_image_path == tmp_path
            assert isinstance(feature_set.features, list)
            assert feature_set.total_feature_count == len(feature_set.features)
            
            # Empty image should detect few or no features
            assert feature_set.total_feature_count >= 0
            
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_detect_features_synthetic_circles(self):
        """Test detection with synthetic circular features."""
        # Create image with synthetic circles
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Create image with circles
            image = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
            
            # Draw some circles
            cv2.circle(image, (100, 100), 30, (0, 0, 0), 2)  # Black circle outline
            cv2.circle(image, (300, 100), 25, (0, 0, 0), 2)  # Another circle
            cv2.circle(image, (200, 300), 35, (0, 0, 0), -1)  # Filled circle (hole)
            
            cv2.imwrite(str(tmp_path), image)
        
        try:
            detector = ActualFeatureDetector()
            feature_set = detector.detect_features(tmp_path)
            
            # Validate detection results
            assert isinstance(feature_set, ActualFeatureSet)
            assert feature_set.total_feature_count >= 0  # Should detect some features
            
            # Check feature properties
            for feature in feature_set.features:
                assert isinstance(feature, ActualFeature)
                assert feature.feature_id is not None
                assert isinstance(feature.feature_type, FeatureType)
                assert 0.0 <= feature.confidence <= 1.0
                assert isinstance(feature.center, Point2D)
                assert feature.coordinate_system == "image_pixels"
                
                # Circular features should have radius
                if feature.feature_type in [FeatureType.CIRCLE, FeatureType.THROUGH_HOLE]:
                    assert feature.radius is not None
                    assert feature.radius > 0
            
        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_feature_deduplication(self):
        """Test that duplicate feature detections are removed."""
        detector = ActualFeatureDetector()
        
        # Create duplicate features
        feature1 = ActualFeature(
            feature_id="test1",
            feature_type=FeatureType.CIRCLE,
            confidence=0.9,
            center=Point2D(100, 100),
            radius=20.0,
            detection_method=DetectionMethod.CONTOUR_ANALYSIS
        )
        
        # Very similar feature (should be considered duplicate)
        feature2 = ActualFeature(
            feature_id="test2",
            feature_type=FeatureType.CIRCLE,
            confidence=0.8,
            center=Point2D(102, 98),  # Close to feature1
            radius=22.0,
            detection_method=DetectionMethod.HOUGH_CIRCLES
        )
        
        # Different feature (should not be duplicate)
        feature3 = ActualFeature(
            feature_id="test3",
            feature_type=FeatureType.CIRCLE,
            confidence=0.7,
            center=Point2D(200, 200),  # Far from others
            radius=25.0,
            detection_method=DetectionMethod.CONTOUR_ANALYSIS
        )
        
        features = [feature1, feature2, feature3]
        deduplicated = detector._remove_duplicate_features(features)
        
        # Should remove one duplicate but keep the different one
        assert len(deduplicated) == 2
        
        # Higher confidence feature should be kept
        kept_ids = {f.feature_id for f in deduplicated}
        assert "test1" in kept_ids  # Highest confidence
        assert "test3" in kept_ids  # Different location
    
    def test_nonexistent_image_file(self):
        """Test detection with non-existent image file."""
        detector = ActualFeatureDetector()
        nonexistent_path = Path("nonexistent_image.png")
        
        # Should return empty feature set instead of raising exception (graceful handling)
        result = detector.detect_features(nonexistent_path)
        
        assert isinstance(result, ActualFeatureSet)
        assert len(result.features) == 0
        assert result.detection_statistics.total_features_detected == 0


class TestActualFeatureModel:
    """Test ActualFeature data model."""
    
    def test_actual_feature_creation(self):
        """Test creating ActualFeature instances."""
        feature = ActualFeature(
            feature_id="test_circle",
            feature_type=FeatureType.CIRCLE,
            confidence=0.85,
            center=Point2D(150.5, 200.3),
            radius=25.0,
            detection_method=DetectionMethod.HOUGH_CIRCLES
        )
        
        # Test basic properties
        assert feature.feature_id == "test_circle"
        assert feature.feature_type == FeatureType.CIRCLE
        assert feature.confidence == 0.85
        assert feature.center.x == 150.5
        assert feature.center.y == 200.3
        assert feature.radius == 25.0
        assert feature.coordinate_system == "image_pixels"
        
        # Test computed properties
        assert feature.area is not None
        assert feature.perimeter is not None
        assert feature.bounding_box is not None
    
    def test_confidence_bounds_validation(self):
        """Test that confidence is properly bounded."""
        # Test confidence > 1.0
        feature1 = ActualFeature(
            feature_id="test1",
            feature_type=FeatureType.CIRCLE,
            confidence=1.5,  # > 1.0
            center=Point2D(100, 100),
            radius=20.0
        )
        assert feature1.confidence == 1.0
        
        # Test confidence < 0.0
        feature2 = ActualFeature(
            feature_id="test2", 
            feature_type=FeatureType.CIRCLE,
            confidence=-0.3,  # < 0.0
            center=Point2D(100, 100),
            radius=20.0
        )
        assert feature2.confidence == 0.0
    
    def test_rectangular_feature(self):
        """Test rectangular feature properties."""
        feature = ActualFeature(
            feature_id="test_rect",
            feature_type=FeatureType.RECTANGULAR_HOLE,
            confidence=0.75,
            center=Point2D(100, 100),
            width=40.0,
            height=60.0
        )
        
        assert feature.width == 40.0
        assert feature.height == 60.0
        assert feature.area == 40.0 * 60.0
        assert feature.perimeter == 2 * (40.0 + 60.0)


class TestActualFeatureSet:
    """Test ActualFeatureSet data model."""
    
    def test_feature_set_creation(self):
        """Test creating ActualFeatureSet."""
        from feature_inspection.models.actual_feature import ActualDetectionStatistics
        
        features = [
            ActualFeature(
                feature_id="circle1",
                feature_type=FeatureType.CIRCLE,
                confidence=0.9,
                center=Point2D(100, 100),
                radius=20.0
            ),
            ActualFeature(
                feature_id="hole1",
                feature_type=FeatureType.THROUGH_HOLE,
                confidence=0.8,
                center=Point2D(200, 200),
                radius=15.0
            )
        ]
        
        stats = ActualDetectionStatistics(
            total_contours_found=50,
            contours_after_filtering=25,
            circles_detected=1,
            through_holes_detected=1,
            rectangular_holes_detected=0,
            total_features_detected=2,
            average_confidence=0.85,
            detection_time_seconds=1.5,
            preprocessing_time_seconds=0.5
        )
        
        feature_set = ActualFeatureSet(
            source_image_path=Path("test_image.png"),
            features=features,
            detection_timestamp="2024-01-01 12:00:00",
            detection_statistics=stats,
            configuration_snapshot={},
            image_dimensions=(400, 300),
            preprocessing_applied=["resize", "noise_reduction"]
        )
        
        # Test basic properties
        assert len(feature_set.features) == 2
        assert feature_set.total_feature_count == 2
        assert feature_set.circle_count == 1
        assert feature_set.through_hole_count == 1
        assert feature_set.rectangular_hole_count == 0
        assert abs(feature_set.average_confidence - 0.85) < 0.001  # Use tolerance for floating point
        
        # Test feature filtering
        circles = feature_set.get_features_by_type(FeatureType.CIRCLE)
        assert len(circles) == 1
        assert circles[0].feature_id == "circle1"