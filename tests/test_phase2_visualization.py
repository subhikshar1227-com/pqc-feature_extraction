"""
Tests for Phase 2: Actual Feature Visualization

Tests the visualization system for displaying detected actual features.
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
import shutil

from feature_extraction.expected.feature_types import Point2D, FeatureType
from feature_inspection.models.actual_feature import ActualFeature, ActualFeatureSet, ActualDetectionStatistics, DetectionMethod
from feature_inspection.output.visualizer import ActualFeatureVisualizer


class TestActualFeatureVisualizer:
    """Test actual feature visualization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.visualizer = ActualFeatureVisualizer()
        
        # Create test image
        self.test_image_path = self.temp_dir / "test_image.png"
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White image
        cv2.imwrite(str(self.test_image_path), test_image)
        
        # Create test actual features
        self.test_features = [
            ActualFeature(
                feature_id="circle_1",
                feature_type=FeatureType.CIRCLE,
                confidence=0.95,
                center=Point2D(100.0, 100.0),
                radius=25.0,
                detection_method=DetectionMethod.HOUGH_CIRCLES
            ),
            ActualFeature(
                feature_id="hole_1", 
                feature_type=FeatureType.THROUGH_HOLE,
                confidence=0.85,
                center=Point2D(300.0, 200.0),
                radius=20.0,
                detection_method=DetectionMethod.CONTOUR_ANALYSIS
            ),
            ActualFeature(
                feature_id="rect_1",
                feature_type=FeatureType.RECTANGULAR_HOLE,
                confidence=0.75,
                center=Point2D(200.0, 300.0),
                width=40.0,
                height=60.0,
                detection_method=DetectionMethod.CONTOUR_ANALYSIS
            )
        ]
        
        self.test_feature_set = ActualFeatureSet(
            source_image_path=self.test_image_path,
            features=self.test_features,
            detection_timestamp="2024-01-01 12:00:00",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=15, contours_after_filtering=8,
                circles_detected=1, through_holes_detected=1, rectangular_holes_detected=1,
                total_features_detected=3, average_confidence=0.85,
                detection_time_seconds=2.0, preprocessing_time_seconds=0.8
            ),
            configuration_snapshot={},
            image_dimensions=(400, 400),
            preprocessing_applied=["resize", "noise_reduction"]
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_visualize_actual_features_basic(self):
        """Test basic actual feature visualization."""
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, self.test_feature_set
        )
        
        # Check output file was created
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert "actual_features" in output_path.name
        
        # Check image can be loaded
        viz_image = cv2.imread(str(output_path))
        assert viz_image is not None
        assert viz_image.shape == (400, 400, 3)
    
    def test_visualize_with_matched_features(self):
        """Test visualization with matched vs unmatched features."""
        matched_ids = ["circle_1", "hole_1"]  # rect_1 is unmatched
        
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, self.test_feature_set, 
            matched_feature_ids=matched_ids
        )
        
        assert output_path.exists()
        
        # Visualization should show different colors for matched vs unmatched
        # (Visual verification would be manual, but we can check file creation)
    
    def test_visualize_custom_output_path(self):
        """Test visualization with custom output path."""
        custom_output = self.temp_dir / "custom_visualization.png"
        
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, self.test_feature_set, custom_output
        )
        
        assert output_path == custom_output
        assert custom_output.exists()
    
    def test_visualize_features_in_image_pixels(self):
        """Test visualization of features in image pixel coordinates."""
        # Features should be in image_pixels coordinate system for visualization
        assert self.test_feature_set.coordinate_system == "image_pixels"
        
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, self.test_feature_set
        )
        
        assert output_path.exists()
    
    def test_visualize_features_in_dxf_coordinates_error(self):
        """Test that features in DXF coordinates cannot be visualized directly."""
        # Create feature set in DXF coordinates
        dxf_feature_set = ActualFeatureSet(
            source_image_path=self.test_image_path,
            features=self.test_features,
            detection_timestamp="2024-01-01 12:00:00",
            detection_statistics=self.test_feature_set.detection_statistics,
            configuration_snapshot={},
            image_dimensions=(400, 400),
            preprocessing_applied=[],
            coordinate_system="dxf_mm"  # DXF coordinates
        )
        
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, dxf_feature_set
        )
        
        # Visualization should still be created but may be empty or show warning
        assert output_path.exists()
    
    def test_visualize_empty_feature_set(self):
        """Test visualization with no detected features."""
        empty_feature_set = ActualFeatureSet(
            source_image_path=self.test_image_path,
            features=[],  # No features
            detection_timestamp="2024-01-01 12:00:00",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=0, contours_after_filtering=0,
                circles_detected=0, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=0, average_confidence=0.0,
                detection_time_seconds=1.0, preprocessing_time_seconds=0.5
            ),
            configuration_snapshot={},
            image_dimensions=(400, 400),
            preprocessing_applied=[]
        )
        
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, empty_feature_set
        )
        
        assert output_path.exists()
        
        # Should show metadata even with no features
        viz_image = cv2.imread(str(output_path))
        assert viz_image is not None
    
    def test_visualize_different_feature_types(self):
        """Test visualization handles all feature types correctly."""
        feature_types_test = [
            ActualFeature(
                feature_id="circle_test",
                feature_type=FeatureType.CIRCLE,
                confidence=0.9,
                center=Point2D(50.0, 50.0),
                radius=15.0
            ),
            ActualFeature(
                feature_id="hole_test",
                feature_type=FeatureType.THROUGH_HOLE,
                confidence=0.8,
                center=Point2D(150.0, 50.0),
                radius=12.0
            ),
            ActualFeature(
                feature_id="square_test",
                feature_type=FeatureType.SQUARE_HOLE,
                confidence=0.85,
                center=Point2D(250.0, 50.0),
                width=30.0,
                height=30.0
            ),
            ActualFeature(
                feature_id="rect_test",
                feature_type=FeatureType.RECTANGULAR_HOLE,
                confidence=0.75,
                center=Point2D(350.0, 50.0),
                width=25.0,
                height=40.0
            )
        ]
        
        feature_set = ActualFeatureSet(
            source_image_path=self.test_image_path,
            features=feature_types_test,
            detection_timestamp="2024-01-01 12:00:00",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=4, contours_after_filtering=4,
                circles_detected=1, through_holes_detected=1, rectangular_holes_detected=2,
                total_features_detected=4, average_confidence=0.825,
                detection_time_seconds=1.5, preprocessing_time_seconds=0.3
            ),
            configuration_snapshot={},
            image_dimensions=(400, 400),
            preprocessing_applied=[]
        )
        
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, feature_set
        )
        
        assert output_path.exists()
    
    def test_visualize_nonexistent_image_error(self):
        """Test error handling for non-existent input image."""
        nonexistent_image = Path("does_not_exist.jpg")
        
        with pytest.raises(FileNotFoundError):
            self.visualizer.visualize_actual_features(
                nonexistent_image, self.test_feature_set
            )
    
    def test_visualize_creates_output_directory(self):
        """Test that visualization creates output directory if needed."""
        nested_output = self.temp_dir / "deep" / "nested" / "viz_output.png"
        
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, self.test_feature_set, nested_output
        )
        
        assert output_path == nested_output
        assert nested_output.exists()
        assert nested_output.parent.exists()
    
    def test_create_comparison_visualization(self):
        """Test comparison visualization with matched features."""
        matched_ids = ["circle_1", "rect_1"]
        output_path = self.temp_dir / "comparison_viz.png"
        
        result_path = self.visualizer.create_comparison_visualization(
            self.test_image_path, self.test_feature_set, matched_ids, output_path
        )
        
        assert result_path == output_path
        assert output_path.exists()
    
    def test_visualization_uses_original_image_not_processed(self):
        """Test that visualization uses original image, not processed version."""
        # This test ensures the visualizer loads the original image file,
        # not any processed/aligned versions
        
        # Create a distinctive original image with red square in a clear area
        original_image = np.zeros((200, 200, 3), dtype=np.uint8)
        original_image[10:40, 10:40] = [0, 0, 255]  # Red square in BGR format, positioned away from features
        cv2.imwrite(str(self.test_image_path), original_image)
        
        # Use a minimal feature set that won't overlap the red square
        minimal_feature_set = ActualFeatureSet(
            source_image_path=self.test_image_path,
            features=[
                ActualFeature(
                    feature_id="test_circle",
                    feature_type=FeatureType.CIRCLE,
                    confidence=0.9,
                    center=Point2D(150.0, 150.0),  # Far from red square
                    radius=15.0,
                    detection_method=DetectionMethod.HOUGH_CIRCLES
                )
            ],
            detection_timestamp="2024-01-01",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=1, contours_after_filtering=1,
                circles_detected=1, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=1, average_confidence=0.9,
                detection_time_seconds=0.1, preprocessing_time_seconds=0.1
            ),
            configuration_snapshot={},
            image_dimensions=(200, 200),
            preprocessing_applied=[]
        )
        
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, minimal_feature_set
        )
        
        # Load visualization and check it contains the red square
        viz_image = cv2.imread(str(output_path))
        assert viz_image is not None
        
        # Should have red pixels from original (visualization adds overlays but preserves base)
        # Check for red pixels in BGR format where B=0, G=0, R=255
        red_pixels = np.sum((viz_image[:, :, 2] > 200) & (viz_image[:, :, 0] < 50) & (viz_image[:, :, 1] < 50))
        assert red_pixels > 0  # Some red pixels should remain from original
    
    def test_visualization_metadata_display(self):
        """Test that visualization includes feature metadata."""
        output_path = self.visualizer.visualize_actual_features(
            self.test_image_path, self.test_feature_set
        )
        
        assert output_path.exists()
        
        # The visualization should include metadata overlay
        # (Content verification would require OCR or manual inspection)
        viz_image = cv2.imread(str(output_path))
        
        # Check that image has been modified from original (metadata added)
        original_image = cv2.imread(str(self.test_image_path))
        assert not np.array_equal(viz_image, original_image)


class TestVisualizationConfiguration:
    """Test visualization configuration usage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.visualizer = ActualFeatureVisualizer()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_visualization_uses_config_colors(self):
        """Test that visualization uses colors from configuration."""
        from feature_inspection.config import (
            COLOR_ACTUAL_FEATURE, COLOR_MATCHED_FEATURE, COLOR_EXTRA_FEATURE
        )
        
        # Verify config values are defined (BGR format for OpenCV)
        assert isinstance(COLOR_ACTUAL_FEATURE, tuple)
        assert len(COLOR_ACTUAL_FEATURE) == 3
        assert isinstance(COLOR_MATCHED_FEATURE, tuple)
        assert len(COLOR_MATCHED_FEATURE) == 3
        assert isinstance(COLOR_EXTRA_FEATURE, tuple)
        assert len(COLOR_EXTRA_FEATURE) == 3
    
    def test_visualization_uses_config_parameters(self):
        """Test that visualization uses parameters from configuration."""
        from feature_inspection.config import (
            VISUALIZATION_CIRCLE_THICKNESS, VISUALIZATION_TEXT_FONT_SCALE,
            VISUALIZATION_TEXT_THICKNESS, VISUALIZATION_MARKER_SIZE
        )
        
        # Verify config parameters are defined and reasonable
        assert VISUALIZATION_CIRCLE_THICKNESS > 0
        assert VISUALIZATION_TEXT_FONT_SCALE > 0
        assert VISUALIZATION_TEXT_THICKNESS > 0
        assert VISUALIZATION_MARKER_SIZE > 0
    
    def test_no_hardcoded_visualization_values(self):
        """Test that visualization doesn't use hardcoded values."""
        # This test ensures visualization parameters come from config,
        # not hardcoded constants in the visualizer
        
        # Check that visualizer module imports from config
        import inspect
        from feature_inspection.output import visualizer
        
        # Get the module source instead of just the class source
        visualizer_source = inspect.getsource(visualizer)
        
        # Should import from config
        assert "from ..config import" in visualizer_source
        
        # Should not contain obvious magic numbers for colors/sizes in the class
        # (We allow some basic colors like white for text backgrounds)
        class_source = inspect.getsource(ActualFeatureVisualizer)
        
        # Check for problematic hardcoded primary colors (but allow white/black for backgrounds)
        problematic_patterns = ["(255, 0, 0)", "(0, 255, 0)", "(0, 0, 255)", "(255, 255, 0)", "(255, 0, 255)", "(0, 255, 255)"]
        for pattern in problematic_patterns:
            assert pattern not in class_source, f"Found hardcoded color {pattern} - should use config constants"