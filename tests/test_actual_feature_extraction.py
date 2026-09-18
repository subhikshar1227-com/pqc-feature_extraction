"""
Tests for Phase 2B Actual Feature Extraction

Tests the actual feature extraction functionality without using
expected features for detection.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from feature_inspection.actual import (
    ActualFeature, ActualFeatureType, GeometricProperties, EvidenceMetrics,
    ActualFeatureExtractionResult, ActualFeatureExtractor, extract_actual_features
)
from feature_inspection.actual.circle_extractor import CircleExtractor
from feature_inspection.actual.rectangle_extractor import RectangleExtractor
from feature_inspection.actual.contour_extractor import ContourExtractor
from feature_inspection.preprocessing import PreprocessingResult


class TestActualFeatureModels:
    """Test actual feature data models."""
    
    def test_actual_feature_creation(self):
        """Test creating ActualFeature objects."""
        # Create test geometry
        geometry = GeometricProperties(
            center=(100.0, 150.0),
            area=314.0,
            perimeter=62.8,
            bounding_box=(90, 140, 20, 20),
            radius=10.0,
            diameter=20.0,
            circularity=1.0,
            solidity=0.95,
            extent=0.785,
            convexity=1.0
        )
        
        # Create test evidence
        evidence = EvidenceMetrics(
            confidence=0.8,
            edge_support=0.7,
            contour_quality=0.9,
            intensity_consistency=0.6,
            geometric_consistency=0.85,
            internal_edge_evidence=1.0
        )
        
        # Create test contour
        angles = np.linspace(0, 2*np.pi, 32)
        contour_x = 100 + 10 * np.cos(angles)
        contour_y = 150 + 10 * np.sin(angles)
        contour = np.column_stack((contour_x, contour_y)).astype(np.int32)
        
        # Create feature
        feature = ActualFeature(
            feature_id="test_circle_1",
            feature_type=ActualFeatureType.CIRCLE,
            geometry=geometry,
            contour=contour,
            evidence=evidence,
            source_representation="internal_geometry_edges",
            detection_method="contour_analysis"
        )
        
        assert feature.feature_id == "test_circle_1"
        assert feature.feature_type == ActualFeatureType.CIRCLE
        assert feature.geometry.center == (100.0, 150.0)
        assert feature.evidence.confidence == 0.8
        assert len(feature.contour) == 32
    
    def test_coordinate_transformation(self):
        """Test coordinate system transformations."""
        # Create test feature in processed coordinates
        geometry = GeometricProperties(
            center=(200.0, 300.0),
            area=100.0,
            perimeter=40.0,
            bounding_box=(190, 290, 20, 20),
            radius=5.64,
            circularity=0.62
        )
        
        evidence = EvidenceMetrics(
            confidence=0.7,
            edge_support=0.6,
            contour_quality=0.8,
            intensity_consistency=0.5,
            geometric_consistency=0.75
        )
        
        contour = np.array([[190, 290], [210, 290], [210, 310], [190, 310]], dtype=np.int32)
        
        feature = ActualFeature(
            feature_id="test_rect_1",
            feature_type=ActualFeatureType.RECTANGLE,
            geometry=geometry,
            contour=contour,
            evidence=evidence,
            source_representation="internal_geometry_edges",
            detection_method="contour_analysis",
            scale_factor=0.5  # Processed image is half size
        )
        
        # Convert to original coordinates
        original_feature = feature.to_original_coordinates(roi_offset=(0, 0))
        
        assert original_feature.coordinate_system == "original"
        assert original_feature.scale_factor == 1.0
        assert original_feature.geometry.center == (400.0, 600.0)  # Scaled up by 2
        assert original_feature.geometry.area == 400.0  # Scaled up by 4 (2^2)


class TestCircleExtractor:
    """Test circle feature extraction."""
    
    def test_circle_extractor_initialization(self):
        """Test circle extractor initialization."""
        extractor = CircleExtractor()
        
        assert extractor.min_radius > 0
        assert extractor.max_radius > extractor.min_radius
        assert extractor.circularity_threshold > 0
        assert extractor.min_confidence > 0

    def test_partial_arc_is_not_accepted_as_full_circle(self):
        """A localized arc should not qualify as a full circular feature."""
        extractor = CircleExtractor()
        size = 200
        edge_image = np.zeros((size, size), dtype=np.uint8)
        center = (100, 100)
        radius = 30.0

        for angle in np.linspace(0, 2 * np.pi, 500, endpoint=False):
            if 0.2 * np.pi < angle < 1.0 * np.pi:
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                if 0 <= x < size and 0 <= y < size:
                    edge_image[y, x] = 255

        validation = extractor._validate_hough_candidate(
            center,
            radius,
            edge_image,
            np.zeros((size, size, 3), dtype=np.uint8),
            np.ones((size, size), dtype=np.uint8) * 255,
        )

        assert validation["valid"] is False
        assert "coverage" in validation["rejection_reason"] or "localized" in validation["rejection_reason"] or "gap" in validation["rejection_reason"]

    def test_hough_only_detection_confidence_is_penalized(self):
        """Weak Hough-only evidence should not receive a high confidence score."""
        extractor = CircleExtractor()
        validation = {
            "valid": True,
            "edge_support": 0.5,
            "angular_coverage": 0.35,
            "sector_coverage": 0.25,
            "geometric_consistency": 0.4,
            "intensity_consistency": 0.2,
            "contour_agreement": False,
        }

        confidence = extractor._calculate_hough_evidence_confidence(validation)
        assert confidence < 0.6

    def test_synthetic_circle_detection(self):
        """Test circle detection on synthetic images."""
        # Create synthetic images
        size = 200
        internal_edges = np.zeros((size, size), dtype=np.uint8)
        isolated_product = np.zeros((size, size, 3), dtype=np.uint8)
        product_mask = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw a clear circle in edges
        cv2.circle(internal_edges, (100, 100), 30, 255, 2)
        cv2.circle(isolated_product, (100, 100), 30, (128, 128, 128), -1)
        
        extractor = CircleExtractor()
        circles = extractor.extract_circles(internal_edges, isolated_product, product_mask)
        
        # Should detect the synthetic circle
        assert len(circles) >= 1
        
        # Check first detected circle
        circle = circles[0]
        assert circle.feature_type == ActualFeatureType.CIRCLE
        assert abs(circle.geometry.center[0] - 100) < 10
        assert abs(circle.geometry.center[1] - 100) < 10
        assert abs(circle.geometry.radius - 30) < 10
        assert circle.evidence.confidence > 0


class TestRectangleExtractor:
    """Test rectangle feature extraction."""
    
    def test_rectangle_extractor_initialization(self):
        """Test rectangle extractor initialization."""
        extractor = RectangleExtractor()
        
        assert extractor.min_area > 0
        assert extractor.max_area > extractor.min_area
        assert extractor.min_side_length > 0
        assert extractor.min_confidence > 0
    
    def test_synthetic_rectangle_detection(self):
        """Test rectangle detection on synthetic images."""
        # Create synthetic images
        size = 200
        internal_edges = np.zeros((size, size), dtype=np.uint8)
        isolated_product = np.zeros((size, size, 3), dtype=np.uint8)
        product_mask = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw a clear rectangle in edges
        cv2.rectangle(internal_edges, (50, 70), (150, 130), 255, 2)
        cv2.rectangle(isolated_product, (50, 70), (150, 130), (128, 128, 128), -1)
        
        extractor = RectangleExtractor()
        rectangles = extractor.extract_rectangles(internal_edges, isolated_product, product_mask)
        
        # Should detect the synthetic rectangle
        assert len(rectangles) >= 1
        
        # Check first detected rectangle  
        rect = rectangles[0]
        assert rect.feature_type in [ActualFeatureType.RECTANGLE, ActualFeatureType.SQUARE]
        # Center should be approximately (100, 100)
        assert abs(rect.geometry.center[0] - 100) < 15
        assert abs(rect.geometry.center[1] - 100) < 15
        assert rect.evidence.confidence > 0


class TestContourExtractor:
    """Test general contour extraction."""
    
    def test_contour_extractor_initialization(self):
        """Test contour extractor initialization."""
        extractor = ContourExtractor()
        
        assert extractor.min_area > 0
        assert extractor.max_area > extractor.min_area
        assert extractor.min_perimeter > 0
        assert extractor.min_confidence > 0
    
    def test_general_contour_detection(self):
        """Test general contour detection."""
        # Create synthetic irregular shape
        size = 200
        internal_edges = np.zeros((size, size), dtype=np.uint8)
        isolated_product = np.zeros((size, size, 3), dtype=np.uint8)
        product_mask = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw an irregular shape (triangle)
        points = np.array([[100, 50], [80, 130], [120, 130]], dtype=np.int32)
        cv2.polylines(internal_edges, [points], True, 255, 2)
        cv2.fillPoly(isolated_product, [points], (128, 128, 128))
        
        extractor = ContourExtractor()
        contours = extractor.extract_contours(internal_edges, isolated_product, product_mask)
        
        # Should detect the contour
        assert len(contours) >= 1
        
        # Check first detected contour
        contour = contours[0]
        assert contour.feature_type == ActualFeatureType.GENERAL_CONTOUR
        assert contour.evidence.confidence > 0


class TestActualFeatureExtractor:
    """Test main feature extractor orchestrator."""
    
    def test_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = ActualFeatureExtractor()
        
        assert extractor.circle_extractor is not None
        assert extractor.rectangle_extractor is not None
        assert extractor.contour_extractor is not None
    
    def test_empty_preprocessing_result(self):
        """Test handling of failed preprocessing."""
        # Create failed preprocessing result
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Create minimal failed preprocessing result
            empty_img = np.zeros((100, 100), dtype=np.uint8)
            empty_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
            
            preprocessing_result = PreprocessingResult(
                source_image_path=temp_path,
                original_image=empty_bgr,
                processed_image=empty_img,
                product_mask=empty_img,
                isolated_product_image=empty_img,
                raw_internal_geometry_edges=empty_img,
                internal_geometry_edges=empty_img,
                outer_boundary_edges=empty_img,
                edge_representation=empty_img,
                original_dimensions=(100, 100),
                processed_dimensions=(100, 100),
                scale_factor=1.0,
                roi_offset=(0, 0),
                preprocessing_successful=False,  # Failed preprocessing
                isolation_successful=False,
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
                silhouette_validation_result="FAILED",
                external_gradient_strength=0.0,
                boundary_gradient_consistency=0.0,
                suspicious_boundary_fraction=0.0,
                boundary_gradient_strength=0.0,
                raw_internal_edge_density=0.0,
                filtered_internal_edge_density=0.0,
                internal_edge_retention_ratio=1.0,
                internal_edge_reduction_ratio=0.0,
                outer_boundary_density=0.0,
                final_edge_density=0.0,
                processing_steps=["failed"],
                configuration_snapshot={}
            )
            
            extractor = ActualFeatureExtractor()
            result = extractor.extract_features(preprocessing_result)
            
            # Should return empty result for failed preprocessing
            assert not result.preprocessing_successful
            assert len(result.features) == 0
            assert result.total_candidates_generated == 0
            assert result.average_confidence == 0.0
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_synthetic_multi_feature_extraction(self):
        """Test extraction with multiple feature types."""
        # Create synthetic scene with multiple features
        size = 300
        internal_edges = np.zeros((size, size), dtype=np.uint8)
        isolated_product = np.zeros((size, size, 3), dtype=np.uint8)
        product_mask = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw circle
        cv2.circle(internal_edges, (80, 80), 25, 255, 2)
        cv2.circle(isolated_product, (80, 80), 25, (100, 100, 100), -1)
        
        # Draw rectangle
        cv2.rectangle(internal_edges, (150, 60), (220, 100), 255, 2)
        cv2.rectangle(isolated_product, (150, 60), (220, 100), (120, 120, 120), -1)
        
        # Draw irregular shape
        triangle = np.array([[100, 150], [80, 200], [120, 200]], dtype=np.int32)
        cv2.polylines(internal_edges, [triangle], True, 255, 2)
        cv2.fillPoly(isolated_product, [triangle], (140, 140, 140))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Create successful preprocessing result
            preprocessing_result = PreprocessingResult(
                source_image_path=temp_path,
                original_image=isolated_product,
                processed_image=cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY),
                product_mask=product_mask,
                isolated_product_image=cv2.cvtColor(isolated_product, cv2.COLOR_BGR2GRAY),
                raw_internal_geometry_edges=internal_edges,
                internal_geometry_edges=internal_edges,
                outer_boundary_edges=np.zeros_like(internal_edges),
                edge_representation=internal_edges,
                original_dimensions=(size, size),
                processed_dimensions=(size, size),
                scale_factor=1.0,
                roi_offset=(0, 0),
                preprocessing_successful=True,
                isolation_successful=True,
                product_area_pixels=5000,
                product_area_fraction=0.05,
                mask_bounding_box=(0, 0, size, size),
                mask_bounding_box_area_fraction=1.0,
                foreground_component_count=1,
                largest_component_fraction=1.0,
                mask_contour_area=5000.0,
                mask_contour_perimeter=200.0,
                mask_extent=0.05,
                mask_solidity=0.9,
                border_touching_foreground=False,
                significant_foreground_regions=1,
                silhouette_validation_result="PASS",
                external_gradient_strength=10.0,
                boundary_gradient_consistency=0.8,
                suspicious_boundary_fraction=0.1,
                boundary_gradient_strength=15.0,
                raw_internal_edge_density=0.02,
                filtered_internal_edge_density=0.015,
                internal_edge_retention_ratio=0.75,
                internal_edge_reduction_ratio=0.25,
                outer_boundary_density=0.01,
                final_edge_density=0.02,
                processing_steps=["test"],
                configuration_snapshot={}
            )
            
            result = extract_actual_features(preprocessing_result)
            
            # Should extract multiple features
            assert result.preprocessing_successful
            assert len(result.features) > 0
            assert result.total_candidates_generated > 0
            
            # Check feature types are detected
            detected_types = set(f.feature_type for f in result.features)
            # Should detect at least some features (exact detection depends on thresholds)
            assert len(detected_types) > 0
            
        finally:
            temp_path.unlink(missing_ok=True)


class TestNoBiasValidation:
    """Test that feature extraction doesn't use expected features."""
    
    def test_no_expected_feature_dependency(self):
        """Verify feature extraction doesn't depend on expected features."""
        # This test verifies that the ActualFeatureExtractor can be instantiated
        # and used without any expected feature information
        
        extractor = ActualFeatureExtractor()
        
        # Check that extractor doesn't have any expected feature attributes
        assert not hasattr(extractor, 'expected_features')
        assert not hasattr(extractor, 'expected_counts')
        assert not hasattr(extractor, 'dxf_features')
        
        # Check that individual extractors also don't depend on expected features
        circle_extractor = extractor.circle_extractor
        rectangle_extractor = extractor.rectangle_extractor
        contour_extractor = extractor.contour_extractor
        
        for sub_extractor in [circle_extractor, rectangle_extractor, contour_extractor]:
            assert not hasattr(sub_extractor, 'expected_features')
            assert not hasattr(sub_extractor, 'expected_counts')
            assert not hasattr(sub_extractor, 'target_counts')
    
    def test_no_hardcoded_coordinates(self):
        """Verify no hardcoded product-specific coordinates."""
        extractor = ActualFeatureExtractor()
        
        # Check configuration parameters are from config, not hardcoded
        from feature_inspection.config import (
            CIRCLE_MIN_RADIUS, RECTANGLE_MIN_AREA, CONTOUR_MIN_AREA
        )
        
        assert extractor.circle_extractor.min_radius == CIRCLE_MIN_RADIUS
        assert extractor.rectangle_extractor.min_area == RECTANGLE_MIN_AREA
        assert extractor.contour_extractor.min_area == CONTOUR_MIN_AREA


class TestCoordinateConsistency:
    """Test coordinate system consistency."""
    
    def test_coordinate_system_tracking(self):
        """Test that coordinate systems are properly tracked."""
        # Create test feature
        geometry = GeometricProperties(
            center=(100.0, 100.0),
            area=100.0,
            perimeter=40.0,
            bounding_box=(90, 90, 20, 20)
        )
        
        evidence = EvidenceMetrics(
            confidence=0.8,
            edge_support=0.7,
            contour_quality=0.9,
            intensity_consistency=0.6,
            geometric_consistency=0.8
        )
        
        contour = np.array([[90, 90], [110, 90], [110, 110], [90, 110]], dtype=np.int32)
        
        feature = ActualFeature(
            feature_id="test_coord",
            feature_type=ActualFeatureType.RECTANGLE,
            geometry=geometry,
            contour=contour,
            evidence=evidence,
            source_representation="internal_geometry_edges",
            detection_method="contour_analysis",
            coordinate_system="processed",
            scale_factor=0.8
        )
        
        assert feature.coordinate_system == "processed"
        assert feature.scale_factor == 0.8
        
        # Convert to original coordinates
        original_feature = feature.to_original_coordinates()
        assert original_feature.coordinate_system == "original"
        assert original_feature.scale_factor == 1.0