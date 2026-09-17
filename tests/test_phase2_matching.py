"""
Tests for Phase 2: Feature Matching

Tests the feature matching engine between expected and actual features.
"""

import pytest
import numpy as np

from feature_extraction.expected.feature_types import ExpectedFeature, ExpectedFeatureSet, FeatureType, Point2D
from feature_inspection.models.actual_feature import ActualFeature, ActualFeatureSet, ActualDetectionStatistics, DetectionMethod
from feature_inspection.models.coordinate_transform import CoordinateTransform
from feature_inspection.matching import FeatureMatcher, GeometricMatcher
from feature_inspection.models.feature_match import FeatureMatch, FeatureMatchSet, MatchType
from pathlib import Path


class TestCoordinateTransform:
    """Test coordinate transformation utilities."""
    
    def test_identity_transform(self):
        """Test coordinate transform with identity matrix."""
        identity_matrix = np.eye(3)
        transform = CoordinateTransform(identity_matrix)
        
        assert transform.is_available()
        
        # Test point transformation (should be unchanged)
        test_point = Point2D(100.0, 200.0)
        transformed = transform.dxf_to_image(test_point)
        
        assert abs(transformed.x - test_point.x) < 0.001
        assert abs(transformed.y - test_point.y) < 0.001
        
        # Test round-trip transformation
        recovered = transform.image_to_dxf(transformed)
        assert abs(recovered.x - test_point.x) < 0.001
        assert abs(recovered.y - test_point.y) < 0.001
    
    def test_scale_transform(self):
        """Test coordinate transform with scaling."""
        # 2x scale transform
        scale_matrix = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        transform = CoordinateTransform(scale_matrix)
        
        assert transform.is_available()
        
        # Test point scaling
        test_point = Point2D(10.0, 20.0)
        scaled = transform.dxf_to_image(test_point)
        
        assert abs(scaled.x - 20.0) < 0.001  # 2x scale
        assert abs(scaled.y - 40.0) < 0.001  # 2x scale
        
        # Test radius scaling
        test_radius = 5.0
        scaled_radius = transform.transform_radius_dxf_to_image(test_radius, test_point)
        assert abs(scaled_radius - 10.0) < 0.001  # 2x scale
    
    def test_no_transform(self):
        """Test behavior when no transformation matrix is provided."""
        transform = CoordinateTransform(None)
        
        assert not transform.is_available()
        
        with pytest.raises(ValueError):
            transform.dxf_to_image(Point2D(0, 0))
    
    def test_invalid_transform_matrix(self):
        """Test handling of invalid transformation matrices."""
        # Wrong shape matrix
        with pytest.raises(ValueError):
            CoordinateTransform(np.ones((2, 2)))
        
        # Singular matrix
        singular_matrix = np.zeros((3, 3))
        transform = CoordinateTransform(singular_matrix)
        # Should not raise error during initialization but may warn


class TestGeometricMatcher:
    """Test geometric feature matching."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create identity transform for testing
        self.identity_transform = CoordinateTransform(np.eye(3))
        self.matcher = GeometricMatcher(self.identity_transform)
    
    def test_exact_match_score(self):
        """Test matching score for identical features."""
        expected = ExpectedFeature(
            feature_id="expected1",
            feature_type=FeatureType.CIRCLE,
            confidence=0.95,
            center=Point2D(100.0, 100.0),
            radius=20.0,
            source_entity_ids=["entity1"],
            source_type="explicit_circle",
            detection_evidence={"test": True},
            geometric_properties={"test": True}
        )
        
        actual = ActualFeature(
            feature_id="actual1",
            feature_type=FeatureType.CIRCLE,
            confidence=0.9,
            center=Point2D(100.0, 100.0),  # Same center
            radius=20.0,  # Same radius
            detection_method=DetectionMethod.HOUGH_CIRCLES
        )
        
        score, evidence = self.matcher.calculate_match_score(expected, actual)
        
        assert score > 0.9  # Should be very high score
        assert evidence["type_compatible"] is True
        assert evidence["coordinate_transform_success"] is True
    
    def test_no_match_different_types(self):
        """Test matching score for incompatible feature types."""
        expected = ExpectedFeature(
            feature_id="expected1",
            feature_type=FeatureType.CIRCLE,
            confidence=0.95,
            center=Point2D(100.0, 100.0),
            radius=20.0,
            source_entity_ids=["entity1"],
            source_type="explicit_circle",
            detection_evidence={"test": True},
            geometric_properties={"test": True}
        )
        
        actual = ActualFeature(
            feature_id="actual1",
            feature_type=FeatureType.RECTANGULAR_HOLE,  # Different type
            confidence=0.9,
            center=Point2D(100.0, 100.0),
            width=40.0,
            height=30.0
        )
        
        score, evidence = self.matcher.calculate_match_score(expected, actual)
        
        assert score < 0.5  # Should be low score for type mismatch
        assert evidence["type_compatible"] is False
    
    def test_substitutable_types(self):
        """Test matching between substitutable feature types."""
        expected = ExpectedFeature(
            feature_id="expected1",
            feature_type=FeatureType.CIRCLE,
            confidence=0.95,
            center=Point2D(100.0, 100.0),
            radius=20.0,
            source_entity_ids=["entity1"],
            source_type="explicit_circle",
            detection_evidence={"test": True},
            geometric_properties={"test": True}
        )
        
        actual = ActualFeature(
            feature_id="actual1",
            feature_type=FeatureType.THROUGH_HOLE,  # Substitutable with CIRCLE
            confidence=0.9,
            center=Point2D(100.0, 100.0),
            radius=20.0
        )
        
        score, evidence = self.matcher.calculate_match_score(expected, actual)
        
        assert score > 0.7  # Should allow substitution but with penalty
        assert evidence["substitutable"] is True
    
    def test_position_deviation_scoring(self):
        """Test scoring based on position differences."""
        expected = ExpectedFeature(
            feature_id="expected1",
            feature_type=FeatureType.CIRCLE,
            confidence=0.95,
            center=Point2D(100.0, 100.0),
            radius=20.0,
            source_entity_ids=["entity1"],
            source_type="explicit_circle",
            detection_evidence={"test": True},
            geometric_properties={"test": True}
        )
        
        # Close position (should score high)
        actual_close = ActualFeature(
            feature_id="actual_close",
            feature_type=FeatureType.CIRCLE,
            confidence=0.9,
            center=Point2D(101.0, 101.0),  # 1.4mm away
            radius=20.0
        )
        
        # Far position (should score low)
        actual_far = ActualFeature(
            feature_id="actual_far",
            feature_type=FeatureType.CIRCLE,
            confidence=0.9,
            center=Point2D(150.0, 150.0),  # ~70mm away
            radius=20.0
        )
        
        score_close, _ = self.matcher.calculate_match_score(expected, actual_close)
        score_far, _ = self.matcher.calculate_match_score(expected, actual_far)
        
        assert score_close > score_far
    
    def test_size_deviation_scoring(self):
        """Test scoring based on size differences."""
        expected = ExpectedFeature(
            feature_id="expected1",
            feature_type=FeatureType.CIRCLE,
            confidence=0.95,
            center=Point2D(100.0, 100.0),
            radius=20.0,
            source_entity_ids=["entity1"],
            source_type="explicit_circle",
            detection_evidence={"test": True},
            geometric_properties={"test": True}
        )
        
        # Similar size (should score high)
        actual_similar = ActualFeature(
            feature_id="actual_similar",
            feature_type=FeatureType.CIRCLE,
            confidence=0.9,
            center=Point2D(100.0, 100.0),
            radius=20.5  # Very close
        )
        
        # Different size (should score lower)
        actual_different = ActualFeature(
            feature_id="actual_different",
            feature_type=FeatureType.CIRCLE,
            confidence=0.9,
            center=Point2D(100.0, 100.0),
            radius=30.0  # 50% larger
        )
        
        score_similar, _ = self.matcher.calculate_match_score(expected, actual_similar)
        score_different, _ = self.matcher.calculate_match_score(expected, actual_different)
        
        assert score_similar > score_different


class TestFeatureMatcher:
    """Test the main feature matching engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = FeatureMatcher()
    
    def create_test_expected_set(self) -> ExpectedFeatureSet:
        """Create a test expected feature set."""
        features = [
            ExpectedFeature(
                feature_id="expected_circle1",
                feature_type=FeatureType.CIRCLE,
                confidence=0.95,
                center=Point2D(100.0, 100.0),
                radius=20.0,
                source_entity_ids=["entity1"],
                source_type="explicit_circle",
                detection_evidence={"test": True},
                geometric_properties={"test": True}
            ),
            ExpectedFeature(
                feature_id="expected_hole1",
                feature_type=FeatureType.THROUGH_HOLE,
                confidence=0.9,
                center=Point2D(200.0, 200.0),
                radius=15.0,
                source_entity_ids=["entity2"],
                source_type="explicit_circle",
                detection_evidence={"test": True},
                geometric_properties={"test": True}
            )
        ]
        
        # Minimal required fields for ExpectedFeatureSet
        from feature_inspection.models.actual_feature import ActualDetectionStatistics
        
        stats = ActualDetectionStatistics(
            total_contours_found=0,
            contours_after_filtering=0,
            circles_detected=1,
            through_holes_detected=1,
            rectangular_holes_detected=0,
            total_features_detected=2,
            average_confidence=0.925,
            detection_time_seconds=0.0,
            preprocessing_time_seconds=0.0
        )
        
        return ExpectedFeatureSet(
            source_dxf_path=Path("test.dxf"),
            dxf_units="mm",
            features=features,
            extraction_timestamp="2024-01-01 12:00:00",
            processing_statistics=stats,  # Reusing ActualDetectionStatistics as placeholder
            configuration_snapshot={},
            raw_entity_count=10,
            normalized_entity_count=8,
            reconstructed_geometry_count=2
        )
    
    def create_test_actual_set(self) -> ActualFeatureSet:
        """Create a test actual feature set."""
        features = [
            ActualFeature(
                feature_id="actual_circle1",
                feature_type=FeatureType.CIRCLE,
                confidence=0.9,
                center=Point2D(102.0, 98.0),  # Close to expected
                radius=19.5,  # Close to expected
                detection_method=DetectionMethod.HOUGH_CIRCLES
            ),
            ActualFeature(
                feature_id="actual_hole1",
                feature_type=FeatureType.THROUGH_HOLE,
                confidence=0.85,
                center=Point2D(198.0, 202.0),  # Close to expected
                radius=14.8,  # Close to expected
                detection_method=DetectionMethod.CONTOUR_ANALYSIS
            ),
            ActualFeature(
                feature_id="actual_extra1",
                feature_type=FeatureType.CIRCLE,
                confidence=0.8,
                center=Point2D(300.0, 300.0),  # No corresponding expected
                radius=25.0,
                detection_method=DetectionMethod.CONTOUR_ANALYSIS
            )
        ]
        
        stats = ActualDetectionStatistics(
            total_contours_found=50,
            contours_after_filtering=25,
            circles_detected=2,
            through_holes_detected=1,
            rectangular_holes_detected=0,
            total_features_detected=3,
            average_confidence=0.85,
            detection_time_seconds=1.5,
            preprocessing_time_seconds=0.5
        )
        
        return ActualFeatureSet(
            source_image_path=Path("test_image.png"),
            features=features,
            detection_timestamp="2024-01-01 12:00:00",
            detection_statistics=stats,
            configuration_snapshot={},
            image_dimensions=(400, 400),
            preprocessing_applied=["resize", "noise_reduction"]
        )
    
    def test_basic_matching(self):
        """Test basic feature matching functionality."""
        expected_set = self.create_test_expected_set()
        actual_set = self.create_test_actual_set()
        
        match_set = self.matcher.match_features(expected_set, actual_set)
        
        # Validate match set structure
        assert isinstance(match_set, FeatureMatchSet)
        assert len(match_set.matches) >= 0
        assert len(match_set.unmatched_expected) >= 0
        assert len(match_set.unmatched_actual) >= 0
        
        # Should find matches for the two corresponding features
        assert len(match_set.matches) == 2
        assert len(match_set.unmatched_expected) == 0  # All expected features matched
        assert len(match_set.unmatched_actual) == 1   # One extra actual feature
        
        # Validate individual matches
        for match in match_set.matches:
            assert isinstance(match, FeatureMatch)
            assert match.expected_feature is not None
            assert match.actual_feature is not None
            assert isinstance(match.match_type, MatchType)
            assert 0.0 <= match.match_confidence <= 1.0
    
    def test_no_matches_scenario(self):
        """Test matching when no features correspond."""
        # Create expected set with circles
        expected_features = [
            ExpectedFeature(
                feature_id="expected_circle1",
                feature_type=FeatureType.CIRCLE,
                confidence=0.95,
                center=Point2D(100.0, 100.0),
                radius=20.0,
                source_entity_ids=["entity1"],
                source_type="explicit_circle",
                detection_evidence={"test": True},
                geometric_properties={"test": True}
            )
        ]
        
        # Create actual set with rectangles (incompatible)
        actual_features = [
            ActualFeature(
                feature_id="actual_rect1",
                feature_type=FeatureType.RECTANGULAR_HOLE,
                confidence=0.9,
                center=Point2D(500.0, 500.0),  # Far away
                width=40.0,
                height=30.0
            )
        ]
        
        # Create minimal feature sets
        from feature_inspection.models.actual_feature import ActualDetectionStatistics
        
        stats = ActualDetectionStatistics(
            total_contours_found=0, contours_after_filtering=0,
            circles_detected=0, through_holes_detected=0, rectangular_holes_detected=1,
            total_features_detected=1, average_confidence=0.9,
            detection_time_seconds=0.0, preprocessing_time_seconds=0.0
        )
        
        expected_set = ExpectedFeatureSet(
            source_dxf_path=Path("test.dxf"), dxf_units="mm", features=expected_features,
            extraction_timestamp="2024-01-01", processing_statistics=stats,
            configuration_snapshot={}, raw_entity_count=1, normalized_entity_count=1,
            reconstructed_geometry_count=0
        )
        
        actual_set = ActualFeatureSet(
            source_image_path=Path("test.png"), features=actual_features,
            detection_timestamp="2024-01-01", detection_statistics=stats,
            configuration_snapshot={}, image_dimensions=(400, 400),
            preprocessing_applied=[]
        )
        
        match_set = self.matcher.match_features(expected_set, actual_set)
        
        # Should find no matches
        assert len(match_set.matches) == 0
        assert len(match_set.unmatched_expected) == 1
        assert len(match_set.unmatched_actual) == 1
    
    def test_match_ordering_independence(self):
        """Test that matching results don't depend on feature ordering."""
        expected_set = self.create_test_expected_set()
        actual_set = self.create_test_actual_set()
        
        # Get initial matching
        match_set1 = self.matcher.match_features(expected_set, actual_set)
        
        # Reverse the order of features
        actual_set.features = list(reversed(actual_set.features))
        match_set2 = self.matcher.match_features(expected_set, actual_set)
        
        # Results should be equivalent (same number of matches)
        assert len(match_set1.matches) == len(match_set2.matches)
        assert len(match_set1.unmatched_expected) == len(match_set2.unmatched_expected)
        assert len(match_set1.unmatched_actual) == len(match_set2.unmatched_actual)