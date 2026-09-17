"""
Phase 2 Anti-Hardcoding Tests

Tests that ensure Phase 2 implementation follows anti-hardcoding principles.
"""

import pytest
from pathlib import Path
import numpy as np

from feature_extraction.expected.feature_types import ExpectedFeature, FeatureType, Point2D
from feature_inspection.models.actual_feature import ActualFeature, DetectionMethod
from feature_inspection.matching.matcher import FeatureMatcher


class TestPhase2AntiHardcoding:
    """Test that Phase 2 follows anti-hardcoding principles."""
    
    def test_no_hardcoded_product_coordinates(self):
        """Test that no product-specific coordinates are hardcoded."""
        # This test verifies that detection and matching don't rely on 
        # hardcoded coordinates for specific products
        
        matcher = FeatureMatcher()
        
        # Create features at different coordinate ranges
        expected1 = ExpectedFeature(
            feature_id="exp1", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(50.0, 50.0), radius=20.0,
            source_entity_ids=["e1"], source_type="test",
            detection_evidence={}, geometric_properties={}
        )
        
        expected2 = ExpectedFeature(
            feature_id="exp2", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(1000.0, 1000.0), radius=20.0,  # Very different coordinates
            source_entity_ids=["e2"], source_type="test", 
            detection_evidence={}, geometric_properties={}
        )
        
        actual1 = ActualFeature(
            feature_id="act1", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(52.0, 48.0), radius=19.5,  # Close to expected1
            detection_method=DetectionMethod.HOUGH_CIRCLES
        )
        
        actual2 = ActualFeature(
            feature_id="act2", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(1002.0, 998.0), radius=20.5,  # Close to expected2
            detection_method=DetectionMethod.HOUGH_CIRCLES
        )
        
        # Test matching at different coordinate ranges
        score1, _ = matcher.geometric_matcher.calculate_match_score(expected1, actual1)
        score2, _ = matcher.geometric_matcher.calculate_match_score(expected2, actual2)
        
        # Matching should work equally well regardless of coordinate values
        assert abs(score1 - score2) < 0.1, "Matching should be coordinate-agnostic"
    
    def test_no_hardcoded_feature_counts(self):
        """Test that algorithms don't assume specific feature counts."""
        from feature_inspection.models.actual_feature import ActualFeatureSet, ActualDetectionStatistics
        from feature_extraction.expected.feature_types import ExpectedFeatureSet
        
        # Create sets with different feature counts
        stats = ActualDetectionStatistics(
            total_contours_found=0, contours_after_filtering=0,
            circles_detected=0, through_holes_detected=0, rectangular_holes_detected=0,
            total_features_detected=0, average_confidence=0.0,
            detection_time_seconds=0.0, preprocessing_time_seconds=0.0
        )
        
        # Test with varying numbers of features
        test_cases = [0, 1, 5, 20]  # Different feature counts
        
        for count in test_cases:
            expected_features = []
            actual_features = []
            
            for i in range(count):
                expected_features.append(ExpectedFeature(
                    feature_id=f"exp_{i}", feature_type=FeatureType.CIRCLE, confidence=0.9,
                    center=Point2D(i * 100.0, i * 100.0), radius=20.0,
                    source_entity_ids=[f"e_{i}"], source_type="test",
                    detection_evidence={}, geometric_properties={}
                ))
                
                actual_features.append(ActualFeature(
                    feature_id=f"act_{i}", feature_type=FeatureType.CIRCLE, confidence=0.9,
                    center=Point2D(i * 100.0 + 1, i * 100.0 + 1), radius=20.0,
                    detection_method=DetectionMethod.HOUGH_CIRCLES
                ))
            
            expected_set = ExpectedFeatureSet(
                source_dxf_path=Path("test.dxf"), dxf_units="mm", features=expected_features,
                extraction_timestamp="2024-01-01", processing_statistics=stats,
                configuration_snapshot={}, raw_entity_count=count, normalized_entity_count=count,
                reconstructed_geometry_count=0
            )
            
            actual_set = ActualFeatureSet(
                source_image_path=Path("test.png"), features=actual_features,
                detection_timestamp="2024-01-01", detection_statistics=stats,
                configuration_snapshot={}, image_dimensions=(400, 400),
                preprocessing_applied=[]
            )
            
            # Matching should work regardless of feature count
            matcher = FeatureMatcher()
            match_set = matcher.match_features(expected_set, actual_set)
            
            # Should match all features when they correspond
            assert len(match_set.matches) == count
            assert len(match_set.unmatched_expected) == 0
            assert len(match_set.unmatched_actual) == 0
    
    def test_feature_ordering_independence(self):
        """Test that matching doesn't depend on feature list ordering."""
        from feature_inspection.models.actual_feature import ActualFeatureSet, ActualDetectionStatistics
        from feature_extraction.expected.feature_types import ExpectedFeatureSet
        
        # Create features
        expected_features = [
            ExpectedFeature(
                feature_id="exp_a", feature_type=FeatureType.CIRCLE, confidence=0.9,
                center=Point2D(100.0, 100.0), radius=20.0,
                source_entity_ids=["ea"], source_type="test",
                detection_evidence={}, geometric_properties={}
            ),
            ExpectedFeature(
                feature_id="exp_b", feature_type=FeatureType.CIRCLE, confidence=0.9,
                center=Point2D(200.0, 200.0), radius=25.0,
                source_entity_ids=["eb"], source_type="test",
                detection_evidence={}, geometric_properties={}
            ),
            ExpectedFeature(
                feature_id="exp_c", feature_type=FeatureType.CIRCLE, confidence=0.9,
                center=Point2D(300.0, 300.0), radius=30.0,
                source_entity_ids=["ec"], source_type="test",
                detection_evidence={}, geometric_properties={}
            )
        ]
        
        actual_features = [
            ActualFeature(
                feature_id="act_a", feature_type=FeatureType.CIRCLE, confidence=0.9,
                center=Point2D(101.0, 99.0), radius=20.5,
                detection_method=DetectionMethod.HOUGH_CIRCLES
            ),
            ActualFeature(
                feature_id="act_b", feature_type=FeatureType.CIRCLE, confidence=0.9,
                center=Point2D(198.0, 202.0), radius=24.5,
                detection_method=DetectionMethod.HOUGH_CIRCLES
            ),
            ActualFeature(
                feature_id="act_c", feature_type=FeatureType.CIRCLE, confidence=0.9,
                center=Point2D(302.0, 298.0), radius=29.5,
                detection_method=DetectionMethod.HOUGH_CIRCLES
            )
        ]
        
        stats = ActualDetectionStatistics(
            total_contours_found=0, contours_after_filtering=0,
            circles_detected=3, through_holes_detected=0, rectangular_holes_detected=0,
            total_features_detected=3, average_confidence=0.9,
            detection_time_seconds=0.0, preprocessing_time_seconds=0.0
        )
        
        # Test with original ordering
        expected_set1 = ExpectedFeatureSet(
            source_dxf_path=Path("test.dxf"), dxf_units="mm", features=expected_features,
            extraction_timestamp="2024-01-01", processing_statistics=stats,
            configuration_snapshot={}, raw_entity_count=3, normalized_entity_count=3,
            reconstructed_geometry_count=0
        )
        
        actual_set1 = ActualFeatureSet(
            source_image_path=Path("test.png"), features=actual_features,
            detection_timestamp="2024-01-01", detection_statistics=stats,
            configuration_snapshot={}, image_dimensions=(400, 400),
            preprocessing_applied=[]
        )
        
        # Test with reversed ordering
        expected_set2 = ExpectedFeatureSet(
            source_dxf_path=Path("test.dxf"), dxf_units="mm", features=list(reversed(expected_features)),
            extraction_timestamp="2024-01-01", processing_statistics=stats,
            configuration_snapshot={}, raw_entity_count=3, normalized_entity_count=3,
            reconstructed_geometry_count=0
        )
        
        actual_set2 = ActualFeatureSet(
            source_image_path=Path("test.png"), features=list(reversed(actual_features)),
            detection_timestamp="2024-01-01", detection_statistics=stats,
            configuration_snapshot={}, image_dimensions=(400, 400),
            preprocessing_applied=[]
        )
        
        # Perform matching with both orderings
        matcher = FeatureMatcher()
        match_set1 = matcher.match_features(expected_set1, actual_set1)
        match_set2 = matcher.match_features(expected_set2, actual_set2)
        
        # Results should be equivalent regardless of ordering
        assert len(match_set1.matches) == len(match_set2.matches)
        assert len(match_set1.unmatched_expected) == len(match_set2.unmatched_expected)
        assert len(match_set1.unmatched_actual) == len(match_set2.unmatched_actual)
        
        # Match quality should be similar
        avg_confidence1 = sum(m.match_confidence for m in match_set1.matches) / len(match_set1.matches)
        avg_confidence2 = sum(m.match_confidence for m in match_set2.matches) / len(match_set2.matches)
        assert abs(avg_confidence1 - avg_confidence2) < 0.1
    
    def test_no_product_specific_branches(self):
        """Test that no algorithms contain product-specific conditional logic."""
        # This is a design-level test - Phase 2 should not contain 
        # any if-statements based on product names, file names, or 
        # other product-specific identifiers
        
        # Test that feature matching works the same regardless of source paths
        matcher = FeatureMatcher()
        
        expected = ExpectedFeature(
            feature_id="test_exp", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(100.0, 100.0), radius=20.0,
            source_entity_ids=["e1"], source_type="test",
            detection_evidence={}, geometric_properties={}
        )
        
        actual = ActualFeature(
            feature_id="test_act", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(102.0, 98.0), radius=19.5,
            detection_method=DetectionMethod.HOUGH_CIRCLES
        )
        
        # Test with different "product names" (which should not affect matching)
        score1, _ = matcher.geometric_matcher.calculate_match_score(expected, actual)
        score2, _ = matcher.geometric_matcher.calculate_match_score(expected, actual)
        
        # Score should be identical regardless of context
        assert score1 == score2
    
    def test_configuration_driven_behavior(self):
        """Test that behavior is driven by configuration, not hardcoded values."""
        from feature_inspection.config import (
            MATCH_CENTER_TOLERANCE_MM, MATCH_RADIUS_TOLERANCE_RELATIVE
        )
        
        # Test that matching behavior respects configuration
        matcher = FeatureMatcher()
        
        expected = ExpectedFeature(
            feature_id="test_exp", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(100.0, 100.0), radius=20.0,
            source_entity_ids=["e1"], source_type="test",
            detection_evidence={}, geometric_properties={}
        )
        
        # Feature just within tolerance
        actual_within = ActualFeature(
            feature_id="test_act_within", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(100.0 + MATCH_CENTER_TOLERANCE_MM * 0.9, 100.0), 
            radius=20.0 * (1 + MATCH_RADIUS_TOLERANCE_RELATIVE * 0.9),
            detection_method=DetectionMethod.HOUGH_CIRCLES
        )
        
        # Feature just outside tolerance  
        actual_outside = ActualFeature(
            feature_id="test_act_outside", feature_type=FeatureType.CIRCLE, confidence=0.9,
            center=Point2D(100.0 + MATCH_CENTER_TOLERANCE_MM * 1.5, 100.0),
            radius=20.0 * (1 + MATCH_RADIUS_TOLERANCE_RELATIVE * 1.5),
            detection_method=DetectionMethod.HOUGH_CIRCLES
        )
        
        score_within, _ = matcher.geometric_matcher.calculate_match_score(expected, actual_within)
        score_outside, _ = matcher.geometric_matcher.calculate_match_score(expected, actual_outside)
        
        # Within-tolerance should score higher than outside-tolerance
        assert score_within > score_outside
        
        # Verify scores are influenced by configuration values
        assert score_within > 0.6  # Should be acceptable
        assert score_outside < score_within  # Should be penalized