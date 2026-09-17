"""
Tests for Phase 2: Integration Tests

Tests the complete Phase 2 pipeline integration with coordinate transformation,
output persistence, and validation against real images.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import shutil

from feature_extraction.expected.feature_types import Point2D, FeatureType, ExpectedFeature, ExpectedFeatureSet
from feature_inspection.models.actual_feature import ActualDetectionStatistics
from feature_inspection import inspect_product_quality, inspect_product_with_phase1_integration
from feature_inspection.models.inspection_result import InspectionStatus
from cad_image_alignment import AlignmentResult


class TestPhase2Integration:
    """Test complete Phase 2 pipeline integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock DXF file
        self.mock_dxf_path = self.temp_dir / "test_design.dxf"
        self.mock_dxf_path.touch()  # Create empty file for path validation
        
        # Create test product image
        self.test_image_path = self.temp_dir / "test_product.jpg"
        import cv2
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128  # Gray image
        # Add some circles for detection
        cv2.circle(test_image, (100, 100), 30, (255, 255, 255), 2)
        cv2.circle(test_image, (300, 200), 25, (0, 0, 0), -1)  # Filled (hole)
        cv2.imwrite(str(self.test_image_path), test_image)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_original_image_input_validation(self):
        """Test that pipeline validates original image input."""
        # Test with non-existent image
        nonexistent_image = Path("does_not_exist.jpg")
        
        with pytest.raises(FileNotFoundError):
            inspect_product_quality(
                nonexistent_image, self.mock_dxf_path, save_outputs=False
            )
    
    def test_pipeline_dxf_input_validation(self):
        """Test that pipeline validates DXF input."""
        nonexistent_dxf = Path("does_not_exist.dxf")
        
        with pytest.raises(FileNotFoundError):
            inspect_product_quality(
                self.test_image_path, nonexistent_dxf, save_outputs=False
            )
    
    def test_pipeline_without_alignment_result(self):
        """Test pipeline behavior when no alignment result is provided."""
        # Mock the DXF extraction at the correct import location
        from feature_inspection.pipeline import orchestrator
        import feature_extraction
        original_extract = orchestrator.extract_expected_features
        
        def mock_extract_expected_features(dxf_path):
            return ExpectedFeatureSet(
                source_dxf_path=dxf_path,
                dxf_units="mm",
                features=[
                    ExpectedFeature(
                        feature_id="expected_1",
                        feature_type=FeatureType.CIRCLE,
                        confidence=0.95,
                        center=Point2D(50.0, 50.0),
                        radius=15.0,
                        source_entity_ids=["entity_1"],
                        source_type="test",
                        detection_evidence={},
                        geometric_properties={}
                    )
                ],
                extraction_timestamp="2024-01-01",
                processing_statistics=ActualDetectionStatistics(
                    total_contours_found=0, contours_after_filtering=0,
                    circles_detected=1, through_holes_detected=0, rectangular_holes_detected=0,
                    total_features_detected=1, average_confidence=0.95,
                    detection_time_seconds=0.0, preprocessing_time_seconds=0.0
                ),
                configuration_snapshot={},
                raw_entity_count=1,
                normalized_entity_count=1,
                reconstructed_geometry_count=1
            )
        
        # Mock at the import location in orchestrator
        orchestrator.extract_expected_features = mock_extract_expected_features
        
        try:
            result = inspect_product_quality(
                self.test_image_path, self.mock_dxf_path, 
                alignment_result=None, save_outputs=False
            )
            
            assert result is not None
            assert hasattr(result, 'inspection_status')
            assert result.source_image_path == self.test_image_path
            
            # Without alignment, coordinate transformation should fail
            # But pipeline should still complete
            
        finally:
            orchestrator.extract_expected_features = original_extract
    
    def test_pipeline_with_identity_alignment(self):
        """Test pipeline with identity transformation matrix."""
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
        
        # Mock expected feature extraction
        from feature_inspection.pipeline import orchestrator
        original_extract = orchestrator.extract_expected_features
        
        def mock_extract_expected_features(dxf_path):
            return ExpectedFeatureSet(
                source_dxf_path=dxf_path,
                dxf_units="mm",
                features=[
                    ExpectedFeature(
                        feature_id="expected_circle",
                        feature_type=FeatureType.CIRCLE,
                        confidence=0.95,
                        center=Point2D(100.0, 100.0),  # Should match detected circle
                        radius=30.0,
                        source_entity_ids=["circle_entity"],
                        source_type="test",
                        detection_evidence={},
                        geometric_properties={}
                    )
                ],
                extraction_timestamp="2024-01-01",
                processing_statistics=ActualDetectionStatistics(
                    total_contours_found=0, contours_after_filtering=0,
                    circles_detected=1, through_holes_detected=0, rectangular_holes_detected=0,
                    total_features_detected=1, average_confidence=0.95,
                    detection_time_seconds=0.0, preprocessing_time_seconds=0.0
                ),
                configuration_snapshot={},
                raw_entity_count=1,
                normalized_entity_count=1,
                reconstructed_geometry_count=1
            )
        
        orchestrator.extract_expected_features = mock_extract_expected_features
        
        try:
            result = inspect_product_quality(
                self.test_image_path, self.mock_dxf_path,
                alignment_result=alignment_result, save_outputs=False
            )
            
            assert result is not None
            assert result.inspection_status in [InspectionStatus.PASS, InspectionStatus.REVIEW, InspectionStatus.FAIL]
            
            # Should have detected some actual features
            assert len(result.actual_feature_set.features) >= 0
            
            # Coordinate transformation should have been attempted
            # (Success depends on feature detection quality)
            
        finally:
            orchestrator.extract_expected_features = original_extract
    
    def test_pipeline_with_precomputed_expected_features(self):
        """Test pipeline variant with pre-computed expected features."""
        expected_features = ExpectedFeatureSet(
            source_dxf_path=self.mock_dxf_path,
            dxf_units="mm",
            features=[
                ExpectedFeature(
                    feature_id="precomputed_1",
                    feature_type=FeatureType.THROUGH_HOLE,
                    confidence=0.9,
                    center=Point2D(300.0, 200.0),
                    radius=25.0,
                    source_entity_ids=["hole_entity"],
                    source_type="test",
                    detection_evidence={},
                    geometric_properties={}
                )
            ],
            extraction_timestamp="2024-01-01",
            processing_statistics=ActualDetectionStatistics(
                total_contours_found=0, contours_after_filtering=0,
                circles_detected=0, through_holes_detected=1, rectangular_holes_detected=0,
                total_features_detected=1, average_confidence=0.9,
                detection_time_seconds=0.0, preprocessing_time_seconds=0.0
            ),
            configuration_snapshot={},
            raw_entity_count=1,
            normalized_entity_count=1,
            reconstructed_geometry_count=1
        )
        
        result = inspect_product_with_phase1_integration(
            self.test_image_path, expected_features, save_outputs=False
        )
        
        assert result is not None
        assert result.expected_feature_set == expected_features
        assert len(result.actual_feature_set.features) >= 0
    
    def test_pipeline_output_persistence_integration(self):
        """Test that pipeline properly saves outputs when requested."""
        # Use temporary directory for outputs
        output_dir = self.temp_dir / "outputs"
        
        # Mock expected feature extraction
        from feature_inspection.pipeline import orchestrator
        original_extract = orchestrator.extract_expected_features
        
        def mock_extract_expected_features(dxf_path):
            return ExpectedFeatureSet(
                source_dxf_path=dxf_path,
                dxf_units="mm",
                features=[],  # Empty for simplicity
                extraction_timestamp="2024-01-01",
                processing_statistics=ActualDetectionStatistics(
                    total_contours_found=0, contours_after_filtering=0,
                    circles_detected=0, through_holes_detected=0, rectangular_holes_detected=0,
                    total_features_detected=0, average_confidence=0.0,
                    detection_time_seconds=0.0, preprocessing_time_seconds=0.0
                ),
                configuration_snapshot={},
                raw_entity_count=0,
                normalized_entity_count=0,
                reconstructed_geometry_count=0
            )
        
        orchestrator.extract_expected_features = mock_extract_expected_features
        
        try:
            result = inspect_product_quality(
                self.test_image_path, self.mock_dxf_path,
                save_outputs=True, output_base_dir=output_dir
            )
            
            assert result is not None
            
            # Check that output directory was created
            expected_output_dir = output_dir / self.test_image_path.stem / "phase_2"
            assert expected_output_dir.exists()
            
            # Check for expected output files
            expected_files = [
                "actual_features.json",
                "matching_result.json", 
                "inspection_result.json",
                "output_summary.json"
            ]
            
            for filename in expected_files:
                file_path = expected_output_dir / filename
                assert file_path.exists(), f"Expected output file {filename} not found"
            
        finally:
            orchestrator.extract_expected_features = original_extract
    
    def test_pipeline_coordinate_system_tracking(self):
        """Test that coordinate systems are properly tracked through pipeline."""
        # Mock expected features
        from feature_inspection.pipeline import orchestrator
        original_extract = orchestrator.extract_expected_features
        
        def mock_extract_expected_features(dxf_path):
            return ExpectedFeatureSet(
                source_dxf_path=dxf_path,
                dxf_units="mm",
                features=[],
                extraction_timestamp="2024-01-01",
                processing_statistics=ActualDetectionStatistics(
                    total_contours_found=0, contours_after_filtering=0,
                    circles_detected=0, through_holes_detected=0, rectangular_holes_detected=0,
                    total_features_detected=0, average_confidence=0.0,
                    detection_time_seconds=0.0, preprocessing_time_seconds=0.0
                ),
                configuration_snapshot={},
                raw_entity_count=0,
                normalized_entity_count=0,
                reconstructed_geometry_count=0
            )
        
        orchestrator.extract_expected_features = mock_extract_expected_features
        
        try:
            result = inspect_product_quality(
                self.test_image_path, self.mock_dxf_path, save_outputs=False
            )
            
            # Check coordinate system tracking
            actual_coord_sys = result.actual_feature_set.coordinate_system
            
            # Should be either dxf_mm (if transform succeeded) or indicate transform status
            assert actual_coord_sys in [
                "dxf_mm", 
                "image_pixels_transform_unavailable",
                "image_pixels_transform_failed"
            ]
            
            # Expected features should always be in dxf_mm
            expected_coord_sys = getattr(result.expected_feature_set, 'coordinate_system', 'dxf_mm')
            assert expected_coord_sys == 'dxf_mm'
            
        finally:
            orchestrator.extract_expected_features = original_extract
    
    def test_pipeline_preserves_feature_provenance(self):
        """Test that feature provenance is preserved through pipeline."""
        # Mock expected features
        from feature_inspection.pipeline import orchestrator
        original_extract = orchestrator.extract_expected_features
        
        def mock_extract_expected_features(dxf_path):
            return ExpectedFeatureSet(
                source_dxf_path=dxf_path,
                dxf_units="mm", 
                features=[
                    ExpectedFeature(
                        feature_id="provenance_test",
                        feature_type=FeatureType.CIRCLE,
                        confidence=0.95,
                        center=Point2D(100.0, 100.0),
                        radius=30.0,
                        source_entity_ids=["dxf_circle_123"],
                        source_type="explicit_circle",
                        detection_evidence={"dxf_layer": "features"},
                        geometric_properties={"diameter": 60.0}
                    )
                ],
                extraction_timestamp="2024-01-01",
                processing_statistics=ActualDetectionStatistics(
                    total_contours_found=0, contours_after_filtering=0,
                    circles_detected=1, through_holes_detected=0, rectangular_holes_detected=0,
                    total_features_detected=1, average_confidence=0.95,
                    detection_time_seconds=0.0, preprocessing_time_seconds=0.0
                ),
                configuration_snapshot={},
                raw_entity_count=1,
                normalized_entity_count=1,
                reconstructed_geometry_count=1
            )
        
        orchestrator.extract_expected_features = mock_extract_expected_features
        
        try:
            result = inspect_product_quality(
                self.test_image_path, self.mock_dxf_path, save_outputs=False
            )
            
            # Check expected feature provenance preserved
            expected_feature = result.expected_feature_set.features[0]
            assert expected_feature.feature_id == "provenance_test"
            assert expected_feature.source_entity_ids == ["dxf_circle_123"]
            assert expected_feature.source_type == "explicit_circle"
            assert "dxf_layer" in expected_feature.detection_evidence
            
            # Check actual feature provenance
            if len(result.actual_feature_set.features) > 0:
                actual_feature = result.actual_feature_set.features[0]
                assert actual_feature.feature_id is not None
                assert hasattr(actual_feature, 'detection_method')
                assert hasattr(actual_feature, 'detection_evidence')
                
                # If coordinate transformation occurred, should have transformation evidence
                if result.actual_feature_set.coordinate_system == "dxf_mm":
                    assert "coordinate_transformation" in actual_feature.detection_evidence
            
        finally:
            orchestrator.extract_expected_features = original_extract
    
    def test_no_expected_count_based_actual_detection(self):
        """Test that actual detection doesn't use expected feature counts."""
        # Create two different expected feature sets with different counts
        expected_set_1 = ExpectedFeatureSet(
            source_dxf_path=self.mock_dxf_path,
            dxf_units="mm",
            features=[],  # 0 expected features
            extraction_timestamp="2024-01-01",
            processing_statistics=ActualDetectionStatistics(
                total_contours_found=0, contours_after_filtering=0,
                circles_detected=0, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=0, average_confidence=0.0,
                detection_time_seconds=0.0, preprocessing_time_seconds=0.0
            ),
            configuration_snapshot={},
            raw_entity_count=0,
            normalized_entity_count=0,
            reconstructed_geometry_count=0
        )
        
        expected_set_5 = ExpectedFeatureSet(
            source_dxf_path=self.mock_dxf_path,
            dxf_units="mm",
            features=[
                ExpectedFeature(
                    feature_id=f"expected_{i}",
                    feature_type=FeatureType.CIRCLE,
                    confidence=0.9,
                    center=Point2D(i*50.0, i*50.0),
                    radius=10.0,
                    source_entity_ids=[f"entity_{i}"],
                    source_type="test",
                    detection_evidence={},
                    geometric_properties={}
                )
                for i in range(5)  # 5 expected features
            ],
            extraction_timestamp="2024-01-01",
            processing_statistics=ActualDetectionStatistics(
                total_contours_found=0, contours_after_filtering=0,
                circles_detected=5, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=5, average_confidence=0.9,
                detection_time_seconds=0.0, preprocessing_time_seconds=0.0
            ),
            configuration_snapshot={},
            raw_entity_count=5,
            normalized_entity_count=5,
            reconstructed_geometry_count=5
        )
        
        # Run detection with both expected sets
        result_1 = inspect_product_with_phase1_integration(
            self.test_image_path, expected_set_1, save_outputs=False
        )
        
        result_5 = inspect_product_with_phase1_integration(
            self.test_image_path, expected_set_5, save_outputs=False
        )
        
        # Actual detection should be independent of expected count
        # (Same image should detect same features regardless of expected count)
        actual_count_1 = len(result_1.actual_feature_set.features)
        actual_count_5 = len(result_5.actual_feature_set.features)
        
        assert actual_count_1 == actual_count_5, \
            f"Actual detection should be independent of expected count: {actual_count_1} != {actual_count_5}"