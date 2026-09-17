"""
Tests for Phase 2: Output Persistence

Tests the output persistence system for saving Phase 2 results.
"""

import pytest
import json
import tempfile
from pathlib import Path
import shutil

from feature_extraction.expected.feature_types import Point2D, FeatureType
from feature_inspection.models.actual_feature import ActualFeature, ActualFeatureSet, ActualDetectionStatistics, DetectionMethod
from feature_inspection.models.feature_match import FeatureMatchSet, MatchingStatistics
from feature_inspection.models.inspection_result import InspectionResult, InspectionStatus, QualityMetrics, InspectionSummary
from feature_inspection.output.persistence import OutputPersistence
from feature_inspection.output.serialization import Phase2JSONEncoder, save_json, load_json


class TestOutputPersistence:
    """Test output persistence functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())
        self.persistence = OutputPersistence(self.temp_dir)
        
        # Create test data
        self.test_image_path = Path("test_product_image.jpg")
        
        self.test_actual_features = ActualFeatureSet(
            source_image_path=self.test_image_path,
            features=[
                ActualFeature(
                    feature_id="actual_1",
                    feature_type=FeatureType.CIRCLE,
                    confidence=0.9,
                    center=Point2D(100.0, 100.0),
                    radius=20.0,
                    detection_method=DetectionMethod.HOUGH_CIRCLES
                )
            ],
            detection_timestamp="2024-01-01 12:00:00",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=10, contours_after_filtering=5,
                circles_detected=1, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=1, average_confidence=0.9,
                detection_time_seconds=1.0, preprocessing_time_seconds=0.5
            ),
            configuration_snapshot={},
            image_dimensions=(400, 400),
            preprocessing_applied=[]
        )
        
        self.test_match_set = FeatureMatchSet(
            matches=[],
            unmatched_expected=[],
            unmatched_actual=[],
            matching_timestamp="2024-01-01 12:01:00",
            matching_statistics=MatchingStatistics(
                total_expected_features=1, total_actual_features=1,
                successful_matches=0, exact_matches=0, good_matches=0,
                acceptable_matches=0, poor_matches=0,
                unmatched_expected=1, unmatched_actual=1,
                average_match_confidence=0.0, average_geometric_accuracy=0.0,
                matching_time_seconds=0.1
            ),
            matching_algorithm="test",
            configuration_snapshot={}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_output_directory(self):
        """Test output directory creation."""
        output_dir = self.persistence.create_output_directory(self.test_image_path)
        
        # Check directory structure
        expected_dir = self.temp_dir / self.test_image_path.stem / "phase_2"
        assert output_dir == expected_dir
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_save_actual_features(self):
        """Test saving actual features to JSON."""
        output_dir = self.persistence.create_output_directory(self.test_image_path)
        
        saved_path = self.persistence.save_actual_features(
            self.test_actual_features, output_dir
        )
        
        # Check file was created
        expected_path = output_dir / "actual_features.json"
        assert saved_path == expected_path
        assert saved_path.exists()
        
        # Check file content
        with open(saved_path, 'r') as f:
            data = json.load(f)
        
        assert "features" in data
        assert len(data["features"]) == 1
        assert data["features"][0]["feature_id"] == "actual_1"
        assert data["coordinate_system"] == "image_pixels"
        assert "_metadata" in data
    
    def test_save_actual_features_with_suffix(self):
        """Test saving actual features with coordinate suffix."""
        output_dir = self.persistence.create_output_directory(self.test_image_path)
        
        saved_path = self.persistence.save_actual_features(
            self.test_actual_features, output_dir, "_transformed"
        )
        
        expected_path = output_dir / "actual_features_transformed.json"
        assert saved_path == expected_path
        assert saved_path.exists()
    
    def test_save_matching_results(self):
        """Test saving feature matching results."""
        output_dir = self.persistence.create_output_directory(self.test_image_path)
        
        saved_path = self.persistence.save_matching_results(
            self.test_match_set, output_dir
        )
        
        expected_path = output_dir / "matching_result.json"
        assert saved_path == expected_path
        assert saved_path.exists()
        
        # Check content
        with open(saved_path, 'r') as f:
            data = json.load(f)
        
        assert "matches" in data
        assert "matching_statistics" in data
        assert data["matching_algorithm"] == "test"
    
    def test_save_transformation_metadata(self):
        """Test saving coordinate transformation metadata."""
        output_dir = self.persistence.create_output_directory(self.test_image_path)
        
        metadata = {
            "transformation_successful": True,
            "input_coordinate_system": "image_pixels",
            "output_coordinate_system": "dxf_mm"
        }
        
        saved_path = self.persistence.save_transformation_metadata(metadata, output_dir)
        
        expected_path = output_dir / "coordinate_transformation.json"
        assert saved_path == expected_path
        assert saved_path.exists()
        
        # Check content
        with open(saved_path, 'r') as f:
            data = json.load(f)
        
        assert data["transformation_successful"] is True
        assert "save_timestamp" in data
    
    def test_output_directory_derived_from_image_path(self):
        """Test that output directory is correctly derived from image path."""
        # Test with different image names
        test_cases = [
            Path("product_A_front.jpg"),
            Path("test_image_123.png"),
            Path("circular_top_sample.tiff")
        ]
        
        for image_path in test_cases:
            output_dir = self.persistence.create_output_directory(image_path)
            expected_dir = self.temp_dir / image_path.stem / "phase_2"
            assert output_dir == expected_dir
    
    def test_no_hardcoded_product_names_in_paths(self):
        """Test that no product names are hardcoded in output paths."""
        # This ensures output paths are derived dynamically from input
        
        different_products = [
            Path("box_front_product_1.jpg"),
            Path("circular_rear_product_2.jpg"),
            Path("unknown_product_xyz.png")
        ]
        
        for product_image in different_products:
            output_dir = self.persistence.create_output_directory(product_image)
            
            # Directory should be based on image stem, not hardcoded names
            assert product_image.stem in str(output_dir)
            assert "product_1" not in str(output_dir) or product_image.stem == "box_front_product_1"
            assert "product_2" not in str(output_dir) or product_image.stem == "circular_rear_product_2"


class TestPhase2JSONEncoder:
    """Test JSON encoding for Phase 2 data structures."""
    
    def test_encode_point2d(self):
        """Test Point2D encoding."""
        point = Point2D(123.456, 789.012)
        encoded = Phase2JSONEncoder.encode_point2d(point)
        
        assert encoded == {"x": 123.456, "y": 789.012}
    
    def test_encode_enum(self):
        """Test enum encoding."""
        feature_type = FeatureType.CIRCLE
        encoded = Phase2JSONEncoder.encode_enum(feature_type)
        
        assert encoded == "circle"
        
        # Test None enum
        encoded_none = Phase2JSONEncoder.encode_enum(None)
        assert encoded_none is None
    
    def test_encode_actual_feature(self):
        """Test ActualFeature encoding."""
        feature = ActualFeature(
            feature_id="test_feature",
            feature_type=FeatureType.THROUGH_HOLE,
            confidence=0.85,
            center=Point2D(100.0, 200.0),
            radius=25.5,
            detection_method=DetectionMethod.CONTOUR_ANALYSIS,
            detection_evidence={"test": "data"},
            quality_metrics={"score": 0.9}
        )
        
        encoded = Phase2JSONEncoder.encode_actual_feature(feature)
        
        assert encoded["feature_id"] == "test_feature"
        assert encoded["feature_type"] == "through_hole"
        assert encoded["confidence"] == 0.85
        assert encoded["center"] == {"x": 100.0, "y": 200.0}
        assert encoded["radius"] == 25.5
        assert encoded["detection_method"] == "contour_analysis"
        assert encoded["detection_evidence"] == {"test": "data"}
        assert encoded["quality_metrics"] == {"score": 0.9}
    
    def test_encode_actual_feature_set(self):
        """Test ActualFeatureSet encoding."""
        feature = ActualFeature(
            feature_id="test",
            feature_type=FeatureType.CIRCLE,
            confidence=0.9,
            center=Point2D(50.0, 50.0),
            radius=10.0
        )
        
        feature_set = ActualFeatureSet(
            source_image_path=Path("test.jpg"),
            features=[feature],
            detection_timestamp="2024-01-01",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=5, contours_after_filtering=3,
                circles_detected=1, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=1, average_confidence=0.9,
                detection_time_seconds=1.0, preprocessing_time_seconds=0.2
            ),
            configuration_snapshot={},
            image_dimensions=(200, 200),
            preprocessing_applied=["resize"]
        )
        
        encoded = Phase2JSONEncoder.encode_actual_feature_set(feature_set)
        
        assert encoded["source_image_path"] == "test.jpg"
        assert len(encoded["features"]) == 1
        assert encoded["coordinate_system"] == "image_pixels"
        assert encoded["total_feature_count"] == 1
        assert encoded["circle_count"] == 1
        assert encoded["preprocessing_applied"] == ["resize"]
    
    def test_serialization_round_trip_compatibility(self):
        """Test that serialization produces valid, loadable JSON."""
        feature_set = ActualFeatureSet(
            source_image_path=Path("round_trip_test.png"),
            features=[
                ActualFeature(
                    feature_id="rt_feature",
                    feature_type=FeatureType.RECTANGULAR_HOLE,
                    confidence=0.75,
                    center=Point2D(150.0, 250.0),
                    width=30.0,
                    height=40.0
                )
            ],
            detection_timestamp="2024-01-01",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=1, contours_after_filtering=1,
                circles_detected=0, through_holes_detected=0, rectangular_holes_detected=1,
                total_features_detected=1, average_confidence=0.75,
                detection_time_seconds=0.5, preprocessing_time_seconds=0.1
            ),
            configuration_snapshot={},
            image_dimensions=(300, 500),
            preprocessing_applied=[]
        )
        
        # Encode to JSON
        encoded = Phase2JSONEncoder.encode_actual_feature_set(feature_set)
        
        # Convert to JSON string and back
        json_str = json.dumps(encoded)
        loaded_data = json.loads(json_str)
        
        # Verify key data preserved
        assert loaded_data["source_image_path"] == "round_trip_test.png"
        assert len(loaded_data["features"]) == 1
        assert loaded_data["features"][0]["feature_id"] == "rt_feature"
        assert loaded_data["features"][0]["width"] == 30.0
        assert loaded_data["features"][0]["height"] == 40.0


class TestSaveLoadJSON:
    """Test save_json and load_json functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_actual_feature_set(self):
        """Test saving and loading ActualFeatureSet."""
        feature_set = ActualFeatureSet(
            source_image_path=Path("save_load_test.jpg"),
            features=[],
            detection_timestamp="2024-01-01",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=0, contours_after_filtering=0,
                circles_detected=0, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=0, average_confidence=0.0,
                detection_time_seconds=0.1, preprocessing_time_seconds=0.05
            ),
            configuration_snapshot={},
            image_dimensions=(100, 100),
            preprocessing_applied=[]
        )
        
        output_path = self.temp_dir / "test_feature_set.json"
        
        # Save
        save_json(feature_set, output_path)
        assert output_path.exists()
        
        # Load and verify
        loaded_data = load_json(output_path)
        assert loaded_data["source_image_path"] == "save_load_test.jpg"
        assert loaded_data["total_feature_count"] == 0
        assert "_metadata" in loaded_data
        assert loaded_data["_metadata"]["data_type"] == "ActualFeatureSet"
    
    def test_save_json_creates_directories(self):
        """Test that save_json creates necessary directories."""
        nested_path = self.temp_dir / "deep" / "nested" / "path" / "test.json"
        
        feature_set = ActualFeatureSet(
            source_image_path=Path("test.jpg"),
            features=[],
            detection_timestamp="2024-01-01",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=0, contours_after_filtering=0,
                circles_detected=0, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=0, average_confidence=0.0,
                detection_time_seconds=0.1, preprocessing_time_seconds=0.05
            ),
            configuration_snapshot={},
            image_dimensions=(100, 100),
            preprocessing_applied=[]
        )
        
        # Should create directories and save file
        save_json(feature_set, nested_path)
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    def test_save_json_metadata_added(self):
        """Test that save_json adds appropriate metadata."""
        feature_set = ActualFeatureSet(
            source_image_path=Path("metadata_test.jpg"),
            features=[],
            detection_timestamp="2024-01-01",
            detection_statistics=ActualDetectionStatistics(
                total_contours_found=0, contours_after_filtering=0,
                circles_detected=0, through_holes_detected=0, rectangular_holes_detected=0,
                total_features_detected=0, average_confidence=0.0,
                detection_time_seconds=0.1, preprocessing_time_seconds=0.05
            ),
            configuration_snapshot={},
            image_dimensions=(100, 100),
            preprocessing_applied=[]
        )
        
        output_path = self.temp_dir / "metadata_test.json"
        save_json(feature_set, output_path)
        
        loaded_data = load_json(output_path)
        metadata = loaded_data["_metadata"]
        
        assert "serialization_timestamp" in metadata
        assert metadata["serializer_version"] == "1.0.0"
        assert metadata["data_type"] == "ActualFeatureSet"
        assert metadata["encoder_method"] == "encode_actual_feature_set"