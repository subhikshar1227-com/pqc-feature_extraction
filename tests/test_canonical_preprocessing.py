"""
Tests for Canonical Phase 2 Preprocessing

Validates the single authoritative preprocessing implementation.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json

from feature_inspection.actual.canonical_preprocessor import CanonicalPreprocessor, PreprocessingResult


class TestCanonicalPreprocessor:
    """Test the canonical Phase 2 preprocessor."""
    
    def setup_method(self):
        """Setup for each test."""
        self.preprocessor = CanonicalPreprocessor()
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes with centralized configuration."""
        assert self.preprocessor.max_resolution > 0
        assert self.preprocessor.gradient_threshold > 0
        assert self.preprocessor.canny_low > 0
        assert self.preprocessor.canny_high > self.preprocessor.canny_low
        assert 0 < self.preprocessor.border_margin_fraction < 1
        assert len(self.preprocessor.blur_kernel) == 2
        
        # Check configuration snapshot exists
        assert isinstance(self.preprocessor._config_snapshot, dict)
        assert "max_resolution" in self.preprocessor._config_snapshot
        assert "gradient_threshold" in self.preprocessor._config_snapshot
    
    def test_nonexistent_image_file(self):
        """Test handling of non-existent image file."""
        fake_path = Path("nonexistent.jpg")
        
        with pytest.raises(FileNotFoundError):
            self.preprocessor.preprocess_image(fake_path)
    
    def test_coordinate_mapping_identity(self):
        """Test coordinate mapping when no resizing occurs."""
        result = PreprocessingResult(
            source_image_path=Path("test.jpg"),
            original_image=np.zeros((100, 200, 3), dtype=np.uint8),
            processed_image=np.zeros((100, 200), dtype=np.uint8),
            product_mask=np.zeros((100, 200), dtype=np.uint8),
            isolated_product_image=np.zeros((100, 200), dtype=np.uint8),
            raw_internal_geometry_edges=np.zeros((100, 200), dtype=np.uint8),
            internal_geometry_edges=np.zeros((100, 200), dtype=np.uint8),
            outer_boundary_edges=np.zeros((100, 200), dtype=np.uint8),
            edge_representation=np.zeros((100, 200), dtype=np.uint8),
            original_dimensions=(200, 100),  # (width, height)
            processed_dimensions=(200, 100),
            scale_factor=1.0,
            roi_offset=(0, 0),
            preprocessing_successful=True,
            isolation_successful=True,
            
            # Updated mask quality diagnostics
            product_area_pixels=1000,
            product_area_fraction=0.5,
            mask_bounding_box=(0, 0, 200, 100),
            mask_bounding_box_area_fraction=1.0,
            foreground_component_count=1,
            largest_component_fraction=1.0,
            mask_contour_area=1000.0,
            mask_contour_perimeter=600.0,
            mask_extent=0.5,
            mask_solidity=1.0,
            border_touching_foreground=False,
            significant_foreground_regions=1,
            
            # Silhouette validation results (required)
            silhouette_validation_result="PASS",
            external_gradient_strength=10.0,
            boundary_gradient_consistency=1.0,
            suspicious_boundary_fraction=0.1,
            boundary_gradient_strength=15.0,
            
            # Updated edge extraction metrics
            raw_internal_edge_density=0.08,
            filtered_internal_edge_density=0.05,
            internal_edge_retention_ratio=0.625,
            internal_edge_reduction_ratio=0.375,
            outer_boundary_density=0.02,
            final_edge_density=0.1,
            
            processing_steps=["test"],
            configuration_snapshot={}
        )
        
        # Test identity transformation
        orig_x, orig_y = result.to_original_coordinates(50, 75)
        assert orig_x == 50.0
        assert orig_y == 75.0
        
        proc_x, proc_y = result.to_processed_coordinates(50, 75)
        assert proc_x == 50.0
        assert proc_y == 75.0
    
    def test_coordinate_mapping_with_scaling(self):
        """Test coordinate mapping when resizing occurs."""
        result = PreprocessingResult(
            source_image_path=Path("test.jpg"),
            original_image=np.zeros((200, 400, 3), dtype=np.uint8),
            processed_image=np.zeros((100, 200), dtype=np.uint8),
            product_mask=np.zeros((100, 200), dtype=np.uint8),
            isolated_product_image=np.zeros((100, 200), dtype=np.uint8),
            raw_internal_geometry_edges=np.zeros((100, 200), dtype=np.uint8),
            internal_geometry_edges=np.zeros((100, 200), dtype=np.uint8),
            outer_boundary_edges=np.zeros((100, 200), dtype=np.uint8),
            edge_representation=np.zeros((100, 200), dtype=np.uint8),
            original_dimensions=(400, 200),  # (width, height)
            processed_dimensions=(200, 100),
            scale_factor=0.5,  # Halved resolution
            roi_offset=(0, 0),
            preprocessing_successful=True,
            isolation_successful=True,
            
            # Updated mask quality diagnostics
            product_area_pixels=1000,
            product_area_fraction=0.5,
            mask_bounding_box=(0, 0, 200, 100),
            mask_bounding_box_area_fraction=1.0,
            foreground_component_count=1,
            largest_component_fraction=1.0,
            mask_contour_area=1000.0,
            mask_contour_perimeter=600.0,
            mask_extent=0.5,
            mask_solidity=1.0,
            border_touching_foreground=False,
            significant_foreground_regions=1,
            
            # Silhouette validation results (required)
            silhouette_validation_result="PASS",
            external_gradient_strength=10.0,
            boundary_gradient_consistency=1.0,
            suspicious_boundary_fraction=0.1,
            boundary_gradient_strength=15.0,
            
            # Updated edge extraction metrics
            raw_internal_edge_density=0.08,
            filtered_internal_edge_density=0.05,
            internal_edge_retention_ratio=0.625,
            internal_edge_reduction_ratio=0.375,
            outer_boundary_density=0.02,
            final_edge_density=0.1,
            
            processing_steps=["test"],
            configuration_snapshot={}
        )
        
        # Test scaling transformation
        orig_x, orig_y = result.to_original_coordinates(50, 25)
        assert orig_x == 100.0  # 50 / 0.5 = 100
        assert orig_y == 50.0   # 25 / 0.5 = 50
        
        proc_x, proc_y = result.to_processed_coordinates(100, 50)
        assert proc_x == 50.0   # 100 * 0.5 = 50
        assert proc_y == 25.0   # 50 * 0.5 = 25
    
    def test_coordinate_mapping_with_roi_offset(self):
        """Test coordinate mapping with ROI offset."""
        result = PreprocessingResult(
            source_image_path=Path("test.jpg"),
            original_image=np.zeros((100, 200, 3), dtype=np.uint8),
            processed_image=np.zeros((100, 200), dtype=np.uint8),
            product_mask=np.zeros((100, 200), dtype=np.uint8),
            isolated_product_image=np.zeros((100, 200), dtype=np.uint8),
            raw_internal_geometry_edges=np.zeros((100, 200), dtype=np.uint8),
            internal_geometry_edges=np.zeros((100, 200), dtype=np.uint8),
            outer_boundary_edges=np.zeros((100, 200), dtype=np.uint8),
            edge_representation=np.zeros((100, 200), dtype=np.uint8),
            original_dimensions=(200, 100),
            processed_dimensions=(200, 100),
            scale_factor=1.0,
            roi_offset=(10, 5),  # ROI cropping offset
            preprocessing_successful=True,
            isolation_successful=True,
            
            # Updated mask quality diagnostics
            product_area_pixels=1000,
            product_area_fraction=0.5,
            mask_bounding_box=(0, 0, 200, 100),
            mask_bounding_box_area_fraction=1.0,
            foreground_component_count=1,
            largest_component_fraction=1.0,
            mask_contour_area=1000.0,
            mask_contour_perimeter=600.0,
            mask_extent=0.5,
            mask_solidity=1.0,
            border_touching_foreground=False,
            significant_foreground_regions=1,
            
            # Silhouette validation results (required)
            silhouette_validation_result="PASS",
            external_gradient_strength=10.0,
            boundary_gradient_consistency=1.0,
            suspicious_boundary_fraction=0.1,
            boundary_gradient_strength=15.0,
            
            # Updated edge extraction metrics
            raw_internal_edge_density=0.08,
            filtered_internal_edge_density=0.05,
            internal_edge_retention_ratio=0.625,
            internal_edge_reduction_ratio=0.375,
            outer_boundary_density=0.02,
            final_edge_density=0.1,
            
            processing_steps=["test"],
            configuration_snapshot={}
        )
        
        # Test offset transformation
        orig_x, orig_y = result.to_original_coordinates(50, 25)
        assert orig_x == 60.0   # (50 + 10) / 1.0 = 60
        assert orig_y == 30.0   # (25 + 5) / 1.0 = 30
        
        proc_x, proc_y = result.to_processed_coordinates(60, 30)
        assert proc_x == 50.0   # 60 * 1.0 - 10 = 50
        assert proc_y == 25.0   # 30 * 1.0 - 5 = 25
    
    def test_failed_preprocessing_result(self):
        """Test creation of failed preprocessing result."""
        fake_path = Path("fake.jpg")
        failed_result = self.preprocessor._create_failed_result(fake_path, "Test error")
        
        assert not failed_result.preprocessing_successful
        assert not failed_result.isolation_successful
        assert failed_result.product_area_pixels == 0
        assert failed_result.product_area_fraction == 0.0
        
        # Updated failed result checks
        assert failed_result.raw_internal_edge_density == 0.0
        assert failed_result.filtered_internal_edge_density == 0.0
        assert failed_result.internal_edge_retention_ratio == 1.0
        assert failed_result.internal_edge_reduction_ratio == 0.0
        assert failed_result.outer_boundary_density == 0.0
        assert failed_result.final_edge_density == 0.0
        
        assert failed_result.processing_steps == ["failed"]
        assert "error" in failed_result.configuration_snapshot
        
        # Should still have valid coordinate mapping
        orig_x, orig_y = failed_result.to_original_coordinates(10, 20)
        assert isinstance(orig_x, float)
        assert isinstance(orig_y, float)
    
    def test_configuration_centralization(self):
        """Test that all configuration comes from centralized config."""
        from feature_inspection.config import (
            PREPROCESSING_MAX_RESOLUTION, PREPROCESSING_GRADIENT_THRESHOLD,
            PREPROCESSING_CANNY_OUTER_LOW, PREPROCESSING_CANNY_OUTER_HIGH,
            PREPROCESSING_BORDER_MARGIN_FRACTION, PREPROCESSING_MIN_COMPONENT_AREA_PX
        )
        
        # Verify preprocessor uses centralized config
        assert self.preprocessor.max_resolution == PREPROCESSING_MAX_RESOLUTION
        assert self.preprocessor.gradient_threshold == PREPROCESSING_GRADIENT_THRESHOLD
        assert self.preprocessor.canny_low == PREPROCESSING_CANNY_OUTER_LOW
        assert self.preprocessor.canny_high == PREPROCESSING_CANNY_OUTER_HIGH
        assert self.preprocessor.border_margin_fraction == PREPROCESSING_BORDER_MARGIN_FRACTION
        assert self.preprocessor.min_component_area == PREPROCESSING_MIN_COMPONENT_AREA_PX
    
    def test_no_hardcoded_values(self):
        """Test that no algorithmic values are hardcoded in preprocessor."""
        # Configuration snapshot should not contain magic numbers
        config = self.preprocessor._config_snapshot
        
        # All values should be positive and reasonable
        assert config["max_resolution"] > 100
        assert config["gradient_threshold"] > 0
        assert config["canny_low"] > 0
        assert config["canny_high"] > config["canny_low"]
        assert 0 < config["border_margin_fraction"] < 1
        
        # Should not contain obvious magic numbers
        magic_numbers = [999, 9999, 12345, 42, 100, 255]  # Common hardcoded values
        for key, value in config.items():
            if isinstance(value, (int, float)):
                assert value not in magic_numbers, f"Suspicious magic number {value} in config key {key}"
    
    def test_preprocessing_deterministic(self):
        """Test that preprocessing is deterministic for same inputs."""
        # Create a simple test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 128  # Gray image
            # Add a simple shape for product isolation
            cv2.rectangle(test_image, (50, 25), (150, 75), (200, 200, 200), -1)
            cv2.imwrite(tmp.name, test_image)
            tmp_path = Path(tmp.name)
        
        try:
            # Run preprocessing twice
            result1 = self.preprocessor.preprocess_image(tmp_path)
            result2 = self.preprocessor.preprocess_image(tmp_path)
            
            # Results should be identical
            assert result1.preprocessing_successful == result2.preprocessing_successful
            assert result1.isolation_successful == result2.isolation_successful
            assert result1.original_dimensions == result2.original_dimensions
            assert result1.processed_dimensions == result2.processed_dimensions
            assert result1.scale_factor == result2.scale_factor
            assert result1.product_area_pixels == result2.product_area_pixels
            
            # Updated edge metric comparisons
            assert result1.raw_internal_edge_density == result2.raw_internal_edge_density
            assert result1.filtered_internal_edge_density == result2.filtered_internal_edge_density
            assert result1.internal_edge_retention_ratio == result2.internal_edge_retention_ratio
            assert result1.internal_edge_reduction_ratio == result2.internal_edge_reduction_ratio
            
            # Arrays should be identical
            np.testing.assert_array_equal(result1.edge_representation, result2.edge_representation)
            np.testing.assert_array_equal(result1.product_mask, result2.product_mask)
            np.testing.assert_array_equal(result1.raw_internal_geometry_edges, result2.raw_internal_geometry_edges)
            np.testing.assert_array_equal(result1.internal_geometry_edges, result2.internal_geometry_edges)
            np.testing.assert_array_equal(result1.outer_boundary_edges, result2.outer_boundary_edges)
            
        finally:
            tmp_path.unlink()  # Clean up
    
    def test_save_preprocessing_outputs(self):
        """Test saving preprocessing outputs."""
        # Create a test result
        test_image = np.ones((50, 100, 3), dtype=np.uint8) * 100
        result = PreprocessingResult(
            source_image_path=Path("test.jpg"),
            original_image=test_image,
            processed_image=np.ones((50, 100), dtype=np.uint8) * 128,
            product_mask=np.ones((50, 100), dtype=np.uint8) * 255,
            isolated_product_image=np.ones((50, 100), dtype=np.uint8) * 128,
            raw_internal_geometry_edges=np.ones((50, 100), dtype=np.uint8) * 80,
            internal_geometry_edges=np.ones((50, 100), dtype=np.uint8) * 100,
            outer_boundary_edges=np.ones((50, 100), dtype=np.uint8) * 150,
            edge_representation=np.ones((50, 100), dtype=np.uint8) * 200,
            original_dimensions=(100, 50),
            processed_dimensions=(100, 50),
            scale_factor=1.0,
            roi_offset=(0, 0),
            preprocessing_successful=True,
            isolation_successful=True,
            
            # Updated comprehensive diagnostics
            product_area_pixels=5000,
            product_area_fraction=1.0,
            mask_bounding_box=(0, 0, 100, 50),
            mask_bounding_box_area_fraction=1.0,
            foreground_component_count=1,
            largest_component_fraction=1.0,
            mask_contour_area=5000.0,
            mask_contour_perimeter=300.0,
            mask_extent=1.0,
            mask_solidity=1.0,
            border_touching_foreground=False,
            significant_foreground_regions=1,
            
            # Silhouette validation results (required)
            silhouette_validation_result="PASS",
            external_gradient_strength=10.0,
            boundary_gradient_consistency=1.0,
            suspicious_boundary_fraction=0.1,
            boundary_gradient_strength=15.0,
            
            # Updated edge extraction metrics
            raw_internal_edge_density=0.3,
            filtered_internal_edge_density=0.2,
            internal_edge_retention_ratio=0.667,
            internal_edge_reduction_ratio=0.333,
            outer_boundary_density=0.3,
            final_edge_density=0.5,
            
            processing_steps=["test_step"],
            configuration_snapshot={"test": "value"}
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            saved_files = self.preprocessor.save_preprocessing_outputs(result, output_dir)
            
            # Check all expected files were saved
            expected_files = ["original", "mask", "mask_overlay", "isolated_product", "internal_edges", 
                            "raw_internal_edges", "outer_boundary", "final_edges", "processed", "montage", "metadata"]
            assert set(saved_files.keys()) == set(expected_files)
            
            # Check files exist and are readable
            for file_type, file_path in saved_files.items():
                assert file_path.exists()
                
                if file_type == "metadata":
                    # Check JSON metadata
                    with open(file_path) as f:
                        metadata = json.load(f)
                    assert "preprocessing_successful" in metadata
                    assert "original_dimensions" in metadata
                    assert "configuration" in metadata
                else:
                    # Check image files
                    loaded_image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                    assert loaded_image is not None


class TestPreprocessingResult:
    """Test the PreprocessingResult dataclass."""
    
    def test_preprocessing_result_creation(self):
        """Test creating a preprocessing result."""
        result = PreprocessingResult(
            source_image_path=Path("test.jpg"),
            original_image=np.zeros((100, 200, 3), dtype=np.uint8),
            processed_image=np.zeros((100, 200), dtype=np.uint8),
            product_mask=np.zeros((100, 200), dtype=np.uint8),
            isolated_product_image=np.zeros((100, 200), dtype=np.uint8),
            raw_internal_geometry_edges=np.zeros((100, 200), dtype=np.uint8),
            internal_geometry_edges=np.zeros((100, 200), dtype=np.uint8),
            outer_boundary_edges=np.zeros((100, 200), dtype=np.uint8),
            edge_representation=np.zeros((100, 200), dtype=np.uint8),
            original_dimensions=(200, 100),
            processed_dimensions=(200, 100),
            scale_factor=1.0,
            roi_offset=(0, 0),
            preprocessing_successful=True,
            isolation_successful=True,
            
            # Updated comprehensive mask quality diagnostics
            product_area_pixels=1000,
            product_area_fraction=0.5,
            mask_bounding_box=(0, 0, 200, 100),
            mask_bounding_box_area_fraction=1.0,
            foreground_component_count=1,
            largest_component_fraction=1.0,
            mask_contour_area=1000.0,
            mask_contour_perimeter=600.0,
            mask_extent=0.5,
            mask_solidity=1.0,
            border_touching_foreground=False,
            significant_foreground_regions=1,
            
            # Silhouette validation results (required)
            silhouette_validation_result="PASS",
            external_gradient_strength=10.0,
            boundary_gradient_consistency=1.0,
            suspicious_boundary_fraction=0.1,
            boundary_gradient_strength=15.0,
            
            # Updated edge extraction metrics
            raw_internal_edge_density=0.08,
            filtered_internal_edge_density=0.05,
            internal_edge_retention_ratio=0.625,
            internal_edge_reduction_ratio=0.375,
            outer_boundary_density=0.02,
            final_edge_density=0.1,
            
            processing_steps=["step1", "step2"],
            configuration_snapshot={"param": "value"}
        )
        
        assert result.source_image_path == Path("test.jpg")
        assert result.preprocessing_successful
        assert result.isolation_successful
        assert result.original_dimensions == (200, 100)
        assert result.processed_dimensions == (200, 100)
        assert result.scale_factor == 1.0
        assert result.product_area_pixels == 1000
        assert result.product_area_fraction == 0.5
        assert result.raw_internal_edge_density == 0.08
        assert result.filtered_internal_edge_density == 0.05
        assert result.internal_edge_retention_ratio == 0.625
        assert result.internal_edge_reduction_ratio == 0.375
        assert result.outer_boundary_density == 0.02
        assert result.final_edge_density == 0.1
        assert len(result.processing_steps) == 2
        assert result.configuration_snapshot["param"] == "value"


class TestCanonicalPreprocessorRefinements:
    """Test the refinements to the canonical preprocessor."""
    
    def setup_method(self):
        """Setup for each test."""
        self.preprocessor = CanonicalPreprocessor()

    def test_separate_edge_representations(self):
        """Test that preprocessing produces separate edge representations."""
        # Create a simple test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 128  # Gray image
            # Add a simple shape for product isolation
            cv2.rectangle(test_image, (50, 25), (150, 75), (200, 200, 200), -1)
            cv2.imwrite(tmp.name, test_image)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.preprocessor.preprocess_image(tmp_path)
            
            # Should have separate representations
            assert result.raw_internal_geometry_edges is not None
            assert result.internal_geometry_edges is not None
            assert result.outer_boundary_edges is not None
            assert result.edge_representation is not None
            assert result.isolated_product_image is not None
            
            # Internal and outer boundary should be different arrays
            assert not np.array_equal(result.internal_geometry_edges, result.outer_boundary_edges)
            
            # Edge densities should be calculated
            assert 0 <= result.raw_internal_edge_density <= 1
            assert 0 <= result.filtered_internal_edge_density <= 1
            assert 0 <= result.internal_edge_retention_ratio <= 1
            assert 0 <= result.internal_edge_reduction_ratio <= 1
            assert 0 <= result.outer_boundary_density <= 1
            assert 0 <= result.final_edge_density <= 1
            
        finally:
            tmp_path.unlink()  # Clean up

    def test_texture_suppression_parameters(self):
        """Test that texture suppression parameters are configurable."""
        # Check that new configuration parameters are loaded
        from feature_inspection.config import (
            PREPROCESSING_INTERNAL_EDGE_COMBINATION_WEIGHT,
            PREPROCESSING_OUTER_BOUNDARY_COMBINATION_WEIGHT,
            PREPROCESSING_NOISE_REDUCTION_KERNEL_SIZE,
            PREPROCESSING_MIN_EDGE_STRENGTH,
            PREPROCESSING_TEXTURE_SUPPRESSION_THRESHOLD
        )
        
        # Verify parameters exist and are reasonable
        assert 0 <= PREPROCESSING_INTERNAL_EDGE_COMBINATION_WEIGHT <= 1
        assert 0 <= PREPROCESSING_OUTER_BOUNDARY_COMBINATION_WEIGHT <= 1
        assert PREPROCESSING_NOISE_REDUCTION_KERNEL_SIZE > 0
        assert PREPROCESSING_MIN_EDGE_STRENGTH > 0
        assert PREPROCESSING_TEXTURE_SUPPRESSION_THRESHOLD > 0
        
        # Verify preprocessor has access to these parameters
        config = self.preprocessor._config_snapshot
        assert "internal_edge_weight" in config
        assert "outer_boundary_weight" in config
        assert "noise_reduction_kernel" in config
        assert "min_edge_strength" in config
        assert "texture_suppression_threshold" in config

    def test_preprocessing_step_documentation(self):
        """Test that preprocessing steps are properly documented."""
        # Create a simple test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image = np.ones((50, 100, 3), dtype=np.uint8) * 128
            cv2.rectangle(test_image, (20, 10), (80, 40), (180, 180, 180), -1)
            cv2.imwrite(tmp.name, test_image)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.preprocessor.preprocess_image(tmp_path)
            
            # Should have detailed processing steps
            assert len(result.processing_steps) >= 10  # Should have many steps
            
            # Check for key steps
            steps_text = " ".join(result.processing_steps)
            assert "noise_reduction" in steps_text
            assert "texture_suppression" in steps_text or "texture_aware" in steps_text
            assert "geometry" in steps_text or "internal_edges" in steps_text
            assert "boundary" in steps_text
            assert "combination" in steps_text
            
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])