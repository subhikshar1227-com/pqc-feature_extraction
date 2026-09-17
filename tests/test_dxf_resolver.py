"""
Tests for DXF Resolution Module

Validates that blueprint identities are correctly mapped to DXF files
without modifying the existing alignment system.
"""

import pytest
from pathlib import Path

from dxf_resolver import resolve_dxf, get_blueprint_dxf_mapping, DXFResolutionError


class TestDXFResolver:
    """Test the DXF resolution functionality."""
    
    def test_resolve_known_blueprints(self):
        """Test that all known blueprints resolve to valid DXF files."""
        expected_mappings = {
            "circular_top": "c_tp.dxf",
            "circular_rear": "c-bp.dxf",
            "box_front": "cad_box_front.dxf", 
            "box_rear": "cad_box_rear (1).dxf",
        }
        
        for blueprint_name, expected_filename in expected_mappings.items():
            dxf_path = resolve_dxf(blueprint_name)
            
            # Should return a Path object
            assert isinstance(dxf_path, Path)
            
            # Should be an absolute path
            assert dxf_path.is_absolute()
            
            # Should point to an existing file
            assert dxf_path.exists(), f"DXF file not found: {dxf_path}"
            assert dxf_path.is_file(), f"DXF path is not a file: {dxf_path}"
            
            # Should have the expected filename
            assert dxf_path.name == expected_filename
            
            # Should be in the dxf directory
            assert dxf_path.parent.name == "dxf"
    
    def test_unknown_blueprint_raises_error(self):
        """Test that unknown blueprint names raise DXFResolutionError."""
        unknown_blueprints = [
            "unknown_blueprint",
            "nonexistent",
            "box_left",  # Similar to valid names but not in mapping
            "circular_bottom",
        ]
        
        for blueprint_name in unknown_blueprints:
            with pytest.raises(DXFResolutionError) as exc_info:
                resolve_dxf(blueprint_name)
            
            # Error should mention the unknown blueprint
            assert blueprint_name in str(exc_info.value)
            # Error should list known blueprints for guidance
            assert "Known blueprints:" in str(exc_info.value)
    
    def test_invalid_input_types_raise_error(self):
        """Test that invalid input types raise DXFResolutionError."""
        invalid_inputs = [None, 123, [], {}, Path("test")]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(DXFResolutionError) as exc_info:
                resolve_dxf(invalid_input)
            
            assert "must be a string" in str(exc_info.value)
    
    def test_empty_blueprint_name_raises_error(self):
        """Test that empty or whitespace-only names raise DXFResolutionError."""
        empty_inputs = ["", "   ", "\t", "\n", "  \t\n  "]
        
        for empty_input in empty_inputs:
            with pytest.raises(DXFResolutionError) as exc_info:
                resolve_dxf(empty_input)
            
            assert "cannot be empty" in str(exc_info.value)
    
    def test_get_blueprint_mapping(self):
        """Test that get_blueprint_dxf_mapping returns expected mappings."""
        mapping = get_blueprint_dxf_mapping()
        
        # Should be a dictionary
        assert isinstance(mapping, dict)
        
        # Should contain expected blueprints
        expected_blueprints = {"circular_top", "circular_rear", "box_front", "box_rear"}
        assert set(mapping.keys()) == expected_blueprints
        
        # Should contain expected DXF filenames
        expected_dxf_files = {"c_tp.dxf", "c-bp.dxf", "cad_box_front.dxf", "cad_box_rear (1).dxf"}
        assert set(mapping.values()) == expected_dxf_files
        
        # Specific mappings should be correct
        assert mapping["circular_top"] == "c_tp.dxf"
        assert mapping["circular_rear"] == "c-bp.dxf"
        assert mapping["box_front"] == "cad_box_front.dxf"
        assert mapping["box_rear"] == "cad_box_rear (1).dxf"
    
    def test_mapping_is_copy(self):
        """Test that get_blueprint_dxf_mapping returns a copy, not the original."""
        mapping1 = get_blueprint_dxf_mapping()
        mapping2 = get_blueprint_dxf_mapping()
        
        # Should be separate objects
        assert mapping1 is not mapping2
        
        # But with same content
        assert mapping1 == mapping2
        
        # Modifying one shouldn't affect the other
        mapping1["test"] = "test.dxf"
        assert "test" not in mapping2
    
    def test_case_sensitivity(self):
        """Test that blueprint names are case-sensitive."""
        case_variations = [
            "Circular_top",
            "CIRCULAR_TOP", 
            "Box_Front",
            "BOX_FRONT",
        ]
        
        for variation in case_variations:
            with pytest.raises(DXFResolutionError):
                resolve_dxf(variation)


class TestIntegrationWithExistingProject:
    """Test integration points with the existing project structure."""
    
    def test_dxf_directory_exists(self):
        """Test that the DXF directory exists in the expected location."""
        # This should work from the test runner's perspective
        project_root = Path(__file__).parent.parent
        dxf_dir = project_root / "data" / "dxf"
        
        assert dxf_dir.exists(), f"DXF directory not found: {dxf_dir}"
        assert dxf_dir.is_dir(), f"DXF path is not a directory: {dxf_dir}"
    
    def test_all_referenced_dxf_files_exist(self):
        """Test that all DXF files referenced in the mapping actually exist."""
        from dxf_resolver.resolver import validate_all_dxf_files
        
        # Should not raise any exception if all files exist
        validate_all_dxf_files()
    
    def test_blueprint_names_match_existing_blueprints(self):
        """Test that blueprint names match the existing blueprint PNG files."""
        project_root = Path(__file__).parent.parent
        blueprints_dir = project_root / "blueprints"
        
        # Get actual blueprint files (without .png extension)
        if blueprints_dir.exists():
            actual_blueprint_names = {
                f.stem for f in blueprints_dir.glob("*.png")
            }
            
            # Get mapped blueprint names
            mapping = get_blueprint_dxf_mapping()
            mapped_blueprint_names = set(mapping.keys())
            
            # They should match exactly
            assert mapped_blueprint_names == actual_blueprint_names, (
                f"Blueprint name mismatch. "
                f"Mapped: {mapped_blueprint_names}, "
                f"Actual files: {actual_blueprint_names}"
            )