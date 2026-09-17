"""
DXF Resolution Implementation

Provides the core functionality to resolve blueprint identities to their
corresponding DXF file paths.
"""

import logging
from pathlib import Path
from typing import Dict

from .config import BLUEPRINT_DXF_MAPPING, DXF_DIR

logger = logging.getLogger(__name__)


class DXFResolutionError(Exception):
    """Raised when a blueprint cannot be resolved to a valid DXF file."""
    pass


def resolve_dxf(blueprint_name: str) -> Path:
    """
    Resolve a blueprint identity to its corresponding DXF file path.
    
    Args:
        blueprint_name: The blueprint identity (e.g., "circular_top", "box_front")
        
    Returns:
        Path: Absolute path to the corresponding DXF file
        
    Raises:
        DXFResolutionError: If the blueprint is unknown or DXF file doesn't exist
        
    Example:
        >>> dxf_path = resolve_dxf("circular_top")
        >>> print(dxf_path)  # /path/to/project/dxf/c_tp.dxf
    """
    if not isinstance(blueprint_name, str):
        raise DXFResolutionError(
            f"Blueprint name must be a string, got {type(blueprint_name).__name__}: {blueprint_name}"
        )
    
    if not blueprint_name.strip():
        raise DXFResolutionError("Blueprint name cannot be empty or whitespace-only")
    
    # Look up the blueprint in our mapping
    dxf_filename = BLUEPRINT_DXF_MAPPING.get(blueprint_name)
    if dxf_filename is None:
        known_blueprints = list(BLUEPRINT_DXF_MAPPING.keys())
        raise DXFResolutionError(
            f"Unknown blueprint '{blueprint_name}'. "
            f"Known blueprints: {known_blueprints}"
        )
    
    # Construct the full path
    dxf_path = DXF_DIR / dxf_filename
    
    # Verify the file exists
    if not dxf_path.exists():
        raise DXFResolutionError(
            f"DXF file for blueprint '{blueprint_name}' not found. "
            f"Expected: {dxf_path} (mapped from filename: '{dxf_filename}')"
        )
    
    logger.debug(f"Resolved blueprint '{blueprint_name}' → {dxf_path}")
    return dxf_path.resolve()  # Return absolute path


def get_blueprint_dxf_mapping() -> Dict[str, str]:
    """
    Get a copy of the current blueprint→DXF filename mapping.
    
    Returns:
        Dict[str, str]: Mapping from blueprint names to DXF filenames
        
    Note:
        This returns filenames only, not full paths. Use resolve_dxf() to get paths.
    """
    return BLUEPRINT_DXF_MAPPING.copy()


def validate_all_dxf_files() -> None:
    """
    Validate that all DXF files referenced in the mapping exist and are accessible.
    
    Raises:
        DXFResolutionError: If any DXF files are missing or inaccessible
    """
    missing_files = []
    inaccessible_files = []
    
    for blueprint_name, dxf_filename in BLUEPRINT_DXF_MAPPING.items():
        try:
            dxf_path = resolve_dxf(blueprint_name)
            # Try to access the file (basic readability check)
            if not dxf_path.is_file():
                inaccessible_files.append((blueprint_name, dxf_filename, str(dxf_path)))
        except DXFResolutionError as e:
            if "not found" in str(e):
                missing_files.append((blueprint_name, dxf_filename))
            else:
                raise  # Re-raise other types of resolution errors
    
    errors = []
    if missing_files:
        errors.append("Missing DXF files:")
        for blueprint_name, dxf_filename in missing_files:
            errors.append(f"  - Blueprint '{blueprint_name}' → '{dxf_filename}'")
    
    if inaccessible_files:
        errors.append("Inaccessible DXF files:")
        for blueprint_name, dxf_filename, full_path in inaccessible_files:
            errors.append(f"  - Blueprint '{blueprint_name}' → {full_path}")
    
    if errors:
        raise DXFResolutionError("\n".join(errors))