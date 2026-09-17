"""
DXF Resolution Configuration

Centralized mapping between blueprint identities and their corresponding DXF files.
This is the single source of truth for blueprint→DXF associations.

To add new products:
1. Add the blueprint→DXF mapping to BLUEPRINT_DXF_MAPPING
2. Ensure the DXF file exists in the dxf/ directory
3. No other code changes should be needed

The mapping is intentionally kept as a simple dictionary to avoid
product-specific conditional logic scattered throughout the codebase.
"""

from pathlib import Path

# Root project directory (parent of this module)
PROJECT_ROOT = Path(__file__).parent.parent

# DXF files directory
DXF_DIR = PROJECT_ROOT / "data" / "dxf"

# Blueprint name to DXF filename mapping
# This is the authoritative source for all blueprint→DXF associations
BLUEPRINT_DXF_MAPPING = {
    "circular_top": "c_tp.dxf",
    "circular_rear": "c-bp.dxf", 
    "box_front": "cad_box_front.dxf",
    "box_rear": "cad_box_rear (1).dxf",  # Note: actual filename has (1) suffix
}

# Validation: ensure all referenced DXF files exist
def _validate_dxf_files() -> None:
    """Validate that all DXF files referenced in the mapping actually exist."""
    missing_files = []
    for blueprint_name, dxf_filename in BLUEPRINT_DXF_MAPPING.items():
        dxf_path = DXF_DIR / dxf_filename
        if not dxf_path.exists():
            missing_files.append((blueprint_name, dxf_filename, str(dxf_path)))
    
    if missing_files:
        error_msg = "Missing DXF files referenced in blueprint mapping:\n"
        for blueprint_name, dxf_filename, full_path in missing_files:
            error_msg += f"  Blueprint '{blueprint_name}' → '{dxf_filename}' (expected at: {full_path})\n"
        raise FileNotFoundError(error_msg)

# Validate configuration on module import
try:
    _validate_dxf_files()
except FileNotFoundError as e:
    import warnings
    warnings.warn(f"DXF configuration validation failed: {e}", UserWarning)