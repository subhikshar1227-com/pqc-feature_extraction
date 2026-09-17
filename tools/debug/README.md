# Debug Tools

This directory contains debugging and analysis tools for the Phase 1 feature extraction system.

**IMPORTANT**: These files are for debugging and analysis only. They are NOT part of the production feature extraction pipeline.

## Production Code vs Debug Code

- **Production Code**: Located in `feature_extraction/`, `dxf_resolver/`, `cad_image_alignment/`
- **Debug/Analysis Code**: Located in this directory and the main project root (files starting with `debug_*`)

## Available Debug Tools

### Core Analysis Tools
- `../geometry_audit.py` - Generic geometric analysis of DXF files (no product expectations)
- `../detailed_feature_report.py` - Shows which DXF geometry produced each detected feature
- `../final_validation.py` - Validates results against acceptance targets

### Legacy Debug Scripts
- `../debug_*.py` files - Various debugging scripts for specific issues
- `../test_*.py` files (not in tests/) - Ad-hoc testing scripts

## Usage Guidelines

1. **For Understanding Results**: Use `geometry_audit.py` and `detailed_feature_report.py`
2. **For Validation**: Use `final_validation.py` 
3. **For Development**: Use the legacy debug scripts as needed

## Important Notes

- Debug tools may contain hardcoded coordinates, filenames, and expected results
- This is acceptable for debugging tools but forbidden in production code
- Production algorithms must be driven by geometry, not by these debug expectations
- All hardcoding is validated to be absent from production code via `tests/test_anti_hardcoding.py`

## Adding New Debug Tools

When creating new debug scripts:

1. Place them in this directory or mark them clearly with `debug_` prefix
2. Include clear documentation about what they analyze
3. Never import production modules and modify their behavior based on debug expectations
4. Use them to understand and validate, not to drive algorithm logic