# PeenyaProjectMSME - Mechanical Part Inspection Pipeline

## Project Purpose

An automated inspection pipeline for manufactured mechanical parts that compares real product images against CAD blueprints to detect geometric deviations and missing features. The system validates that manufactured parts match their engineering specifications.

## Current Implementation Status

**CURRENT HANDOFF POINT: Phase 2B Actual Feature Extraction**

### Phase Status:

✅ **Phase 0: DXF Resolution** - Complete and frozen  
✅ **Phase 1: Blueprint Identification + Expected Feature Extraction** - Complete and frozen  
✅ **Phase 2A: Canonical Preprocessing** - Complete and frozen  
⚠️ **Phase 2B: Actual Feature Extraction** - Implemented but under refinement  
❌ **Phase 3: Expected-vs-Actual Matching** - Not implemented  
❌ **Phase 4: Geometric Comparison** - Not implemented  
❌ **Phase 5: Quality Inspection** - Not implemented  
❌ **Phase 6: Final Reporting** - Not implemented  

## Known Limitations

- **Phase 2B Issue**: Real-image circle detection still produces too many Hough-derived false-positive candidates, especially on dense/noisy samples
- **Coordinate Transformation**: Validated pixel → CAD/mm transformation source is currently unavailable
- **Incomplete Pipeline**: Downstream geometric matching/comparison/inspection phases are not yet complete  

## Architecture Overview

### Phase 0: DXF Resolution
```
Blueprint name → dxf_resolver/ → Matched DXF file path
```

### Phase 1: Blueprint Identification (Frozen)
```
Input image → cad_image_alignment/ → Best matching blueprint
            ↓
DXF file → feature_extraction/ → Expected features list
```

### Phase 2A: Canonical Preprocessing (Frozen)
```
Input image → feature_inspection/preprocessing/ → PreprocessingResult
    ↓
Outputs:
- product_mask: Complete physical silhouette
- isolated_product_image: Product region for feature detection
- internal_geometry_edges: Texture-suppressed structural edges
- raw_internal_geometry_edges: Raw edges before filtering
- outer_boundary_edges: Product contour
- Preprocessing montage for visual validation
```

### Phase 2B: Actual Feature Extraction (Current Handoff Point)
```
PreprocessingResult → feature_inspection/actual/ → List[ActualFeature]
    ↓
Feature extractors:
- CircleExtractor: Detects circular features via contour + Hough methods
- RectangleExtractor: Detects rectangular/square features
- ContourExtractor: Detects general geometric contours
- Duplicate suppression and validation
```

## Directory Structure

```
PeenyaProjectMSME/
├── cad_image_alignment/        # Phase 1: Blueprint identification
├── dxf_resolver/              # Phase 0: DXF file resolution
├── feature_extraction/        # Phase 1: Expected feature extraction
├── feature_inspection/        # Phase 2: Actual feature processing
│   ├── preprocessing/         #   Phase 2A: Canonical preprocessing
│   ├── actual/               #   Phase 2B: Actual feature extraction ← HANDOFF POINT
│   ├── matching/             #   Phase 3: Feature matching (empty)
│   ├── comparison/           #   Phase 4: Geometric comparison (empty) 
│   ├── inspection/           #   Phase 5: Quality inspection (empty)
│   └── config.py             #   Centralized configuration
├── tests/                    # Test suite (147 tests)
├── data/
│   ├── blueprints/          # Reference blueprint images
│   ├── dxf/                 # Authoritative DXF files
│   └── inputs/              # Real product photographs
├── docs/                    # Project documentation
├── tools/                   # Debug/audit/validation utilities
├── outputs/                 # Runtime outputs (git-ignored)
├── quick_test.py           # Phase 0→1→2A pipeline demo
└── validate_actual_feature_extraction.py  # Phase 2B validation
```

## Installation and Setup

### Prerequisites
- Python 3.14+ (tested with Python 3.14.6)
- Required packages listed in requirements.txt

### Install Dependencies
```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. Run Complete Pipeline (Phase 0→1→2A)
```bash
python quick_test.py
```
**What it does:**
- Blueprint identification and DXF resolution
- Expected feature extraction from DXF
- Phase 2A canonical preprocessing
- **Stops at preprocessing** - does not run Phase 2B feature detection

### 2. Run Phase 2B Actual Feature Extraction Validation
```bash
python validate_actual_feature_extraction.py
```
**What it does:**
- Runs Phase 2A preprocessing on all input images
- Extracts actual features using image evidence only
- Saves detection overlays and metadata to `outputs/actual_feature_extraction/`
- **Important**: This is purely image-driven detection with NO expected feature dependency

### 3. Run Test Suite
```bash
python -m pytest tests -q
```
**Expected result**: 147 tests pass

## Data Structure

### Input Images (`data/inputs/`)
Real product photographs to be inspected. Supported formats: PNG, JPG, JPEG.

### Blueprint References (`data/blueprints/`)
Reference blueprint images used for Phase 1 product identification. These are NOT detector outputs.

### DXF Files (`data/dxf/`)
Authoritative CAD files containing expected geometric features. These define WHAT SHOULD EXIST in the manufactured parts.

**Critical distinction:**
- **Expected features** = extracted from DXF files (what should exist)  
- **Actual features** = detected from product photographs (what is actually present)

## Current Handoff Point: Phase 2B Feature Extraction

### What's Implemented:
- **Multi-method detection**: Contour analysis, Hough transforms, edge validation
- **Feature types**: Circles, rectangles/squares, general contours
- **Evidence-based validation**: Features must have actual image evidence
- **Duplicate suppression**: Enhanced clustering and consolidation
- **Configuration-driven**: 100+ tunable parameters in `feature_inspection/config.py`

### What Needs Work:
The next developer should focus on `feature_inspection/actual/` and specifically:

1. **Reduce Hough false positives** in `circle_extractor.py`
2. **Improve validation thresholds** for real-world images
3. **Enhanced texture discrimination** to avoid detecting surface patterns as features

### Preservation Requirements:
- **DO NOT modify** Phase 0, 1, or 2A preprocessing
- **Maintain the preprocessing contract**: `product_mask`, `isolated_product`, `internal_geometry_edges`, etc.
- **Keep configuration-driven approach**: All thresholds in `config.py`
- **No product-specific logic**: Detection must work on any product type
- **No expected feature dependency**: Phase 2B uses only image evidence

## Configuration

All algorithmic parameters are centralized in configuration files:
- `feature_inspection/config.py`: Phase 2A preprocessing + Phase 2B feature extraction
- `feature_extraction/config.py`: Phase 1 expected feature extraction
- `dxf_resolver/config.py`: Phase 0 DXF resolution
- `cad_image_alignment/constants.py`: Phase 1 blueprint alignment

## Output Locations

All runtime outputs are saved to `outputs/` (git-ignored):
- `outputs/{image_name}/phase_2a_preprocessing/`: Preprocessing results
- `outputs/actual_feature_extraction/{image_name}/`: Feature detection results
- `outputs/geometry_preserving_validation/{image_name}/`: Validation outputs

## Important Developer Rules

### ✅ ALLOWED:
- Tune parameters in `feature_inspection/config.py`
- Modify feature extraction algorithms in `feature_inspection/actual/`
- Add new feature types or detection methods
- Improve validation and duplicate suppression
- Add tests for new functionality

### ❌ FORBIDDEN:
- Modify Phase 0, 1, or 2A preprocessing behavior
- Add product-specific or filename-specific logic  
- Hardcode coordinates, feature counts, or thresholds
- Break the preprocessing API contract
- Use expected features to guide actual feature detection
- Change the directory structure

## Testing

The test suite validates:
- **Behavioral invariance**: Core algorithms produce consistent results
- **Anti-hardcoding compliance**: No product-specific logic detected
- **API contracts**: All phases maintain their interfaces
- **Feature extraction**: Synthetic and real image validation

Run specific test categories:
```bash
# Full suite
python -m pytest tests -q

# Phase 2B feature extraction only  
python -m pytest tests/test_actual_feature_extraction.py -v

# Anti-hardcoding audit
python -m pytest tests/test_anti_hardcoding.py -v
```

## Git Repository Notes

**⚠️ IMPORTANT FOR GITHUB INITIALIZATION:**
This directory contains an existing `.git` folder that should be removed before initializing a new GitHub repository.

When setting up the new repository:
1. Remove the existing `.git` directory
2. Initialize fresh: `git init`  
3. Add files: `git add .`
4. Initial commit: `git commit -m "Initial commit - Phase 2B handoff"`
5. Add GitHub remote and push

## License and Development Notes

This is an active development project for mechanical part inspection automation. The codebase is structured for handoff between developers while preserving critical working components.

**Next Developer Focus**: Phase 2B actual feature extraction in `feature_inspection/actual/`