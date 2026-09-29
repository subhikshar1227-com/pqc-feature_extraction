# PeenyaProjectMSME — Mechanical Part Inspection Pipeline

## Project Purpose

An automated inspection pipeline for manufactured mechanical parts that compares
real product images against CAD blueprints to detect geometric deviations and
missing features. The system validates that manufactured parts match their
engineering specifications.

## Current Implementation Status

**CURRENT HANDOFF POINT: Phase 3 — Expected vs Actual Feature Matching**

Phase 2B actual feature extraction is now complete and frozen following a
12-step evidence-driven remediation (see [Phase 2B Changes](#phase-2b-changes)
below).

| Phase | Status |
|-------|--------|
| Phase 0: DXF Resolution | ✅ Complete and frozen |
| Phase 1: Blueprint Identification + Expected Feature Extraction | ✅ Complete and frozen |
| Phase 2A: Canonical Preprocessing | ✅ Complete and frozen |
| **Phase 2B: Actual Feature Extraction** | **✅ Complete and frozen** |
| Phase 3: Expected-vs-Actual Matching | ❌ Not implemented |
| Phase 4: Geometric Comparison | ❌ Not implemented |
| Phase 5: Quality Inspection | ❌ Not implemented |
| Phase 6: Final Reporting | ❌ Not implemented |

---

## Phase 2B Changes

Phase 2B underwent a 12-step evidence-architecture remediation to resolve
systematic false-positive inflation in circle detection.  The original pipeline
accepted hundreds of texture/scratch/rust candidates per image because Hough ran
on the appearance image and confidence was computed from correlated edge metrics.
All 12 steps are now complete and the freeze certificate has been issued.

### What changed and why

#### Step 1–4 — Evidence observability and geometric signals

`EvidenceMetrics` was extended with 9 new fields that make every acceptance or
rejection decision inspectable:

```
angular_coverage           sector_coverage        angular_uniformity
max_gap_ratio              radial_consistency     radial_error_median
radial_error_p95           local_contrast         gradient_orientation_consistency
```

The edge-support calculation was replaced with a **radial band search** that
measures how close detected edges are to the predicted circumference, not just
whether any edge exists nearby.  Circumferential gap detection was rewritten
with a correct double-pass loop that handles the 0°/360° wraparound boundary.

#### Step 5 — Gradient orientation consistency

A new independent signal: at a real circular boundary the local image gradient
direction should be radial.  For each circumference sample the method computes
`abs(dot(normalised_gradient, outward_radial_unit))` using Sobel filters on the
appearance image.  Random texture has random gradient directions and scores near
0.5; a genuine circle scores 0.7–0.9.

#### Step 6 — Geometry-first Hough input

Hough candidate generation now runs on `internal_geometry_edges` (the
preprocessing edge map) instead of `isolated_product` (the appearance image).
The edge map contains only confirmed geometric boundaries; appearance textures,
rust marks, and machining scratches do not appear in it.  The appearance image
is still used — but only for **validation** of Hough proposals, never for
generating them.

A config flag `HOUGH_GEOMETRY_FIRST_INPUT = True` allows A/B comparison between
the two modes without code changes.

#### Step 7 — Independent evidence groups in the confidence model

The old confidence formula counted the same edge signal 2–3× under different
names (`edge_support`, `angular_coverage`, `geometric_consistency` all derived
from the same edge image).  The new `_calculate_grouped_confidence` method
organises evidence into five independent groups:

| Group | Signals | Weight |
|-------|---------|--------|
| A — Circumferential geometry | angular_coverage, sector_coverage, gap, uniformity | 0.28 |
| B — Radial geometry | radial_consistency, error_median, error_p95 | 0.27 |
| C — Edge orientation | gradient_orientation_consistency | 0.18 |
| D — Appearance | local_contrast, intensity_consistency | 0.17 |
| E — Detector agreement | contour/Hough agreement flag | 0.10 |

A hard geometric floor caps confidence when Groups A+B are both weak, preventing
strong appearance evidence from rescuing a geometrically unsound candidate.

#### Step 8 — Generation cap is a resource guard, not a correctness gate

`MAX_CANDIDATES_PER_TYPE` was previously applied **after** all validation had
run, silently discarding validated survivors when > 50 per type passed.  The
pipeline was restructured so:

```
extraction → validation → generation_cap → deduplication → accepted features
```

The cap now fires before the expensive per-extractor validation passes and
returns a `(kept, dropped)` tuple.  Any drop is logged as a WARNING and counted
in the new `ActualFeatureExtractionResult.candidates_dropped_by_generation_cap`
field.  The constant was raised from 50 to 200 to stop it acting as a
feature-count gate on normal images.

#### Step 9 — Duplicate suppression correctness

Three bugs were fixed in the clustering/deduplication logic:

- **Concentric guard inverted**: two circles with the same centre but very
  different radii (e.g. inner/outer bore edges) were being **merged** instead of
  kept as distinct features.  Fixed to `return False`.
- **Greedy pivot loop non-transitive**: if A≈B and B≈C but A≉C, C was placed in
  a separate cluster despite being a duplicate.  Replaced with a
  **union-find** (path-compressing) algorithm that correctly closes the
  transitive relation.
- **Selection formula re-counted correlated metrics**: `_select_best_from_cluster`
  used a hand-crafted weighted sum of edge metrics.  Replaced with
  `evidence.confidence` directly — the already-computed grouped score.

#### Step 10 — Adversarial and regression tests

Fourteen synthetic scenarios were added to `TestAdversarialAndRegression`:

| # | Scenario | Expected outcome |
|---|----------|-----------------|
| 1 | Perfect circle | Accepted, high geometric evidence |
| 2 | Quarter arc | Rejected |
| 3 | Random edges | Rejected |
| 4 | Circular-looking texture | Lower confidence than real circle |
| 5 | Real circle + heavy texture | Circle retained |
| 6 | Circle with crossing scratches | Circle retained |
| 7 | Straight edge | Rejected |
| 8 | Ellipse | Rejected or low confidence |
| 9 | Multiple real circles | All retained |
| **10** | **Real circle + many fake circles** | **Real retained, fakes rejected** |
| 11 | Wraparound gap | Correctly measured |
| 12 | Concentric circles | Both retained |
| 13 | Nearby distinct circles | Not merged |
| 14 | Scale variation (r=15, 45, 90) | All detected |

Scenario 10 is the key regression test that reproduces the original
false-positive explosion.

#### Step 11 — Data-driven threshold tuning

A measurement tool (`tools/measure_evidence_distributions.py`) ran the full
pipeline on 40 real images and recorded all evidence fields for every Hough
candidate (accepted and rejected) before the final gate.  16 357 candidates were
measured.  Four thresholds were adjusted based on the resulting distributions:

| Constant | Before | After | Measurement basis |
|----------|--------|-------|------------------|
| `HOUGH_MIN_LOCAL_CONTRAST` | 6 | **4** | accepted p10 = 3.8; old value rejected 533 genuine circles |
| `HOUGH_MAX_EDGE_GAP_RATIO` | 0.45 | **0.30** | accepted p90 = 0.234; 0.22 dead zone above genuine population |
| `HOUGH_RADIAL_SAMPLES` | 8 | **16** | 2 414 rejections used this gate; finer sampling reduces noise |
| `HOUGH_MIN_ANGULAR_COVERAGE` | 0.55 | **0.50** | accepted p10 = 0.49; old value cut ~2 % of genuine circles |

No threshold was changed without a measured distribution justifying the
direction and magnitude of the change.

#### Step 12 — Full validation and freeze

`validate_actual_feature_extraction.py` was rewritten as a nine-criterion freeze
check that runs on every image in `data/inputs/`:

1. All 229 unit/adversarial tests pass
2. Median accepted circles per image ≤ 60
3. No high-confidence candidates with suspicious angular coverage
4. Accepted Hough circles have populated angular_coverage p10 ≥ 0.40
5. No expected-count dependence in source code
6. Tuned thresholds have measurement-based justification comments
7. Generation cap never triggered across the test set
8. Accepted Hough circles meet radial consistency requirement (p10 ≥ 0.50)
9. Gradient orientation consistency is non-trivial (median > 0.30)

All nine criteria passed on the 40-image validation set.  The freeze certificate
was issued.

### Measured results at freeze (40 images)

| Metric | Value |
|--------|-------|
| Total Hough candidates evaluated | ~16 000 |
| Accepted circles (median per image) | 35.5 |
| Accepted circles (p90 per image) | 50 |
| Suspicious high-confidence candidates | 0 |
| Angular coverage p10 (Hough) | 0.508 |
| Radial consistency p10 (Hough) | 0.578 |
| Gradient orientation median | 0.676 |
| Radius range (accepted) | 5 – 52 px |
| Median pipeline time | 5.1 s/image |

---

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
- product_mask:                 Complete physical silhouette
- isolated_product_image:       Product region for feature detection
- internal_geometry_edges:      Texture-suppressed structural edges
- raw_internal_geometry_edges:  Raw edges before filtering
- outer_boundary_edges:         Product contour
```

### Phase 2B: Actual Feature Extraction (Frozen)
```
PreprocessingResult → feature_inspection/actual/ → List[ActualFeature]
    ↓
CircleExtractor:
  1. Hough on internal_geometry_edges (geometry-first input)
  2. Validate every candidate against evidence gates:
       angular coverage  ·  radial consistency  ·  gradient orientation
       local contrast    ·  edge continuity      ·  sector coverage
  3. Score with grouped confidence model (5 independent groups)
  4. Deduplicate via union-find clustering

RectangleExtractor:  contour-based rectangle/square detection
ContourExtractor:    general closed-shape detection
ActualFeatureExtractor:  orchestrates all three + deduplication
```

---

## Directory Structure

```
PeenyaProjectMSME/
├── cad_image_alignment/        # Phase 1: Blueprint identification (frozen)
├── dxf_resolver/               # Phase 0: DXF file resolution (frozen)
├── feature_extraction/         # Phase 1: Expected feature extraction (frozen)
├── feature_inspection/         # Phase 2: Actual feature processing
│   ├── preprocessing/          #   Phase 2A: Canonical preprocessing (frozen)
│   ├── actual/                 #   Phase 2B: Actual feature extraction (frozen)
│   │   ├── circle_extractor.py         # Geometry-first Hough + evidence gates
│   │   ├── rectangle_extractor.py      # Rectangle/square detection
│   │   ├── contour_extractor.py        # General contour detection
│   │   ├── actual_feature_extractor.py # Orchestrator
│   │   └── feature_models.py           # ActualFeature, EvidenceMetrics dataclasses
│   ├── matching/               #   Phase 3: Feature matching (not implemented)
│   ├── comparison/             #   Phase 4: Geometric comparison (not implemented)
│   ├── inspection/             #   Phase 5: Quality inspection (not implemented)
│   └── config.py               #   Centralized configuration (all thresholds here)
├── tests/
│   └── test_actual_feature_extraction.py  # 229 tests (unit + adversarial)
├── tools/
│   └── measure_evidence_distributions.py  # Step 11 measurement tool
├── data/
│   ├── blueprints/             # Reference blueprint images
│   ├── dxf/                    # Authoritative CAD/DXF files
│   └── inputs/                 # Real product photographs (69 images)
├── docs/                       # Project documentation and phase docs
├── outputs/                    # Runtime outputs (git-ignored)
├── quick_test.py               # Phase 0→1→2A pipeline demo
└── validate_actual_feature_extraction.py  # Phase 2B freeze validation script
```

---

## Installation and Setup

### Prerequisites
- Python 3.13+ (tested with Python 3.13.14)
- Required packages in `requirements.txt`

### Install Dependencies
```bash
python -m pip install -r requirements.txt
```

---

## Usage

### 1. Run Phase 2B validation (freeze check)
```bash
python validate_actual_feature_extraction.py
```
Runs the full preprocessing + feature extraction pipeline on all images in
`data/inputs/`, evaluates 9 freeze criteria, and writes a report to
`outputs/phase2b_validation/phase2b_validation_report.txt`.

Exit code `0` means all freeze criteria are satisfied.

Optional: limit to the first N images for a quick smoke test:
```bash
python validate_actual_feature_extraction.py --max-images 10
```

### 2. Run the full test suite
```bash
python -m pytest tests/ -q
# Expected: 229 passed
```

### 3. Run Phase 2B tests only
```bash
python -m pytest tests/test_actual_feature_extraction.py -v
```

### 4. Re-run threshold measurement on real images (Step 11 tool)
```bash
python tools/measure_evidence_distributions.py
# Output: tools/evidence_distributions.csv
#         tools/threshold_report.txt
```

### 5. Run complete pipeline demo (Phase 0→1→2A)
```bash
python quick_test.py
```
This stops at preprocessing and does not run Phase 2B.

---

## Key Configuration Parameters

All thresholds are in `feature_inspection/config.py`.  The Phase 2B parameters
most likely to need adjustment for a new product type are:

| Constant | Current value | Description |
|----------|---------------|-------------|
| `HOUGH_GEOMETRY_FIRST_INPUT` | `True` | Use edge map for Hough input (set `False` for appearance mode) |
| `HOUGH_MIN_ANGULAR_COVERAGE` | `0.50` | Minimum circumference fraction with edge support |
| `HOUGH_MAX_EDGE_GAP_RATIO` | `0.30` | Maximum continuous gap as fraction of circumference |
| `HOUGH_MIN_RADIAL_AGREEMENT` | `0.50` | Minimum radial consistency score |
| `HOUGH_RADIAL_TOLERANCE_PIXELS` | `3` | Tolerance band around predicted circumference |
| `HOUGH_MIN_LOCAL_CONTRAST` | `4` | Minimum interior-exterior contrast |
| `HOUGH_RADIAL_SAMPLES` | `16` | Angular samples for radial consistency check |
| `MAX_CANDIDATES_PER_TYPE` | `200` | Generation safety cap (resource guard, not correctness gate) |

To retune thresholds for a new image set, run
`tools/measure_evidence_distributions.py` first, examine the separation between
accepted and rejected populations in `tools/threshold_report.txt`, and only then
adjust constants — with the measurement data as justification.

---

## Data Structure

### Input Images (`data/inputs/`)
Real product photographs. 69 JPEG files, all used in Phase 2B validation.

### Blueprint References (`data/blueprints/`)
Reference blueprint images used for Phase 1 product identification.

### DXF Files (`data/dxf/`)
Authoritative CAD files containing expected geometric features.

**Critical distinction:**
- **Expected features** = extracted from DXF files — what *should* exist
- **Actual features** = detected from product photographs — what *is* present

Phase 2B detection uses **only image evidence**.  CAD/DXF expected counts are
never used to decide how many circles to accept or reject.

---

## Important Developer Rules

### ✅ ALLOWED
- Tune parameters in `feature_inspection/config.py` (with measurement justification)
- Modify or extend `feature_inspection/actual/` for Phase 3 integration work
- Add new feature types or detection methods
- Add tests for new functionality

### ❌ FORBIDDEN
- Modify Phase 0, 1, or 2A preprocessing
- Add product-specific, filename-specific, or image-specific logic
- Hardcode coordinates, feature counts, or thresholds without measurement basis
- Break the `PreprocessingResult` API contract
- Use expected features or CAD/DXF counts to guide actual feature detection
- Use top-N selection as a proxy for feature existence

### Preserved modules (do not modify without a regression test showing necessity)
- `feature_inspection/preprocessing/`
- `cad_image_alignment/`
- `feature_extraction/`
- `dxf_resolver/`

---

## Testing

| Test class | What it covers |
|------------|----------------|
| `TestActualFeatureModels` | `EvidenceMetrics`, `ActualFeature` dataclasses |
| `TestCircleExtractor` | Hough validation, evidence fields, radial/gap/gradient checks |
| `TestHoughGeometryFirstInput` | Geometry-first vs appearance mode A/B |
| `TestGroupedConfidenceModel` | 4-scenario acceptance criteria for confidence model |
| `TestCandidateLimitNotCorrectnessGate` | Cap is resource guard, not feature-count gate |
| `TestDuplicateSuppression` | Union-find clustering, concentric, nearby-distinct, tiebreak |
| `TestAdversarialAndRegression` | 14 synthetic scenarios including key FP regression |
| `TestThresholdCalibration` | Pins tuned constants, boundary-condition acceptance |

```bash
# Full suite
python -m pytest tests/ -q
# 229 passed

# Adversarial tests only
python -m pytest tests/test_actual_feature_extraction.py::TestAdversarialAndRegression -v

# Threshold calibration only
python -m pytest tests/test_actual_feature_extraction.py::TestThresholdCalibration -v
```

---

## Output Locations

```
outputs/
├── phase2b_validation/
│   ├── phase2b_validation_report.txt   # Freeze certificate (9-criterion report)
│   ├── accepted_circles.csv            # All accepted circles with full evidence fields
│   ├── per_image_summary.json          # Per-image metrics
│   └── {image_name}/
│       └── overlay.jpg                 # Feature detection overlay
└── actual_feature_extraction/          # Legacy per-image metadata (old validator)
```

---

## Known Limitations

1. **False positive rate is measured, not zero.** The pipeline removes the
   systematic false-positive explosion from appearance-driven Hough but some
   false positives (median 35.5 circles/image) may remain. Not all accepted
   circles correspond to real manufactured features — Phase 3 matching will
   cross-reference against CAD expectations to make that determination.

2. **Small circles (r < 10 px)** have higher rejection rates. The contour path
   is more reliable than Hough for small features due to accumulator
   discretisation.

3. **Validation set is the 69 images in `data/inputs/`.** New product
   geometries or lighting conditions may require re-running
   `tools/measure_evidence_distributions.py` before tuning thresholds.

4. **Preprocessing warnings** ("Product mask touches image border") appear on
   all current images. This is an upstream Phase 2A characteristic, not
   modified here.

5. **Phase 3 onwards is not implemented.** The `feature_inspection/matching/`,
   `comparison/`, and `inspection/` directories are empty stubs.

---

## Next Developer Focus: Phase 3

With Phase 2B frozen, the next step is `feature_inspection/matching/` —
comparing the `List[ActualFeature]` output against the `List[ExpectedFeature]`
output from Phase 1 to determine which manufactured features are present,
missing, or geometrically deviated.

The `EvidenceMetrics` fields added in Phase 2B
(`angular_coverage`, `radial_consistency`, `gradient_orientation_consistency`,
etc.) are available for the matcher to use when assessing detection reliability.

---

## Git Repository Notes

**⚠️ IMPORTANT FOR GITHUB INITIALIZATION:**
This directory contains an existing `.git` folder. Remove it before initializing
a new GitHub repository:

```bash
# Remove existing git history
Remove-Item -Recurse -Force .git    # PowerShell
# or
rm -rf .git                          # bash

git init
git add .
git commit -m "Initial commit: Phase 2B frozen"
# Add remote and push
```
