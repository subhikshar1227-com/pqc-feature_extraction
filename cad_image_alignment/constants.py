# ===========================================================================
# Alignment pipeline — tuning constants
#
# All magic numbers live here. To adapt to a new camera, part type, or
# lighting environment, only this file needs to change.
# ===========================================================================

import cv2

# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

# Minimum contour area as a fraction of the total image area.
# Lowered to 0.001 so small parts (box) photographed from a distance are
# still detected. Raise toward 0.01 if you get false contours from noise.
MIN_CONTOUR_AREA_FRACTION = 0.001

# Margin added around the CAD bounding box before scaling (fraction of bbox size).
CAD_BBOX_MARGIN_FRACTION = 0.05

# ---------------------------------------------------------------------------
# Silhouette fill
# ---------------------------------------------------------------------------

# If the flood-filled region exceeds this fraction of image area the fill
# escaped an open contour — fall back to morphologically closed edges.
FILL_ESCAPE_THRESHOLD = 0.80

# Morphological close kernel size used in _fill_silhouette and _count_holes.
FILL_MORPH_KERNEL_SIZE = 5

# ---------------------------------------------------------------------------
# Edge overlap score
# ---------------------------------------------------------------------------

# Dilation radius (px) applied to both edge maps before computing IoU.
# Gives slack for minor positional residuals.
EDGE_DILATE_PX = 3

# ---------------------------------------------------------------------------
# Hole counting
# ---------------------------------------------------------------------------

# Fraction of image area used to compute the default min_area_px in _count_holes.
# 0.00015 × 1920000 = 288 px at 1600×1200 — captures bolt holes ~380–685 px.
HOLE_MIN_AREA_FRACTION = 0.00015

# Absolute minimum for min_area_px regardless of image size (avoids zero on tiny images).
HOLE_MIN_AREA_PX_FLOOR = 10

# ---------------------------------------------------------------------------
# Coarse grid search
# ---------------------------------------------------------------------------

# Working resolution for the coarse search (fraction of input size).
COARSE_SEARCH_SCALE = 0.5

# Second downscale for the fast IoU grid (fraction of coarse size).
# Net resolution = COARSE_SEARCH_SCALE × COARSE_GRID_SCALE = 0.25 of input.
COARSE_GRID_SCALE = 0.5

# Angle step for the coarse grid search (degrees).
# 10° for the initial wide sweep — Stage 3 refines to ±FINE_ANGLE_HALF_WINDOW
# at 1° steps, so the effective precision is still 1° despite the coarse seed.
COARSE_ANGLE_STEP = 10

# Scale multipliers applied to base_scale during coarse search.
# Symmetric band covers ±15% from the diagonal-ratio estimate.
COARSE_SCALE_BAND = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

# IoU score at which the coarse fast-grid search exits early.
# At 12.5% resolution, perfect-match IoU tops out around 0.35–0.45.
COARSE_EARLY_EXIT_SCORE = 0.40

# Number of top grid candidates to keep for the hybrid-score verify pass.
COARSE_TOP_N_CANDIDATES = 20

# Scale offsets tested around best_sf_coarse in Stage 3.
COARSE_STAGE3_SCALE_OFFSETS = [-0.03, 0.0, +0.03]

# ---------------------------------------------------------------------------
# Fine angle sweep (around the best coarse candidate)
# ---------------------------------------------------------------------------

# Half-width of the 1° angle sweep window around the coarse best angle.
FINE_ANGLE_HALF_WINDOW = 8   # sweeps ±8° at 1° steps

# Scale offsets applied in the final Stage 4 refinement pass.
FINE_SCALE_OFFSETS = [-0.05, -0.025, 0.0, +0.025, +0.05]

# Angle offsets applied in the final Stage 4 refinement pass.
FINE_ANGLE_OFFSETS = [-2, -1, 0, +1, +2]

# ---------------------------------------------------------------------------
# Hybrid scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------

# Weight for filled-silhouette IoU (shape/scale fit).
HYBRID_FILLED_WEIGHT = 0.50

# Weight for raw-edge IoU (hole/internal-feature phase sensitivity).
HYBRID_EDGE_WEIGHT   = 0.50

# ---------------------------------------------------------------------------
# Size penalty (combined_score adjustment)
# ---------------------------------------------------------------------------

# CAD-to-real silhouette size ratio below which the size penalty activates.
# At 0.90: CAD must not be more than ~11% larger than the real silhouette.
SIZE_PENALTY_THRESHOLD = 0.90

# Minimum size_factor — limits the maximum penalty to 20% of combined_score.
SIZE_PENALTY_FLOOR = 0.80

# ---------------------------------------------------------------------------
# Hole-match penalty (match_best_template ranking adjustment)
# ---------------------------------------------------------------------------

# Ratio of min(real,cad)/max(real,cad) at or above which hole counts are
# considered "within range" and the factor is set to 1.0 (neutral).
# 0.5 means counts must differ by more than 2× to trigger a penalty.
HOLE_MATCH_NEUTRAL_RATIO = 0.50

# hole_match_factor floor — the worst a template can score due to hole mismatch.
HOLE_MATCH_FLOOR = 0.75

# hole_match_factor when exactly one side has zero holes.
HOLE_MATCH_ONE_SIDE_ZERO = 0.75

# ---------------------------------------------------------------------------
# ECC fine alignment
# ---------------------------------------------------------------------------

ECC_MAX_ITERATIONS = 50
ECC_EPSILON        = 1e-4
ECC_WARP_MODE      = cv2.MOTION_EUCLIDEAN   # rotation + translation only

# ECC scale sanity bounds — residual correction must keep scale near 1.
ECC_SCALE_MIN = 0.92
ECC_SCALE_MAX = 1.08

# Gaussian filter size passed to findTransformECC (must be positive odd integer).
ECC_GAUSS_FILT_SIZE = 5

# ---------------------------------------------------------------------------
# ECC dilation (applied before findTransformECC)
# ---------------------------------------------------------------------------

# Ellipse kernel radius for dilating edge maps before ECC.
ECC_DILATE_KERNEL_SIZE = 3
ECC_DILATE_ITERATIONS  = 2

# ---------------------------------------------------------------------------
# Identification thresholds
# ---------------------------------------------------------------------------

# Alignment score (filled-silhouette IoU) below which we warn.
HIGH_CONFIDENCE_THRESHOLD = 0.80

# Coverage (fraction of real silhouette covered by CAD) required to PASS.
COVERAGE_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Transform validity (used by _validate_similarity shim)
# ---------------------------------------------------------------------------

SCALE_MIN = 0.5
SCALE_MAX = 2.0

# ---------------------------------------------------------------------------
# Threshold used when binarising resized edge maps inside alignment.py
# ---------------------------------------------------------------------------

RESIZE_BINARISE_THRESHOLD = 20
