"""
CAD-Image Alignment module.

Public API
----------
  align(cad_edge_map, real_edge_map)           -> AlignmentResult
  match_best_template(templates, real_edge_map) -> list[TemplateMatch]
  apply_transform(edge_map, matrix, output_shape) -> np.ndarray

All tuning parameters live in constants.py. A small number of structural
constants (kernel sizes, fill thresholds) that are local to a single function
are also in constants.py — see each function's imports for the full list.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class ContourDescriptor:
    """Geometric descriptor for the primary contour of an edge map."""
    contour: np.ndarray
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    bbox_diagonal: float
    pca_angle_deg: float


@dataclass
class AlignmentResult:
    """
    Output of a single CAD-vs-real alignment attempt.

    Fields
    ------
    aligned_image   : warped CAD edge map at real image resolution
    transform_matrix: 3×3 homogeneous CAD→real transform
    alignment_score : filled-silhouette IoU (symmetric)
    coverage        : intersection / real_silhouette_area (one-sided,
                      gates the identified flag)
    edge_score      : dilated raw-edge IoU
    hole_diff       : |real_hole_count − cad_hole_count in aligned image|
    combined_score  : (0.5·coverage + 0.5·edge_score) × size_factor
                      — includes size penalty; used for ranking in
                      match_best_template
    strategy        : "ecc_fine" | "affine_coarse_only" | "identity"
    high_confidence : alignment_score >= HIGH_CONFIDENCE_THRESHOLD
    identified      : coverage >= COVERAGE_THRESHOLD
    """
    aligned_image: np.ndarray
    transform_matrix: np.ndarray
    alignment_score: float
    coverage: float
    edge_score: float
    hole_diff: int
    combined_score: float
    strategy: str
    high_confidence: bool
    identified: bool


@dataclass
class TemplateMatch:
    """One ranked result from match_best_template."""
    name: str
    result: AlignmentResult
    rank: int


# ===========================================================================
# Input validation
# ===========================================================================

def _validate_inputs(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> np.ndarray:
    """
    Validate dtypes/ndim/non-empty, then resize CAD to match the real canvas
    if their shapes differ (preserving design geometry via uniform scaling).

    Returns the (possibly resized) cad_edge_map.
    Raises ValueError on invalid inputs.
    """
    from .constants import CAD_BBOX_MARGIN_FRACTION, RESIZE_BINARISE_THRESHOLD

    if cad_edge_map.dtype != np.uint8:
        raise ValueError(
            f"cad_edge_map has invalid dtype {cad_edge_map.dtype}, expected uint8"
        )
    if real_edge_map.dtype != np.uint8:
        raise ValueError(
            f"real_edge_map has invalid dtype {real_edge_map.dtype}, expected uint8"
        )
    if cad_edge_map.ndim != 2:
        raise ValueError(
            f"cad_edge_map has invalid ndim {cad_edge_map.ndim}, expected 2"
        )
    if real_edge_map.ndim != 2:
        raise ValueError(
            f"real_edge_map has invalid ndim {real_edge_map.ndim}, expected 2"
        )
    if not np.any(cad_edge_map):
        raise ValueError("cad_edge_map is empty (no non-zero pixels)")
    if not np.any(real_edge_map):
        raise ValueError("real_edge_map is empty (no non-zero pixels)")

    if cad_edge_map.shape != real_edge_map.shape:
        rh, rw = real_edge_map.shape
        ch, cw = cad_edge_map.shape
        logger.debug(
            f"Resolution mismatch: cad {cad_edge_map.shape} vs real {real_edge_map.shape}. "
            "Cropping to CAD bbox then uniform-scaling."
        )
        contours, _ = cv2.findContours(
            cad_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            margin = int(max(bw, bh) * CAD_BBOX_MARGIN_FRACTION)
            x1 = max(0, bx - margin)
            y1 = max(0, by - margin)
            x2 = min(cw, bx + bw + margin)
            y2 = min(ch, by + bh + margin)
            cad_edge_map = cad_edge_map[y1:y2, x1:x2]
        ch2, cw2 = cad_edge_map.shape
        fit_scale = min(rw / cw2, rh / ch2)
        new_w = int(round(cw2 * fit_scale))
        new_h = int(round(ch2 * fit_scale))
        interp = cv2.INTER_AREA if fit_scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(cad_edge_map, (new_w, new_h), interpolation=interp)
        _, scaled = cv2.threshold(scaled, RESIZE_BINARISE_THRESHOLD, 255, cv2.THRESH_BINARY)
        canvas = np.zeros((rh, rw), dtype=np.uint8)
        canvas[(rh - new_h) // 2:(rh - new_h) // 2 + new_h,
               (rw - new_w) // 2:(rw - new_w) // 2 + new_w] = scaled
        cad_edge_map = canvas

    return cad_edge_map


# ===========================================================================
# Geometry helpers
# ===========================================================================

def _compute_pca_angle(contour: np.ndarray) -> float:
    """Return the principal axis angle of a contour in [0°, 360°)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    centered = pts - pts.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    principal = eigenvectors[:, -1]
    return np.degrees(np.arctan2(principal[1], principal[0])) % 360.0


def _extract_primary_contour(edge_map: np.ndarray) -> Optional[ContourDescriptor]:
    """
    Find the dominant contour in an edge map.

    Filters out contours smaller than MIN_CONTOUR_AREA_FRACTION × image area,
    computes the convex hull of all remaining ones, and returns a
    ContourDescriptor. Returns None if no valid contour is found.
    """
    from .constants import MIN_CONTOUR_AREA_FRACTION

    contours, _ = cv2.findContours(
        edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = edge_map.shape[0] * edge_map.shape[1] * MIN_CONTOUR_AREA_FRACTION
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid:
        logger.warning(
            f"No contour found with area >= {MIN_CONTOUR_AREA_FRACTION * 100:.2f}% "
            f"of image ({min_area:.0f} px)"
        )
        return None

    hull = cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in valid]))
    M = cv2.moments(hull)
    if M["m00"] == 0:
        primary = max(valid, key=cv2.contourArea)
        M = cv2.moments(primary)
        hull = primary

    centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(hull)

    return ContourDescriptor(
        contour=hull.reshape(-1, 1, 2).astype(np.int32),
        centroid=centroid,
        bbox=(x, y, w, h),
        bbox_diagonal=float(np.hypot(w, h)),
        pca_angle_deg=_compute_pca_angle(max(valid, key=cv2.contourArea)),
    )


def _build_affine_matrix(
    scale: float,
    angle_deg: float,
    src_centroid: tuple[float, float],
    dst_centroid: tuple[float, float],
) -> np.ndarray:
    """Build a 3×3 similarity matrix (uniform scale + rotation + translation)."""
    cx_src, cy_src = src_centroid
    cx_dst, cy_dst = dst_centroid
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    sc, ss = scale * c, scale * s
    tx = cx_dst - cx_src
    ty = cy_dst - cy_src
    return np.array([
        [ sc, -ss,  cx_src * (1 - sc) + cy_src * ss + tx],
        [ ss,  sc,  cy_src * (1 - sc) - cx_src * ss + ty],
        [0.0, 0.0,  1.0],
    ], dtype=np.float64)


# ===========================================================================
# Silhouette and scoring
# ===========================================================================

def _fill_silhouette(edge_map: np.ndarray) -> np.ndarray:
    """
    Flood-fill the interior of an edge contour to produce a solid silhouette.

    Falls back to morphologically closed edges when the flood fill escapes
    an open contour (detected by the filled fraction exceeding
    FILL_ESCAPE_THRESHOLD).
    """
    from .constants import FILL_ESCAPE_THRESHOLD, FILL_MORPH_KERNEL_SIZE

    h, w = edge_map.shape
    closed = cv2.morphologyEx(
        edge_map, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (FILL_MORPH_KERNEL_SIZE, FILL_MORPH_KERNEL_SIZE)),
    )
    canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    canvas[1:h + 1, 1:w + 1] = closed
    inv = cv2.bitwise_not(canvas)
    cv2.floodFill(inv, np.zeros((h + 4, w + 4), dtype=np.uint8), (0, 0), 0)
    filled = inv[1:h + 1, 1:w + 1]

    if float(np.count_nonzero(filled)) / float(h * w) > FILL_ESCAPE_THRESHOLD:
        logger.debug("_fill_silhouette: flood-fill escaped open contour, using closed edges")
        return closed
    return filled


def _compute_edge_overlap_score(
    aligned_edges: np.ndarray,
    real_edges: np.ndarray,
) -> float:
    """
    IoU on dilated raw edge maps.

    Sensitive to internal features (holes, cutouts) that _fill_silhouette
    erases. Dilation radius is controlled by EDGE_DILATE_PX in constants.py.
    """
    from .constants import EDGE_DILATE_PX

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_DILATE_PX, EDGE_DILATE_PX))
    cad_d  = cv2.dilate(aligned_edges, kernel)
    real_d = cv2.dilate(real_edges,    kernel)
    inter = np.logical_and(cad_d > 0, real_d > 0).sum()
    union = np.logical_or(cad_d  > 0, real_d > 0).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _count_holes(edge_map: np.ndarray, min_area_px: int = None) -> tuple[int, float]:
    """
    Count enclosed regions (holes) using 2-level contour hierarchy (RETR_CCOMP).

    min_area_px is auto-scaled to image size using HOLE_MIN_AREA_FRACTION if
    not provided. Pass an explicit value to override (e.g. when calling on
    images of very different resolutions).

    Returns (hole_count, total_hole_area_fraction).
    """
    from .constants import HOLE_MIN_AREA_FRACTION, HOLE_MIN_AREA_PX_FLOOR, FILL_MORPH_KERNEL_SIZE

    total_area = edge_map.shape[0] * edge_map.shape[1]
    if min_area_px is None:
        min_area_px = max(HOLE_MIN_AREA_PX_FLOOR, int(total_area * HOLE_MIN_AREA_FRACTION))

    closed = cv2.morphologyEx(
        edge_map, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (FILL_MORPH_KERNEL_SIZE, FILL_MORPH_KERNEL_SIZE))
    )
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0.0
    holes = [
        c for c, h in zip(contours, hierarchy[0])
        if h[3] != -1 and cv2.contourArea(c) >= min_area_px
    ]
    return len(holes), sum(cv2.contourArea(c) for c in holes) / total_area


def _hybrid_score(
    warped: np.ndarray,
    real_bin: np.ndarray,
    real_filled: np.ndarray,
) -> float:
    """
    Composite score used during coarse search.

    Combines filled-silhouette IoU (shape/scale) with raw-edge IoU
    (hole/feature phase). Weights from HYBRID_FILLED_WEIGHT /
    HYBRID_EDGE_WEIGHT in constants.py.

    real_filled is precomputed once per real image and passed in to avoid
    recomputing it on every candidate in the search loop.
    """
    from .constants import HYBRID_FILLED_WEIGHT, HYBRID_EDGE_WEIGHT

    warped_filled = _fill_silhouette(warped)
    fi = np.logical_and(warped_filled > 0, real_filled > 0).sum()
    fu = np.logical_or(warped_filled  > 0, real_filled > 0).sum()
    filled_iou = float(fi) / float(fu) if fu > 0 else 0.0

    ei = np.logical_and(warped > 0, real_bin > 0).sum()
    eu = np.logical_or(warped  > 0, real_bin > 0).sum()
    edge_iou = float(ei) / float(eu) if eu > 0 else 0.0

    return HYBRID_FILLED_WEIGHT * filled_iou + HYBRID_EDGE_WEIGHT * edge_iou


def _compute_scores(
    aligned_image: np.ndarray,
    real_edge_map: np.ndarray,
) -> tuple[float, float, float, float, int, float]:
    """
    Compute all scoring metrics for an aligned CAD image vs the real edge map.

    Returns
    -------
    (alignment_score, coverage, edge_score, combined_score, hole_diff, size_factor)

    alignment_score : symmetric silhouette IoU
    coverage        : intersection / real_silhouette_area  (gates identification)
    edge_score      : dilated raw-edge IoU
    combined_score  : size-penalised (0.5·coverage + 0.5·edge_score)
    hole_diff       : |real_holes − cad_holes_in_aligned|
    size_factor     : the multiplier applied to raw combined_score
    """
    from .constants import SIZE_PENALTY_THRESHOLD, SIZE_PENALTY_FLOOR

    cad_filled  = _fill_silhouette(aligned_image)
    real_filled = _fill_silhouette(real_edge_map)

    intersection    = int(np.logical_and(cad_filled > 0, real_filled > 0).sum())
    union           = int(np.logical_or( cad_filled > 0, real_filled > 0).sum())
    alignment_score = float(intersection) / float(union) if union > 0 else 0.0

    real_area = int((real_filled > 0).sum())
    coverage  = float(intersection) / float(real_area) if real_area > 0 else 0.0

    edge_score = _compute_edge_overlap_score(aligned_image, real_edge_map)

    real_holes, _ = _count_holes(real_edge_map)
    cad_holes,  _ = _count_holes(aligned_image)
    hole_diff = abs(real_holes - cad_holes)

    # Size penalty: reduce combined_score when the aligned CAD silhouette is
    # notably larger than the real part. A correct template should fit tightly.
    cad_area = int((cad_filled > 0).sum())
    if cad_area > 0 and real_area > 0:
        size_ratio  = float(real_area) / float(cad_area)
        size_factor = max(SIZE_PENALTY_FLOOR,
                          min(1.0, size_ratio / SIZE_PENALTY_THRESHOLD))
    else:
        size_factor = 1.0

    combined_score = (0.5 * coverage + 0.5 * edge_score) * size_factor
    return alignment_score, coverage, edge_score, combined_score, hole_diff, size_factor


# ===========================================================================
# Coarse transform (multi-resolution grid search)
# ===========================================================================

def _estimate_base_scale(
    cad_desc: ContourDescriptor,
    real_desc: ContourDescriptor,
) -> float:
    """
    Estimate the CAD→real scale factor.

    Uses bounding-box diagonal ratio as the primary estimator — more robust
    than sqrt(area ratio) when the real edge map is sparse (missing corners
    shrink the hull area but the diagonal remains correct).
    Falls back to sqrt(area ratio) when the diagonal is degenerate.
    """
    if cad_desc.bbox_diagonal > 0:
        return real_desc.bbox_diagonal / cad_desc.bbox_diagonal
    cad_area = cv2.contourArea(cad_desc.contour)
    if cad_area > 0:
        return float(np.sqrt(cv2.contourArea(real_desc.contour) / cad_area))
    logger.warning("Both diagonal and area are zero — defaulting base_scale to 1.0")
    return 1.0


def _compute_coarse_transform(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Multi-resolution grid search for the best similarity transform.

    Four stages, all parameters from constants.py:

    Stage 1 — fast edge-IoU on tiny grid (COARSE_GRID_SCALE² of input)
              over all COARSE_SCALE_BAND × COARSE_ANGLE_STEP candidates.
    Stage 2 — hybrid score (filled + edge IoU) on top COARSE_TOP_N_CANDIDATES
              from Stage 1.
    Stage 3 — 1° sweep ±FINE_ANGLE_HALF_WINDOW around the Stage 2 winner,
              with COARSE_STAGE3_SCALE_OFFSETS around best scale.
    Stage 4 — final ±FINE_SCALE_OFFSETS / ±FINE_ANGLE_OFFSETS pass at
              COARSE_SEARCH_SCALE resolution.

    Returns a 3×3 similarity matrix scaled to full resolution, or None on failure.
    """
    from .constants import (
        COARSE_SEARCH_SCALE, COARSE_GRID_SCALE,
        COARSE_ANGLE_STEP, COARSE_SCALE_BAND,
        COARSE_EARLY_EXIT_SCORE, COARSE_TOP_N_CANDIDATES,
        COARSE_STAGE3_SCALE_OFFSETS,
        FINE_ANGLE_HALF_WINDOW, FINE_SCALE_OFFSETS, FINE_ANGLE_OFFSETS,
        RESIZE_BINARISE_THRESHOLD,
    )

    h_full, w_full = real_edge_map.shape

    def _resize_thresh(src: np.ndarray, h: int, w: int) -> np.ndarray:
        r = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
        _, r = cv2.threshold(r, RESIZE_BINARISE_THRESHOLD, 255, cv2.THRESH_BINARY)
        return r

    # Half-resolution working images
    h_half = max(1, int(h_full * COARSE_SEARCH_SCALE))
    w_half = max(1, int(w_full * COARSE_SEARCH_SCALE))
    cad_half  = _resize_thresh(cad_edge_map,  h_half, w_half)
    real_half = _resize_thresh(real_edge_map, h_half, w_half)

    cad_desc  = _extract_primary_contour(cad_half)
    real_desc = _extract_primary_contour(real_half)

    if cad_desc is None:
        logger.warning("No valid contour in CAD edge map — cannot compute coarse transform")
        return None
    if real_desc is None:
        logger.warning("No valid contour in real edge map — cannot compute coarse transform")
        return None

    base_scale = _estimate_base_scale(cad_desc, real_desc)
    pca_diff   = real_desc.pca_angle_deg - cad_desc.pca_angle_deg

    logger.debug(
        f"Coarse: base_scale={base_scale:.3f}  "
        f"cad_pca={cad_desc.pca_angle_deg:.1f}°  "
        f"real_pca={real_desc.pca_angle_deg:.1f}°"
    )

    # Candidate angles: uniform grid + PCA-seeded hints (+ 180° ambiguity)
    grid_angles = list(set(
        list(range(0, 360, COARSE_ANGLE_STEP)) +
        [round(pca_diff) % 360, round(pca_diff + 180) % 360]
    ))

    # ── Stage 1: fast edge-IoU on tiny grid ────────────────────────────────
    h_grid = max(1, int(h_half * COARSE_GRID_SCALE))
    w_grid = max(1, int(w_half * COARSE_GRID_SCALE))
    cad_grid  = _resize_thresh(cad_half,  h_grid, w_grid)
    real_grid = _resize_thresh(real_half, h_grid, w_grid)

    grid_cad_desc  = _extract_primary_contour(cad_grid)
    grid_real_desc = _extract_primary_contour(real_grid)
    if grid_cad_desc is None or grid_real_desc is None:
        grid_cad_desc, grid_real_desc = cad_desc, real_desc
        cad_grid, real_grid = cad_half, real_half

    real_grid_bool = real_grid.astype(np.bool_)

    def _fast_iou(warped: np.ndarray) -> float:
        wb = warped.astype(np.bool_)
        inter = int(np.logical_and(wb, real_grid_bool).sum())
        union = int(np.logical_or( wb, real_grid_bool).sum())
        return float(inter) / float(union) if union > 0 else 0.0

    best_fast = -1.0
    best_angle_g, best_sf_g = grid_angles[0], COARSE_SCALE_BAND[0]
    candidates: list[tuple[float, int, float]] = []
    done = False

    for sf in COARSE_SCALE_BAND:
        if done:
            break
        s = base_scale * sf
        for angle in grid_angles:
            M     = _build_affine_matrix(s, angle, grid_cad_desc.centroid, grid_real_desc.centroid)
            score = _fast_iou(apply_transform(cad_grid, M, real_grid.shape))
            if score > best_fast:
                best_fast = score
                best_angle_g, best_sf_g = angle, sf
            candidates.append((score, angle, sf))
            if best_fast >= COARSE_EARLY_EXIT_SCORE:
                done = True
                break

    # ── Stage 2: hybrid score on top-N candidates ──────────────────────────
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_angles = list(set(int(c[1]) for c in candidates[:COARSE_TOP_N_CANDIDATES]))
    top_sfs    = list(set(c[2]      for c in candidates[:COARSE_TOP_N_CANDIDATES]))

    real_grid_filled = _fill_silhouette(real_grid)
    best_hybrid = -1.0
    best_angle_h, best_sf_h = best_angle_g, best_sf_g

    for sf in top_sfs:
        s = base_scale * sf
        for angle in top_angles:
            M     = _build_affine_matrix(s, angle, grid_cad_desc.centroid, grid_real_desc.centroid)
            score = _hybrid_score(apply_transform(cad_grid, M, real_grid.shape), real_grid, real_grid_filled)
            if score > best_hybrid:
                best_hybrid = score
                best_angle_h, best_sf_h = angle, sf

    # ── Stage 3: ±FINE_ANGLE_HALF_WINDOW sweep around Stage 2 winner ───────
    sweep_angles = list(set(
        list(range(best_angle_h - FINE_ANGLE_HALF_WINDOW, best_angle_h + FINE_ANGLE_HALF_WINDOW + 1)) +
        list(range(int(pca_diff) - FINE_ANGLE_HALF_WINDOW, int(pca_diff) + FINE_ANGLE_HALF_WINDOW + 1)) +
        list(range(int(pca_diff + 180) - FINE_ANGLE_HALF_WINDOW, int(pca_diff + 180) + FINE_ANGLE_HALF_WINDOW + 1))
    ))

    best_sweep = -1.0
    best_angle_s, best_sf_s = best_angle_h, best_sf_h

    for sf_off in COARSE_STAGE3_SCALE_OFFSETS:
        s = base_scale * (best_sf_h + sf_off)
        for angle in sweep_angles:
            M     = _build_affine_matrix(s, angle, grid_cad_desc.centroid, grid_real_desc.centroid)
            score = _hybrid_score(apply_transform(cad_grid, M, real_grid.shape), real_grid, real_grid_filled)
            if score > best_sweep:
                best_sweep = score
                best_angle_s, best_sf_s = angle, sf_off + best_sf_h

    # ── Stage 4: final fine pass at half-resolution ─────────────────────────
    real_half_filled = _fill_silhouette(real_half)
    best_M    = np.eye(3, dtype=np.float64)
    best_score = -1.0

    for sf_offset in FINE_SCALE_OFFSETS:
        s = base_scale * (best_sf_s + sf_offset)
        for a_offset in FINE_ANGLE_OFFSETS:
            M     = _build_affine_matrix(
                s, best_angle_s + a_offset,
                cad_desc.centroid, real_desc.centroid,
            )
            score = _hybrid_score(apply_transform(cad_half, M, real_half.shape), real_half, real_half_filled)
            if score > best_score:
                best_score, best_M = score, M

    logger.debug(f"Coarse best score={best_score:.4f}")

    S_down = np.diag([COARSE_SEARCH_SCALE, COARSE_SEARCH_SCALE, 1.0])
    S_up   = np.diag([1 / COARSE_SEARCH_SCALE, 1 / COARSE_SEARCH_SCALE, 1.0])
    return S_up @ best_M @ S_down


# ===========================================================================
# ECC fine alignment
# ===========================================================================

def _compute_fine_transform(
    coarsely_aligned_cad: np.ndarray,
    real_edge_map: np.ndarray,
    coarse_matrix: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Refine the coarse transform using ECC on dilated edge maps.

    ECC is seeded with identity because coarse_matrix already warped
    coarsely_aligned_cad — ECC only corrects the small residual.
    Uses MOTION_EUCLIDEAN (rotation + translation only, scale preserved).

    Returns the refined 3×3 matrix on success, or None when:
      - cv2.findTransformECC fails to converge
      - The ECC scale deviates outside [ECC_SCALE_MIN, ECC_SCALE_MAX]
        (indicates divergence rather than a genuine correction)
    """
    from .constants import (
        ECC_MAX_ITERATIONS, ECC_EPSILON, ECC_WARP_MODE,
        ECC_SCALE_MIN, ECC_SCALE_MAX,
        ECC_GAUSS_FILT_SIZE, ECC_DILATE_KERNEL_SIZE, ECC_DILATE_ITERATIONS,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (ECC_DILATE_KERNEL_SIZE, ECC_DILATE_KERNEL_SIZE))
    src_f = cv2.dilate(coarsely_aligned_cad, kernel,
                       iterations=ECC_DILATE_ITERATIONS).astype(np.float32)
    dst_f = cv2.dilate(real_edge_map,         kernel,
                       iterations=ECC_DILATE_ITERATIONS).astype(np.float32)

    warp_init = np.eye(2, 3, dtype=np.float32)
    criteria  = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITERATIONS,
        ECC_EPSILON,
    )

    try:
        _, warp_ecc = cv2.findTransformECC(
            dst_f, src_f, warp_init, ECC_WARP_MODE, criteria, None, ECC_GAUSS_FILT_SIZE
        )
    except cv2.error as exc:
        logger.warning(f"ECC fine alignment failed: {exc}")
        return None

    M_ecc = np.eye(3, dtype=np.float64)
    M_ecc[:2, :] = warp_ecc.astype(np.float64)

    scale = float(np.hypot(M_ecc[0, 0], M_ecc[1, 0]))
    if not (ECC_SCALE_MIN <= scale <= ECC_SCALE_MAX):
        logger.warning(f"ECC rejected: scale={scale:.3f} outside [{ECC_SCALE_MIN}, {ECC_SCALE_MAX}]")
        return None

    logger.debug(f"ECC succeeded: residual_scale={scale:.4f}")
    return M_ecc @ coarse_matrix


# ===========================================================================
# Main align function
# ===========================================================================

def align(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> AlignmentResult:
    """
    Align a CAD edge map to a real photo edge map.

    Pipeline:
      1. Validate + resize inputs (_validate_inputs)
      2. Multi-resolution coarse grid search (_compute_coarse_transform)
      3. ECC fine alignment (_compute_fine_transform)
         Falls back to coarse-only when ECC fails or diverges.
      4. Compute all scores (_compute_scores) and return AlignmentResult.

    All parameters are read from constants.py.
    """
    from .constants import HIGH_CONFIDENCE_THRESHOLD, COVERAGE_THRESHOLD

    cad_edge_map = _validate_inputs(cad_edge_map, real_edge_map)

    M_coarse = _compute_coarse_transform(cad_edge_map, real_edge_map)

    if M_coarse is None:
        logger.warning("Coarse alignment failed — falling back to identity transform.")
        final_matrix = np.eye(3, dtype=np.float64)
        strategy = "identity"
    else:
        coarsely_aligned = apply_transform(cad_edge_map, M_coarse, real_edge_map.shape)
        M_fine = _compute_fine_transform(coarsely_aligned, real_edge_map, M_coarse)

        if M_fine is not None:
            final_matrix = M_fine
            strategy = "ecc_fine"
        else:
            logger.warning("ECC fine alignment failed — using coarse transform only.")
            final_matrix = M_coarse
            strategy = "affine_coarse_only"

    aligned_image = apply_transform(cad_edge_map, final_matrix, real_edge_map.shape)

    alignment_score, coverage, edge_score, combined_score, hole_diff, _ = \
        _compute_scores(aligned_image, real_edge_map)

    high_confidence = alignment_score >= HIGH_CONFIDENCE_THRESHOLD
    identified      = coverage >= COVERAGE_THRESHOLD

    if not high_confidence and not identified:
        logger.warning(
            f"Low confidence: alignment_score={alignment_score:.4f} < {HIGH_CONFIDENCE_THRESHOLD}"
        )

    logger.debug(
        f"align: strategy={strategy}  score={alignment_score:.4f}  "
        f"coverage={coverage:.4f}  edge={edge_score:.4f}  "
        f"holes_diff={hole_diff}  combined={combined_score:.4f}"
    )

    return AlignmentResult(
        aligned_image=aligned_image,
        transform_matrix=final_matrix,
        alignment_score=alignment_score,
        coverage=coverage,
        edge_score=edge_score,
        hole_diff=hole_diff,
        combined_score=combined_score,
        strategy=strategy,
        high_confidence=high_confidence,
        identified=identified,
    )


# ===========================================================================
# Transform application
# ===========================================================================

def apply_transform(
    edge_map: np.ndarray,
    matrix: np.ndarray,
    output_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Warp an edge map by a 3×3 homogeneous matrix."""
    if output_shape is None:
        output_shape = edge_map.shape
    return cv2.warpPerspective(
        edge_map, matrix,
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ===========================================================================
# Template matching
# ===========================================================================

def match_best_template(
    templates: list[tuple[str, np.ndarray]],
    real_edge_map: np.ndarray,
) -> list[TemplateMatch]:
    """
    Align every CAD template against the real edge map and rank by final_score.

    Ranking formula
    ---------------
    final_score = combined_score × hole_match_factor

    combined_score already includes the size penalty (see _compute_scores).

    hole_match_factor ∈ [HOLE_MATCH_FLOOR, 1.0]:
      - 1.0  (neutral): both have no holes, or counts are within 2× of each
               other — too ambiguous to penalise when detection is incomplete.
      - [HOLE_MATCH_FLOOR, 1.0) (weak penalty): strong mismatch (one count
               > 2× the other).
      - HOLE_MATCH_ONE_SIDE_ZERO: exactly one side has zero holes.

    The real image hole count is computed once and reused for all templates.
    """
    from .constants import (
        HOLE_MATCH_NEUTRAL_RATIO,
        HOLE_MATCH_FLOOR,
        HOLE_MATCH_ONE_SIDE_ZERO,
    )

    if not templates:
        raise ValueError("templates list is empty")

    real_holes, _ = _count_holes(real_edge_map)
    logger.debug(f"match_best_template: real_holes={real_holes}")

    results = []
    for name, cad_edge_map in templates:
        logger.debug(f"Aligning template '{name}'...")
        result = align(cad_edge_map, real_edge_map)

        cad_holes_template, _ = _count_holes(cad_edge_map)

        if real_holes == 0 and cad_holes_template == 0:
            hole_match_factor = 1.0
        elif real_holes == 0 or cad_holes_template == 0:
            hole_match_factor = HOLE_MATCH_ONE_SIDE_ZERO
        else:
            ratio = min(real_holes, cad_holes_template) / max(real_holes, cad_holes_template)
            if ratio >= HOLE_MATCH_NEUTRAL_RATIO:
                hole_match_factor = 1.0
            else:
                # Penalise proportionally between HOLE_MATCH_FLOOR and 1.0
                hole_match_factor = HOLE_MATCH_FLOOR + (1.0 - HOLE_MATCH_FLOOR) * (ratio / HOLE_MATCH_NEUTRAL_RATIO)

        final_score = result.combined_score * hole_match_factor

        logger.debug(
            f"  '{name}': coverage={result.coverage:.4f}  "
            f"edge={result.edge_score:.4f}  combined={result.combined_score:.4f}  "
            f"real_holes={real_holes}  cad_holes={cad_holes_template}  "
            f"hole_factor={hole_match_factor:.3f}  final={final_score:.4f}  "
            f"strategy={result.strategy}"
        )
        results.append((name, result, final_score))

    results.sort(key=lambda x: x[2], reverse=True)
    return [
        TemplateMatch(name=name, result=result, rank=i + 1)
        for i, (name, result, _) in enumerate(results)
    ]


# ===========================================================================
# Backward compatibility shims (kept so existing tests don't break)
# ===========================================================================

def _compute_alignment_score(
    aligned_image: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Filled-silhouette IoU. Kept for test compatibility."""
    cad_filled  = _fill_silhouette(aligned_image)
    real_filled = _fill_silhouette(reference)
    intersection = int(np.logical_and(cad_filled > 0, real_filled > 0).sum())
    union        = int(np.logical_or( cad_filled > 0, real_filled > 0).sum())
    return float(intersection) / float(union) if union > 0 else 0.0


def _validate_similarity(M: np.ndarray) -> bool:
    """Scale-range check on a similarity matrix. Kept for test compatibility."""
    from .constants import SCALE_MIN, SCALE_MAX
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    valid = SCALE_MIN <= scale <= SCALE_MAX
    if not valid:
        logger.debug(f"_validate_similarity: scale {scale:.3f} outside [{SCALE_MIN}, {SCALE_MAX}]")
    return valid
