"""
quick_test.py  —  Inspection pipeline entry point.

Pipeline:
  Stage 1: Identification & Alignment
    - Load + cap resolution
    - Preprocess: isolate part → edge map
    - Align CAD blueprints against the edge map
    - Rank candidates, gate on coverage threshold
    - Save alignment overlays

To adapt to new cameras, lighting, or part types:
  - Tune MAX_LONG_EDGE, GRADIENT_THRESHOLD, BORDER_MARGIN_FRACTION below.
  - All alignment parameters live in cad_image_alignment/constants.py.

Usage:
    python quick_test.py
"""

import logging

import cv2
import numpy as np
from pathlib import Path

from cad_image_alignment import align, match_best_template
from dxf_resolver import resolve_dxf, DXFResolutionError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUTS_DIR     = Path("data/inputs")
BLUEPRINTS_DIR = Path("data/blueprints")
OUTPUTS_DIR    = Path("outputs")
IMAGE_EXTS     = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Preprocessing parameters
# (change these to adapt to a new camera or environment)
# ---------------------------------------------------------------------------

# Downsample input images so the long edge is at most this many pixels.
# Larger = more accurate but slower. The coarse grid runs at 12.5% of this,
# so 1600 → 200px coarse resolution, which is sufficient for most parts.
MAX_LONG_EDGE = 1600

# Morphological gradient threshold for extracting internal edges from the
# masked part. Lower = more edges from lightly-lit surfaces; higher = fewer
# but cleaner edges. Range: 5–40 (default 20 works for most metal parts).
GRADIENT_THRESHOLD = 12

# Canny edge thresholds for the outer boundary of the part mask.
CANNY_OUTER_LOW  = 50
CANNY_OUTER_HIGH = 150

# Minimum component area (pixels) to consider during part isolation.
# Prevents tiny noise blobs from being selected as the part.
MIN_COMPONENT_AREA_PX = 200

# Components within this fraction of the short image edge from the image
# border are rejected (they are floor/table/wall, not the part).
BORDER_MARGIN_FRACTION = 0.01

# Morphological kernel size used to close/open the Otsu mask.
MASK_MORPH_KERNEL_SIZE = 9
MASK_MORPH_CLOSE_ITERS = 3
MASK_MORPH_OPEN_ITERS  = 2

# CLAHE parameters for contrast normalisation inside the part mask.
# clipLimit controls contrast enhancement strength; tileGridSize controls
# the local neighbourhood size. Increase clipLimit for very dark/uneven parts.
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# Maximum fraction of mask area that internal edges may cover.
# Auto-raises gradient threshold if exceeded to suppress texture noise.
INTERNAL_EDGE_MAX_DENSITY = 0.08

# Gradient threshold step multiplier used when density is too high.
GRADIENT_STEP_MULTIPLIER  = 1.5

# Maximum number of threshold-raising retries in extract_internal_edges.
GRADIENT_MAX_RETRIES = 4

# Component scoring weights: area weight + centrality weight must sum to 1.
COMPONENT_AREA_WEIGHT       = 0.95
COMPONENT_CENTRALITY_WEIGHT = 0.05

# Minimum border margin enforced regardless of BORDER_MARGIN_FRACTION.
BORDER_MARGIN_MIN_PX = 3

# Morphological kernel sizes used inside extract_internal_edges.
GRADIENT_KERNEL_SIZE = 3   # for morphological gradient
EDGE_CLOSE_KERNEL_SIZE = 2  # for closing small gaps in internal edges

# CAD blueprint preprocessing parameters.
CAD_BLUR_KERNEL    = (3, 3)
CAD_CANNY_LOW      = 20
CAD_CANNY_HIGH     = 80
CAD_DILATE_KERNEL  = (2, 2)

# Real image blur kernel size for Otsu mask extraction.
REAL_BLUR_KERNEL = (5, 5)


# ===========================================================================
# ── IMAGE LOADING ─────────────────────────────────────────────────────────
# ===========================================================================

def load_image(path: Path) -> np.ndarray | None:
    """Load a grayscale image, returning None if the file can't be read."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"cv2.imread returned None for {path}")
    return img


def cap_resolution(img: np.ndarray, max_long_edge: int = MAX_LONG_EDGE) -> np.ndarray:
    """
    Downsample so the long edge ≤ max_long_edge.
    Preserves aspect ratio. Returns the original if already small enough.
    Change max_long_edge for higher-resolution cameras.
    """
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / long_edge
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


# ===========================================================================
# ── PART ISOLATION ────────────────────────────────────────────────────────
# ===========================================================================

def _score_component(
    x0: int, y0: int, x1: int, y1: int,
    area: int,
    img_w: int, img_h: int,
    border_margin: int,
) -> float:
    """
    Score a connected component as a part candidate.
    Returns -1 if the component touches the image border (floor/table/wall).
    Otherwise returns area as the primary score, with centrality used only
    as a small tiebreaker (5%) so a slightly off-centre part isn't penalised.
    """
    if x0 <= border_margin or y0 <= border_margin or \
       x1 >= img_w - border_margin or y1 >= img_h - border_margin:
        return -1.0   # touches border → reject

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    dist = np.hypot(cx - img_w / 2, cy - img_h / 2)
    max_dist = np.hypot(img_w / 2, img_h / 2)
    centrality = 1.0 - dist / max_dist
    return float(area) * (COMPONENT_AREA_WEIGHT + COMPONENT_CENTRALITY_WEIGHT * centrality)


def _threshold_and_pick_part(
    blur: np.ndarray,
    thresh_flags: int,
    kernel_size: int,
    close_iters: int,
    open_iters: int,
    min_area: int,
    border_margin: int,
) -> tuple[np.ndarray, float]:
    """
    Apply Otsu threshold with given polarity, morphologically clean the mask,
    then pick the single connected component that best represents the part.

    Returns (best_mask, best_score) where best_score = -1 if nothing found.
    """
    _, raw = cv2.threshold(blur, 0, 255, thresh_flags)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=close_iters)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  k, iterations=open_iters)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    h, w = raw.shape
    border_margin_px = max(BORDER_MARGIN_MIN_PX, int(min(h, w) * border_margin))

    best_mask  = np.zeros_like(raw)
    best_score = -1.0

    for lbl in range(1, num_labels):
        s    = stats[lbl]
        area = s[cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x0, y0 = s[cv2.CC_STAT_LEFT],  s[cv2.CC_STAT_TOP]
        x1      = x0 + s[cv2.CC_STAT_WIDTH]
        y1      = y0 + s[cv2.CC_STAT_HEIGHT]
        score  = _score_component(x0, y0, x1, y1, area, w, h, border_margin_px)
        if score > best_score:
            best_score = score
            best_mask  = np.where(labels == lbl, np.uint8(255), np.uint8(0))

    return best_mask, best_score


def isolate_part(
    blur: np.ndarray,
    kernel_size: int = MASK_MORPH_KERNEL_SIZE,
    close_iters: int = MASK_MORPH_CLOSE_ITERS,
    open_iters:  int = MASK_MORPH_OPEN_ITERS,
    min_area:    int = MIN_COMPONENT_AREA_PX,
    border_margin: float = BORDER_MARGIN_FRACTION,
) -> np.ndarray:
    """
    Isolate the part from the background using adaptive Otsu thresholding.

    Tries both polarities (BINARY_INV for dark parts on bright backgrounds,
    BINARY for bright parts on dark backgrounds) and picks whichever gives
    the highest-scoring non-border component.

    Falls back to the raw largest-component approach if both fail to find a
    component that doesn't touch the image border.

    Parameters can all be overridden to handle unusual shooting conditions.
    """
    inv_mask, inv_score = _threshold_and_pick_part(
        blur, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        kernel_size, close_iters, open_iters, min_area, border_margin,
    )
    fwd_mask, fwd_score = _threshold_and_pick_part(
        blur, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        kernel_size, close_iters, open_iters, min_area, border_margin,
    )

    if inv_score >= fwd_score and np.any(inv_mask):
        return inv_mask
    if np.any(fwd_mask):
        return fwd_mask

    # Both polarities found only border-touching components — fall back to
    # largest-component without border rejection (better than nothing).
    logger.warning("isolate_part: no non-border component found, using largest component")
    _, raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=close_iters)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  k, iterations=open_iters)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, np.uint8(255), np.uint8(0))
    return raw


# ===========================================================================
# ── EDGE EXTRACTION ───────────────────────────────────────────────────────
# ===========================================================================

def extract_internal_edges(
    img: np.ndarray,
    mask: np.ndarray,
    gradient_threshold: int = GRADIENT_THRESHOLD,
) -> np.ndarray:
    """
    Extract edges from inside the part mask.

    Pipeline:
      1. CLAHE on the masked region — normalises local brightness so bolt holes
         on the dark side of the part have the same contrast as the bright side.
         Without this, holes under uneven lighting produce gradients below the
         threshold and are silently dropped from the edge map.
      2. Morphological gradient on the CLAHE-enhanced image.
      3. Adaptive threshold — starts at gradient_threshold, auto-raises if
         edge density exceeds 8% of mask area (prevents texture flooding on
         flat box parts while keeping the lower threshold for circular holes).
    """
    masked = cv2.bitwise_and(img, img, mask=mask)

    # CLAHE: equalise local contrast so all bolt holes are equally visible
    # regardless of where the lighting hotspot falls on the part surface.
    clahe  = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(masked)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (GRADIENT_KERNEL_SIZE, GRADIENT_KERNEL_SIZE))
    grad = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, k)

    mask_area = int(np.count_nonzero(mask))

    thresh = gradient_threshold
    for _ in range(GRADIENT_MAX_RETRIES):
        _, edges = cv2.threshold(grad, thresh, 255, cv2.THRESH_BINARY)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (EDGE_CLOSE_KERNEL_SIZE, EDGE_CLOSE_KERNEL_SIZE))
        )
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        density = float(np.count_nonzero(edges)) / float(mask_area) if mask_area > 0 else 0.0
        if density <= INTERNAL_EDGE_MAX_DENSITY:
            break
        thresh = int(thresh * GRADIENT_STEP_MULTIPLIER)

    return edges


def extract_outer_boundary(mask: np.ndarray,
                           low: int = CANNY_OUTER_LOW,
                           high: int = CANNY_OUTER_HIGH) -> np.ndarray:
    """
    Extract the outer boundary of the part mask using Canny.
    Adjust low/high thresholds for thinner or thicker boundary lines.
    """
    return cv2.Canny(mask, low, high)


# ===========================================================================
# ── FULL PREPROCESSING PIPELINE ───────────────────────────────────────────
# ===========================================================================

def preprocess_cad(path: Path) -> np.ndarray:
    """
    Prepare a CAD blueprint PNG as an edge map for alignment.
    Inverts the image (blueprints are dark lines on white), blurs, Canny, dilates.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load blueprint: {path}")
    inv   = cv2.bitwise_not(img)
    blur  = cv2.GaussianBlur(inv, CAD_BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, CAD_CANNY_LOW, CAD_CANNY_HIGH)
    edges = cv2.dilate(edges, np.ones(CAD_DILATE_KERNEL, np.uint8))
    return edges


def preprocess_real(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare a real product photo as an edge map for alignment.

    Returns
    -------
    (real_edges, mask)
      real_edges : uint8 binary edge map — combination of outer boundary
                   and internal morphological-gradient edges
      mask       : uint8 binary part silhouette mask

    To adapt:
      - Change GRADIENT_THRESHOLD for more/fewer internal edges.
      - Change MASK_MORPH_KERNEL_SIZE / CLOS_ITERS for noisier masks.
      - Change BORDER_MARGIN_FRACTION if the part is very close to the frame edge.
    """
    blur = cv2.GaussianBlur(img, REAL_BLUR_KERNEL, 0)
    mask = isolate_part(blur)

    internal_edges = extract_internal_edges(img, mask)
    outer_boundary = extract_outer_boundary(mask)
    real_edges     = cv2.bitwise_or(internal_edges, outer_boundary)

    return real_edges, mask


# ===========================================================================
# ── OUTPUT HELPERS ────────────────────────────────────────────────────────
# ===========================================================================

def collect_images(folder: Path) -> list[Path]:
    """Return all image files in a folder, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def print_result(name: str, result, rank: int = None) -> None:
    """Print a formatted alignment result to stdout."""
    prefix = f"  {name}" if rank == 1 else f"  {rank}. {name}" if rank else f"   {name}"
    print(f"\n{prefix}")
    print(f"   Coverage:        {result.coverage:.4f}  ({result.coverage:.1%})")
    print(f"   Edge Score:      {result.edge_score:.4f}  ({result.edge_score:.1%})")
    print(f"   Combined Score:  {result.combined_score:.4f}  ({result.combined_score:.1%})  <- ranked on this")
    print(f"   Hole Diff:       {result.hole_diff}")
    print(f"   Alignment Score: {result.alignment_score:.4f}")
    print(f"   Strategy:        {result.strategy}")


def save_outputs(
    out_dir: Path,
    blueprint_stem: str,
    real_edges: np.ndarray,
    mask: np.ndarray,
    result,
    is_best: bool = False,
    debug_written: bool = False,
) -> bool:
    """
    Write alignment outputs for one blueprint.

    Outputs:
      <blueprint>_aligned.png   — warped CAD edge map
      <blueprint>_overlay.png   — red=CAD, green=real, yellow=overlap
      best_aligned.png / best_overlay.png  — copies for the winning blueprint
      debug_mask.png / debug_real_edges.png — written once per input image
    """
    prefix = f"{blueprint_stem}_"
    cv2.imwrite(str(out_dir / f"{prefix}aligned.png"), result.aligned_image)

    overlay = np.zeros((*real_edges.shape, 3), dtype=np.uint8)
    overlay[:, :, 2] = result.aligned_image   # red  = CAD
    overlay[:, :, 1] = real_edges             # green = real
    cv2.imwrite(str(out_dir / f"{prefix}overlay.png"), overlay)

    if not debug_written:
        cv2.imwrite(str(out_dir / "debug_mask.png"),       mask)
        cv2.imwrite(str(out_dir / "debug_real_edges.png"), real_edges)

    if is_best:
        cv2.imwrite(str(out_dir / "best_aligned.png"), result.aligned_image)
        cv2.imwrite(str(out_dir / "best_overlay.png"), overlay)

    return True


# ===========================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("Inspection Pipeline  --  Batch Mode")
    print("=" * 70)

    # ── Pre-flight ──────────────────────────────────────────────────────────
    for folder in (INPUTS_DIR, BLUEPRINTS_DIR):
        if not folder.exists():
            print(f"\n[FAIL] Folder not found: {folder.resolve()}")
            return

    input_images    = collect_images(INPUTS_DIR)
    blueprint_paths = collect_images(BLUEPRINTS_DIR)

    if not input_images:
        print(f"\n[FAIL] No images found in '{INPUTS_DIR}'")
        return
    if not blueprint_paths:
        print(f"\n[FAIL] No blueprints found in '{BLUEPRINTS_DIR}'")
        return

    print(f"\nFound {len(input_images)} input(s) in '{INPUTS_DIR}'")
    print(f"Found {len(blueprint_paths)} blueprint(s) in '{BLUEPRINTS_DIR}'")

    # ── Load CAD templates ──────────────────────────────────────────────────
    print("\nLoading blueprints...")
    templates: list[tuple[str, np.ndarray]] = []
    for bp in blueprint_paths:
        try:
            edges = preprocess_cad(bp)
            templates.append((bp.stem, edges))
            print(f"   [OK] {bp.name}  ({np.count_nonzero(edges)} edge px)")
        except FileNotFoundError as exc:
            print(f"   [FAIL] {exc}")

    if not templates:
        print("\n[FAIL] No valid blueprints loaded.")
        return

    # ── Per-image loop ───────────────────────────────────────────────────────
    for idx, inp in enumerate(input_images, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(input_images)}]  Input: {inp.name}")
        print(f"{'=' * 70}")

        # Load + cap resolution
        real = load_image(inp)
        if real is None:
            print(f"\n   [FAIL] Cannot load image: {inp.name}")
            continue
        real = cap_resolution(real)
        print(f"   Shape: {real.shape}")

        # Preprocess
        print(f"\n   Stage 1: Identification")
        print(f"   Preprocessing...")
        real_edges, mask = preprocess_real(real)
        print(f"   Real edges: {np.count_nonzero(real_edges)} pixels")

        # Output folder
        out_dir = OUTPUTS_DIR / inp.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output folder: {out_dir.resolve()}")

        # Always write debug files so failures are diagnosable
        cv2.imwrite(str(out_dir / "debug_mask.png"),       mask)
        cv2.imwrite(str(out_dir / "debug_real_edges.png"), real_edges)

        # Align
        print(f"   Aligning against {len(templates)} blueprint(s)...")
        matches = match_best_template(templates, real_edges)

        print(f"\n   Results -- ranked by combined score (coverage + edge overlap):")
        for m in matches:
            print_result(m.name, m.result, rank=m.rank)

        best = matches[0]

        # Gate
        if not best.result.identified:
            print(f"\n   [FAIL] Stage 1: Identification")
            print(f"          Reason: '{best.name}' coverage {best.result.coverage:.1%} "
                  f"is below threshold")
            continue

        print(f"\n   [PASS] Stage 1: Identification")
        print(f"          Matched: '{best.name}'  coverage {best.result.coverage:.1%}")

        # PHASE 0: Resolve blueprint to corresponding DXF file
        try:
            dxf_path = resolve_dxf(best.name)
            print(f"          DXF resolved: {dxf_path.name}")
            logger.info(f"Blueprint '{best.name}' resolved to DXF: {dxf_path}")
            
            # PHASE 1: Extract expected features from DXF
            try:
                from feature_extraction import extract_expected_features
                from feature_extraction.visualization import visualize_expected_features
                
                print(f"\n   Stage 2: Expected Feature Extraction")
                print(f"   Processing DXF: {dxf_path.name}...")
                
                expected_features = extract_expected_features(dxf_path)
                
                print(f"   [PASS] Stage 2: Expected Feature Extraction")
                print(f"          Features found: {expected_features.total_feature_count}")
                print(f"          Circles: {expected_features.circle_count}")
                print(f"          Through holes: {expected_features.through_hole_count}")
                print(f"          Average confidence: {expected_features.average_confidence:.3f}")
                
                # Generate feature visualization
                print(f"   Generating expected feature visualization...")
                viz_path = visualize_expected_features(expected_features, 
                                                     output_path=out_dir / "expected_features.png")
                print(f"   [OK] Expected features visualization: {viz_path.name}")
                
                # Store complete results for future phases
                identification_result = {
                    "identified_blueprint": best.name,
                    "dxf_path": dxf_path,
                    "transform_matrix": best.result.transform_matrix,
                    "alignment_result": best.result,
                    "coverage": best.result.coverage,
                    "alignment_score": best.result.alignment_score,
                    "strategy": best.result.strategy,
                    "expected_features": expected_features,  # Phase 1 output
                }
                
                # PHASE 2: STEP 1 - Preprocessing Only
                print(f"\n   Stage 3: Phase 2 Preprocessing")
                print(f"   Applying geometry-preserving Phase 2 preprocessing...")
                
                try:
                    from feature_inspection.actual.canonical_preprocessor import CanonicalPreprocessor
                    
                    # Initialize canonical Phase 2 preprocessor
                    preprocessor = CanonicalPreprocessor()
                    
                    # Apply preprocessing using geometry-preserving algorithms
                    preprocessing_result = preprocessor.preprocess_image(inp)
                    
                    # Create Phase 2 output directory
                    phase2_output_dir = out_dir.parent / inp.stem / "phase_2"
                    
                    # Save preprocessing outputs
                    saved_files = preprocessor.save_preprocessing_outputs(
                        result=preprocessing_result,
                        output_dir=phase2_output_dir
                    )
                    
                    print(f"   [PASS] Stage 3: Phase 2 Preprocessing")
                    print(f"          Preprocessing: {'SUCCESS' if preprocessing_result.preprocessing_successful else 'FAILED'}")
                    
                    if preprocessing_result.preprocessing_successful:
                        print(f"          Product isolation: {preprocessing_result.product_area_pixels} pixels ({preprocessing_result.product_area_fraction:.1%} of image)")
                        print(f"          Raw internal edges: {preprocessing_result.raw_internal_edge_density:.3f}")
                        print(f"          Filtered internal edges: {preprocessing_result.filtered_internal_edge_density:.3f}")
                        print(f"          Texture reduction: {preprocessing_result.internal_edge_reduction_ratio:.1%}")
                        print(f"          Outer boundary edges: {preprocessing_result.outer_boundary_density:.3f}")
                        print(f"          Final edge density: {preprocessing_result.final_edge_density:.3f}")
                        
                        if preprocessing_result.border_touching_foreground:
                            print(f"          ⚠ WARNING: Product mask touches image border")
                        if preprocessing_result.mask_solidity < 0.7:
                            print(f"          ⚠ WARNING: Low mask solidity ({preprocessing_result.mask_solidity:.2f})")
                    else:
                        print(f"          Error: {preprocessing_result.configuration_snapshot.get('error', 'Unknown preprocessing failure')}")
                    
                    print(f"          Phase 2 preprocessing outputs: {phase2_output_dir}")
                    print(f"          - original_image.png")
                    print(f"          - product_mask.png") 
                    print(f"          - mask_overlay.png")
                    print(f"          - isolated_product.png")
                    print(f"          - internal_geometry_edges.png")
                    print(f"          - outer_boundary_edges.png")
                    print(f"          - final_preprocessed_edges.png")
                    print(f"          - preprocessing_montage.png")
                    print(f"          - preprocessing_metadata.json")
                    
                    # STOP HERE - NO FEATURE DETECTION YET
                    print(f"   [INFO] Phase 2 Step 1 Complete - Ready for Step 2 (Feature Detection)")
                    
                except Exception as e:
                    print(f"\n   [FAIL] Stage 3: Phase 2 Preprocessing")
                    print(f"          Error: {e}")
                    logger.error(f"Phase 2 preprocessing failed for {inp.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with remaining images even if Phase 2 fails
                
            except Exception as e:
                print(f"\n   [FAIL] Stage 2: Expected Feature Extraction")
                print(f"          Error: {e}")
                logger.error(f"Failed to extract expected features from {dxf_path}: {e}")
                # Continue with alignment outputs even if feature extraction fails
                
        except DXFResolutionError as e:
            print(f"\n   [FAIL] DXF Resolution")
            print(f"          Error: {e}")
            logger.error(f"Failed to resolve DXF for blueprint '{best.name}': {e}")
            continue

        # Save outputs
        print(f"\n   Saving alignment outputs...")
        for m in matches:
            save_outputs(
                out_dir=out_dir, blueprint_stem=m.name,
                real_edges=real_edges, mask=mask,
                result=m.result, is_best=(m.rank == 1),
                debug_written=True,
            )
            tag = " <- best" if m.rank == 1 else ""
            print(f"   [OK] {m.name}_aligned.png  |  {m.name}_overlay.png{tag}")

    print(f"\n{'=' * 70}")
    print(f"[OK] Done!  Outputs in '{OUTPUTS_DIR.resolve()}'")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
