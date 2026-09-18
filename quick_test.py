"""
quick_test.py — End-to-End Pipeline Verification Entry Point

PIPELINE FLOW:
INPUT IMAGE
    ↓
Phase 1: Blueprint identification/alignment (using frozen Phase 1 API)
    ↓
DXF resolution
    ↓
Expected feature extraction  
    ↓
Phase 2A: Canonical preprocessing
    ↓
STOP (Pipeline intentionally stops here)

This verification entry point processes through Phase 2A preprocessing only.
Uses the existing frozen Phase 1 API and canonical Phase 2A preprocessor.

Usage:
    python quick_test.py
"""

import logging
import cv2
import numpy as np
from pathlib import Path

from cad_image_alignment import match_best_template, TemplateMatch
from cad_image_alignment.constants import COVERAGE_THRESHOLD
from dxf_resolver import resolve_dxf, DXFResolutionError
from feature_extraction import extract_expected_features
from feature_inspection.preprocessing import CanonicalPreprocessor
from feature_inspection.config import PREPROCESSING_MAX_RESOLUTION

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUTS_DIR = Path("data/inputs")
BLUEPRINTS_DIR = Path("data/blueprints")
OUTPUTS_DIR = Path("outputs")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def collect_images(folder: Path) -> list[Path]:
    """Return all image files in a folder, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def prepare_blueprint_edge_map(path: Path) -> np.ndarray:
    """
    Prepare a CAD blueprint PNG as an edge map for Phase 1 alignment.
    Uses minimal preprocessing for alignment only.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load blueprint: {path}")
    
    # Basic blueprint preprocessing: invert, blur, Canny, dilate
    inv = cv2.bitwise_not(img)
    blur = cv2.GaussianBlur(inv, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return edges


def prepare_real_image_edge_map(img: np.ndarray) -> np.ndarray:
    """
    Prepare a real product photo as an edge map for Phase 1 alignment.
    Uses minimal preprocessing - Phase 2A uses CanonicalPreprocessor separately.
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def print_result(name: str, result) -> None:
    """Print a formatted alignment result to stdout."""
    print(f"   Coverage:        {result.coverage:.4f}  ({result.coverage:.1%})")
    print(f"   Edge Score:      {result.edge_score:.4f}  ({result.edge_score:.1%})")
    print(f"   Combined Score:  {result.combined_score:.4f}  ({result.combined_score:.1%})")
    print(f"   Hole Diff:       {result.hole_diff}")
    print(f"   Alignment Score: {result.alignment_score:.4f}")
    print(f"   Strategy:        {result.strategy}")



def main() -> None:
    print("=" * 70)
    print("Phase 0 → Phase 1 → Phase 2A Pipeline Verification")
    print("=" * 70)

    inputs = collect_images(INPUTS_DIR)
    blueprints = collect_images(BLUEPRINTS_DIR)
    
    if not inputs:
        print(f"No images found in '{INPUTS_DIR}'")
        return
    if not blueprints:
        print(f"No blueprints found in '{BLUEPRINTS_DIR}'")
        return

    print(f"\nFound {len(inputs)} input(s) in '{INPUTS_DIR}'")
    print(f"Found {len(blueprints)} blueprint(s) in '{BLUEPRINTS_DIR}'")

    # Load blueprints using minimal preprocessing for alignment
    print(f"\nLoading blueprints...")
    blueprint_templates = []
    for bp in blueprints:
        try:
            edges = prepare_blueprint_edge_map(bp)
            edge_count = int(np.count_nonzero(edges))
            blueprint_templates.append((bp.stem, edges))
            print(f"   [OK] {bp.name}  ({edge_count} edge px)")
        except Exception as e:
            print(f"   [FAIL] {bp.name}  ({e})")

    if not blueprint_templates:
        print("No blueprints could be loaded")
        return

    # Initialize canonical Phase 2A preprocessor
    preprocessor = CanonicalPreprocessor()

    # Process each input
    for i, inp in enumerate(inputs, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(inputs)}]  Input: {inp.name}")
        print(f"{'=' * 70}")
        
        try:
            # Load input image
            img = cv2.imread(str(inp), cv2.IMREAD_COLOR)
            if img is None:
                print(f"   [FAIL] Cannot load image: {inp}")
                continue
                
            h, w = img.shape[:2]
            print(f"   Shape: ({h}, {w})")
            
            # Resize if needed (use centralized configuration)
            if max(w, h) > PREPROCESSING_MAX_RESOLUTION:
                scale = PREPROCESSING_MAX_RESOLUTION / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"   Resized to: ({new_h}, {new_w})")
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Create output directory
            out_dir = OUTPUTS_DIR / inp.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # PHASE 1: Blueprint identification/alignment using frozen API
            print(f"\n   Phase 1: Blueprint identification/alignment")
            print(f"   Preparing real image edge map...")
            
            real_edges = prepare_real_image_edge_map(gray)
            edge_count = int(np.count_nonzero(real_edges))
            print(f"   Real edges: {edge_count} pixels")
            
            # Use frozen Phase 1 API for alignment
            print(f"   Aligning against {len(blueprint_templates)} blueprint(s)...")
            matches = match_best_template(blueprint_templates, real_edges)
            
            if not matches:
                print(f"   [FAIL] Phase 1: No successful alignments")
                continue
            
            best_match = matches[0]
            print(f"   Results (ranked by final score):")
            for i, match in enumerate(matches, 1):
                print(f"\n   {i}. {match.name}")
                print_result(match.name, match.result)
            
            # Check coverage threshold from frozen Phase 1
            if best_match.result.coverage < COVERAGE_THRESHOLD:
                print(f"\n   [FAIL] Phase 1: Identification failed")
                print(f"          Best match '{best_match.name}' coverage {best_match.result.coverage:.1%} < {COVERAGE_THRESHOLD:.1%}")
                continue
                
            print(f"\n   [PASS] Phase 1: Blueprint identification")
            print(f"          Matched: '{best_match.name}' (coverage {best_match.result.coverage:.1%})")
            
            # DXF resolution
            try:
                dxf_path = resolve_dxf(best_match.name)
                print(f"          DXF resolved: {dxf_path.name}")
            except DXFResolutionError as e:
                print(f"          DXF resolution failed: {e}")
                dxf_path = None
            
            # Expected Feature Extraction
            if dxf_path:
                print(f"\n   Expected Feature Extraction")
                print(f"   Processing DXF: {dxf_path.name}...")
                try:
                    expected_features = extract_expected_features(dxf_path)
                    
                    print(f"   [PASS] Expected Feature Extraction")
                    print(f"          Features found: {len(expected_features.features)}")
                    
                    # Count feature types
                    circles = sum(1 for f in expected_features.features if f.feature_type.name == "CIRCLE")
                    holes = sum(1 for f in expected_features.features if f.feature_type.name == "THROUGH_HOLE")
                    
                    print(f"          Circles: {circles}")
                    print(f"          Through holes: {holes}")
                    
                    if expected_features.features:
                        avg_confidence = sum(f.confidence for f in expected_features.features) / len(expected_features.features)
                        print(f"          Average confidence: {avg_confidence:.3f}")
                        
                except Exception as e:
                    print(f"   [FAIL] Expected Feature Extraction")
                    print(f"          Error: {e}")
            
            # PHASE 2A: Canonical Preprocessing ONLY
            print(f"\n   Phase 2A: Canonical Preprocessing")
            print(f"   Applying geometry-preserving preprocessing...")
            
            try:
                # Apply canonical Phase 2A preprocessing
                preprocessing_result = preprocessor.preprocess_image(inp)
                
                print(f"   [PASS] Phase 2A: Canonical Preprocessing")
                print(f"          Preprocessing: {'SUCCESS' if preprocessing_result.preprocessing_successful else 'FAILED'}")
                print(f"          Product isolation: {preprocessing_result.product_area_pixels} pixels ({preprocessing_result.product_area_fraction:.1%})")
                print(f"          Raw internal edges: {preprocessing_result.raw_internal_edge_density:.3f}")
                print(f"          Filtered internal edges: {preprocessing_result.filtered_internal_edge_density:.3f}")
                print(f"          Texture reduction: {preprocessing_result.internal_edge_reduction_ratio:.1%}")
                print(f"          Final edge density: {preprocessing_result.final_edge_density:.3f}")
                
                # Show warnings if any
                if preprocessing_result.border_touching_foreground:
                    print(f"          ⚠ WARNING: Product mask touches image border")
                if preprocessing_result.mask_solidity < 0.5:
                    print(f"          ⚠ WARNING: Low mask solidity ({preprocessing_result.mask_solidity:.2f})")
                
                # Save Phase 2A preprocessing outputs
                phase2_dir = out_dir / "phase_2a_preprocessing"
                saved_files = preprocessor.save_preprocessing_outputs(preprocessing_result, phase2_dir)
                
                print(f"          Preprocessing outputs saved to: {phase2_dir}")
                print(f"          - preprocessing_montage.png (visual validation)")
                print(f"          - mask_overlay.png (mask inspection)")
                print(f"          - final_preprocessed_edges.png (ready for feature detection)")
                
                # CLEAR STOP MESSAGE
                print(f"\n   Phase 2A preprocessing complete.")
                print(f"   Note: Phase 2B actual feature extraction is implemented but not run by quick_test.py")
                print(f"   Use 'python validate_actual_feature_extraction.py' to run Phase 2B.")
                
            except Exception as e:
                print(f"\n   [FAIL] Phase 2A: Canonical Preprocessing")
                print(f"          Error: {e}")
                logger.error(f"Phase 2A preprocessing failed for {inp.name}: {e}")
                continue
                
        except Exception as e:
            print(f"   [FAIL] Processing failed for {inp.name}: {e}")
            logger.error(f"Processing failed for {inp.name}: {e}")
            continue
    
    print(f"\n{'=' * 70}")
    print(f"Phase 0→1→2A pipeline verification complete.")
    print(f"For Phase 2B feature extraction, run: python validate_actual_feature_extraction.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()