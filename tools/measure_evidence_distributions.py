"""
Step 11 — Threshold Tuning Measurement Tool

Runs the full preprocessing + circle-extraction pipeline over every image in
data/inputs, intercepts every Hough candidate (accepted AND rejected) before
the final confidence gate, and records all evidence metrics.

Run from the repo root:
    python tools/measure_evidence_distributions.py

Output:
  - tools/evidence_distributions.csv   (raw per-candidate rows)
  - tools/threshold_report.txt         (summary statistics)

The CSV is the primary artefact: load it in a spreadsheet or pandas to
examine the separation between genuine and false candidates.
"""

import csv
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

# Make sure the repo root is on sys.path when running as a script
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from feature_inspection.preprocessing.canonical_preprocessor import CanonicalPreprocessor
from feature_inspection.actual.circle_extractor import CircleExtractor

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

INPUT_DIR = _REPO_ROOT / "data" / "inputs"
OUTPUT_CSV = _REPO_ROOT / "tools" / "evidence_distributions.csv"
OUTPUT_REPORT = _REPO_ROOT / "tools" / "threshold_report.txt"

# Collect at most this many images to keep runtime reasonable
MAX_IMAGES = 40

# ------------------------------------------------------------------ #
# Evidence interception                                                #
# ------------------------------------------------------------------ #

_intercepted: list[dict] = []   # list of per-candidate dicts


def _patched_validate(original_validate):
    """Wrap _validate_hough_candidate so every call is recorded."""
    def wrapper(self, center, radius, edge_image, intensity_image, product_mask):
        result = original_validate(self, center, radius,
                                   edge_image, intensity_image, product_mask)
        # Record ALL candidates regardless of valid/invalid
        _intercepted.append({
            "center_x":   center[0],
            "center_y":   center[1],
            "radius":     radius,
            "valid":      result.get("valid", False),
            "reason":     result.get("rejection_reason", ""),
            "edge_support":                  result.get("edge_support",           0.0),
            "angular_coverage":              result.get("angular_coverage",        0.0),
            "sector_coverage":               result.get("sector_coverage",         0.0),
            "angular_uniformity":            result.get("angular_uniformity",      0.0),
            "max_gap_ratio":                 result.get("max_gap_ratio",           1.0),
            "radial_consistency":            result.get("radial_consistency",      0.0),
            "radial_error_median":           result.get("radial_error_median",     0.0),
            "radial_error_p95":              result.get("radial_error_p95",        0.0),
            "gradient_orientation":          result.get("gradient_orientation_consistency", 0.0),
            "local_contrast":                result.get("local_contrast",          0.0),
            "intensity_consistency":         result.get("intensity_consistency",   0.0),
            "geometric_consistency":         result.get("geometric_consistency",   0.0),
            "contour_quality":               result.get("contour_quality",         0.0),
            "circularity_evidence":          result.get("circularity_evidence",    0.0),
            "solidity_evidence":             result.get("solidity_evidence",       0.0),
        })
        return result
    return wrapper


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": float("nan"), "p10": float("nan"),
                "p25": float("nan"), "median": float("nan"),
                "p75": float("nan"), "p90": float("nan"), "max": float("nan")}
    a = np.array(values)
    return {
        "n":      len(a),
        "min":    float(np.min(a)),
        "p10":    float(np.percentile(a, 10)),
        "p25":    float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75":    float(np.percentile(a, 75)),
        "p90":    float(np.percentile(a, 90)),
        "max":    float(np.max(a)),
    }


def _fmt(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    return (f"n={s['n']}  min={s['min']:.3f}  p10={s['p10']:.3f}  "
            f"p25={s['p25']:.3f}  med={s['median']:.3f}  "
            f"p75={s['p75']:.3f}  p90={s['p90']:.3f}  max={s['max']:.3f}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    images = sorted(INPUT_DIR.glob("*.jpeg"))[:MAX_IMAGES]
    if not images:
        print(f"No .jpeg images found in {INPUT_DIR}")
        sys.exit(1)
    print(f"Processing {len(images)} images …")

    preprocessor = CanonicalPreprocessor()
    extractor    = CircleExtractor()

    _intercepted.clear()
    image_tag: dict[int, str] = {}   # maps intercepted-index → image filename

    # Patch _validate_hough_candidate so every candidate is recorded
    with patch.object(
        CircleExtractor,
        "_validate_hough_candidate",
        _patched_validate(CircleExtractor._validate_hough_candidate),
    ):
        for img_path in images:
            print(f"  {img_path.name}", end="", flush=True)
            idx_before = len(_intercepted)
            try:
                prep = preprocessor.preprocess_image(img_path)
                if not prep.preprocessing_successful or not prep.isolation_successful:
                    print(" [skip: preprocessing failed]")
                    continue

                extractor.extract_circles(
                    prep.internal_geometry_edges,
                    prep.isolated_product_image,
                    prep.product_mask,
                    prep.raw_internal_geometry_edges,
                    prep.outer_boundary_edges,
                )
                n_new = len(_intercepted) - idx_before
                # Tag each new record with the image filename
                for k in range(idx_before, len(_intercepted)):
                    _intercepted[k]["image"] = img_path.name
                print(f" → {n_new} candidates")
            except Exception as exc:
                print(f" [ERROR: {exc}]")

    if not _intercepted:
        print("No candidates collected — check that images preprocessed correctly.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Write CSV                                                           #
    # ------------------------------------------------------------------ #
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image", "center_x", "center_y", "radius", "valid", "reason",
        "edge_support", "angular_coverage", "sector_coverage",
        "angular_uniformity", "max_gap_ratio",
        "radial_consistency", "radial_error_median", "radial_error_p95",
        "gradient_orientation", "local_contrast", "intensity_consistency",
        "geometric_consistency", "contour_quality",
        "circularity_evidence", "solidity_evidence",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_intercepted)
    print(f"\nCSV written to {OUTPUT_CSV}")

    # ------------------------------------------------------------------ #
    # Compute summary statistics                                          #
    # ------------------------------------------------------------------ #
    accepted  = [r for r in _intercepted if r["valid"]]
    rejected  = [r for r in _intercepted if not r["valid"]]

    print(f"\nTotal candidates : {len(_intercepted)}")
    print(f"  Accepted       : {len(accepted)}")
    print(f"  Rejected       : {len(rejected)}")

    metrics = [
        "edge_support", "angular_coverage", "sector_coverage",
        "angular_uniformity", "max_gap_ratio",
        "radial_consistency", "radial_error_median", "radial_error_p95",
        "gradient_orientation", "local_contrast", "intensity_consistency",
    ]

    lines = []
    lines.append("=" * 80)
    lines.append("EVIDENCE DISTRIBUTION REPORT — Step 11 Threshold Tuning")
    lines.append(f"Images processed : {len(images)}")
    lines.append(f"Total candidates : {len(_intercepted)}  "
                 f"(accepted={len(accepted)}  rejected={len(rejected)})")
    lines.append("=" * 80)

    lines.append("\n--- ACCEPTED candidates ---")
    for m in metrics:
        vals = [r[m] for r in accepted if r[m] is not None]
        lines.append(f"  {m:<32}  {_fmt(_stats(vals))}")

    lines.append("\n--- REJECTED candidates ---")
    for m in metrics:
        vals = [r[m] for r in rejected if r[m] is not None]
        lines.append(f"  {m:<32}  {_fmt(_stats(vals))}")

    # Rejection reason frequency
    lines.append("\n--- Rejection reasons (top 10) ---")
    from collections import Counter
    reasons = Counter(r["reason"].split(":")[0].strip() for r in rejected)
    for reason, count in reasons.most_common(10):
        lines.append(f"  {count:5d}  {reason}")

    # Radius distribution (accepted)
    lines.append("\n--- Radius distribution (accepted) ---")
    radii = [r["radius"] for r in accepted]
    lines.append(f"  {_fmt(_stats(radii))}")

    # Suggested thresholds based on p10 of accepted / p90 of rejected separation
    lines.append("\n--- Threshold guidance ---")
    lines.append("  (p10 of accepted candidates = conservative safe minimum)")
    lines.append("  (p90 of rejected candidates = aggressive reject ceiling)")
    lines.append("  A threshold between p90_rejected and p10_accepted separates populations.")
    lines.append("")
    for m in ["angular_coverage", "sector_coverage", "radial_consistency",
              "gradient_orientation", "local_contrast"]:
        a_vals = [r[m] for r in accepted if r[m] is not None]
        r_vals = [r[m] for r in rejected if r[m] is not None]
        a_p10 = float(np.percentile(a_vals, 10)) if a_vals else float("nan")
        r_p90 = float(np.percentile(r_vals, 90)) if r_vals else float("nan")
        sep = a_p10 - r_p90
        lines.append(f"  {m:<32}  accepted_p10={a_p10:.3f}  "
                     f"rejected_p90={r_p90:.3f}  separation={sep:+.3f}")

    # max_gap_ratio: smaller is better for accepted
    m = "max_gap_ratio"
    a_vals = [r[m] for r in accepted if r[m] is not None]
    r_vals = [r[m] for r in rejected if r[m] is not None]
    a_p90 = float(np.percentile(a_vals, 90)) if a_vals else float("nan")
    r_p10 = float(np.percentile(r_vals, 10)) if r_vals else float("nan")
    lines.append(f"  {m:<32}  accepted_p90={a_p90:.3f}  "
                 f"rejected_p10={r_p10:.3f}  (lower=better for accepted)")

    report = "\n".join(lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {OUTPUT_REPORT}\n")
    print(report)


if __name__ == "__main__":
    main()
