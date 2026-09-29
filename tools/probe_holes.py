"""Quick diagnostic: show intensity stats and dark-blob counts for one image."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from feature_inspection.preprocessing.canonical_preprocessor import CanonicalPreprocessor

img = Path("data/inputs/WhatsApp Image 2026-09-21 at 11.56.01 PM.jpeg")
prep = CanonicalPreprocessor().preprocess_image(img)

app  = prep.isolated_product_image
gray = cv2.cvtColor(app, cv2.COLOR_BGR2GRAY) if len(app.shape) == 3 else app.copy()
mask = prep.product_mask > 0
pix  = gray[mask]

print("Product pixel intensity distribution:")
for p in [1, 5, 10, 15, 20, 25, 50, 75]:
    print(f"  p{p:2d} = {np.percentile(pix, p):.0f}")

print()
for base_t in [40, 50, 60, 70, 80, 90, 100, 120]:
    dark = (gray < base_t) & mask
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_closed = cv2.morphologyEx(dark.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3)
    contours, _ = cv2.findContours(dark_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good = []
    for c in contours:
        a = cv2.contourArea(c)
        p2 = cv2.arcLength(c, True)
        if p2 == 0 or a < 28:
            continue
        circ = 4 * 3.14159 * a / (p2 * p2)
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        sol = a / ha if ha > 0 else 0
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if circ >= 0.45 and sol >= 0.60 and 3 <= r <= 150:
            int_pix = gray[dark_closed > 0]
            df = float(np.mean(int_pix < base_t)) if len(int_pix) > 0 else 0
            good.append((r, circ, sol, df, cx, cy))
    print(f"threshold={base_t:3d}: total_dark={dark.sum():7d}  candidate_holes={len(good)}")
    for r, circ, sol, df, cx, cy in sorted(good, key=lambda x: -x[0])[:6]:
        print(f"   r={r:5.1f}  circ={circ:.2f}  sol={sol:.2f}  dark_frac={df:.2f}  "
              f"center=({cx:.0f},{cy:.0f})")
