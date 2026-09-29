"""
Probe why the bottom-left bolt hole and centre bore are missed.
Shows all dark blobs at progressively higher thresholds with their properties.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from feature_inspection.preprocessing.canonical_preprocessor import CanonicalPreprocessor

img = Path("data/inputs/WhatsApp Image 2026-09-21 at 11.56.01 PM.jpeg")
prep = CanonicalPreprocessor().preprocess_image(img)

app  = prep.isolated_product_image
gray = app.copy() if app.ndim == 2 else cv2.cvtColor(app, cv2.COLOR_BGR2GRAY)
mask = prep.product_mask > 0
pix  = gray[mask]
scale = prep.scale_factor
roi_x, roi_y = prep.roi_offset

print(f"Image shape: {gray.shape}   scale: {scale:.3f}   roi: {prep.roi_offset}")
print(f"Product pixel stats: min={pix.min()} p5={np.percentile(pix,5):.0f} "
      f"p10={np.percentile(pix,10):.0f} p15={np.percentile(pix,15):.0f} "
      f"p25={np.percentile(pix,25):.0f} median={np.median(pix):.0f}")

# The two known holes are at processed coords ~(462,454) and (667,689)
# Convert to original coords for reference
for name, (px, py) in [("hole1", (462, 454)), ("hole2", (667, 689))]:
    ox = int((px + roi_x) / scale)
    oy = int((py + roi_y) / scale)
    print(f"  {name} processed=({px},{py}) original=({ox},{oy}) intensity={gray[py,px]}")

print()
print("Scanning for ALL dark blobs at different thresholds:")
print("=" * 70)

for thr in range(20, 90, 5):
    dark = (gray < thr) & mask
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dm  = cv2.morphologyEx(dark.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3)
    contours, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 28:
            continue
        p = cv2.arcLength(c, True)
        if p == 0:
            continue
        circ = 4 * np.pi * a / (p * p)
        hull = cv2.convexHull(c)
        sol  = a / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < 3 or r > 200:
            continue
        hi = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(hi, [c], -1, 255, -1)
        ip = gray[hi > 0]
        df = float(np.mean(ip < thr)) if len(ip) > 0 else 0
        candidates.append((r, circ, sol, df, cx, cy))
    
    if candidates:
        print(f"\nthreshold={thr}: {len(candidates)} blobs (circ>=0, sol>=0, r 3-200)")
        for r, circ, sol, df, cx, cy in sorted(candidates, key=lambda x: -x[0])[:10]:
            # also show in original image coords
            ox = int((cx + roi_x) / scale)
            oy = int((cy + roi_y) / scale)
            flag = ""
            if circ < 0.45: flag += " LOW_CIRC"
            if sol < 0.65:  flag += " LOW_SOL"
            if df < 0.80:   flag += " LOW_DARK"
            print(f"   r={r:5.1f} circ={circ:.2f} sol={sol:.2f} df={df:.2f} "
                  f"proc=({cx:.0f},{cy:.0f}) orig=({ox},{oy}){flag}")
