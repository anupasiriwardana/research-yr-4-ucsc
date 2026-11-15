"""
Three-panel picture per image (detect_patches.py with save_visualizations on):

    1. image      GT patch boxes in cyan (when a coordinate file is found)
    2. per-layer  nearest-profile Mahalanobis distance; thin white outline = hot cells (distance > layer threshold);
                  GREEN boxes = panel-2 patches
    3. per-class  the SAME distance map; thin white outline = hot cells (distance > threshold of the cell's nearest
                  profile); MAGENTA boxes = panel-3 patches
    Colour scale (same in both heatmaps): blue = clean median distance, red = viz_scale_max_ratio x layer threshold.
"""

from pathlib import Path

import cv2
import numpy as np

PANEL = 640
GT_COLOR = (255, 255, 0)        # BGR cyan
P2_COLOR = (0, 255, 0)          # green
P3_COLOR = (255, 0, 255)        # magenta


def _up(a):
    return cv2.resize(a.astype(np.float32), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def _heat(a, base, lo, hi):
    heat = cv2.applyColorMap((np.clip((_up(a) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 0.3, heat, 0.7, 0)


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _panel(base, dist, lo, hi, hot, regions, color, outline):
    img = _heat(dist, base, lo, hi)
    if outline:
        contours, _ = cv2.findContours(_up(hot).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
    for r in regions:
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{r['mass']:.0f} t{r['scale']}", (x1, max(y1 - 4, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def render(resized_rgb, layer, gt, dist, lo, thr, p2, p3, out_path, outline=True, vmax_ratio=2.0):
    """dist = nearest-profile distance map; lo = clean median, thr = layer threshold (colour scale lo .. vmax_ratio x thr).
    p2 / p3 = dict(hot, regions, label)."""
    base = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
    hi = vmax_ratio * thr
    left = base.copy()
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(left, (x1, y1), (x2, y2), GT_COLOR, 2)
    _label(left, "image  (cyan = GT patch)" if gt else "image  (no GT found)")
    mid = _panel(base, dist, lo, hi, p2["hot"], p2["regions"], P2_COLOR, outline)
    _label(mid, f"L{layer} Mahalanobis, {p2['label']}  (green = patch)")
    right = _panel(base, dist, lo, hi, p3["hot"], p3["regions"], P3_COLOR, outline)
    _label(right, f"L{layer} Mahalanobis, {p3['label']}  (magenta = patch)")
    canvas = np.hstack([left, mid, right])
    header = (f"GT patches: {len(gt)}   |   panel 2: {len(p2['regions'])} box(es)"
              + "".join(f"  [mass {r['mass']:.0f}, t{r['scale']}, {r['cells']} cells]" for r in p2["regions"][:3])
              + f"   |   panel 3: {len(p3['regions'])} box(es)"
              + "".join(f"  [mass {r['mass']:.0f}, t{r['scale']}, {r['cells']} cells]" for r in p3["regions"][:3])
              + f"   |   colour: blue {lo:.1f} .. red {hi:.1f}")
    band = np.full((34, canvas.shape[1], 3), 30, np.uint8)
    cv2.putText(band, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([band, canvas]))
