"""
Two-panel picture per image (extract_mahalanobis_maps.py with save_visualizations on):

    left   the image; ground-truth patch boxes in cyan when a coordinate file is found
    right  nearest-profile Mahalanobis distance at the chosen layer.
           Colour scale per layer, from the calibration: blue = clean median (q50), red = viz_scale_max_ratio x threshold.
           White outline (optional) = cells above the threshold (the clean `threshold_quantile`, default 99.5%):
           threshold_mode "per_layer" = one threshold for the layer; "per_class" = each cell is compared with the
           threshold of its nearest profile. The colour scale always uses the layer threshold, so colours mean the
           same in both modes; only the outline and the header change.

Header: with GT boxes, the distance / threshold ratio inside the patch vs far (> 48 px) from it, and the share of
cells above the threshold in each; without GT (clean images), overall numbers.
"""

from pathlib import Path

import cv2
import numpy as np

from maha_core import cell_masks

PANEL = 640
GT_COLOR = (255, 255, 0)      # BGR cyan


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render(resized_rgb, dist, layer, stride, lo, thr, gt, out_path, vmax_ratio=2.0, outline=True,
           thr_map=None, mode="per_layer"):
    """thr = layer threshold (sets the colour scale). thr_map = per-cell threshold for the outline and the header
    (per_class mode); None = the layer threshold everywhere."""
    if thr_map is None:
        thr_map = np.full(dist.shape, thr, dtype=np.float32)
    up = cv2.resize(dist.astype(np.float32), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    up_thr = cv2.resize(thr_map.astype(np.float32), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    hi = vmax_ratio * thr
    base = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

    left = base.copy()
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(left, (x1, y1), (x2, y2), GT_COLOR, 2)
    _label(left, "image  (cyan = GT patch)" if gt else "image  (no GT found)")

    heat = cv2.applyColorMap((np.clip((up - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    right = cv2.addWeighted(base, 0.3, heat, 0.7, 0)
    if outline:
        contours, _ = cv2.findContours((up > up_thr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(right, contours, -1, (255, 255, 255), 1)
    white = f"white: > {thr:.1f}" if mode == "per_layer" else "white: > per-class thr"
    _label(right, f"nearest-profile distance, layer {layer}  (blue {lo:.1f} .. red {hi:.1f}"
                  + (f", {white})" if outline else ")"))

    canvas = np.hstack([left, right])
    r = dist / thr_map
    if gt:
        inside, far = cell_masks(*dist.shape, stride, gt, 48)
        if inside.any():
            header = (f"patch cells={int(inside.sum())}  d/thr median={np.median(r[inside]):.2f} max={r[inside].max():.2f}  "
                      f"above thr: {100 * (r[inside] > 1).mean():.0f}%  |  far cells: above thr {100 * (r[far] > 1).mean():.2f}%")
        else:
            header = "GT boxes found but no cell centre inside them"
    else:
        header = f"d/thr median={np.median(r):.2f} max={r.max():.2f}  cells above thr: {int((r > 1).sum())} of {r.size}"
    band = np.full((34, canvas.shape[1], 3), 30, np.uint8)
    cv2.putText(band, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([band, canvas]))
