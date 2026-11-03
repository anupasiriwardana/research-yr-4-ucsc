"""
Per-image score heatmap for Mech-7 (opt-in: batch_settings.save_score_heatmaps).

One PNG per image, three panels side by side:

    1. image      : ground-truth patch boxes (cyan, if a GT file is found) and
                    Mech-7's detected boxes (red)
    2. raw        : the per-cell nearest-profile distance on a FIXED colour scale
                    (0 .. heatmap_max_distance) so images are comparable; white
                    outline = cells above their own threshold
    3. margin     : raw - the cell's own threshold, blue (< 0, normal) to red
                    (> 0, flagged), clipped to +-MARGIN_LIM

When GT boxes exist, a header line reports how the map behaves inside vs
outside the patch: median / max raw inside, % of patch cells flagged, and
% of far-away cells flagged (a cell counts as "far" if its centre is more
than HALO px from every GT box).
"""

import re
from pathlib import Path

import cv2
import numpy as np

BOX_RE = re.compile(r"box=\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)")
NAME_PREFIX_RE = re.compile(r"^(\d+-)?(multi-)?patched[_-]")
MARGIN_LIM = 10.0
HALO = 48
PANEL = 640
GT_COLOR = (255, 255, 0)      # BGR cyan
DET_COLOR = (0, 0, 255)       # BGR red


def find_gt_boxes(gt_dir, img_name):
    """Look for the GT file of a patched image, tolerating the naming prefixes
    the apply scripts add ('patched_', 'multi-patched_', '15-multi-patched_').
    Returns [] if the folder is unset or no file matches."""
    if not gt_dir:
        return []
    stem = Path(img_name).stem
    for key in (stem, NAME_PREFIX_RE.sub("", stem)):
        p = Path(gt_dir) / f"{key}.txt"
        if p.exists():
            return [tuple(map(int, m.groups())) for m in BOX_RE.finditer(p.read_text())]
    return []


def _cell_masks(H, W, stride, boxes):
    cx = (np.arange(W) + 0.5) * stride
    cy = (np.arange(H) + 0.5) * stride
    X, Y = np.meshgrid(cx, cy)
    inside = np.zeros((H, W), bool)
    near = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in boxes:
        inside |= (X >= x1) & (X < x2) & (Y >= y1) & (Y < y2)
        near |= (X >= x1 - HALO) & (X < x2 + HALO) & (Y >= y1 - HALO) & (Y < y2 + HALO)
    return inside, ~near


def _up(a, interp=cv2.INTER_NEAREST):
    return cv2.resize(a, (PANEL, PANEL), interpolation=interp)


def _boxes(img, gt_boxes, det_boxes):
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
    for e in det_boxes:
        x1, y1, x2, y2 = e["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), DET_COLOR, 2)


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _stats_line(raw, margin, stride, gt_boxes):
    inside, far = _cell_masks(*raw.shape, stride, gt_boxes)
    if not inside.any():
        return "GT boxes found but no cell centre falls inside them"
    flagged = margin > 0
    return (f"patch cells={int(inside.sum())}  raw median={np.median(raw[inside]):.1f} max={raw[inside].max():.1f}  "
            f"flagged={100 * flagged[inside].mean():.0f}%  |  "
            f"far cells={int(far.sum())}  raw median={np.median(raw[far]):.1f}  flagged={100 * flagged[far].mean():.1f}%")


def render_heatmap(resized_rgb, raw, margin, stride, det_boxes, gt_boxes, vmax, out_path):
    base = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
    flagged_up = _up((margin > 0).astype(np.uint8))

    # panel 1: image + boxes
    p1 = base.copy()
    _boxes(p1, gt_boxes, det_boxes)
    _label(p1, "image  (cyan=GT patch, red=detected)" if gt_boxes else "image  (red=detected, no GT found)")

    # panel 2: raw distance, fixed scale
    norm = np.clip(raw / vmax, 0, 1)
    heat = cv2.applyColorMap(_up((norm * 255).astype(np.uint8)), cv2.COLORMAP_JET)
    p2 = cv2.addWeighted(base, 0.3, heat, 0.7, 0)
    contours, _ = cv2.findContours(flagged_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p2, contours, -1, (255, 255, 255), 1)
    _boxes(p2, gt_boxes, [])
    _label(p2, f"raw distance  (0 blue .. {vmax:g} red, white=flagged)")

    # panel 3: margin over own threshold
    m = np.clip(margin / MARGIN_LIM, -1, 1)
    rgb = np.empty(raw.shape + (3,), np.float32)       # BGR
    pos, neg = np.clip(m, 0, 1), np.clip(-m, 0, 1)
    rgb[..., 0] = 255 * (1 - pos)
    rgb[..., 1] = 255 * (1 - pos - neg)
    rgb[..., 2] = 255 * (1 - neg)
    p3 = cv2.addWeighted(base, 0.3, _up(rgb.astype(np.uint8)), 0.7, 0)
    _boxes(p3, gt_boxes, [])
    _label(p3, f"margin over threshold  (blue <0, red >0, +-{MARGIN_LIM:g})")

    canvas = np.hstack([p1, p2, p3])
    if gt_boxes:
        header = _stats_line(raw, margin, stride, gt_boxes)
    else:
        header = (f"raw median={np.median(raw):.1f} max={raw.max():.1f}  flagged cells="
                  f"{int((margin > 0).sum())} of {raw.size}")
    band = np.full((34, canvas.shape[1], 3), 30, np.uint8)
    cv2.putText(band, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([band, canvas]))
