"""
Per-image energy visualization for Mech-6 (opt-in: extract_energy_maps.py --save-visualizations).

One PNG with three panels:
    1. image    : ground-truth patch boxes in cyan when a GT file is found
    2. energy   : z-score of the log-energy at the chosen layer (Mech-4's signal), fixed colour scale
                  0 .. VMAX; white outline = cells above `flag_z`
    3. Mech-6   : distance to the nearest class / background energy profile in std units of log-energy
                  (`nearest_abs`), same colour scale, so the two panels are directly comparable

Header (when GT boxes exist): median / max of the z-score inside the patch vs far from it, and the
share of cells above `flag_z` in each.
"""

from pathlib import Path

import cv2
import numpy as np

from diagnose_energy import cell_masks
from mech6_energy import LAYER_STRIDE, log_energy

PANEL = 640
VMAX = 8.0
GT_COLOR = (255, 255, 0)      # BGR cyan


def _up(a):
    return cv2.resize(a.astype(np.float32), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def _heat(a, base):
    heat = cv2.applyColorMap((np.clip(_up(a) / VMAX, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 0.3, heat, 0.7, 0)


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render(resized_rgb, e, layer, profiles, gt, out_path, flag_z=3.5):
    g = profiles.stats[layer]["global"]
    z = np.clip((log_energy(e) - g["mean_log"]) / g["std_log"], 0, None)
    near = profiles.scores(layer, e)["nearest_abs"]
    base = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

    p1 = base.copy()
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(p1, (x1, y1), (x2, y2), GT_COLOR, 2)
    _label(p1, "image  (cyan = GT patch)" if gt else "image  (no GT found)")

    p2 = _heat(z, base)
    contours, _ = cv2.findContours(_up(z > flag_z).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p2, contours, -1, (255, 255, 255), 1)
    p3 = _heat(near, base)
    _label(p2, f"layer {layer} energy z-score  (0 blue .. {VMAX:g} red, white: z > {flag_z:g})")
    _label(p3, f"Mech-6 nearest-profile distance, layer {layer}")

    canvas = np.hstack([p1, p2, p3])
    if gt:
        inside, far = cell_masks(*z.shape, LAYER_STRIDE[layer], gt, 48)
        header = (f"patch cells={int(inside.sum())}  z median={np.median(z[inside]):.1f} max={z[inside].max():.1f}  "
                  f"above {flag_z:g}: {100 * (z[inside] > flag_z).mean():.0f}%  |  far cells={int(far.sum())}  "
                  f"z median={np.median(z[far]):.1f}  above {flag_z:g}: {100 * (z[far] > flag_z).mean():.1f}%")
    else:
        header = f"z median={np.median(z):.1f} max={z.max():.1f}  cells above {flag_z:g}: {int((z > flag_z).sum())} of {z.size}"
    band = np.full((34, canvas.shape[1], 3), 30, np.uint8)
    cv2.putText(band, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([band, canvas]))
