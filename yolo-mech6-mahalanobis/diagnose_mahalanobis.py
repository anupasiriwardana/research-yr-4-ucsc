"""
Cell-level check: how well does the nearest-profile distance separate patch cells from normal cells?
Works on saved maps (extract_mahalanobis_maps.py with save_maps on), no model needed.

positives = cells whose centre is inside a GT patch box (patched maps + coordinate files)
negatives = every cell of the clean maps   (column "vs clean")
            and cells > 48 px from any patch in the patched maps   (column "vs far")

Per layer: AUROC, TPR at 0.5% FPR (threshold taken from the negatives themselves), and at the CALIBRATED threshold
(from calibrate_mahalanobis.py, per_layer or per_class as set by threshold_mode): share of patch cells above it, share of clean cells above it, and share of clean
images with at least one cell above it.

Usage:
    python diagnose_mahalanobis.py --config config_yolov8.json --patched-maps maps/yolov8/patched
        --clean-maps maps/yolov8/clean --gt-dir <coordinates>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from maha_core import MahaProfiles, cell_masks, gt_boxes, load_config, profile_path, resolve

MAX_NEG = 2_000_000


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--patched-maps", default=None, help="folder of .npz maps of patched images (default: config diagnose.patched_maps_dir)")
    ap.add_argument("--clean-maps", default=None, help="folder of .npz maps of clean images (default: config diagnose.clean_maps_dir)")
    ap.add_argument("--gt-dir", default=None, help="patch coordinate files (default: config gt_coordinates_dir)")
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    ap.add_argument("--threshold-mode", choices=["per_layer", "per_class"], default=None, help="override threshold_mode")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg.get("diagnose", {})
    pdir = Path(args.patched_maps) if args.patched_maps else resolve(cfg, dcfg.get("patched_maps_dir"))
    cdir = Path(args.clean_maps) if args.clean_maps else resolve(cfg, dcfg.get("clean_maps_dir"))
    gdir = Path(args.gt_dir) if args.gt_dir else resolve(cfg, cfg.get("gt_coordinates_dir"))
    if not (pdir and cdir and gdir):
        raise ValueError("Need patched maps, clean maps and a GT folder (flags or config).")

    mode = args.threshold_mode or cfg.get("threshold_mode", "per_layer")
    prof = MahaProfiles(profile_path(cfg), "cpu")
    rng = np.random.default_rng(0)
    patched = sorted(pdir.glob("*.npz"))
    clean = sorted(cdir.glob("*.npz"))
    print(f"{len(patched)} patched maps, {len(clean)} clean maps; threshold = clean {prof.data['threshold_quantile_key']}, threshold_mode {mode}")
    print(f"{'layer':>5} {'patch cells':>11} {'AUROC clean':>11} {'TPR@0.5% clean':>14} {'AUROC far':>9} "
          f"{'TPR@0.5% far':>12} | {'patch>thr':>9} {'clean cells>thr':>15} {'clean imgs>thr':>14}")
    results = {}
    for l in prof.layers:
        stride, thr = prof.data["layers"][l]["stride"], prof.threshold(l)
        pos, far, pos_t, used = [], [], [], 0
        for p in patched:
            boxes = gt_boxes(gdir, p.stem)
            if not boxes:
                continue
            z = np.load(p)
            d, t = z[f"d{l}"], prof.threshold_map(l, z[f"c{l}"], mode)
            inside, farm = cell_masks(*d.shape, stride, boxes, 48)
            pos.append(d[inside]); far.append(d[farm]); pos_t.append(t[inside]); used += 1
        cl, cl_t = [], []
        for p in clean:
            z = np.load(p)
            cl.append(z[f"d{l}"]); cl_t.append(prof.threshold_map(l, z[f"c{l}"], mode))
        if not pos or not cl:
            raise RuntimeError("No patched maps with GT boxes, or no clean maps.")
        pos, far, pos_t = np.concatenate(pos), np.concatenate(far), np.concatenate(pos_t)
        neg = np.concatenate([c.ravel() for c in cl])
        neg_t = np.concatenate([t.ravel() for t in cl_t])
        imgs_flagged = float(np.mean([(c > t).any() for c, t in zip(cl, cl_t)]))
        row = {"patched_images_with_gt": used, "patch_cells": int(pos.size)}
        for name, n in (("clean", neg), ("far", far)):
            n = n if n.size <= MAX_NEG else rng.choice(n, MAX_NEG, replace=False)
            row[f"auroc_{name}"] = auroc(pos, n)
            row[f"tpr_at_0.5fpr_{name}"] = float((pos > np.quantile(n, 0.995)).mean())
        row.update({"threshold_mode": mode, "layer_threshold": thr, "patch_cells_above_thr": float((pos > pos_t).mean()),
                    "clean_cells_above_thr": float((neg > neg_t).mean()), "clean_images_with_cell_above_thr": imgs_flagged})
        results[l] = row
        print(f"{l:>5} {row['patch_cells']:>11} {row['auroc_clean']:>11.3f} {row['tpr_at_0.5fpr_clean']:>14.3f} "
              f"{row['auroc_far']:>9.3f} {row['tpr_at_0.5fpr_far']:>12.3f} | {row['patch_cells_above_thr']:>9.3f} "
              f"{row['clean_cells_above_thr']:>15.4f} {imgs_flagged:>14.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
