"""
Score the boxes written by detect_patches.py against the GT patch boxes, for panel 2 and panel 3 side by side.

Patched images (detections + GT):
    a predicted box and a GT box match when IoU >= match_iou (greedy, best IoU first, one-to-one)
    precision = matched boxes / predicted boxes      recall = matched GT / all GT      F1
    mean IoU  = over matched pairs
    centre hit = share of GT patches whose centre lies inside some predicted box (looser than IoU)
    image recall = share of patched images with at least one box
Clean images (detections only):
    clean image FPR = share of clean images with at least one box;  boxes per clean image

Usage:
    python evaluate_detector.py --config config_yolov8.json --out results/yolov8.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mech6_core import gt_boxes, load_config, read_boxes, resolve


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(pred, gt, thr):
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); ious.append(v)
    return ious


def evaluate_panel(patched_dir, clean_dir, gt_dir, match_iou):
    tp = n_pred = n_gt = hits = img_hit = n_img = 0
    ious = []
    for f in sorted(Path(patched_dir).glob("*.txt")):
        gt = gt_boxes(gt_dir, f.stem)
        if not gt:
            continue
        pred = read_boxes(f)
        n_img += 1
        n_pred += len(pred); n_gt += len(gt)
        m = match(pred, gt, match_iou)
        tp += len(m); ious += m
        img_hit += bool(pred)
        for g in gt:
            cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            hits += any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in pred)
    clean = [read_boxes(f) for f in sorted(Path(clean_dir).glob("*.txt"))] if clean_dir else []
    if n_img == 0:
        raise RuntimeError(f"No detection files in {patched_dir} with a GT file in {gt_dir}.")
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gt
    return {"patched_images": n_img, "gt_patches": n_gt, "predicted_boxes": n_pred, "matched": tp,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "mean_iou": float(np.mean(ious)) if ious else 0.0, "centre_hit": hits / n_gt, "image_recall": img_hit / n_img,
            "clean_images": len(clean),
            "clean_image_fpr": float(np.mean([bool(c) for c in clean])) if clean else None,
            "boxes_per_clean_image": float(np.mean([len(c) for c in clean])) if clean else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--patched-detections", default=None, help="default: evaluate.patched_detections_dir")
    ap.add_argument("--clean-detections", default=None, help="default: evaluate.clean_detections_dir")
    ap.add_argument("--gt-dir", default=None, help="default: evaluate.gt_coordinates_dir")
    ap.add_argument("--match-iou", type=float, default=None)
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()
    cfg = load_config(args.config)
    e = cfg.get("evaluate", {})
    pdir = Path(args.patched_detections) if args.patched_detections else resolve(cfg, e.get("patched_detections_dir"))
    cdir = Path(args.clean_detections) if args.clean_detections else resolve(cfg, e.get("clean_detections_dir"))
    gdir = Path(args.gt_dir) if args.gt_dir else resolve(cfg, e.get("gt_coordinates_dir"))
    thr = args.match_iou if args.match_iou is not None else e.get("match_iou", 0.3)
    if pdir is None or gdir is None:
        raise ValueError("Need patched detections and a GT folder (flags or config 'evaluate').")

    results = {}
    for panel in ("panel2", "panel3"):
        results[panel] = evaluate_panel(pdir / panel, cdir / panel if cdir else None, gdir, thr)
    keys = [("precision", "{:.3f}"), ("recall", "{:.3f}"), ("f1", "{:.3f}"), ("mean_iou", "{:.3f}"),
            ("centre_hit", "{:.3f}"), ("image_recall", "{:.3f}"), ("clean_image_fpr", "{:.3f}"),
            ("boxes_per_clean_image", "{:.2f}"), ("predicted_boxes", "{}"), ("gt_patches", "{}")]
    print(f"match IoU >= {thr}; {results['panel2']['patched_images']} patched images, "
          f"{results['panel2']['clean_images']} clean images")
    print(f"{'':>22} {'panel 2 (energy z)':>20} {'panel 3 (nearest)':>20}")
    for k, fmt in keys:
        vals = [results[p][k] for p in ("panel2", "panel3")]
        print(f"{k:>22} " + " ".join(f"{('-' if v is None else fmt.format(v)):>20}" for v in vals))
    if args.out:
        out = resolve(cfg, args.out) if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        results["match_iou"] = thr
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
