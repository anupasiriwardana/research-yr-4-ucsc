"""
Step 1 diagnostic: can the cell score tell patch cells from normal cells AT ALL?

Uses the patched images + their ground-truth patch boxes (from
apply_saved_multi_patch_batch.py). Every grid cell is labelled:

    patch  : cell centre lies inside a ground-truth patch box
    normal : cell centre is farther than --halo pixels from every patch box
             (the ring in between is skipped: the tapped layer's receptive
             field makes cells next to a patch score high for a legitimate
             reason, which would blur the comparison)

For each candidate score it reports:
    auroc        : chance that a random patch cell outscores a random normal
                   cell (1.0 perfect, 0.5 coin flip)
    tpr@fpr0.5%  : fraction of patch cells flagged by the threshold that flags
                   only 0.5% of normal cells (same for 1% and 5%)
    median_patch / median_normal : typical score inside / outside

Scores compared (all box-independent):
    raw       : minimum Mahalanobis distance over all profiles (Mech-1 / global)
    ratio     : that distance / the winning class's learned threshold (Mech-7)
    zscore    : (distance - class mean) / class std (Mech-7 Variant B)
    bg_only   : distance to the background profile only

Optional --clean-dir adds a second "normal" set made of cells from clean images
(no patch anywhere), reported as auroc_vs_clean.

It scores ONE tap scale per run (the one in the config). To compare P3/P4/P5,
make one config per scale (tap_scale + stride + its own profile/threshold
filenames), run calibrate_profiles.py + learn_thresholds.py for it, then run
this with --config.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from mech7_core import (load_config, validated_settings, list_images, load_image_tensor,
                        ProfileBank)
from ultralytics_cls_head_adapter import UltralyticsClsHeadAdapter

DEFAULT_GT_DIR = r"D:\GitHub\yr-4-research\test-patch-det-multi-img\gen-patched-img\yolov11-patch_coordinates"
BOX_RE = re.compile(r"box=\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)")


def read_boxes(path):
    return [tuple(map(int, m.groups())) for m in BOX_RE.finditer(Path(path).read_text())] if path.exists() else []


def cell_geometry(H, W, stride, boxes, halo):
    """-> (patch_mask, normal_mask) boolean [H, W] from cell centres in pixels."""
    cx = (np.arange(W) + 0.5) * stride
    cy = (np.arange(H) + 0.5) * stride
    X, Y = np.meshgrid(cx, cy)
    patch = np.zeros((H, W), bool)
    near = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in boxes:
        patch |= (X >= x1) & (X < x2) & (Y >= y1) & (Y < y2)
        near |= (X >= x1 - halo) & (X < x2 + halo) & (Y >= y1 - halo) & (Y < y2 + halo)
    return patch, ~near


def auroc(pos, neg):
    """Rank-based AUROC (Mann-Whitney U), ties averaged."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size)
    sorted_v = allv[order]
    # average ranks over ties
    uniq, start, counts = np.unique(sorted_v, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def summarize(pos, neg, clean=None):
    out = {"auroc": auroc(pos, neg), "median_patch": float(np.median(pos)), "median_normal": float(np.median(neg))}
    for fpr in (0.005, 0.01, 0.05):
        thr = np.quantile(neg, 1 - fpr)
        out[f"tpr@fpr{fpr * 100:g}%"] = float((pos > thr).mean())
    if clean is not None:
        out["auroc_vs_clean"] = auroc(pos, clean)
        for fpr in (0.005, 0.01):
            thr = np.quantile(clean, 1 - fpr)
            out[f"tpr@clean_fpr{fpr * 100:g}%"] = float((pos > thr).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--patched-dir", default=None, help="default: batch_settings.input_dir")
    ap.add_argument("--gt-dir", default=DEFAULT_GT_DIR)
    ap.add_argument("--clean-dir", default=None, help="optional clean images for a second 'normal' set")
    ap.add_argument("--filename-prefix", default="multi-patched_", help="prefix of patched image names to strip to find the GT file")
    ap.add_argument("--halo", type=int, default=48, help="pixels around a patch box excluded from 'normal'")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--max-clean-images", type=int, default=100)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    config = load_config(args.config)
    s = validated_settings(config)
    scale, stride = s.get("tap_scale", "P3"), int(s["stride"])
    bg_id = s.get("background_class_id", -1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(config["model_path"]).to(device)
    adapter = UltralyticsClsHeadAdapter(model.model, scales=(scale,))
    profiles_dir = Path(config["profiles_dir"])
    bank = ProfileBank.from_file(profiles_dir / config["calibration_profile_filename"], device)

    import pickle
    with open(profiles_dir / config["threshold_filename"], "rb") as f:
        tdata = pickle.load(f)
    thr_k = torch.tensor([tdata["per_class"].get(c, tdata["pooled"])["threshold"] for c in bank.class_ids], device=device)
    mean_k = torch.tensor([tdata["per_class"].get(c, tdata["pooled"])["mean"] for c in bank.class_ids], device=device)
    std_k = torch.tensor([max(tdata["per_class"].get(c, tdata["pooled"])["std"], 1e-6) for c in bank.class_ids], device=device)
    bg_index = bank.class_ids.index(bg_id) if bg_id in bank.class_ids else None

    def cell_scores(tensor):
        feat = adapter.get_activations(tensor)[scale]
        _, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
        dist = bank.distances(flat)
        best, idx = dist.min(dim=1)
        sc = {"raw": best, "ratio": best / thr_k[idx], "zscore": (best - mean_k[idx]) / std_k[idx]}
        if bg_index is not None:
            sc["bg_only"] = dist[:, bg_index]
        return H, W, {k: v.cpu().numpy() for k, v in sc.items()}

    patched_dir = args.patched_dir or config["batch_settings"]["input_dir"]
    gt_dir = Path(args.gt_dir)
    images = list_images(patched_dir, args.max_images)
    pos_all, neg_all = {}, {}
    n_img = n_patches = 0
    for n, img in enumerate(images, 1):
        key = img.stem[len(args.filename_prefix):] if img.stem.startswith(args.filename_prefix) else img.stem
        boxes = read_boxes(gt_dir / f"{key}.txt")
        if not boxes:
            continue
        _, tensor = load_image_tensor(img, device)
        H, W, sc = cell_scores(tensor)
        patch, normal = cell_geometry(H, W, stride, boxes, args.halo)
        if not patch.any():
            continue
        n_img += 1
        n_patches += len(boxes)
        for name, arr in sc.items():
            a = arr.reshape(H, W)
            pos_all.setdefault(name, []).append(a[patch])
            neg_all.setdefault(name, []).append(a[normal])
        if n % 25 == 0:
            print(f"  {n}/{len(images)} images")

    clean_all = {}
    if args.clean_dir:
        for img in list_images(args.clean_dir, args.max_clean_images):
            _, tensor = load_image_tensor(img, device)
            _, _, sc = cell_scores(tensor)
            for name, arr in sc.items():
                clean_all.setdefault(name, []).append(arr)

    report = {"tap_scale": scale, "stride": stride, "halo_px": args.halo, "patched_images_used": n_img,
              "patches": n_patches, "profile": config["calibration_profile_filename"], "scores": {}}
    for name in pos_all:
        pos, neg = np.concatenate(pos_all[name]), np.concatenate(neg_all[name])
        clean = np.concatenate(clean_all[name]) if name in clean_all else None
        report["scores"][name] = summarize(pos, neg, clean)
        report["scores"][name]["n_patch_cells"] = int(pos.size)
        report["scores"][name]["n_normal_cells"] = int(neg.size)

    print(f"\nTap {scale} (stride {stride}): {n_img} patched images, {n_patches} patches, halo {args.halo}px")
    hdr = f"{'score':<9}{'AUROC':>8}{'TPR@0.5%':>10}{'TPR@1%':>9}{'TPR@5%':>9}{'med patch':>11}{'med normal':>12}"
    print(hdr)
    for name, r in report["scores"].items():
        print(f"{name:<9}{r['auroc']:>8.3f}{r['tpr@fpr0.5%']:>10.3f}{r['tpr@fpr1%']:>9.3f}{r['tpr@fpr5%']:>9.3f}"
              f"{r['median_patch']:>11.2f}{r['median_normal']:>12.2f}")
        if "auroc_vs_clean" in r:
            print(f"{'':<9}vs clean-image cells: AUROC {r['auroc_vs_clean']:.3f}, "
                  f"TPR@0.5% {r['tpr@clean_fpr0.5%']:.3f}, TPR@1% {r['tpr@clean_fpr1%']:.3f}")

    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "diagnostics" / f"separability_{scale}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
