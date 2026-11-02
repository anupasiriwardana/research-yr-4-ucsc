"""
Step 2 of 2 for Mech-7: learn per-class thresholds from a SECOND clean image
set (`threshold_images_dir`, disjoint from the profile set).

For every cell of every image (not just cells inside boxes -- the runtime
scores all of them) it finds the nearest profile, exactly as the detector will,
and records that best distance under the winning class. Then, per class:

    threshold = percentile(distances, threshold_percentile)
    mean/std  = of those distances (for the zscore variant)

A class with fewer than `min_class_samples_for_own_threshold` cells falls back
to the pooled values over all cells. For the zscore variant, the single z
threshold is the same percentile of the pooled z-scores.

`adjust_threshold` is NOT baked in here; it is added at detection time, so it
can be tuned without re-running this script.
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from mech7_core import (load_config, validated_settings, list_images, load_image_tensor, ProfileBank)
from ultralytics_cls_head_adapter import UltralyticsClsHeadAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-images", type=int, default=None, help="override detector_settings.max_threshold_images")
    args = ap.parse_args()

    config = load_config(args.config)
    s = validated_settings(config)
    scale = s.get("tap_scale", "P3")
    pct = float(s.get("threshold_percentile", 99.5))
    min_own = int(s.get("min_class_samples_for_own_threshold", 30))
    bg_id = s.get("background_class_id", -1)
    max_images = args.max_images or s.get("max_threshold_images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(config["model_path"]).to(device)
    adapter = UltralyticsClsHeadAdapter(model.model, scales=(scale,))
    print(adapter.describe())

    bank = ProfileBank.from_file(Path(config["profiles_dir"]) / config["calibration_profile_filename"], device)
    images = list_images(config["threshold_images_dir"], max_images)
    print(f"Learning thresholds from {len(images)} images in {config['threshold_images_dir']} "
          f"(tap={scale}, percentile={pct}, {len(bank.class_ids)} profiles)")

    per_class = defaultdict(list)   # class id -> list of np arrays of best distances
    for n, img_path in enumerate(images, 1):
        _, tensor = load_image_tensor(img_path, device)
        feat = adapter.get_activations(tensor)[scale]
        B, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
        best, idx = bank.nearest(flat)
        best, idx = best.cpu().numpy(), idx.cpu().numpy()
        for k in np.unique(idx):
            per_class[bank.class_ids[k]].append(best[idx == k])
        if n % 200 == 0:
            print(f"  {n}/{len(images)} images")

    per_class = {c: np.concatenate(v) for c, v in per_class.items()}
    all_d = np.concatenate(list(per_class.values()))
    pooled = {"threshold": float(np.percentile(all_d, pct)), "mean": float(all_d.mean()),
              "std": float(all_d.std()), "n": int(all_d.size)}

    table, z_all = {}, []
    for cid in bank.class_ids:
        d = per_class.get(cid)
        n = 0 if d is None else d.size
        label = "background" if cid == bg_id else f"class {cid}"
        if n >= min_own:
            e = {"threshold": float(np.percentile(d, pct)), "mean": float(d.mean()),
                 "std": float(d.std()), "n": int(n), "source": "own"}
        else:
            e = {**pooled, "n": int(n), "source": "pooled"}
        table[cid] = e
        if n:
            z_all.append((d - e["mean"]) / max(e["std"], 1e-6))
        print(f"  {label:>12}: n={n:>8}  threshold={e['threshold']:.2f}  ({e['source']})")

    z_threshold = float(np.percentile(np.concatenate(z_all), pct))
    out = {"percentile": pct, "min_class_samples": min_own, "tap_scale": scale,
           "pooled": pooled, "per_class": table, "zscore_threshold": z_threshold}
    path = Path(config["profiles_dir"]) / config["threshold_filename"]
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nPooled threshold (single global at p{pct}): {pooled['threshold']:.2f}")
    print(f"Z-score threshold: {z_threshold:.2f}")
    print(f"Saved thresholds to {path}")


if __name__ == "__main__":
    main()
