"""
Step 2 of 2 for Mech-5: learn per-class thresholds, for every configured scale,
from a SECOND clean image set (`threshold_images_dir`, disjoint from the
profile set).

For every cell of every image (not just cells inside boxes -- the runtime
scores all of them) it finds the nearest profile of that scale, exactly as the
detector will, and records that best distance under the winning class. Then,
per scale and per class:

    threshold = percentile(distances, threshold_percentile)

A class with fewer than `min_class_samples_for_own_threshold` cells falls back
to the pooled value over all cells of that scale.

`adjust_threshold` and the rule's `offset` are NOT baked in here; they are
applied at detection time, so they can be tuned without re-running this script.
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from mech5_core import load_config, validated_settings, ordered_scales, list_images, load_image_tensor, ProfileBank
from ultralytics_cls_head_adapter import UltralyticsClsHeadAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-images", type=int, default=None, help="override detector_settings.max_threshold_images")
    args = ap.parse_args()

    config = load_config(args.config)
    s = validated_settings(config)
    scales = ordered_scales(config)
    pct = float(s.get("threshold_percentile", 99.5))
    min_own = int(s.get("min_class_samples_for_own_threshold", 30))
    bg_id = s.get("background_class_id", -1)
    max_images = args.max_images or s.get("max_threshold_images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(config["model_path"]).to(device)
    adapter = UltralyticsClsHeadAdapter(model.model, scales=tuple(scales))
    print(adapter.describe())

    profiles_dir = Path(config["profiles_dir"])
    banks = {sc: ProfileBank.from_file(profiles_dir / config["scales"][sc]["profile_filename"], device) for sc in scales}
    images = list_images(config["threshold_images_dir"], max_images)
    print(f"Learning thresholds from {len(images)} images in {config['threshold_images_dir']} "
          f"(scales={scales}, percentile={pct})")

    per_class = {sc: defaultdict(list) for sc in scales}    # scale -> class id -> arrays of best distances
    for n, img_path in enumerate(images, 1):
        _, tensor = load_image_tensor(img_path, device)
        acts = adapter.get_activations(tensor)
        for sc in scales:
            feat = acts[sc]
            _, C, H, W = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
            best, idx = banks[sc].nearest(flat)
            best, idx = best.cpu().numpy(), idx.cpu().numpy()
            for k in np.unique(idx):
                per_class[sc][banks[sc].class_ids[k]].append(best[idx == k])
        if n % 200 == 0:
            print(f"  {n}/{len(images)} images")

    for sc in scales:
        bank = banks[sc]
        arrays = {c: np.concatenate(v) for c, v in per_class[sc].items()}
        all_d = np.concatenate(list(arrays.values()))
        pooled = {"threshold": float(np.percentile(all_d, pct)), "mean": float(all_d.mean()),
                  "std": float(all_d.std()), "n": int(all_d.size)}
        table = {}
        for cid in bank.class_ids:
            d = arrays.get(cid)
            n = 0 if d is None else d.size
            label = "background" if cid == bg_id else f"class {cid}"
            if n >= min_own:
                e = {"threshold": float(np.percentile(d, pct)), "mean": float(d.mean()),
                     "std": float(d.std()), "n": int(n), "source": "own"}
            else:
                e = {**pooled, "n": int(n), "source": "pooled"}
            table[cid] = e
            print(f"  [{sc}] {label:>12}: n={n:>8}  threshold={e['threshold']:.2f}  ({e['source']})")
        out = {"percentile": pct, "min_class_samples": min_own, "tap_scale": sc, "pooled": pooled, "per_class": table}
        path = profiles_dir / config["scales"][sc]["threshold_filename"]
        with open(path, "wb") as f:
            pickle.dump(out, f)
        print(f"[{sc}] pooled threshold (p{pct}): {pooled['threshold']:.2f}; saved to {path}\n")


if __name__ == "__main__":
    main()
