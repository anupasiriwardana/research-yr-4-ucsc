"""
Step 1 for Mech-6: per-class (+ background) energy profiles, per tapped layer.

For each clean image in `profile_images_dir`, the detector's boxes (offline only) are projected
onto each layer's grid with that layer's stride; log-energies of the cells inside a box feed that
class's profile, and a capped random sample of uncovered cells feeds the background profile
(same procedure as Mech-1 Option B). Also stores the global mean/std of raw and log energy over
every cell (what Mech-4 uses for its single threshold).
"""

import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from mech6_energy import (LAYER_STRIDE, BG_ID, load_config, list_images, load_image_tensor, BackboneTap,
                          energy_map, log_energy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    layers = cfg["tap_layers"]
    min_samples = cfg.get("min_calibration_samples", 100)
    bg_per_image = cfg.get("background_samples_per_image", 250)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(cfg["model_path"]).to(device)
    tap = BackboneTap(model.model, layers)
    images = list_images(cfg["profile_images_dir"], 0, args.max_images or cfg.get("max_calibration_images"))
    print(f"Energy profiles from {len(images)} images, layers {layers}")

    per_class = {l: defaultdict(list) for l in layers}
    g_n = {l: 0 for l in layers}
    g_s = {l: np.zeros(4) for l in layers}                  # sum E, sum E^2, sum logE, sum logE^2
    for n, path in enumerate(images, 1):
        _, tensor = load_image_tensor(path, device)
        boxes = model.predict(source=tensor, verbose=False)[0].boxes
        acts = tap.get_activations(tensor)
        for l in layers:
            e = energy_map(acts[l]).cpu().numpy().astype(np.float64)
            le = log_energy(e)
            H, W = e.shape
            g_n[l] += e.size
            g_s[l] += [e.sum(), (e ** 2).sum(), le.sum(), (le ** 2).sum()]
            covered = np.zeros((H, W), bool)
            s = LAYER_STRIDE[l]
            for box, cls in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = (box / s).astype(int)
                x1, x2 = min(max(x1, 0), W - 1), min(max(x2, 0), W)
                y1, y2 = min(max(y1, 0), H - 1), min(max(y2, 0), H)
                if x2 <= x1 or y2 <= y1:
                    continue
                covered[y1:y2, x1:x2] = True
                per_class[l][int(cls)].append(le[y1:y2, x1:x2].ravel())
            ys, xs = np.nonzero(~covered)
            if ys.size:
                pick = random.sample(range(ys.size), min(bg_per_image, ys.size))
                per_class[l][BG_ID].append(le[ys[pick], xs[pick]])
        if n % 200 == 0:
            print(f"  {n}/{len(images)}")

    stats = {}
    for l in layers:
        n, (s1, s2, l1, l2) = g_n[l], g_s[l]
        mean, mean_l = s1 / n, l1 / n
        stats[l] = {"global": {"mean": mean, "std": float(np.sqrt(s2 / n - mean ** 2)),
                               "mean_log": mean_l, "std_log": float(np.sqrt(l2 / n - mean_l ** 2)), "n": int(n)}}
        for cid, parts in sorted(per_class[l].items()):
            v = np.concatenate(parts)
            if v.size <= min_samples:
                continue
            stats[l][cid] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size)}
        label = lambda c: "bg" if c == BG_ID else str(c)
        print(f"layer {l}: global log-energy mean {stats[l]['global']['mean_log']:.2f} std {stats[l]['global']['std_log']:.2f}; "
              f"{len(stats[l]) - 1} profiles: " + ", ".join(f"{label(c)}:{stats[l][c]['mean']:.2f}/{stats[l][c]['std']:.2f}"
                                                            for c in sorted(k for k in stats[l] if k != 'global')))
    out = Path(cfg["profiles_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / cfg["profile_filename"], "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved {out / cfg['profile_filename']}")


if __name__ == "__main__":
    main()
