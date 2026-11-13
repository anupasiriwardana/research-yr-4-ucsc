"""
Calibration for Mech-6-Mahalanobis (run once per model).

Pass 1 - profiles (`profile_images_dir`, first `max_calibration_images` images):
    the model's own detections (offline, conf 0.25) are projected onto each layer's grid; every cell inside a box
    adds its feature vector to that class's running sums, and `background_samples_per_image` random uncovered
    cells add to the background profile. Then per class: mean, shrunk covariance, inverse.
    Classes with fewer than max(min_calibration_samples, 2 * channels) cells are dropped.

Pass 2 - clean threshold (`threshold_images_dir`, first `threshold_num_images` images, different from pass 1):
    nearest-profile distance of EVERY cell; stores quantiles q50 .. q99.9 and max per layer. `threshold_quantile`
    (default 0.995) is the threshold the visualization outlines and its colour scale is built around.
    It also stores a threshold PER PROFILE (class / background): the same quantile over only the clean cells whose
    nearest profile is that class (needs `min_threshold_cells` cells, else that class uses the layer threshold).
    Which one is used is chosen later by `threshold_mode` ("per_layer" or "per_class"), so one calibration serves both.

Usage:
    python calibrate_mahalanobis.py --config config_yolov8.json
    python calibrate_mahalanobis.py --config config_yolov8.json --max-images 50 --threshold-images 20   (quick test)
    python calibrate_mahalanobis.py --config config_yolov8.json --threshold-only   (redo pass 2 with existing profiles)
"""

import argparse
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch

from maha_core import (BG_ID, BackboneTap, Detector, MahaProfiles, layer_stride, list_images, load_config,
                       load_image_tensor, profile_path, resolve)

QUANTILES = {"q50": 0.50, "q90": 0.90, "q99": 0.99, "q99.5": 0.995, "q99.9": 0.999}


def accumulate(acc, key, x):
    """x [n, C] float64 on GPU -> running count, sum, sum of outer products."""
    a = acc[key]
    if a is None:
        C = x.shape[1]
        a = acc[key] = [0, torch.zeros(C, dtype=torch.float64, device=x.device),
                        torch.zeros(C, C, dtype=torch.float64, device=x.device)]
    a[0] += x.shape[0]
    a[1] += x.sum(0)
    a[2] += x.T @ x


def pass1_profiles(cfg, det, tap, device, max_images):
    layers = cfg["tap_layers"]
    bg_per_image = cfg.get("background_samples_per_image", 250)
    shrink = cfg.get("shrinkage", 0.01)
    images = list_images(resolve(cfg, cfg["profile_images_dir"]), 0, max_images or cfg.get("max_calibration_images", 2000))
    print(f"Pass 1: profiles from {len(images)} images, layers {layers}, shrinkage {shrink}")
    acc = {l: defaultdict(lambda: None) for l in layers}
    strides, t0 = {}, time.time()
    for n, path in enumerate(images, 1):
        _, x = load_image_tensor(path, device)
        xyxy, cls = det.predict_boxes(x)
        acts = tap.get_activations(x)
        for l in layers:
            f = acts[l][0].double()                                  # [C, H, W]
            C, H, W = f.shape
            s = strides[l] = layer_stride(acts[l])
            covered = torch.zeros(H, W, dtype=torch.bool, device=device)
            for (bx1, by1, bx2, by2), c in zip(xyxy, cls):
                x1, y1, x2, y2 = int(bx1 / s), int(by1 / s), int(bx2 / s), int(by2 / s)
                x1, x2 = min(max(x1, 0), W - 1), min(max(x2, 0), W)
                y1, y2 = min(max(y1, 0), H - 1), min(max(y2, 0), H)
                if x2 <= x1 or y2 <= y1:
                    continue
                covered[y1:y2, x1:x2] = True
                accumulate(acc[l], int(c), f[:, y1:y2, x1:x2].reshape(C, -1).T)
            free = (~covered).nonzero(as_tuple=False)
            if free.shape[0]:
                pick = free[torch.randperm(free.shape[0], device=device)[:bg_per_image]]
                accumulate(acc[l], BG_ID, f[:, pick[:, 0], pick[:, 1]].T)
        if n % 200 == 0:
            print(f"  {n}/{len(images)}  ({time.time() - t0:.0f}s)")

    out = {}
    for l in layers:
        ids, means, invs, counts = [], [], [], []
        for cid in sorted(acc[l], key=lambda c: (c != BG_ID, c)):
            cnt, s1, s2 = acc[l][cid]
            C = s1.shape[0]
            need = max(cfg.get("min_calibration_samples", 1000), 2 * C)
            if cnt < need:
                continue
            mu = s1 / cnt
            cov = (s2 - cnt * torch.outer(mu, mu)) / (cnt - 1)
            cov = (1 - shrink) * cov + shrink * cov.diagonal().mean() * torch.eye(C, dtype=cov.dtype, device=cov.device)
            ids.append(cid)
            means.append(mu.cpu().numpy().astype(np.float32))
            invs.append(torch.linalg.inv(cov).cpu().numpy().astype(np.float32))
            counts.append(int(cnt))
        if not ids:
            raise RuntimeError(f"Layer {l}: no class had enough samples; use more images or lower min_calibration_samples.")
        out[l] = {"class_ids": ids, "means": np.stack(means), "inv_covs": np.stack(invs), "counts": counts,
                  "stride": strides[l], "channels": int(means[0].shape[0])}
        names = ", ".join(f"{'bg' if c == BG_ID else c}:{k}" for c, k in zip(ids, counts))
        print(f"layer {l} (stride {strides[l]}, {out[l]['channels']} ch): {len(ids)} profiles  [{names}]")
    return {"layers": out, "model": cfg["model_path"], "shrinkage": shrink, "profile_images": len(images),
            "threshold_quantile_key": f"q{100 * cfg.get('threshold_quantile', 0.995):g}"}


def pass2_threshold(cfg, data, tap, device, num_images, path):
    with open(path, "wb") as f:                        # MahaProfiles reads from file
        pickle.dump(data, f)
    prof = MahaProfiles(path, device)
    q = dict(QUANTILES)
    tq = cfg.get("threshold_quantile", 0.995)
    q[data["threshold_quantile_key"]] = tq
    images = list_images(resolve(cfg, cfg["threshold_images_dir"]), 0, num_images or cfg.get("threshold_num_images", 300))
    print(f"Pass 2: clean distances on {len(images)} images from {cfg['threshold_images_dir']}")
    vals = {l: [] for l in prof.layers}
    cids = {l: [] for l in prof.layers}
    for n, p in enumerate(images, 1):
        _, x = load_image_tensor(p, device)
        acts = tap.get_activations(x)
        for l in prof.layers:
            d, c = prof.distances(l, acts[l])
            vals[l].append(d.ravel())
            cids[l].append(c.ravel())
        if n % 100 == 0:
            print(f"  {n}/{len(images)}")
    min_cells = cfg.get("min_threshold_cells", 1000)
    for l in prof.layers:
        v = np.concatenate(vals[l])
        c = np.concatenate(cids[l])
        # per_class mode: threshold of each profile = quantile of the clean cells that are NEAREST to that profile
        per_class = {}
        for cid in data["layers"][l]["class_ids"]:
            sel = v[c == cid]
            if sel.size >= min_cells:
                per_class[int(cid)] = {"threshold": float(np.quantile(sel, tq)), "n": int(sel.size)}
        data["layers"][l]["class_thresholds"] = per_class
        data["layers"][l]["clean_quantiles"] = {k: float(np.quantile(v, qq)) for k, qq in q.items()}
        data["layers"][l]["clean_quantiles"]["max"] = float(v.max())
        cq = data["layers"][l]["clean_quantiles"]
        print(f"layer {l}: clean distance median {cq['q50']:.2f}, q99 {cq['q99']:.2f}, "
              f"threshold {data['threshold_quantile_key']} = {cq[data['threshold_quantile_key']]:.2f}, max {cq['max']:.2f}")
        missing = [cid for cid in data["layers"][l]["class_ids"] if int(cid) not in per_class]
        print(f"         per-class thresholds: " + ", ".join(
            f"{'bg' if k == BG_ID else k}:{t['threshold']:.2f} (n={t['n']})" for k, t in sorted(per_class.items()))
              + (f"; fall back to the layer threshold (< {min_cells} clean cells): "
                 + ", ".join('bg' if k == BG_ID else str(k) for k in missing) if missing else ""))
    data["threshold_images"] = len(images)
    data["threshold_quantile"] = tq
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-images", type=int, default=None, help="override max_calibration_images (pass 1)")
    ap.add_argument("--threshold-images", type=int, default=None, help="override threshold_num_images (pass 2)")
    ap.add_argument("--threshold-only", action="store_true", help="keep the existing profiles, redo pass 2 only")
    args = ap.parse_args()
    cfg = load_config(args.config)
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = profile_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)

    det = Detector(cfg, device)
    tap = BackboneTap(det.net, cfg["tap_layers"])
    print(f"Model {det.name}, device {device}")
    if args.threshold_only:
        with open(out, "rb") as f:
            data = pickle.load(f)
    else:
        data = pass1_profiles(cfg, det, tap, device, args.max_images)
    data = pass2_threshold(cfg, data, tap, device, args.threshold_images, out)
    with open(out, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
