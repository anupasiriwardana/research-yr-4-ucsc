"""
Calibrate the Mech-6-Mahalanobis detector for one model. The results are WRITTEN INTO THE CONFIG FILE
(panel2.calibrated and panel3.calibrated), so they can be read and edited by hand. The previous config is kept as
<config>.json.bak.

Three stages:
  pass 1  profiles      per class + background per layer: mean vector, covariance (shrunk towards a scaled identity by
                        `shrinkage`), inverse. From `profile_images_dir` (BDD train). The detector's own boxes (offline
                        only) say which cells belong to which class; `background_samples_per_image` random uncovered
                        cells feed the background profile. Running sums, so memory stays flat. A class is kept only with
                        >= max(min_calibration_samples, 2 x channels) cells.
                        Saved to `profile_file` (.pkl, same format as yolo-mech6-mahalanobis, so those files can be reused).
  pass 2  thresholds    on clean `threshold_images_dir` images (BDD test, different images), nearest-profile distance:
                          panel 2: threshold = `threshold_quantile` (0.995) of all clean cells, per layer
                          panel 3: class_thresholds = the same quantile over the clean cells whose NEAREST profile is
                                   that class (only classes with >= `min_threshold_cells` such cells; the others use
                                   the layer threshold, which is the panel-2 threshold)
  regions               one mass threshold per thickness scale (`thickness_scales`), on the same clean images, per
                        panel. Detection tries every scale, so the thresholds are set TOGETHER: the same strictness at
                        every scale, tightened until at most `target_clean_image_fpr` of clean images keep a region at
                        ANY scale. Uses the thresholds and join_cells in the config.

Usage:
    python calibrate_detector.py --config config_yolov8.json                    # everything
    python calibrate_detector.py --config config_yolov8.json --thresholds-only  # keep profiles; redo pass 2 + regions
    python calibrate_detector.py --config config_yolov8.json --regions-only     # keep profiles + thresholds; redo regions
    python calibrate_detector.py --config config_yolov8.json --max-images 60 --threshold-images 20   # quick test
"""

import argparse
import datetime
import pickle
import time
from collections import defaultdict

import numpy as np
import torch

from maha_core import (BG_ID, BackboneTap, Detector, MahaProfiles, class_key, hot_and_joined, layer_stride,
                       list_images, load_config, load_image_tensor, panel_maps, regions_at_scale, resolve, save_config)

PANELS = ("panel2", "panel3")


def det_device(det):
    return next(det.net.parameters()).device


def accumulate(acc, key, x):
    """x [n, C] float64 on the device -> running count, sum, sum of outer products."""
    a = acc[key]
    if a is None:
        C = x.shape[1]
        a = acc[key] = [0, torch.zeros(C, dtype=torch.float64, device=x.device),
                        torch.zeros(C, C, dtype=torch.float64, device=x.device)]
    a[0] += x.shape[0]
    a[1] += x.sum(0)
    a[2] += x.T @ x


def pass1_profiles(cfg, det, tap, ccfg, max_images):
    layers = tap.layers
    device = det_device(det)
    bg_per_image = ccfg.get("background_samples_per_image", 250)
    shrink = ccfg.get("shrinkage", 0.01)
    images = list_images(resolve(cfg, ccfg["profile_images_dir"]), 0, max_images or ccfg.get("max_calibration_images"))
    print(f"Pass 1: profiles from {len(images)} images, layers {layers}, shrinkage {shrink}")
    acc = {l: defaultdict(lambda: None) for l in layers}
    strides, t0 = {}, time.time()
    for n, path in enumerate(images, 1):
        _, x, _ = load_image_tensor(path, device)
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
            if cnt < max(ccfg.get("min_calibration_samples", 1000), 2 * C):
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
        print(f"  layer {l} (stride {strides[l]}, {out[l]['channels']} ch): {len(ids)} profiles: "
              + ", ".join(f"{class_key(c, det.names)}:{k}" for c, k in zip(ids, counts)))
    data = {"layers": out, "model": cfg["model_path"], "shrinkage": shrink, "profile_images": len(images)}
    path = resolve(cfg, cfg["profile_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  saved {path}")
    return data


def clean_distances(cfg, det, tap, prof, ccfg, num_images):
    """Nearest-profile distance (float32) and nearest class id (int16) of every cell of the clean images, per layer."""
    images = list_images(resolve(cfg, ccfg["threshold_images_dir"]), 0, num_images or ccfg.get("threshold_num_images"))
    print(f"Clean images for thresholds / region masses: {len(images)} from {ccfg['threshold_images_dir']}")
    maps, strides = {l: [] for l in tap.layers}, {}
    for n, path in enumerate(images, 1):
        _, x, _ = load_image_tensor(path, det_device(det))
        acts = tap.get_activations(x)
        for l in tap.layers:
            d, ids = prof.nearest(l, acts[l])
            maps[l].append((d.astype(np.float32), ids.astype(np.int16)))
            strides[l] = layer_stride(acts[l])
        if n % 200 == 0:
            print(f"  {n}/{len(images)}")
    return maps, strides


def pass2_thresholds(cfg, prof, maps, names, ccfg):
    q = ccfg.get("threshold_quantile", 0.995)
    min_cells = ccfg.get("min_threshold_cells", 1000)
    print(f"Pass 2: thresholds = clean quantile {q} of the nearest-profile distance")
    for l, lst in maps.items():
        d = np.concatenate([a.ravel() for a, _ in lst])
        ids = np.concatenate([b.ravel() for _, b in lst])
        t = float(np.quantile(d, q))
        table, fallback = {}, []
        for cid in prof.class_ids[l]:
            sel = d[ids == cid]
            if sel.size >= min_cells:
                table[class_key(cid, names)] = round(float(np.quantile(sel, q)), 4)
            else:
                fallback.append(f"{class_key(cid, names)} ({sel.size} cells)")
        cfg["panel2"].setdefault("calibrated", {})[str(l)] = {"threshold": round(t, 4),
                                                             "clean_median": round(float(np.median(d)), 4)}
        cfg["panel3"].setdefault("calibrated", {})[str(l)] = {"layer_threshold": round(t, 4),
                                                             "class_thresholds": table}
        print(f"  layer {l}: clean median {np.median(d):.2f} | layer threshold {t:.3f} | per class: "
              + ", ".join(f"{k} {v:.2f}" for k, v in table.items())
              + (f"; using the layer threshold (too few cells): {', '.join(fallback)}" if fallback else ""))


def joint_mass_thresholds(M, fpr):
    """M: (images, scales) largest region mass per clean image at each scale (0 = no region).
    Same strictness at every scale: threshold_t = the (k+1)-th largest value of column t, so roughly the top k clean
    images of EACH scale are above it. k is raised as far as possible while the share of clean images flagged at ANY
    scale stays <= fpr. Returns (thresholds per scale, flagged share per scale, flagged share overall)."""
    n = M.shape[0]
    desc = -np.sort(-M, axis=0)

    def limits(k):
        return np.ceil(desc[min(k, n - 1)] * 1000) / 1000

    k_best = 0
    for k in range(n):
        if (M > limits(k)).any(axis=1).mean() <= fpr:
            k_best = k
        else:
            break
    lim = limits(k_best)
    return lim, (M > lim).mean(axis=0), float((M > lim).any(axis=1).mean())


def region_calibration(cfg, maps, strides, ccfg):
    """Per layer and panel, from clean images only: one mass threshold per thickness scale, set jointly so that at most
    `target_clean_image_fpr` of clean images keep a region at any scale."""
    fpr = ccfg.get("target_clean_image_fpr", 0.02)
    scales = sorted(int(t) for t in ccfg.get("thickness_scales", [0, 1, 2, 3]))
    print(f"Regions: thickness scales {scales}; mass thresholds set together so that at most {fpr:.1%} of clean "
          f"images keep a region at any scale")
    joins = {p: int(cfg[p]["region"].get("join_cells", 1)) for p in PANELS}
    for l, lst in maps.items():
        s = strides[l]
        best = {p: [] for p in PANELS}
        for d, ids in lst:
            pm = panel_maps(cfg, l, d.astype(np.float64), ids)
            for p in PANELS:
                ratio, hot, joined = hot_and_joined(*pm[p], joins[p])
                row = []
                for t in scales:
                    r = regions_at_scale(ratio, hot, joined, s, t, -np.inf) if hot.any() else []
                    row.append(max((x["mass"] for x in r), default=0.0))
                best[p].append(row)
        line = []
        for p in PANELS:
            lim, per_scale, overall = joint_mass_thresholds(np.asarray(best[p]), fpr)
            cfg[p]["calibrated"][str(l)]["region"] = {
                "mass_thresholds": {str(t): float(m) for t, m in zip(scales, lim)},
                "clean_images_flagged": round(overall, 4),
                "clean_images_flagged_per_scale": {str(t): round(float(f), 4) for t, f in zip(scales, per_scale)}}
            line.append(f"{p}: mass > " + ", ".join(f"t{t} {m:.2f}" for t, m in zip(scales, lim))
                        + f" (clean flagged {overall:.1%})")
        print(f"  layer {l} (stride {s}): " + " | ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--thresholds-only", action="store_true", help="keep the profile file; redo pass 2 and region masses")
    ap.add_argument("--regions-only", action="store_true", help="keep profiles and the thresholds in the config; redo the per-scale mass thresholds")
    ap.add_argument("--max-images", type=int, default=None, help="pass-1 images (override max_calibration_images)")
    ap.add_argument("--threshold-images", type=int, default=None, help="clean images (override threshold_num_images)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg = cfg["calibration"]
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = Detector(cfg, device)
    tap = BackboneTap(det.net, cfg["tap_layers"])
    print(f"Model {det.name}, device {device}")

    if args.thresholds_only or args.regions_only:
        with open(resolve(cfg, cfg["profile_file"]), "rb") as f:
            data = pickle.load(f)
        print(f"Using existing profiles {cfg['profile_file']}")
    else:
        data = pass1_profiles(cfg, det, tap, ccfg, args.max_images)
    prof = MahaProfiles(data, device)
    missing = [l for l in tap.layers if l not in prof.layers]
    if missing:
        raise ValueError(f"The profile file has no layers {missing}.")

    maps, strides = clean_distances(cfg, det, tap, prof, ccfg, args.threshold_images)
    if not args.regions_only:
        pass2_thresholds(cfg, prof, maps, det.names, ccfg)
    region_calibration(cfg, maps, strides, ccfg)

    info = cfg.setdefault("calibration_info", {})
    today = datetime.date.today().isoformat()
    if not (args.thresholds_only or args.regions_only):
        info["profiles"] = f"{today}: {cfg['profile_file']} from {data['profile_images']} images"
    if not args.regions_only:
        info["thresholds"] = (f"{today}: quantile {ccfg.get('threshold_quantile', 0.995)} of "
                              f"{len(next(iter(maps.values())))} clean images")
    info["regions"] = (f"{today}: target clean image FPR {ccfg.get('target_clean_image_fpr', 0.02)}, "
                       f"join_cells at that time, thickness scales {ccfg.get('thickness_scales', [0, 1, 2, 3])}")
    save_config(cfg)
    print(f"Wrote the calibrated values into {cfg['_path']} (previous version: {cfg['_path']}.bak)")


if __name__ == "__main__":
    main()
