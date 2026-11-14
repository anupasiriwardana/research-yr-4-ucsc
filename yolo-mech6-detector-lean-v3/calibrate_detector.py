"""
Calibrate the Mech-6 detector for one model. The results are WRITTEN INTO THE CONFIG FILE (panel2.calibrated and
panel3.calibrated), so they can be read and edited by hand. The previous config is kept as <config>.json.bak.

Three stages:
  pass 1  profiles      log-energy mean/std per class + background per layer, from `profile_images_dir` (BDD train).
                        The detector's own boxes (offline only) say which cells belong to which class.
                        Saved to `profile_file` (.pkl, same format as yolo-mech6, so those files can be reused).
  pass 2  thresholds    on clean `threshold_images_dir` images (BDD test, different images):
                          panel 2: threshold = `threshold_quantile` (0.995) of the clean energy z, per layer
                          panel 3: layer_threshold = the same quantile of the clean nearest-profile score;
                                   class_thresholds = the same quantile over the clean cells whose NEAREST profile is
                                   that class (only classes with >= `min_threshold_cells` such cells; others fall back
                                   to the layer threshold)
  regions  one mass threshold per thickness scale (`thickness_scales`), on the same clean images. Detection tries
                        every scale, so the thresholds are set TOGETHER: the same strictness at every scale, tightened
                        until at most `target_clean_image_fpr` of clean images keep a region at ANY scale.
                        No thickness is chosen here: the detector picks the scale per region.
                        Uses the thresholds and join_cells in the config.

Usage:
    python calibrate_detector.py --config config_yolov8.json                    # everything
    python calibrate_detector.py --config config_yolov8.json --thresholds-only  # keep profiles; redo pass 2 + regions
    python calibrate_detector.py --config config_yolov8.json --regions-only     # keep profiles + thresholds; redo regions
                                                                                # (after editing join_cells, a threshold or the scales)
"""

import argparse
import datetime
import pickle
import random
from collections import defaultdict

import numpy as np
import torch

from mech6_core import (BG_ID, BackboneTap, Detector, EnergyProfiles, class_key, layer_stride, list_images,
                        hot_and_joined, load_config, load_image_tensor, log_energy_map, panel3_threshold_map, panel_maps,
                        regions_at_scale, resolve, save_config)

PANEL3_MODES = ("per_layer", "per_class")


def pass1_profiles(cfg, det, tap, ccfg, max_images):
    layers = tap.layers
    min_samples = ccfg.get("min_calibration_samples", 100)
    bg_per_image = ccfg.get("background_samples_per_image", 250)
    images = list_images(resolve(cfg, ccfg["profile_images_dir"]), 0, max_images or ccfg.get("max_calibration_images"))
    print(f"Pass 1: profiles from {len(images)} images, layers {layers}")
    random.seed(0)
    per_class = {l: defaultdict(list) for l in layers}
    g_n = {l: 0 for l in layers}
    g_s = {l: np.zeros(2) for l in layers}
    for n, path in enumerate(images, 1):
        _, x, _ = load_image_tensor(path, det_device(det))
        xyxy, cls = det.predict_boxes(x)
        acts = tap.get_activations(x)
        for l in layers:
            le = log_energy_map(acts[l])
            H, W = le.shape
            s = layer_stride(acts[l])
            g_n[l] += le.size
            g_s[l] += [le.sum(), (le ** 2).sum()]
            covered = np.zeros((H, W), bool)
            for box, c in zip(xyxy, cls):
                x1, y1, x2, y2 = (box / s).astype(int)
                x1, x2 = min(max(x1, 0), W - 1), min(max(x2, 0), W)
                y1, y2 = min(max(y1, 0), H - 1), min(max(y2, 0), H)
                if x2 <= x1 or y2 <= y1:
                    continue
                covered[y1:y2, x1:x2] = True
                per_class[l][int(c)].append(le[y1:y2, x1:x2].ravel())
            ys, xs = np.nonzero(~covered)
            if ys.size:
                pick = random.sample(range(ys.size), min(bg_per_image, ys.size))
                per_class[l][BG_ID].append(le[ys[pick], xs[pick]])
        if n % 200 == 0:
            print(f"  {n}/{len(images)}")
    stats = {}
    for l in layers:
        n, (l1, l2) = g_n[l], g_s[l]
        mean_l = l1 / n
        stats[l] = {"global": {"mean_log": float(mean_l), "std_log": float(np.sqrt(l2 / n - mean_l ** 2)), "n": int(n)}}
        for cid, parts in sorted(per_class[l].items()):
            v = np.concatenate(parts)
            if v.size > min_samples:
                stats[l][cid] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size)}
        print(f"  layer {l}: {len(stats[l]) - 1} profiles: "
              + ", ".join(f"{class_key(c, det.names)} {stats[l][c]['mean']:.2f}/{stats[l][c]['std']:.2f}"
                          for c in sorted(k for k in stats[l] if k != "global")))
    path = resolve(cfg, cfg["profile_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    print(f"  saved {path}")
    return stats


def det_device(det):
    return next(det.net.parameters()).device


def clean_log_energy(cfg, det, tap, ccfg, num_images):
    images = list_images(resolve(cfg, ccfg["threshold_images_dir"]), 0, num_images or ccfg.get("threshold_num_images"))
    print(f"Clean images for thresholds / region sizes: {len(images)} from {ccfg['threshold_images_dir']}")
    maps, strides = {l: [] for l in tap.layers}, {}
    for n, path in enumerate(images, 1):
        _, x, _ = load_image_tensor(path, det_device(det))
        acts = tap.get_activations(x)
        for l in tap.layers:
            maps[l].append(log_energy_map(acts[l]).astype(np.float32))
            strides[l] = layer_stride(acts[l])
        if n % 200 == 0:
            print(f"  {n}/{len(images)}")
    return maps, strides


def pass2_thresholds(cfg, prof, maps, names, ccfg):
    q = ccfg.get("threshold_quantile", 0.995)
    min_cells = ccfg.get("min_threshold_cells", 1000)
    score_name = cfg["panel3"].get("score", "nearest_abs")
    print(f"Pass 2: thresholds = clean quantile {q}; panel 3 score '{score_name}'")
    for l, lst in maps.items():
        # one image at a time (float32 / int16), so memory stays small however many images or profiles there are
        z, s, ids = [], [], []
        for le in lst:
            le = le.astype(np.float64)
            z.append(prof.energy_z(l, le).astype(np.float32).ravel())
            si, ii = prof.nearest(l, le, score_name)
            s.append(si.astype(np.float32).ravel())
            ids.append(ii.astype(np.int16).ravel())
        z, s, ids = np.concatenate(z), np.concatenate(s), np.concatenate(ids)
        t2 = float(np.quantile(z, q))
        del z
        t3 = float(np.quantile(s, q))
        table, fallback = {}, []
        for cid in prof.class_ids[l]:
            sel = s[ids == cid]
            if sel.size >= min_cells:
                table[class_key(cid, names)] = round(float(np.quantile(sel, q)), 4)
            else:
                fallback.append(f"{class_key(cid, names)} ({sel.size} cells)")
        cfg["panel2"].setdefault("calibrated", {})[str(l)] = {"threshold": round(t2, 4)}
        cfg["panel3"].setdefault("calibrated", {})[str(l)] = {"score": score_name, "layer_threshold": round(t3, 4),
                                                             "class_thresholds": table}
        print(f"  layer {l}: panel 2 z threshold {t2:.3f} | panel 3 layer threshold {t3:.3f}; per class: "
              + ", ".join(f"{k} {v:.3f}" for k, v in table.items())
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


def region_calibration(cfg, prof, maps, strides, ccfg):
    """Per layer, for panel 2 and for panel 3 in both modes, from clean images only: one mass threshold per thickness
    scale, set jointly so that at most `target_clean_image_fpr` of clean images keep a region at any scale."""
    fpr = ccfg.get("target_clean_image_fpr", 0.02)
    scales = sorted(int(t) for t in ccfg.get("thickness_scales", [0, 1, 2, 3]))
    print(f"Regions: thickness scales {scales}; mass thresholds set together so that at most {fpr:.1%} of clean "
          f"images keep a region at any scale")
    keys = ("panel2",) + PANEL3_MODES
    joins = {"panel2": int(cfg["panel2"]["region"].get("join_cells", 1)),
             **{m: int(cfg["panel3"]["region"].get("join_cells", 1)) for m in PANEL3_MODES}}
    for l, lst in maps.items():
        s = strides[l]
        best = {k: [] for k in keys}
        for le in lst:
            pm = panel_maps(cfg, prof, l, le.astype(np.float64))
            score3, _, ids = pm["panel3"]
            setups = {"panel2": pm["panel2"]}
            for mode in PANEL3_MODES:
                setups[mode] = (score3, panel3_threshold_map(cfg["panel3"]["calibrated"][str(l)], ids, mode))
            for k, (sc, thr) in setups.items():
                ratio, hot, joined = hot_and_joined(sc, thr, joins[k])
                row = []
                for t in scales:
                    r = regions_at_scale(ratio, hot, joined, s, t, -np.inf) if hot.any() else []
                    row.append(max((x["mass"] for x in r), default=0.0))
                best[k].append(row)
        line = []
        for k in keys:
            lim, per_scale, overall = joint_mass_thresholds(np.asarray(best[k]), fpr)
            region = {"mass_thresholds": {str(t): float(m) for t, m in zip(scales, lim)},
                      "clean_images_flagged": round(overall, 4),
                      "clean_images_flagged_per_scale": {str(t): round(float(f), 4) for t, f in zip(scales, per_scale)}}
            if k == "panel2":
                cfg["panel2"]["calibrated"][str(l)]["region"] = region
            else:
                cfg["panel3"]["calibrated"][str(l)].setdefault("region", {})[k] = region
            line.append(f"{k}: mass > " + ", ".join(f"t{t} {m:.2f}" for t, m in zip(scales, lim))
                        + f" (clean flagged {overall:.1%})")
        print(f"  layer {l} (stride {s}): " + " | ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--thresholds-only", action="store_true", help="keep the profile file; redo pass 2 and region sizes")
    ap.add_argument("--regions-only", action="store_true", help="keep profiles and the thresholds in the config; redo the per-scale mass thresholds")
    ap.add_argument("--max-images", type=int, default=None, help="pass-1 images (override max_calibration_images)")
    ap.add_argument("--threshold-images", type=int, default=None, help="clean images (override threshold_num_images)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg = cfg["calibration"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = Detector(cfg, device)
    tap = BackboneTap(det.net, cfg["tap_layers"])
    print(f"Model {det.name}")

    if args.thresholds_only or args.regions_only:
        with open(resolve(cfg, cfg["profile_file"]), "rb") as f:
            stats = pickle.load(f)
        print(f"Using existing profiles {cfg['profile_file']}")
    else:
        stats = pass1_profiles(cfg, det, tap, ccfg, args.max_images)
    prof = EnergyProfiles(stats)
    missing = [l for l in tap.layers if l not in prof.layers]
    if missing:
        raise ValueError(f"The profile file has no layers {missing}.")

    maps, strides = clean_log_energy(cfg, det, tap, ccfg, args.threshold_images)
    if not args.regions_only:
        pass2_thresholds(cfg, prof, maps, det.names, ccfg)
    region_calibration(cfg, prof, maps, strides, ccfg)

    info = cfg.setdefault("calibration_info", {})
    today = datetime.date.today().isoformat()
    if not (args.thresholds_only or args.regions_only):
        info["profiles"] = f"{today}: {cfg['profile_file']}"
    if not args.regions_only:
        info["thresholds"] = (f"{today}: quantile {ccfg.get('threshold_quantile', 0.995)} of "
                              f"{len(next(iter(maps.values())))} clean images")
    info.pop("min_region_cells", None)
    info["regions"] = (f"{today}: target clean image FPR {ccfg.get('target_clean_image_fpr', 0.02)}, "
                       f"join_cells at that time, thickness scales {ccfg.get('thickness_scales', [0, 1, 2, 3])}")
    save_config(cfg)
    print(f"Wrote the calibrated values into {cfg['_path']} (previous version: {cfg['_path']}.bak)")


if __name__ == "__main__":
    main()
