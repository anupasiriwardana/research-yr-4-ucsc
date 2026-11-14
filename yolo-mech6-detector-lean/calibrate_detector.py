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
  regions  min size     on the same clean images, run the localizer with every region rule EXCEPT the minimum size and
                        record the largest region per image. min_region_cells = the size that only
                        `target_clean_image_fpr` of clean images reach (at least `min_region_cells_floor`).
                        Uses the thresholds and region settings currently in the config.

Usage:
    python calibrate_detector.py --config config_yolov8.json                    # everything
    python calibrate_detector.py --config config_yolov8.json --thresholds-only  # keep profiles; redo pass 2 + regions
    python calibrate_detector.py --config config_yolov8.json --regions-only     # keep profiles + thresholds; redo regions
                                                                                # (after editing a panel's region settings)
"""

import argparse
import datetime
import pickle
import random
from collections import defaultdict

import numpy as np
import torch

from mech6_core import (BG_ID, BackboneTap, Detector, EnergyProfiles, class_key, layer_stride, list_images,
                        load_config, load_image_tensor, localize, log_energy_map, panel3_threshold_map, panel_maps,
                        resolve, save_config)

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


def region_sizes(cfg, prof, maps, strides, ccfg):
    fpr = ccfg.get("target_clean_image_fpr", 0.02)
    floor = ccfg.get("min_region_cells_floor", 4)
    print(f"Regions: min_region_cells so that at most {fpr:.1%} of clean images keep a region (floor {floor})")

    def pick(largest):
        largest = np.asarray(largest)
        m = max(floor, int(np.quantile(largest, 1 - fpr, method="higher")) + 1)
        return m, float((largest >= m).mean())

    for l, lst in maps.items():
        s = strides[l]
        big2, big3 = [], {m: [] for m in PANEL3_MODES}
        p2 = cfg["panel2"]["region"]
        p3 = cfg["panel3"]["region"]
        for le in lst:
            pm = panel_maps(cfg, prof, l, le.astype(np.float64))
            r, _ = localize(*pm["panel2"], s, p2, 0)
            big2.append(max((x["cells"] for x in r), default=0))
            score, _, ids = pm["panel3"]
            for mode in PANEL3_MODES:
                thr = panel3_threshold_map(cfg["panel3"]["calibrated"][str(l)], ids, mode)
                r, _ = localize(score, thr, s, p3, 0)
                big3[mode].append(max((x["cells"] for x in r), default=0))
        m2, f2 = pick(big2)
        cfg["panel2"]["calibrated"][str(l)]["min_region_cells"] = m2
        res3 = {mode: pick(v) for mode, v in big3.items()}
        cfg["panel3"]["calibrated"][str(l)]["min_region_cells"] = {mode: v[0] for mode, v in res3.items()}
        print(f"  layer {l} (stride {s}): panel 2 min_region_cells {m2} (clean images still flagged {f2:.1%}) | panel 3 "
              + ", ".join(f"{mode} {v[0]} ({v[1]:.1%})" for mode, v in res3.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--thresholds-only", action="store_true", help="keep the profile file; redo pass 2 and region sizes")
    ap.add_argument("--regions-only", action="store_true", help="keep profiles and the thresholds in the config; redo region sizes")
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
    region_sizes(cfg, prof, maps, strides, ccfg)

    info = cfg.setdefault("calibration_info", {})
    today = datetime.date.today().isoformat()
    if not (args.thresholds_only or args.regions_only):
        info["profiles"] = f"{today}: {cfg['profile_file']}"
    if not args.regions_only:
        info["thresholds"] = (f"{today}: quantile {ccfg.get('threshold_quantile', 0.995)} of "
                              f"{len(next(iter(maps.values())))} clean images")
    info["min_region_cells"] = (f"{today}: target clean image FPR {ccfg.get('target_clean_image_fpr', 0.02)}, "
                                f"with the region settings at that time")
    save_config(cfg)
    print(f"Wrote the calibrated values into {cfg['_path']} (previous version: {cfg['_path']}.bak)")


if __name__ == "__main__":
    main()
