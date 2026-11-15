"""
Measure the detection latency of the two lean-v3 localizers for one model, without writing any box files.

What is timed (same definition as the Mech-1 detector, cls_head_detector.py):
    starts immediately before feature extraction and ends after the boxes are found.
      shared    : model forward pass with the backbone hook (BackboneTap) + log-energy map of the detect layer
      panel 2   : energy z score + threshold map + multi-scale localization
      panel 3   : nearest-profile score + per-class threshold map + multi-scale localization
    panel 2 latency = shared + panel 2;  panel 3 latency = shared + panel 3 (each panel used on its own).
    Excluded: image loading and preprocessing, visualization, writing files.
GPU work is synchronized before every clock read. The first `--warmup` images are run but not counted.

Usage:
    python measure_latency.py --config config_yolov8.json --input-dir ../test-patch-det-multi-img/gen-patched-img/yolov8-patched-imgs --out latency/yolov8.json
"""

import argparse
import json
import pickle
import statistics
import time
from pathlib import Path

import torch

from mech6_core import (BackboneTap, Detector, EnergyProfiles, layer_stride, list_images, load_config, load_image_tensor,
                        localize, log_energy_map, panel3_threshold_map, region_settings, resolve)
import numpy as np


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(values):
    return {"mean_ms": statistics.fmean(values), "median_ms": statistics.median(values),
            "std_ms": statistics.pstdev(values), "min_ms": min(values), "max_ms": max(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", required=True, help="images to time (e.g. the patched images)")
    ap.add_argument("--num-images", type=int, default=None, help="default: all")
    ap.add_argument("--warmup", type=int, default=5, help="images run first and not counted")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", required=True, help="JSON file to write")
    args = ap.parse_args()

    cfg = load_config(args.config)
    layer = args.layer or cfg.get("detect_layer", 1)
    mode = cfg["panel3"].get("threshold_mode", "per_class")
    score_name = cfg["panel3"].get("score", "nearest_abs")
    with open(resolve(cfg, cfg["profile_file"]), "rb") as f:
        prof = EnergyProfiles(pickle.load(f))
    rs2 = region_settings(cfg, "panel2", layer)
    rs3 = region_settings(cfg, "panel3", layer, mode)
    thr2 = float(cfg["panel2"]["calibrated"][str(layer)]["threshold"])
    c3 = cfg["panel3"]["calibrated"][str(layer)]
    if c3.get("score", score_name) != score_name:
        raise ValueError(f"panel3 was calibrated for '{c3.get('score')}' but the config says '{score_name}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = Detector(cfg, device)
    tap = BackboneTap(det.net, [layer])
    images = list_images(Path(args.input_dir), 0, args.num_images)
    if len(images) <= args.warmup:
        raise ValueError(f"Need more than {args.warmup} images, got {len(images)}.")
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"Model {det.name}; layer {layer}; panel 3 {score_name}/{mode}; {len(images)} image(s) "
          f"({args.warmup} warm-up); device {gpu}")

    shared, p2, p3 = [], [], []
    for n, path in enumerate(images):
        _, x, _ = load_image_tensor(path, device)                           # not timed
        sync(device)
        t0 = time.perf_counter()
        acts = tap.get_activations(x)
        stride = layer_stride(acts[layer])
        le = log_energy_map(acts[layer])
        sync(device)
        t1 = time.perf_counter()
        z = prof.energy_z(layer, le)
        localize(z, np.full(z.shape, thr2), stride, *rs2)
        t2 = time.perf_counter()
        s, ids = prof.nearest(layer, le, score_name)
        localize(s, panel3_threshold_map(c3, ids, mode), stride, *rs3)
        t3 = time.perf_counter()
        if n >= args.warmup:
            shared.append((t1 - t0) * 1000.0)
            p2.append((t2 - t1) * 1000.0)
            p3.append((t3 - t2) * 1000.0)

    panel2 = [a + b for a, b in zip(shared, p2)]
    panel3 = [a + b for a, b in zip(shared, p3)]
    result = {"model": det.name, "config": str(Path(args.config).name), "layer": layer,
              "panel3": {"score": score_name, "threshold_mode": mode}, "device": gpu,
              "input_dir": str(args.input_dir), "images_timed": len(shared), "warmup_images": args.warmup,
              "definition": "forward pass + log-energy + score + threshold map + multi-scale localization; "
                            "excludes image loading/preprocessing, visualization and file writing",
              "shared_forward_and_energy": summarize(shared),
              "panel2_only_step": summarize(p2), "panel3_only_step": summarize(p3),
              "panel2_latency": summarize(panel2), "panel3_latency": summarize(panel3)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  panel 2: {result['panel2_latency']['mean_ms']:.2f} ms (median {result['panel2_latency']['median_ms']:.2f}) | "
          f"panel 3: {result['panel3_latency']['mean_ms']:.2f} ms (median {result['panel3_latency']['median_ms']:.2f}) | "
          f"shared forward+energy {result['shared_forward_and_energy']['mean_ms']:.2f} ms")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
