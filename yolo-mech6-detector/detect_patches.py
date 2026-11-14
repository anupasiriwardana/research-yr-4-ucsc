"""
Run the two Mech-6 patch localizers on a folder of images. Thresholds and minimum region sizes come from the config
(written there by calibrate_detector.py; edit them by hand to test other values).

Per image:
    <detections_dir>/panel2/<stem>.txt   panel-2 (energy z) patch boxes
    <detections_dir>/panel3/<stem>.txt   panel-3 (nearest profile) patch boxes
        one line per box:  box=(x1, y1, x2, y2) score=.. peak=.. cells=.. fill=.. orig_box=(..)
        box is in 640x640 pixels (same as the GT coordinate files); orig_box in the original image size.
        An empty file = no patch found.
    <visualizations_dir>/det_<stem>.png  (if save_visualizations) the three-panel picture
    <detections_dir>/summary.csv         one row per image: number of boxes per panel

Usage:
    python detect_patches.py --config config_yolov8.json
    python detect_patches.py --config config_yolov8.json --input-dir <clean folder> --num-images 100
        --detections-dir detections/yolov8/clean --visualizations-dir viz/yolov8/clean
"""

import argparse
import csv
import pickle
from pathlib import Path

import torch

from detector_viz import render
from mech6_core import (BackboneTap, Detector, EnergyProfiles, gt_boxes, layer_stride, list_images, load_config,
                        load_image_tensor, localize, log_energy_map, min_cells_for, panel_maps, resolve,
                        write_detections)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--detections-dir", default=None)
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument("--num-images", type=int, default=None, help="0 = all")
    ap.add_argument("--save-visualizations", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--visualizations-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--layer", type=int, default=None, help="override detect_layer")
    ap.add_argument("--panel3-mode", choices=["per_layer", "per_class"], default=None, help="override panel3.threshold_mode")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["detect"]
    if args.panel3_mode:
        cfg["panel3"]["threshold_mode"] = args.panel3_mode
    mode = cfg["panel3"].get("threshold_mode", "per_class")

    in_dir = Path(args.input_dir) if args.input_dir else resolve(cfg, d.get("input_dir"))
    out_dir = Path(args.detections_dir) if args.detections_dir else resolve(cfg, d.get("detections_dir"))
    start = d.get("start_index", 0) if args.start_index is None else args.start_index
    count = d.get("num_images") if args.num_images is None else args.num_images
    viz = d.get("save_visualizations", True) if args.save_visualizations is None else args.save_visualizations
    viz_dir = Path(args.visualizations_dir) if args.visualizations_dir else resolve(cfg, d.get("visualizations_dir"))
    gt_dir = Path(args.gt_dir) if args.gt_dir else resolve(cfg, d.get("gt_coordinates_dir"))
    layer = args.layer or cfg.get("detect_layer", 1)
    if in_dir is None or out_dir is None:
        raise ValueError("Set detect.input_dir and detect.detections_dir (or pass the flags).")
    if viz and viz_dir is None:
        raise ValueError("save_visualizations is on but detect.visualizations_dir is not set.")

    with open(resolve(cfg, cfg["profile_file"]), "rb") as f:
        prof = EnergyProfiles(pickle.load(f))
    min2 = min_cells_for(cfg, "panel2", layer)
    min3 = min_cells_for(cfg, "panel3", layer, mode)
    p2cfg, p3cfg = cfg["panel2"]["region"], cfg["panel3"]["region"]
    thr2 = cfg["panel2"]["calibrated"][str(layer)]["threshold"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = Detector(cfg, device)
    tap = BackboneTap(det.net, [layer])
    images = list_images(in_dir, start, count)
    print(f"Model {det.name}; layer {layer}; {len(images)} image(s) from {in_dir}")
    print(f"  panel 2: z threshold {thr2}, min_region_cells {min2} | panel 3: {mode}, min_region_cells {min3}")

    rows = []
    for n, path in enumerate(images, 1):
        rgb, x, orig = load_image_tensor(path, device)
        acts = tap.get_activations(x)
        stride = layer_stride(acts[layer])
        pm = panel_maps(cfg, prof, layer, log_energy_map(acts[layer]))
        r2, hot2 = localize(*pm["panel2"], stride, p2cfg, min2)
        r3, hot3 = localize(pm["panel3"][0], pm["panel3"][1], stride, p3cfg, min3)
        write_detections(out_dir / "panel2" / f"{path.stem}.txt", r2, orig)
        write_detections(out_dir / "panel3" / f"{path.stem}.txt", r3, orig)
        rows.append({"image": path.name, "panel2_boxes": len(r2), "panel3_boxes": len(r3),
                     "panel2_best": round(r2[0]["score"], 3) if r2 else "", "panel3_best": round(r3[0]["score"], 3) if r3 else ""})
        if viz:
            render(rgb, layer, gt_boxes(gt_dir, path.stem),
                   {"score": pm["panel2"][0], "hot": hot2, "regions": r2, "label": f"(white: z > {thr2:.2f})", "thr": thr2},
                   {"score": pm["panel3"][0], "hot": hot3, "regions": r3, "label": f"(white: > {mode.replace('_', '-')} thr)",
                    "thr": cfg["panel3"]["calibrated"][str(layer)]["layer_threshold"]},
                   viz_dir / f"det_{path.stem}.png", cfg.get("viz_outline", True), cfg.get("viz_scale_max_ratio", 2.0))
        if n % 50 == 0:
            print(f"  {n}/{len(images)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    flagged2 = sum(r["panel2_boxes"] > 0 for r in rows)
    flagged3 = sum(r["panel3_boxes"] > 0 for r in rows)
    print(f"Done. Images with a patch: panel 2 {flagged2}/{len(rows)}, panel 3 {flagged3}/{len(rows)}. "
          f"Boxes in {out_dir}" + (f", pictures in {viz_dir}" if viz else ""))


if __name__ == "__main__":
    main()
