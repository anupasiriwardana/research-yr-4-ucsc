"""
Run Mech-6-Mahalanobis on a folder of images (needs the calibration file from calibrate_mahalanobis.py).

Per image:
  * <out_dir>/<stem>.npz            (if save_maps) keys d1, d2, d3 = nearest-profile distance per tapped layer
                                    (float32 [H, W]) and c1, c2, c3 = id of the nearest profile (-1 = background)
  * <visualizations_dir>/maha_<stem>.png   (if save_visualizations) the two-panel picture for `viz_layer`

Folders, slice and switches come from the config; the flags override them for one run.

Usage:
    python extract_mahalanobis_maps.py --config config_yolov8.json
    python extract_mahalanobis_maps.py --config config_yolov8.json --input-dir <patched> --out-dir maps/yolov8/patched
        --visualizations-dir viz/yolov8/patched --gt-dir <coordinates>
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from maha_core import (BackboneTap, Detector, MahaProfiles, gt_boxes, layer_stride, list_images, load_config,
                       load_image_tensor, profile_path, resolve)
from maha_viz import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", default=None, help="override input_dir")
    ap.add_argument("--out-dir", default=None, help="override out_dir (maps)")
    ap.add_argument("--start-index", type=int, default=None, help="override start_index")
    ap.add_argument("--num-images", type=int, default=None, help="override num_images (null/0 = all)")
    ap.add_argument("--save-maps", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--save-visualizations", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--visualizations-dir", default=None, help="override visualizations_dir")
    ap.add_argument("--viz-layer", type=int, default=None, help="override viz_layer")
    ap.add_argument("--gt-dir", default=None, help="override gt_coordinates_dir (patch coordinate files)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    in_dir = Path(args.input_dir) if args.input_dir else resolve(cfg, cfg.get("input_dir"))
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg, cfg.get("out_dir"))
    start = cfg.get("start_index", 0) if args.start_index is None else args.start_index
    count = cfg.get("num_images") if args.num_images is None else args.num_images
    save_maps = cfg.get("save_maps", True) if args.save_maps is None else args.save_maps
    viz = cfg.get("save_visualizations", True) if args.save_visualizations is None else args.save_visualizations
    viz_dir = Path(args.visualizations_dir) if args.visualizations_dir else resolve(cfg, cfg.get("visualizations_dir"))
    viz_layer = args.viz_layer or cfg.get("viz_layer", 1)
    gt_dir = Path(args.gt_dir) if args.gt_dir else resolve(cfg, cfg.get("gt_coordinates_dir"))
    if in_dir is None:
        raise ValueError("Set input_dir in the config (or pass --input-dir).")
    if save_maps and out_dir is None:
        raise ValueError("save_maps is on but no out_dir is set.")
    if viz and viz_dir is None:
        raise ValueError("save_visualizations is on but no visualizations_dir is set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppath = profile_path(cfg)
    if not ppath.exists():
        raise FileNotFoundError(f"No calibration at {ppath}; run calibrate_mahalanobis.py first.")
    prof = MahaProfiles(ppath, device)
    if viz and viz_layer not in prof.layers:
        raise ValueError(f"viz_layer {viz_layer} is not a calibrated layer {prof.layers}.")
    det = Detector(cfg, device)
    tap = BackboneTap(det.net, prof.layers)
    if save_maps:
        out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(in_dir, start, count)
    print(f"Model {det.name}; {len(images)} image(s) from {in_dir}; maps={'on' if save_maps else 'off'}, "
          f"pictures={'layer ' + str(viz_layer) if viz else 'off'}")
    for n, path in enumerate(images, 1):
        rgb, x = load_image_tensor(path, device)
        acts = tap.get_activations(x)
        res = {l: prof.distances(l, acts[l]) for l in prof.layers}
        if save_maps:
            arrays = {f"d{l}": d for l, (d, _) in res.items()}
            arrays.update({f"c{l}": c for l, (_, c) in res.items()})
            np.savez_compressed(out_dir / f"{path.stem}.npz", **arrays)
        if viz:
            render(rgb, res[viz_layer][0], viz_layer, layer_stride(acts[viz_layer]), prof.quantile(viz_layer, "q50"),
                   prof.threshold(viz_layer), gt_boxes(gt_dir, path.stem), viz_dir / f"maha_{path.stem}.png",
                   cfg.get("viz_scale_max_ratio", 2.0), cfg.get("viz_outline", True))
        if n % 50 == 0:
            print(f"  {n}/{len(images)}")
    print(f"Done: {len(images)} image(s)." + (f" Maps in {out_dir}." if save_maps else "") + (f" Pictures in {viz_dir}." if viz else ""))


if __name__ == "__main__":
    main()
