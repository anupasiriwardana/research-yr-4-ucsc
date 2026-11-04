"""
Save the raw per-cell energy maps of every tapped layer for a folder of images, one
`<stem>.npz` per image (keys e1, e2, e3 = energy at backbone layers 1, 2, 3 as float32).
Scoring variants are computed offline from these plus the profiles, so many ideas can be
tested without re-running the model.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from mech6_models import load_detector, verify_strides
from diagnose_energy import gt_boxes
from energy_viz import render
from mech6_energy import load_config, list_images, load_image_tensor, BackboneTap, energy_map, EnergyProfiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", default=None, help="override config input_dir")
    ap.add_argument("--out-dir", default=None, help="override config out_dir")
    ap.add_argument("--start-index", type=int, default=None, help="override config start_index (default 0)")
    ap.add_argument("--num-images", type=int, default=None, help="override config num_images (default: all)")
    ap.add_argument("--save-visualizations", action=argparse.BooleanOptionalAction, default=None,
                    help="override config save_visualizations (off by default); needs a profile file")
    ap.add_argument("--visualizations-dir", default=None, help="override config visualizations_dir")
    ap.add_argument("--viz-layer", type=int, default=None, help="backbone layer shown (default 1)")
    ap.add_argument("--gt-dir", default=None, help="patch coordinates folder; draws the GT boxes and a stats header")
    args = ap.parse_args()
    cfg = load_config(args.config)
    layers = cfg["tap_layers"]
    input_dir = args.input_dir or cfg.get("input_dir")
    out_dir = args.out_dir or cfg.get("out_dir")
    if not input_dir or not out_dir:
        raise ValueError("Set input_dir and out_dir in the config (or pass --input-dir / --out-dir).")
    start_index = cfg.get("start_index", 0) if args.start_index is None else args.start_index
    num_images = cfg.get("num_images") if args.num_images is None else args.num_images
    viz =cfg.get("save_visualizations", False) if args.save_visualizations is None else args.save_visualizations
    viz_dir = args.visualizations_dir or cfg.get("visualizations_dir")
    viz_layer = args.viz_layer or cfg.get("viz_layer", 1)
    gt_dir = args.gt_dir or cfg.get("gt_coordinates_dir")
    if viz:
        if not viz_dir:
            raise ValueError("save_visualizations is on but no visualizations_dir is set.")
        with open(Path(cfg["profiles_dir"]) / cfg["profile_filename"], "rb") as f:
            profiles = EnergyProfiles(pickle.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = load_detector(cfg, device)
    tap = BackboneTap(det.net, layers)
    verify_strides(tap, device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = list_images(input_dir, start_index, num_images)
    for n, path in enumerate(images, 1):
        rgb, tensor = load_image_tensor(path, device)
        acts = tap.get_activations(tensor)
        maps = {l: energy_map(acts[l]).cpu().numpy().astype(np.float32) for l in layers}
        np.savez_compressed(out / f"{path.stem}.npz", **{f"e{l}": m for l, m in maps.items()})
        if viz:
            render(rgb, maps[viz_layer], viz_layer, profiles, gt_boxes(gt_dir, path.stem) if gt_dir else [],
                   Path(viz_dir) / f"mech6_{path.stem}.png")
        if n % 50 == 0:
            print(f"  {n}/{len(images)}")
    print(f"Saved {len(images)} energy maps to {out}")


if __name__ == "__main__":
    main()
