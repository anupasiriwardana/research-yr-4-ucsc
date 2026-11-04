"""
Save the raw per-cell energy maps of every tapped layer for a folder of images, one
`<stem>.npz` per image (keys e1, e2, e3 = energy at backbone layers 1, 2, 3 as float32).
Scoring variants are computed offline from these plus the profiles, so many ideas can be
tested without re-running the model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from mech6_energy import load_config, list_images, load_image_tensor, BackboneTap, energy_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    layers = cfg["tap_layers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(cfg["model_path"]).to(device)
    tap = BackboneTap(model.model, layers)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = list_images(args.input_dir, args.start_index, args.num_images)
    for n, path in enumerate(images, 1):
        _, tensor = load_image_tensor(path, device)
        acts = tap.get_activations(tensor)
        np.savez_compressed(out / f"{path.stem}.npz",
                            **{f"e{l}": energy_map(acts[l]).cpu().numpy().astype(np.float32) for l in layers})
        if n % 50 == 0:
            print(f"  {n}/{len(images)}")
    print(f"Saved {len(images)} energy maps to {out}")


if __name__ == "__main__":
    main()
