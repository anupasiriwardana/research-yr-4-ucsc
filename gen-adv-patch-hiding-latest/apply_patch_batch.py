"""
Paste a saved patch on many images at once.

For each image the model (from the config) finds up to `apply.max_patches` distinct target objects
(most confident first, at least `apply.min_separation` apart) and a patch is pasted on each.
Per image it writes
  * <paths.patched_dir>/<apply.prefix><original file name>      the patched image (640x640)
  * <paths.coordinates_dir>/<original stem>.txt                 one `box=(x1, y1, x2, y2)` line per patch
    (640-px coordinates; the file is empty when no target was found). The coordinate file is named after
    the ORIGINAL image so the existing detection / metric tools can match it; set
    apply.coordinates_use_prefix to true to name it after the patched image instead.

Usage:
    python apply_patch_batch.py --config config_yolov8.json
    python apply_patch_batch.py --config config_yolov8.json --max-patches 3 --num-images 50 --prefix multi-patched_
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from patch_common import (DetModel, apply_patches, list_images, load_config, load_image, patch_file, resolve,
                          save_image, section, select_all_target_objects, write_coordinate_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-patches", type=int, default=None, help="override apply.max_patches")
    ap.add_argument("--prefix", default=None, help="override apply.prefix (patched image name prefix)")
    ap.add_argument("--patch-name", default=None, help="override the patch to use (name in patch_dir)")
    ap.add_argument("--num-images", type=int, default=None, help="override apply.num_images")
    ap.add_argument("--start-index", type=int, default=None, help="override apply.start_index")
    ap.add_argument("--input-dir", default=None, help="override paths.clean_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ap_cfg = dict(section(cfg, "apply"))
    paths = section(cfg, "paths")
    if args.patch_name:
        ap_cfg["patch_name"] = args.patch_name
    max_patches = args.max_patches or ap_cfg.get("max_patches", 4)
    prefix = ap_cfg.get("prefix", "multi-patched_") if args.prefix is None else args.prefix
    num_images = args.num_images or ap_cfg.get("num_images", 100)
    start = ap_cfg.get("start_index", 0) if args.start_index is None else args.start_index
    scale, rotation = ap_cfg.get("scale", 0.15), ap_cfg.get("rotation", 0.0)
    min_sep, conf, target_class = ap_cfg.get("min_separation", 0.3), ap_cfg.get("conf_thresh", 0.25), ap_cfg.get("target_class")
    if max_patches <= 0 or scale <= 0 or num_images <= 0 or start < 0:
        raise ValueError("max_patches, scale and num_images must be > 0 and start_index >= 0.")

    in_dir = Path(args.input_dir) if args.input_dir else resolve(cfg, paths["clean_dir"])
    out_dir = resolve(cfg, paths["patched_dir"])
    coord_dir = resolve(cfg, paths["coordinates_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    coord_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(in_dir, start, num_images, ap_cfg.get("image_extensions"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pfile = patch_file(cfg, ap_cfg)
    if not pfile.exists():
        raise FileNotFoundError(f"Patch not found: {pfile} (run generate_hiding_patch.py, or set apply.patch_name / patch_path).")
    patch = torch.from_numpy(np.load(pfile)).float().to(device)
    if patch.ndim != 3 or patch.shape[0] != 3:
        raise ValueError(f"Expected a [3, h, w] patch array, got {tuple(patch.shape)}.")
    model = DetModel(cfg, device)
    print(f"Patch {pfile.name}; model {cfg['model']['weights']}; {len(images)} image(s); "
          f"max_patches={max_patches}, target_class={target_class}, prefix='{prefix}'")

    counts, failed = [], []
    for n, path in enumerate(images, 1):
        try:
            image = load_image(path, device)
            if image is None:
                raise ValueError("could not read image")
            with torch.no_grad():
                targets = select_all_target_objects(model.raw_preds(image.unsqueeze(0)), conf, target_class, max_patches, min_sep)
            patched, boxes = apply_patches(image, patch, targets, scale, rotation)
            out_name = f"{prefix}{path.name}"
            save_image(patched, out_dir / out_name)
            coord_name = (Path(out_name).stem if ap_cfg.get("coordinates_use_prefix") else path.stem) + ".txt"
            write_coordinate_file(coord_dir / coord_name, boxes)
            counts.append(len(boxes))
            print(f"[{n}/{len(images)}] {path.name}: {len(boxes)} patch(es) -> {out_name}")
        except (OSError, ValueError, RuntimeError) as e:
            failed.append(path.name)
            print(f"[{n}/{len(images)}] FAILED {path.name}: {e}")

    print(f"\nDone. {len(counts)} image(s) written to {out_dir}; coordinates in {coord_dir}")
    if counts:
        print(f"Patches per image: mean {np.mean(counts):.2f}, max {max(counts)}, images with no target {counts.count(0)}")
    if failed:
        raise RuntimeError(f"{len(failed)} image(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
