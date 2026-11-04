"""
Train an adversarial "hiding" patch against one YOLO model (YOLOv5 / v8 / v9 / 11).

Same method as mechanisms-by-claude/gen-adv-patch-hiding/generate_hiding_patch-V2.py (spatially targeted,
detector-aware loss, corrected patch placement), but:
  * works on all four model families through patch_common.DetModel;
  * every name, folder and setting comes from a config file (see config_*.json and README.md).

How it works, in short:
  1. Run the CLEAN image once (no gradient). Pick the most confident target object (e.g. a car) and the
     anchors inside its box: these are pseudo-labels, no annotations needed.
  2. Paste the patch on that object with a random rotation / scale / small position jitter (EoT).
  3. Run the PATCHED image and minimise the model's confidence at exactly those anchors.
  4. Adam updates the patch pixels only; the model is frozen.

Usage:
    python generate_hiding_patch.py --config config_yolov8.json
    python generate_hiding_patch.py --config config_yolov8.json --steps 20 --max-images 40   (quick test)
"""

import argparse
import json
import random
import time

import numpy as np
import torch

from patch_common import (DetModel, INPUT_SIZE, apply_patches, build_patch_overlay, list_images, load_config,
                          load_image, resolve, save_image, section, select_target_object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override generate.num_steps")
    ap.add_argument("--max-images", type=int, default=None, help="override generate.max_images")
    ap.add_argument("--patch-name", default=None, help="override patch_name (output file name, no extension)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen = section(cfg, "generate")
    paths = section(cfg, "paths")
    patch_name = (args.patch_name or cfg["patch_name"]).removesuffix(".npy")
    patch_dir = resolve(cfg, paths["patch_dir"])
    patch_dir.mkdir(parents=True, exist_ok=True)

    patch_size = gen.get("patch_size", 150)
    conf_thresh = gen.get("conf_thresh", 0.25)
    num_steps = args.steps or gen.get("num_steps", 500)
    batch = gen.get("gpu_batch_size", 4)
    lr = gen.get("learning_rate", 0.02)
    rot_max = gen.get("rotation_max", 22.5)
    scale_min, scale_max = gen.get("scale_min", 0.10), gen.get("scale_max", 0.20)
    jitter = gen.get("placement_jitter", 0.1)
    target_class = gen.get("target_class")
    seed = gen.get("seed")
    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetModel(cfg, device)
    print(f"Model: {cfg['model']['weights']} ({model.family}), device {device}")

    clean_dir = resolve(cfg, paths["clean_dir"])
    files = list_images(clean_dir, 0, args.max_images or gen.get("max_images", 100), gen.get("image_extensions"))
    print(f"{len(files)} candidate training images in {clean_dir}; {batch} per step are loaded to the GPU.")

    patch = torch.rand(3, patch_size, patch_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=lr)

    t0 = time.time()
    for step in range(num_steps):
        optimizer.zero_grad()
        picks = np.random.choice(len(files), size=min(batch, len(files)), replace=False)
        total, valid = 0.0, 0
        for idx in picks:
            image = load_image(files[idx], device)
            if image is None:
                continue
            with torch.no_grad():
                found = select_target_object(model.raw_preds(image.unsqueeze(0)), conf_thresh, target_class)
            if found is None:
                continue
            mask, (ox, oy), _ = found
            angle = np.random.uniform(-rot_max, rot_max)
            scale = np.random.uniform(scale_min, scale_max)
            cx = float(np.clip(ox + np.random.uniform(-jitter, jitter), -0.9, 0.9))
            cy = float(np.clip(oy + np.random.uniform(-jitter, jitter), -0.9, 0.9))
            warped, wmask = build_patch_overlay(torch.clamp(patch, 0, 1), image.shape[1:], cx, cy, angle, scale)
            patched = image * (1 - wmask) + warped * wmask
            probs = model.raw_preds(patched.unsqueeze(0))[0, :, 4:].max(dim=-1)[0]
            total = total + probs[mask].mean()      # confidence at the real object's anchors
            valid += 1
        if valid == 0:
            continue
        total = total / valid
        total.backward()
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if step % 25 == 0 or step == num_steps - 1:
            print(f"  step {step:4d}/{num_steps}  mean targeted confidence: {total.item():.4f}  ({time.time() - t0:.0f}s)")

    patch_np = patch.detach().cpu().numpy()
    npy_path = patch_dir / f"{patch_name}.npy"
    np.save(npy_path, patch_np)
    save_image(patch.detach(), patch_dir / f"{patch_name}.png")
    with (patch_dir / f"{patch_name}.json").open("w", encoding="utf-8") as f:
        json.dump({"model": cfg["model"], "generate": {**gen, "num_steps": num_steps}, "images_used": len(files),
                   "seconds": round(time.time() - t0)}, f, indent=2)
    print(f"\nSaved patch array     : {npy_path}")
    print(f"Saved patch image     : {patch_dir / (patch_name + '.png')}")
    print(f"Saved training record : {patch_dir / (patch_name + '.json')}")

    # demo: patch pasted (no rotation, generate.patch_scale) on the first image that has a target
    demo_name = gen.get("demo_image")
    demo_files = [clean_dir / demo_name] if demo_name else files
    for demo_path in demo_files:
        demo = load_image(demo_path, device)
        if demo is None:
            continue
        with torch.no_grad():
            found = select_target_object(model.raw_preds(demo.unsqueeze(0)), conf_thresh, target_class)
        if found is None:
            continue
        _, (dx, dy), _ = found
        out, _ = apply_patches(demo, patch.detach(), [{"cx": dx, "cy": dy}], gen.get("patch_scale", 0.15), 0.0)
        save_image(out, patch_dir / f"{patch_name}_demo.jpg")
        print(f"Saved demo (on {demo_path.name}): {patch_dir / (patch_name + '_demo.jpg')}")
        break
    print("\nNext: apply_patch_batch.py to paste it on many images, then verify_patch_efficacy_batch.py.")


if __name__ == "__main__":
    main()
