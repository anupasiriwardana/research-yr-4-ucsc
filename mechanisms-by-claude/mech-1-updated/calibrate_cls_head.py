"""
Calibration for Mechanism 1, Option B: per-class Mahalanobis statistics
PLUS a background class, so detection-time scoring never has to rely
on the (possibly-fooled) detector's own box output to decide what to
compare a cell against. See box-dependency-problem-and-solutions.md
for the full rationale.

The object-class collection logic is unchanged from before. The new
part: every cell NOT covered by any detected box is a background
candidate; a random subsample of those (capped per image, since most
of any image is background) is collected the same way as an object
class, under a reserved class id (default -1).
"""

import torch
import cv2
import os
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO
from cls_head_adapter import YOLOv8ClsHeadAdapter

# 1. Load Configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATA_DIR = Path(config["clean_data_dir"])
PROFILE_DIR = Path(config["profiles_dir"])
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

profile_path = PROFILE_DIR / config["calibration_profile_filename"]
stride = config["detector_settings"]["stride"]
min_samples = config["detector_settings"]["min_calibration_samples"]
BACKGROUND_ID = config["detector_settings"].get("background_class_id", -1)
BG_SAMPLES_PER_IMAGE = config["detector_settings"].get("background_samples_per_image", 50)
INPUT_SIZE = 640  # keep in sync with cls_head_detector.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(config["model_path"]).to(device)
adapter = YOLOv8ClsHeadAdapter(model.model)

per_class_features = defaultdict(list)

print("Starting offline calibration on clean dataset (Option B: object classes + background)...")
n_images = 0
for img_file in os.listdir(DATA_DIR):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = str(DATA_DIR / img_file)
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"  Skipping unreadable file: {img_file}")
        continue

    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(orig_img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    results = model.predict(source=img_tensor, verbose=False)[0]
    activations = adapter.get_activations(img_tensor)
    feat = activations["P3"]  # [1, C, H, W]
    H, W = feat.shape[2], feat.shape[3]

    # Tracks which grid cells are covered by a real object box, so the
    # remaining cells can be sampled as background candidates below.
    covered = torch.zeros((H, W), dtype=torch.bool)

    if len(results.boxes) > 0:
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            cls_id = int(cls.item())

            # Box coords already share feat's coordinate system (see
            # earlier coordinate-system fix) -- no rescale needed.
            x1, y1, x2, y2 = (box / stride).int().tolist()
            x1 = min(max(x1, 0), W - 1)
            x2 = min(max(x2, 0), W)
            y1 = min(max(y1, 0), H - 1)
            y2 = min(max(y2, 0), H)

            if x2 <= x1 or y2 <= y1:
                continue

            covered[y1:y2, x1:x2] = True

            # Sample every interior grid cell of the box (matches what
            # the detector will score at inference time), not just the
            # center.
            for gy in range(y1, y2):
                for gx in range(x1, x2):
                    per_class_features[cls_id].append(feat[0, :, gy, gx].cpu())

    # --- Background sampling (new in Option B) ---
    # A random, CAPPED subsample of non-object cells per image. Capping
    # matters: an 80x80 P3 grid has 6,400 cells, and most of any image
    # is background, so collecting every uncovered cell across a
    # multi-thousand-image dataset would dwarf the object-class data
    # and risk the same kind of memory blowup that hit the patch
    # generator's original dataset-preloading bug.
    bg_coords = (~covered).nonzero(as_tuple=False).tolist()
    random.shuffle(bg_coords)
    for gy, gx in bg_coords[:BG_SAMPLES_PER_IMAGE]:
        per_class_features[BACKGROUND_ID].append(feat[0, :, gy, gx].cpu())

    n_images += 1
    torch.cuda.empty_cache()

print(f"Processed {n_images} clean images.")

# Compute Mean and Inverse Covariance per class, including background
stats = {}
for cls_id, feats in per_class_features.items():
    if len(feats) > min_samples:
        feats_tensor = torch.stack(feats)
        mean = feats_tensor.mean(dim=0)
        cov = torch.cov(feats_tensor.T) + 1e-6 * torch.eye(feats_tensor.shape[1])
        inv_cov = torch.linalg.inv(cov)
        stats[cls_id] = {'mean': mean, 'inv_cov': inv_cov}
        label = "background" if cls_id == BACKGROUND_ID else f"class {cls_id}"
        print(f"  {label}: {len(feats)} samples")
    else:
        label = "background" if cls_id == BACKGROUND_ID else f"class {cls_id}"
        print(f"  Skipping {label}: only {len(feats)} samples (< {min_samples})")

with open(profile_path, 'wb') as f:
    pickle.dump(stats, f)

print(f"\nCalibration complete. Stats for {len(stats)} classes (including background) saved to {profile_path}")
print("This profile is for OPTION B scoring (minimum Mahalanobis distance "
      "across all classes + background, no box-gating). Make sure "
      "cls_head_detector.py has been updated to match -- this profile is "
      "NOT compatible with the old box-gated detector, and vice versa.")
print("Also re-derive `threshold` from a fresh clean-image run once this "
      "profile is in place: the scoring rule changed, so previously-tuned "
      "threshold values are not guaranteed to still be meaningful.")