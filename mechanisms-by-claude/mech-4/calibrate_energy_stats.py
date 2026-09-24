"""
Offline calibration for Mechanism 4 -- Clean Feature Energy (CFE),
matching Kim, Yu & Ro (2022)'s actual method.

This is the fix for the instability observed with a purely
self-referential, per-image threshold: a single image's own local
energy statistics vary a lot with scene content (a cluttered street
vs. an empty road), so no fixed k relative to THAT image's own
mean/std can work consistently across different images -- the
reference itself moves around. A threshold derived from a large,
diverse clean dataset is a stable reference that doesn't have this
problem, exactly like Mechanism 1's calibration.

Computes, per tapped layer: mean and std of energy = (L1 norm across
channels)^2, aggregated over EVERY spatial cell of EVERY clean image.
"""

import json
import pickle
import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from stem_activation_adapter import YOLOv8StemAdapter

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATA_DIR = Path(config["clean_data_dir"])
PROFILE_DIR = Path(config["profiles_dir"])
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
profile_path = PROFILE_DIR / config["calibration_profile_filename"]

tap_layers = config["detector_settings"]["tap_layers"]
INPUT_SIZE = 640

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(config["model_path"]).to(device)
model.model.eval()
adapter = YOLOv8StemAdapter(model.model, layer_indices=tap_layers)


def compute_energy(feat: torch.Tensor) -> torch.Tensor:
    """Paper's definition: energy = (L1 norm across channels)^2,
    NOT the sum-of-squares (squared L2 norm) used in the earlier,
    incorrect version of this mechanism."""
    l1_norm = feat[0].abs().sum(dim=0)  # [H, W]
    return l1_norm ** 2


per_layer_energies = {layer_idx: [] for layer_idx in tap_layers}

print(f"Profiling Clean Feature Energy (CFE) for layers {tap_layers}...")
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

    activations = adapter.get_activations(img_tensor)
    for layer_idx in tap_layers:
        energy_map = compute_energy(activations[layer_idx])
        per_layer_energies[layer_idx].append(energy_map.flatten().cpu())

    n_images += 1
    torch.cuda.empty_cache()

print(f"Processed {n_images} clean images.")

stats = {}
for layer_idx, energies in per_layer_energies.items():
    all_values = torch.cat(energies)  # every cell, every image, this layer
    mean = all_values.mean().item()
    std = all_values.std().item()
    stats[layer_idx] = {"mean": mean, "std": std}
    print(f"  layer {layer_idx}: mean={mean:.4f}  std={std:.4f}  "
          f"3.5-sigma threshold={mean + 3.5 * std:.4f}")

with open(profile_path, "wb") as f:
    pickle.dump(stats, f)

print(f"\nCalibration complete. CFE stats for {len(stats)} layer(s) saved to {profile_path}")
print("Run feature_energy_detector.py next -- it loads this profile "
      "instead of computing statistics from each test image.")