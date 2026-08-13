import torch
import cv2
import os
import json
import pickle
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
INPUT_SIZE = 640  # keep in sync with cls_head_detector.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(config["model_path"]).to(device)
adapter = YOLOv8ClsHeadAdapter(model.model)

per_class_features = defaultdict(list)

print("Starting offline calibration on clean dataset...")
n_images = 0
for img_file in os.listdir(DATA_DIR):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = str(DATA_DIR / img_file)
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"  Skipping unreadable file: {img_file}")
        continue

    # --- FIX 1: build ONE tensor and use it for both prediction and
    # activation extraction, so boxes and feat share a coordinate system.
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(orig_img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    results = model.predict(source=img_tensor, verbose=False)[0]
    activations = adapter.get_activations(img_tensor)
    feat = activations["P3"]  # [1, C, H, W]; H=W=INPUT_SIZE/stride for stride-8 P3

    if len(results.boxes) > 0:
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            cls_id = int(cls.item())

            # Box coords are already in the same 640x640 space as feat,
            # so dividing by stride maps directly onto the P3 grid --
            # no separate rescale factor is needed.
            x1, y1, x2, y2 = (box / stride).int().tolist()
            x1 = min(max(x1, 0), feat.shape[3] - 1)
            x2 = min(max(x2, 0), feat.shape[3])
            y1 = min(max(y1, 0), feat.shape[2] - 1)
            y2 = min(max(y2, 0), feat.shape[2])

            # Guard against a degenerate (zero-area) box after clamping
            if x2 <= x1 or y2 <= y1:
                continue

            # --- FIX 2: sample every interior grid cell of the box, not
            # just its center, so calibration matches what the detector
            # scores at inference time (whole box, including edges).
            for gy in range(y1, y2):
                for gx in range(x1, x2):
                    per_class_features[cls_id].append(feat[0, :, gy, gx].cpu())

    n_images += 1
    torch.cuda.empty_cache()

print(f"Processed {n_images} clean images.")

# Compute Mean and Inverse Covariance per class
stats = {}
for cls_id, feats in per_class_features.items():
    if len(feats) > min_samples:  # Require minimum sample size
        feats_tensor = torch.stack(feats)
        mean = feats_tensor.mean(dim=0)

        # Add a small identity matrix to prevent singular matrix errors during inversion
        cov = torch.cov(feats_tensor.T) + 1e-6 * torch.eye(feats_tensor.shape[1])
        inv_cov = torch.linalg.inv(cov)

        stats[cls_id] = {'mean': mean, 'inv_cov': inv_cov}
        print(f"  Class {cls_id}: {len(feats)} samples")
    else:
        print(f"  Skipping class {cls_id}: only {len(feats)} samples (< {min_samples})")

# Persist the calibration artifact
with open(profile_path, 'wb') as f:
    pickle.dump(stats, f)

print(f"\nCalibration complete. Stats for {len(stats)} classes saved to {profile_path}")
print("Remember: re-run cls_head_detector.py against a FRESH clean-image "
      "validation set afterward to re-derive `threshold` statistically -- "
      "see the earlier discussion on percentile / mean+k*std thresholding.")