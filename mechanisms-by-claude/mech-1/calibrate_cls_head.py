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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(config["model_path"]).to(device)
adapter = YOLOv8ClsHeadAdapter(model.model)

per_class_features = defaultdict(list)

print("Starting offline calibration on clean dataset...")
for img_file in os.listdir(DATA_DIR):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    img_path = str(DATA_DIR / img_file)
    
    # Run standard prediction to get clean class locations
    results = model(img_path, verbose=False)[0]
    
    # Resize image to 640x640 so spatial dimensions are strictly divisible by 32
    resized_img = cv2.resize(results.orig_img, (640, 640))
    img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    activations = adapter.get_activations(img_tensor)
    feat = activations["P3"] 
    
    # Map detection boxes to the P3 grid
    if len(results.boxes) > 0:
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            cls_id = int(cls.item())
            
            # Map center of the bounding box to the P3 grid resolution
            center_x = int(((box[0] + box[2]) / 2) // stride)
            center_y = int(((box[1] + box[3]) / 2) // stride)
            
            # Ensure grid indices are within bounds
            center_x = min(max(center_x, 0), feat.shape[3] - 1)
            center_y = min(max(center_y, 0), feat.shape[2] - 1)
            
            per_class_features[cls_id].append(feat[0, :, center_y, center_x].cpu())
            
    torch.cuda.empty_cache()

# Compute Mean and Inverse Covariance per class
stats = {}
for cls_id, feats in per_class_features.items():
    if len(feats) > min_samples: # Require minimum sample size
        feats_tensor = torch.stack(feats)
        mean = feats_tensor.mean(dim=0)
        
        # Add a small identity matrix to prevent singular matrix errors during inversion
        cov = torch.cov(feats_tensor.T) + 1e-6 * torch.eye(feats_tensor.shape[1])
        inv_cov = torch.linalg.inv(cov)
        
        stats[cls_id] = {'mean': mean, 'inv_cov': inv_cov}

# Persist the calibration artifact
with open(profile_path, 'wb') as f:
    pickle.dump(stats, f)

print(f"Calibration complete. Stats for {len(stats)} classes saved to {profile_path}")