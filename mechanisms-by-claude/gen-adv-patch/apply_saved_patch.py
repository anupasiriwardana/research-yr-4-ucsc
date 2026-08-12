import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from art.estimators.object_detection import PyTorchYolo
from art.attacks.evasion import AdversarialPatchPyTorch

# 1. Load Configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATA_DIR = Path(config["clean_dir"])
OUTPUT_DIR = Path(config["output_dir"])
PATCH_PATH = OUTPUT_DIR / config["specific_patch_name"]

if not PATCH_PATH.exists():
    raise FileNotFoundError(f"Could not find saved patch at {PATCH_PATH}.")

print(f"Loading pre-optimized patch from: {PATCH_PATH}")
patch = np.load(PATCH_PATH)

# 2. Load YOLOv8 Model & Setup Wrapper
yolo_model = YOLO(config["model_path"])
yolo_model.model.eval()

class ART_YOLOv8_Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x, y=None, *args, **kwargs):
        self.model.eval()
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict):
            if 'one2many' in out:
                out = out['one2many']
            else:
                out = list(out.values())[0]
                if isinstance(out, tuple): 
                    out = out[0]
        preds = out.transpose(1, 2)
        if y is None:
            return preds
        class_probs = preds[:, :, 4:] 
        max_class_probs, _ = torch.max(class_probs, dim=-1)
        return {"loss_total": torch.mean(max_class_probs)}

wrapped_model = ART_YOLOv8_Wrapper(yolo_model.model)

detector = PyTorchYolo(
    model=wrapped_model,
    input_shape=(3, 640, 640),
    clip_values=(0.0, 1.0),
    channels_first=True,
    attack_losses=("loss_total",)
)

attack = AdversarialPatchPyTorch(
    estimator=detector,
    rotation_max=22.5,
    scale_min=0.2,
    scale_max=0.4,
    learning_rate=0.01,
    max_iter=config["patch_settings"]["max_iter"],
    batch_size=config["patch_settings"]["batch_size"],
    patch_shape=(3, 150, 150),
    targeted=False
)

# 3. Resolve Target Image
if config["specific_clean_image"]:
    target_filename = config["specific_clean_image"]
else:
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError(f"No valid image files found in {DATA_DIR}")
    target_filename = image_files[0]

target_img_path = str(DATA_DIR / target_filename)

print(f"Processing target image: {target_filename}")
img = cv2.imread(target_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))

img_tensor = img_resized.astype(np.float32) / 255.0
img_tensor = np.transpose(img_tensor, (2, 0, 1))
test_batch = np.array([img_tensor], dtype=np.float32)

# 4. Apply Saved Patch & Save Output
patched_batch = attack.apply_patch(x=test_batch, scale=config["patch_settings"]["patch_scale"], patch_external=patch)

patched_img_np = np.transpose(patched_batch[0], (1, 2, 0)) * 255.0
patched_img_bgr = cv2.cvtColor(patched_img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

out_file_path = OUTPUT_DIR / f"applied_art_patched_{target_filename}"
cv2.imwrite(str(out_file_path), patched_img_bgr)

print(f"Successfully applied patch! Saved image to: {out_file_path}")