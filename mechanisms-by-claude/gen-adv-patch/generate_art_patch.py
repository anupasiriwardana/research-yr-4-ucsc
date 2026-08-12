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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NEW_PATCH_NAME = "art_patch.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load and Prepare YOLOv8 Model
yolo_model = YOLO(config["model_path"])
yolo_model.model.eval()

# 3. Model Wrapper for ART Compatibility
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
        proxy_loss = torch.mean(max_class_probs)
        
        return {"loss_total": proxy_loss}

wrapped_model = ART_YOLOv8_Wrapper(yolo_model.model)

# 4. Wrap Model in ART PyTorchYolo Estimator
detector = PyTorchYolo(
    model=wrapped_model,
    input_shape=(3, 640, 640),
    clip_values=(0.0, 1.0),
    channels_first=True,
    attack_losses=("loss_total",)
)

# 5. Image Preprocessing Helper
def load_and_preprocess_batch(folder_path, max_images=16, target_size=(640, 640)):
    batch = []
    
    # Priority: Check if a specific clean image is defined in config
    if config["specific_clean_image"]:
        files = [config["specific_clean_image"]]
    else:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_images]
    
    for f in files:
        img_path = str(folder_path / f)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)
        
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        batch.append(img_tensor)
        
    return np.array(batch, dtype=np.float32), files

print("Loading calibration batch for patch optimization...")
train_images, filenames = load_and_preprocess_batch(DATA_DIR, max_images=config["patch_settings"]["max_images"])

# 6. Configure and Execute Adversarial Patch Attack (EoT)
print("Configuring AdversarialPatchPyTorch attack...")
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

print("Optimizing adversarial patch over batch (this may take a few minutes)...")
patch, patch_mask = attack.generate(x=train_images)

# Save raw optimized patch array
np.save(OUTPUT_DIR / NEW_PATCH_NAME, patch)

# 7. Apply Patch to Test Image
test_img_tensor = train_images[0:1]
patched_batch = attack.apply_patch(x=test_img_tensor, scale=config["patch_settings"]["patch_scale"], patch_external=patch)

# 8. Convert Back to BGR and Save Image
patched_img_np = np.transpose(patched_batch[0], (1, 2, 0)) * 255.0
patched_img_bgr = cv2.cvtColor(patched_img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

out_file_path = OUTPUT_DIR / f"art_patched_{filenames[0]}"
cv2.imwrite(str(out_file_path), patched_img_bgr)

print("\nPatch generation complete!")
print(f"Patched test image saved to: {out_file_path}")