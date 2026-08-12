import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from art.estimators.object_detection import PyTorchYolo
from art.attacks.evasion import AdversarialPatchPyTorch

# 1. Paths & Directory Setup
DATA_DIR = Path(r"D:\GitHub\experiment_data\bdd100k_images_10k\train")
OUTPUT_DIR = Path(r"D:\GitHub\yr-4-research\mechanisms-by-claude\gen-adv-patch\bdd100k_patched")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load and Prepare YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")
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
train_images, filenames = load_and_preprocess_batch(DATA_DIR, max_images=16)

# 6. Configure and Execute Adversarial Patch Attack (EoT)
print("Configuring AdversarialPatchPyTorch attack...")
attack = AdversarialPatchPyTorch(
    estimator=detector,
    rotation_max=22.5,        # EoT rotation jitter in degrees
    scale_min=0.2,            # EoT scale jitter minimum
    scale_max=0.4,            # EoT scale jitter maximum
    learning_rate=0.01,
    max_iter=500,             # Optimization iterations
    batch_size=4,
    patch_shape=(3, 150, 150),# Patch dimensions in pixels (CHW)
    targeted=False            # False = general disruption/suppression
)

print("Optimizing adversarial patch over batch (this may take a few minutes)...")
patch, patch_mask = attack.generate(x=train_images)

# Save raw optimized patch array
np.save(OUTPUT_DIR / "art_patch.npy", patch)

# 7. Apply Patch to Test Image with scale parameter
test_img_tensor = train_images[0:1]
patched_batch = attack.apply_patch(x=test_img_tensor, scale=0.3, patch_external=patch)

# 8. Convert Back to BGR and Save Image
patched_img_np = np.transpose(patched_batch[0], (1, 2, 0)) * 255.0
patched_img_bgr = cv2.cvtColor(patched_img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

out_file_path = OUTPUT_DIR / f"art_patched_{filenames[0]}"
cv2.imwrite(str(out_file_path), patched_img_bgr)

print("\nPatch generation complete!")
print(f"Patched test image saved to: {out_file_path}")