"""
Adversarial patch generator for YOLOv8 — spatially-targeted version.

WHY THIS LOOKS DIFFERENT FROM THE PREVIOUS SCRIPT:
The previous version's loss ("mean of max class-confidence across ALL
anchors in the image") was a classifier-style, spatially-unaware
objective -- it had no notion of WHERE the real object is, so the
optimizer was free to satisfy it by hallucinating new confident
detections anywhere in the image, rather than suppressing the real
object specifically. That's exactly the behavior you observed
(spurious persons/umbrellas appearing, real object confidence barely
moving).

This version fixes that by building a proper detector-aware objective:
  1. Run the CLEAN (unpatched) image through the model once, with no
     gradient, and record which anchors currently fire confidently for
     a real object. These are pseudo-labels -- no ground-truth
     annotations needed, since we're using the model's own predictions.
  2. When optimizing the patch, only look at confidence AT THOSE
     SPECIFIC ANCHORS, and directly minimize it. This is the same idea
     as targeting an "objectness" score in classic anchor-based YOLO
     (suppress confidence at the real object's location specifically),
     adapted for YOLOv8's anchor-free head, which has no separate
     objectness branch -- so the class confidence itself is the
     quantity to suppress.

This does NOT use ART's AdversarialPatchPyTorch attack class -- it's a
direct, transparent training loop, so there's no ambiguity about what
loss is being optimized or which direction ART pushes it. EoT
(random rotation/scale/position each step) is still applied manually,
so the resulting patch is still robust to placement variation.
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from ultralytics import YOLO

# 1. Load Configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATA_DIR = Path(config["clean_dir"])
OUTPUT_DIR = Path(config["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_DIR = Path(config["output_dir"])
PATCH_DIR.mkdir(parents=True, exist_ok=True)
NEW_PATCH_NAME = "art_patch_v6.npy"
STANDALONE_PATCH_NAME = "art_patch_standalone_v6.png"

PS = config["patch_settings"]
INPUT_SIZE = 640
PATCH_SIZE = PS.get("patch_size", 150)
CONF_THRESH = PS.get("conf_thresh", 0.25)
NUM_STEPS = PS.get("num_steps", 500)
GPU_BATCH_SIZE = PS.get("gpu_batch_size", 4)
LEARNING_RATE = PS.get("learning_rate", 0.02)
ROTATION_MAX = PS.get("rotation_max", 22.5)
SCALE_MIN = PS.get("scale_min", 0.2)
SCALE_MAX = PS.get("scale_max", 0.4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Load YOLOv8 and freeze it -- only the patch gets gradients
yolo_model = YOLO(config["model_path"]).to(device)
yolo_model.model.eval()
for p in yolo_model.model.parameters():
    p.requires_grad_(False)


def get_raw_preds(model, x):
    """Raw YOLOv8 forward pass -> decoded per-anchor predictions.
    Shape [B, num_anchors, 4 + num_classes]: columns 0:4 are the
    decoded box (cx, cy, w, h) in pixel space, columns 4: are the
    per-class sigmoid confidences (no separate objectness channel --
    YOLOv8's anchor-free head doesn't have one)."""
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("one2many", list(out.values())[0])
        if isinstance(out, tuple):
            out = out[0]
    return out.transpose(1, 2)


def select_target_object(preds, conf_thresh=CONF_THRESH, target_class=None):
    """From a CLEAN (unpatched) forward pass, pick ONE object to attack:
    the highest-confidence detection of `target_class` if specified
    (e.g. COCO class id 2 = 'car'), otherwise simply the single
    highest-confidence detection in the image. This is the
    pseudo-labeling step -- no ground-truth annotations needed, we
    trust the clean model's own confident predictions.

    Returns (anchor_mask, (cx_norm, cy_norm), box) or None if nothing
    suitable is found in this image:
      - anchor_mask: boolean [A] -- every anchor whose OWN predicted
        box center falls inside the chosen object's box extent (not
        just the single best anchor -- there are usually several
        overlapping anchors around a real object, and we want the
        patch to suppress all of them).
      - (cx_norm, cy_norm): the object's box center in normalized
        [-1, 1] canvas coordinates -- this is what tells the patch
        WHERE to place itself, both during training and at
        application time.
      - box: (x1, y1, x2, y2) in pixel space, for reference.
    """
    boxes_cxcywh = preds[0, :, 0:4]          # [A, 4]  (cx, cy, w, h), pixel space
    class_probs = preds[0, :, 4:]            # [A, nc]
    max_probs, max_classes = class_probs.max(dim=-1)

    candidate_mask = max_probs > conf_thresh
    if target_class is not None:
        candidate_mask = candidate_mask & (max_classes == target_class)

    if candidate_mask.sum() == 0:
        return None

    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    best_idx = candidate_indices[max_probs[candidate_indices].argmax()]
    bx, by, bw, bh = boxes_cxcywh[best_idx].tolist()
    x1, y1, x2, y2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2

    ax, ay = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1]
    anchor_mask = (ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2) & (max_probs > conf_thresh)

    cx_norm = (bx / INPUT_SIZE) * 2 - 1
    cy_norm = (by / INPUT_SIZE) * 2 - 1
    return anchor_mask, (cx_norm, cy_norm), (x1, y1, x2, y2)


def build_patch_overlay(patch, canvas_hw, cx, cy, angle_deg, scale):
    """Differentiably places `patch` (C,ph,pw) onto a (C,H,W) canvas via
    an affine warp -- rotated by angle_deg, scaled by `scale` (relative
    to canvas size), centered at normalized coords (cx, cy) in [-1, 1].
    Returns (warped_patch, warped_mask), both shaped for direct
    compositing onto the target image. Differentiable w.r.t. `patch`,
    so gradients flow back into the patch's pixels during training."""
    C, ph, pw = patch.shape
    H, W = canvas_hw
    theta = torch.deg2rad(torch.tensor(float(angle_deg), device=patch.device))
    cos, sin = torch.cos(theta), torch.sin(theta)

    # grid_sample convention: this matrix maps OUTPUT (canvas) coords
    # to INPUT (patch) coords, so dividing by `scale` here makes the
    # patch appear LARGER on the canvas as `scale` increases.
    affine = torch.stack([
        torch.stack([cos / scale, -sin / scale, torch.tensor(cx, device=patch.device)]),
        torch.stack([sin / scale,  cos / scale, torch.tensor(cy, device=patch.device)]),
    ]).unsqueeze(0).float()

    grid = F.affine_grid(affine, size=(1, C, H, W), align_corners=False)
    warped_patch = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False, padding_mode="zeros")
    mask_src = torch.ones((1, 1, ph, pw), device=patch.device)
    warped_mask = F.grid_sample(mask_src, grid, align_corners=False, padding_mode="zeros")
    return warped_patch.squeeze(0), warped_mask.squeeze(0)


def discover_image_files(folder_path, max_images):
    """Lists candidate image filenames WITHOUT loading any pixel data --
    this is what keeps memory flat regardless of dataset size. Actual
    images are read from disk and moved to GPU one small batch at a
    time during training (see load_single_image below), then released
    once that step is done."""
    return [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))][:max_images]


def load_single_image(path, target_size=INPUT_SIZE):
    """Loads ONE image from disk straight to a GPU tensor, on demand.
    Called fresh each time an image is sampled for a training step --
    a few thousand small JPEG reads over a full training run is cheap
    on disk I/O, and it means the dataset never sits in GPU (or CPU)
    memory all at once."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)


print("Indexing training images (not loading pixel data yet)...")
train_files = discover_image_files(DATA_DIR, PS.get("max_images", 100))
print(f"Found {len(train_files)} candidate images. "
      f"{GPU_BATCH_SIZE} will be loaded to GPU at a time during training.")

# 3. Initialize the patch (random noise, trainable)
patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)
optimizer = torch.optim.Adam([patch], lr=LEARNING_RATE)

print("Optimizing adversarial patch (spatially-targeted, detector-aware objective)...")
for step in range(NUM_STEPS):
    optimizer.zero_grad()

    batch_idx = np.random.choice(len(train_files), size=min(GPU_BATCH_SIZE, len(train_files)), replace=False)
    total_loss = 0.0
    valid = 0

    for idx in batch_idx:
        # Load THIS ONE image from disk to GPU now -- not before, not
        # for the whole batch at once, and it's discarded (no lingering
        # reference) once this iteration of the loop ends.
        image = load_single_image(DATA_DIR / train_files[idx])
        if image is None:
            continue

        # --- Step 1: pseudo-label the target object from the CLEAN pass ---
        with torch.no_grad():
            clean_preds = get_raw_preds(yolo_model.model, image.unsqueeze(0))
            result = select_target_object(clean_preds, target_class=PS.get("target_class"))
        del clean_preds  # no-grad tensor, not needed past this point -- free it now

        if result is None:
            continue  # no matching object confidently detected in this image -- skip it this step
        target_mask, (obj_cx, obj_cy), _ = result

        # --- Step 2: random EoT transform, JITTERED AROUND THE OBJECT'S
        # OWN LOCATION -- not sampled uniformly across the whole canvas.
        # This is what actually ties the patch's learned behavior to
        # "when placed on this object, suppress it", matching how the
        # patch will be placed/tested afterward. A physical attacker
        # sticking a patch on a vehicle wouldn't land pixel-perfect
        # either, so some jitter is realistic, not just a training trick.
        angle = np.random.uniform(-ROTATION_MAX, ROTATION_MAX)
        scale = np.random.uniform(SCALE_MIN, SCALE_MAX)
        jitter = PS.get("placement_jitter", 0.1)
        cx = float(np.clip(obj_cx + np.random.uniform(-jitter, jitter), -0.9, 0.9))
        cy = float(np.clip(obj_cy + np.random.uniform(-jitter, jitter), -0.9, 0.9))

        warped_patch, warped_mask = build_patch_overlay(
            torch.clamp(patch, 0, 1), image.shape[1:], cx, cy, angle, scale
        )
        patched_image = image * (1 - warped_mask) + warped_patch * warped_mask

        # --- Step 3: patched forward pass, loss ONLY at the pseudo-labeled anchors ---
        patched_preds = get_raw_preds(yolo_model.model, patched_image.unsqueeze(0))
        max_probs, _ = patched_preds[0, :, 4:].max(dim=-1)
        targeted_conf = max_probs[target_mask]

        loss = targeted_conf.mean()  # directly minimize confidence at the real object's anchors
        total_loss = total_loss + loss
        valid += 1

    if valid == 0:
        continue

    total_loss = total_loss / valid
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        patch.clamp_(0, 1)

    # The forward/backward graph for this step is done (backward() has
    # already freed it) -- explicitly release any cached GPU memory
    # before the next step loads its batch, rather than relying on the
    # allocator to do it lazily. Cheap relative to a training step, and
    # worth the safety margin on a 4GB-class card.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if step % 25 == 0 or step == NUM_STEPS - 1:
        print(f"  step {step:4d}/{NUM_STEPS}  mean targeted confidence: {total_loss.item():.4f}")

# 4. Save the raw optimized patch (.npy, for apply_saved_patch.py / ART-style
# reuse) AND as a standalone image file, so you can paste it manually
# onto any image in an editor, or hand it to someone without a Python
# dependency to open it.
np.save(PATCH_DIR / NEW_PATCH_NAME, patch.detach().cpu().numpy())
print(f"\nSaved optimized patch (array) to {PATCH_DIR / NEW_PATCH_NAME}")

patch_img_np = (patch.detach().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
patch_img_bgr = cv2.cvtColor(patch_img_np, cv2.COLOR_RGB2BGR)
patch_image_path = PATCH_DIR / STANDALONE_PATCH_NAME
cv2.imwrite(str(patch_image_path), patch_img_bgr)
print(f"Saved standalone patch image to {patch_image_path}")

# 5. Apply the trained patch to a demo image, placed ON the target
# object's own location for THAT image -- not a hardcoded canvas
# position. `specific_clean_image`, if set, controls this demo image;
# otherwise fall back to the first file in the discovered training set
# (loaded fresh from disk here too, same as during training).
if config["specific_clean_image"]:
    demo_filename = config["specific_clean_image"]
    test_image = load_single_image(DATA_DIR / demo_filename)
    if test_image is None:
        print(f"Warning: could not load specific_clean_image "
              f"'{demo_filename}' from {DATA_DIR} -- falling back to "
              f"the first training image for the demo instead.")
        demo_filename = train_files[0]
        test_image = load_single_image(DATA_DIR / demo_filename)
else:
    demo_filename = train_files[0]
    test_image = load_single_image(DATA_DIR / demo_filename)

with torch.no_grad():
    demo_preds = get_raw_preds(yolo_model.model, test_image.unsqueeze(0))
    demo_result = select_target_object(demo_preds, target_class=PS.get("target_class"))

if demo_result is not None:
    _, (demo_cx, demo_cy), _ = demo_result
else:
    print("Warning: no target object detected in the demo test image -- "
          "falling back to center placement, but pick a different demo "
          "image (or a different target_class) for a meaningful demo.")
    demo_cx, demo_cy = 0.0, 0.0

warped_patch, warped_mask = build_patch_overlay(
    patch.detach(), test_image.shape[1:], cx=demo_cx, cy=demo_cy, angle_deg=0.0,
    scale=PS.get("patch_scale", 0.3)
)
patched_test = test_image * (1 - warped_mask) + warped_patch * warped_mask

out_np = (patched_test.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
out_path = OUTPUT_DIR / f"art_patched_v2_{demo_filename}"
cv2.imwrite(str(out_path), out_bgr)
print(f"Patched test image saved to: {out_path}")
print("\nNext: run verify_patch_efficacy.py to check whether the real "
      "object's detection is actually suppressed now, rather than "
      "spurious detections appearing elsewhere.")