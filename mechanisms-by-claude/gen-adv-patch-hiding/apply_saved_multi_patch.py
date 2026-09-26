"""
Apply a pre-optimized patch to any image -- rewritten to match
generate_art_patch.py's own compositing function (build_patch_overlay)
instead of ART's attack.apply_patch(). That mismatch was the likely
cause of weak results when testing the patch outside the training
script: the patch was OPTIMIZED under one geometric convention
(rotation/scale/position handled by build_patch_overlay), so applying
it through a different function's conventions -- or by hand in an
image editor, at a size/position outside what it was trained for --
silently changes its effective potency, even when it "looks right".

PLACEMENT: three modes, controlled by `apply_settings.placement_mode`
in config.json:
  - "auto" (default, easiest): re-detects the target object (same
    target_class as training) on THIS image and centers the patch on
    it automatically -- no manual coordinates needed.
  - "manual": you specify exact pixel coordinates (in the 640x640
    working space) yourself, via `manual_x` / `manual_y`. Useful for
    testing a specific spot, or when the object isn't confidently
    detected in the clean image to auto-target.
  - "multi": places a SEPARATE copy of the patch on several DISTINCT
    detected objects in the same image, each small, with an explicit
    minimum-separation check between them. This is NOT the same as
    calling "auto" placement repeatedly -- "auto" always returns the
    single highest-confidence detection, so looping it would keep
    re-placing on the SAME object every time, stacking copies on top
    of each other into one unnaturally large blob. "multi" instead
    finds several distinct objects up front, from ONE detection pass
    on the clean image, and rejects any candidate too close to one
    already selected.
"""

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
PATCH_DIR = Path(config["patch_dir"])
PATCH_PATH = PATCH_DIR / config["specific_patch_name"]
INPUT_SIZE = 640

AS = config.get("apply_settings", {})
PLACEMENT_MODE = AS.get("placement_mode", "auto")
MANUAL_X = AS.get("manual_x", INPUT_SIZE // 2)
MANUAL_Y = AS.get("manual_y", INPUT_SIZE // 2)
APPLY_SCALE = AS.get("scale", config.get("patch_settings", {}).get("patch_scale", 0.3))
APPLY_ROTATION = AS.get("rotation", 0.0)
TARGET_CLASS = AS.get("target_class", config.get("patch_settings", {}).get("target_class"))
CONF_THRESH = AS.get("conf_thresh", config.get("patch_settings", {}).get("conf_thresh", 0.25))
MULTI_MAX_PATCHES = AS.get("multi_max_patches", 4)
# Minimum distance (normalized [-1,1] units) between two selected
# objects' centers before a candidate is rejected as "too close" to
# one already chosen. This is what prevents overlap/stacking -- as a
# starting heuristic, keep this at least ~2x APPLY_SCALE, since two
# patches placed closer than roughly the sum of their half-widths will
# visually overlap regardless of how different their host objects are.
MULTI_MIN_SEPARATION = AS.get("multi_min_separation", 0.3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not PATCH_PATH.exists():
    raise FileNotFoundError(f"Could not find saved patch at {PATCH_PATH}. Run generate_art_patch.py first.")

print(f"Loading pre-optimized patch from: {PATCH_PATH}")
patch_np = np.load(PATCH_PATH)                      # [3, ph, pw], float in [0,1]
patch = torch.from_numpy(patch_np).float().to(device)

# 2. Load YOLOv8 model (frozen -- we're not training here, just running inference)
yolo_model = YOLO(config["model_path"]).to(device)
yolo_model.model.eval()
for p in yolo_model.model.parameters():
    p.requires_grad_(False)


def get_raw_preds(model, x):
    """Same parsing as generate_art_patch.py -- raw YOLOv8 forward pass
    to decoded per-anchor predictions [B, A, 4+nc]."""
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("one2many", list(out.values())[0])
        if isinstance(out, tuple):
            out = out[0]
    return out.transpose(1, 2)


def select_target_object(preds, conf_thresh=CONF_THRESH, target_class=None):
    """Identical logic to generate_art_patch.py's version -- finds the
    single highest-confidence detection (optionally restricted to
    `target_class`) and returns its center in normalized [-1,1]
    coordinates. Returns None if nothing suitable is found."""
    boxes_cxcywh = preds[0, :, 0:4]
    class_probs = preds[0, :, 4:]
    max_probs, max_classes = class_probs.max(dim=-1)

    candidate_mask = max_probs > conf_thresh
    if target_class is not None:
        candidate_mask = candidate_mask & (max_classes == target_class)

    if candidate_mask.sum() == 0:
        return None

    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    best_idx = candidate_indices[max_probs[candidate_indices].argmax()]
    bx, by, bw, bh = boxes_cxcywh[best_idx].tolist()

    cx_norm = (bx / INPUT_SIZE) * 2 - 1
    cy_norm = (by / INPUT_SIZE) * 2 - 1
    return (cx_norm, cy_norm), (bx, by, bw, bh)


def select_all_target_objects(preds, conf_thresh=CONF_THRESH, target_class=None,
                               max_objects=MULTI_MAX_PATCHES, min_separation=MULTI_MIN_SEPARATION):
    """Finds UP TO max_objects DISTINCT confident detections (optionally
    restricted to target_class), from ONE detection pass -- not by
    calling select_target_object repeatedly. Candidates are considered
    in descending confidence order, and any candidate whose center is
    within min_separation (normalized units) of an already-selected
    object is skipped. This explicit separation check is what prevents
    two patches from landing on/near each other and merging into one
    unnaturally large blob."""
    boxes_cxcywh = preds[0, :, 0:4]
    class_probs = preds[0, :, 4:]
    max_probs, max_classes = class_probs.max(dim=-1)

    candidate_mask = max_probs > conf_thresh
    if target_class is not None:
        candidate_mask = candidate_mask & (max_classes == target_class)

    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    if len(candidate_indices) == 0:
        return []

    order = max_probs[candidate_indices].argsort(descending=True)
    sorted_indices = candidate_indices[order]

    selected = []
    for idx in sorted_indices:
        bx, by, bw, bh = boxes_cxcywh[idx].tolist()
        cx_norm = (bx / INPUT_SIZE) * 2 - 1
        cy_norm = (by / INPUT_SIZE) * 2 - 1

        too_close = any(
            ((cx_norm - s["cx"]) ** 2 + (cy_norm - s["cy"]) ** 2) ** 0.5 < min_separation
            for s in selected
        )
        if too_close:
            continue

        selected.append({"cx": cx_norm, "cy": cy_norm, "box": (bx, by, bw, bh)})
        if len(selected) >= max_objects:
            break

    return selected


def build_patch_overlay(patch, canvas_hw, cx, cy, angle_deg, scale):
    """FIXED: the translation terms now correctly divide by `scale` and
    combine with rotation, so the patch actually renders centered at
    canvas position (cx, cy) with half-width `scale` (canvas-normalized
    units) -- as it was always meant to. The previous version used cx/cy
    directly as the translation, which put the ACTUAL rendered center at
    roughly -scale*cx instead of cx: for small scale values (like the
    0.1-0.2 range trained here), every placement collapsed toward the
    canvas center regardless of how far apart the intended positions
    were -- this is what caused multiple patches to overlap even though
    select_all_target_objects had already picked well-separated objects.
    """
    C, ph, pw = patch.shape
    H, W = canvas_hw
    theta_rad = torch.deg2rad(torch.tensor(float(angle_deg), device=patch.device))
    cos, sin = torch.cos(theta_rad), torch.sin(theta_rad)
    cx_t = torch.tensor(float(cx), device=patch.device)
    cy_t = torch.tensor(float(cy), device=patch.device)

    tx = -(cos * cx_t + sin * cy_t) / scale
    ty = (sin * cx_t - cos * cy_t) / scale

    affine = torch.stack([
        torch.stack([cos / scale, sin / scale, tx]),
        torch.stack([-sin / scale, cos / scale, ty]),
    ]).unsqueeze(0).float()

    grid = F.affine_grid(affine, size=(1, C, H, W), align_corners=False)
    warped_patch = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False, padding_mode="zeros")
    mask_src = torch.ones((1, 1, ph, pw), device=patch.device)
    warped_mask = F.grid_sample(mask_src, grid, align_corners=False, padding_mode="zeros")
    return warped_patch.squeeze(0), warped_mask.squeeze(0)


def load_single_image(path, target_size=INPUT_SIZE):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)


# 3. Resolve target image
if config["specific_clean_image"]:
    target_filename = config["specific_clean_image"]
else:
    image_files = [f for f in DATA_DIR.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not image_files:
        raise FileNotFoundError(f"No valid image files found in {DATA_DIR}")
    target_filename = image_files[0].name

print(f"Processing target image: {target_filename}")
image = load_single_image(DATA_DIR / target_filename)
if image is None:
    raise FileNotFoundError(f"Could not read image: {DATA_DIR / target_filename}")

# 4. Determine placement and apply
if PLACEMENT_MODE == "manual":
    cx = (MANUAL_X / INPUT_SIZE) * 2 - 1
    cy = (MANUAL_Y / INPUT_SIZE) * 2 - 1
    print(f"Manual placement: pixel ({MANUAL_X}, {MANUAL_Y}) -> normalized ({cx:.3f}, {cy:.3f})")

    warped_patch, warped_mask = build_patch_overlay(
        patch, image.shape[1:], cx=cx, cy=cy, angle_deg=APPLY_ROTATION, scale=APPLY_SCALE
    )
    patched_image = image * (1 - warped_mask) + warped_patch * warped_mask

elif PLACEMENT_MODE == "multi":
    # Detect ALL target objects ONCE, on the clean image -- never
    # re-detect after each placement, which would risk an
    # already-placed patch interfering with detections of nearby
    # objects and is unnecessary anyway since every location is fixed
    # up front from this single pass.
    with torch.no_grad():
        preds = get_raw_preds(yolo_model.model, image.unsqueeze(0))
        targets = select_all_target_objects(preds, target_class=TARGET_CLASS)

    if not targets:
        print(f"Warning: no confident detections of target_class={TARGET_CLASS} found -- "
              f"no patches placed. Consider lowering conf_thresh or checking target_class.")
        patched_image = image.clone()
    else:
        print(f"Found {len(targets)} distinct target object(s) for patch placement "
              f"(requested up to {MULTI_MAX_PATCHES}, min separation {MULTI_MIN_SEPARATION}):")
        patched_image = image.clone()
        for i, t in enumerate(targets):
            bx, by, bw, bh = t["box"]
            print(f"  [{i+1}] pixel center ({bx:.0f}, {by:.0f}), box size ({bw:.0f}x{bh:.0f})")
            warped_patch, warped_mask = build_patch_overlay(
                patch, patched_image.shape[1:], cx=t["cx"], cy=t["cy"],
                angle_deg=APPLY_ROTATION, scale=APPLY_SCALE,
            )
            # Composite onto the RUNNING image, so each new patch is
            # drawn on top of previous ones -- harmless here since
            # min_separation already keeps them apart, but this is also
            # why sequential compositing (not a single combined mask)
            # is used: it stays correct even if separation is set too
            # tight and two patches end up slightly touching.
            patched_image = patched_image * (1 - warped_mask) + warped_patch * warped_mask

else:  # "auto"
    with torch.no_grad():
        preds = get_raw_preds(yolo_model.model, image.unsqueeze(0))
        result = select_target_object(preds, target_class=TARGET_CLASS)

    if result is None:
        print(f"Warning: no confident detection of target_class={TARGET_CLASS} found in "
              f"this image -- falling back to image center. Consider switching to "
              f"\"placement_mode\": \"manual\" with explicit manual_x/manual_y for this image.")
        cx, cy = 0.0, 0.0
    else:
        (cx, cy), (bx, by, bw, bh) = result
        print(f"Auto-detected target at pixel center ({bx:.0f}, {by:.0f}), "
              f"box size ({bw:.0f}x{bh:.0f}) -> normalized ({cx:.3f}, {cy:.3f})")

    warped_patch, warped_mask = build_patch_overlay(
        patch, image.shape[1:], cx=cx, cy=cy, angle_deg=APPLY_ROTATION, scale=APPLY_SCALE
    )
    patched_image = image * (1 - warped_mask) + warped_patch * warped_mask

# 6. Save output
out_np = (patched_image.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
out_file_path = OUTPUT_DIR / f"20-multi-patched_{target_filename}"
cv2.imwrite(str(out_file_path), out_bgr)

print(f"Successfully applied patch! Saved image to: {out_file_path}")