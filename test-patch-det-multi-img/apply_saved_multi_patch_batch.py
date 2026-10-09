"""
Apply the saved multi-object patch to a configurable number of images.

This follows apply_saved_multi_patch.py's detection, target-selection, and
corrected affine-compositing logic. For every processed image it writes:

* a patched image in batch_settings.output_dir; and
* a coordinate file in batch_settings.coordinates_dir.

Coordinate files contain one line per rendered patch in the same pixel-box
shape reported by cls_head_detector.py:

    box=(x1, y1, x2, y2)

Coordinates are measured in the 640x640 working image space used by the
detector. A file is still created when no target is found; it is empty in that
case. This makes the output set suitable for later per-image metric scripts.
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


CONFIG_PATH = Path(__file__).parent / "config.json"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    config = json.load(config_file)

INPUT_SIZE = 640
DATA_DIR = Path(config["clean_dir"])
PATCH_DIR = Path(config["patch_dir"])
PATCH_PATH = PATCH_DIR / config["specific_patch_name"]

PATCH_SETTINGS = config.get("patch_settings", {})
APPLY_SETTINGS = config.get("apply_settings", {})
BATCH_SETTINGS = config.get("batch_settings", {})

CONF_THRESH = APPLY_SETTINGS.get(
    "conf_thresh", PATCH_SETTINGS.get("conf_thresh", 0.25)
)
TARGET_CLASS = APPLY_SETTINGS.get(
    "target_class", PATCH_SETTINGS.get("target_class")
)
APPLY_SCALE = APPLY_SETTINGS.get(
    "scale", PATCH_SETTINGS.get("patch_scale", 0.3)
)
APPLY_ROTATION = APPLY_SETTINGS.get("rotation", 0.0)
MULTI_MAX_PATCHES = APPLY_SETTINGS.get("multi_max_patches", 4)
MULTI_MIN_SEPARATION = APPLY_SETTINGS.get("multi_min_separation", 0.3)

NUM_IMAGES = BATCH_SETTINGS.get("num_images", 100)
START_INDEX = BATCH_SETTINGS.get("start_index", 0)
OUTPUT_DIR = Path(BATCH_SETTINGS["output_dir"])
COORDINATES_DIR = Path(BATCH_SETTINGS["coordinates_dir"])
IMAGE_EXTENSIONS = {
    extension.lower() for extension in BATCH_SETTINGS.get(
        "image_extensions", [".jpg", ".jpeg", ".png"]
    )
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_raw_preds(model, image_tensor):
    """Return decoded per-anchor predictions with shape [B, A, 4 + nc]."""
    output = model(image_tensor)
    if isinstance(output, tuple):
        output = output[0]
    elif isinstance(output, dict):
        output = output.get("one2many", list(output.values())[0])
        if isinstance(output, tuple):
            output = output[0]
    return output.transpose(1, 2)


def select_all_target_objects(
    preds,
    conf_thresh=CONF_THRESH,
    target_class=TARGET_CLASS,
    max_objects=MULTI_MAX_PATCHES,
    min_separation=MULTI_MIN_SEPARATION,
):
    """Select distinct targets from one clean forward pass."""
    boxes_cxcywh = preds[0, :, 0:4]
    class_probs = preds[0, :, 4:]
    max_probs, max_classes = class_probs.max(dim=-1)

    candidate_mask = max_probs > conf_thresh
    if target_class is not None:
        candidate_mask = candidate_mask & (max_classes == target_class)

    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    if len(candidate_indices) == 0:
        return []

    sorted_indices = candidate_indices[
        max_probs[candidate_indices].argsort(descending=True)
    ]

    selected = []
    for index in sorted_indices:
        bx, by, bw, bh = boxes_cxcywh[index].tolist()
        cx = (bx / INPUT_SIZE) * 2 - 1
        cy = (by / INPUT_SIZE) * 2 - 1

        too_close = any(
            ((cx - item["cx"]) ** 2 + (cy - item["cy"]) ** 2) ** 0.5
            < min_separation
            for item in selected
        )
        if too_close:
            continue

        selected.append({"cx": cx, "cy": cy, "box": (bx, by, bw, bh)})
        if len(selected) >= max_objects:
            break

    return selected


def build_patch_overlay(patch, canvas_hw, cx, cy, angle_deg, scale):
    """Use the corrected compositor shared by the current multi-patch path."""
    channels, patch_height, patch_width = patch.shape
    height, width = canvas_hw
    theta_rad = torch.deg2rad(
        torch.tensor(float(angle_deg), device=patch.device)
    )
    cosine, sine = torch.cos(theta_rad), torch.sin(theta_rad)
    cx_tensor = torch.tensor(float(cx), device=patch.device)
    cy_tensor = torch.tensor(float(cy), device=patch.device)

    tx = -(cosine * cx_tensor + sine * cy_tensor) / scale
    ty = (sine * cx_tensor - cosine * cy_tensor) / scale
    affine = torch.stack(
        [
            torch.stack([cosine / scale, sine / scale, tx]),
            torch.stack([-sine / scale, cosine / scale, ty]),
        ]
    ).unsqueeze(0).float()

    grid = F.affine_grid(
        affine, size=(1, channels, height, width), align_corners=False
    )
    warped_patch = F.grid_sample(
        patch.unsqueeze(0),
        grid,
        align_corners=False,
        padding_mode="zeros",
    )
    mask_source = torch.ones(
        (1, 1, patch_height, patch_width), device=patch.device
    )
    warped_mask = F.grid_sample(
        mask_source,
        grid,
        align_corners=False,
        padding_mode="zeros",
    )
    return warped_patch.squeeze(0), warped_mask.squeeze(0)


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(
        2, 0, 1
    ).to(device)


def rendered_mask_box(mask):
    """Convert the visible rendered mask to detector-style pixel coordinates."""
    mask_np = mask.squeeze(0).detach().cpu().numpy()
    visible = (mask_np > 0.5).astype(np.uint8)
    ys, xs = np.where(visible)
    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def write_coordinate_file(path, boxes):
    with path.open("w", encoding="utf-8") as coordinate_file:
        for x1, y1, x2, y2 in boxes:
            coordinate_file.write(f"box=({x1}, {y1}, {x2}, {y2})\n")


def discover_images(start_index, num_images):
    candidates = sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = candidates[start_index:start_index + num_images]
    if not selected:
        raise FileNotFoundError(
            f"No images selected from {DATA_DIR} with extensions "
            f"{sorted(IMAGE_EXTENSIONS)} and start_index={start_index}."
        )
    return selected


def process_image(image_path, patch, model):
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    with torch.no_grad():
        predictions = get_raw_preds(model.model, image.unsqueeze(0))
        targets = select_all_target_objects(predictions)

    patched_image = image.clone()
    coordinate_boxes = []
    for target in targets:
        warped_patch, warped_mask = build_patch_overlay(
            patch,
            patched_image.shape[1:],
            cx=target["cx"],
            cy=target["cy"],
            angle_deg=APPLY_ROTATION,
            scale=APPLY_SCALE,
        )
        patched_image = (
            patched_image * (1 - warped_mask) + warped_patch * warped_mask
        )
        box = rendered_mask_box(warped_mask)
        if box is not None:
            coordinate_boxes.append(box)

    output_name = f"multi-patched_{image_path.name}"
    output_path = OUTPUT_DIR / output_name
    output_array = (
        patched_image.permute(1, 2, 0).cpu().numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR),
    )

    coordinate_path = COORDINATES_DIR / f"{image_path.stem}.txt"
    write_coordinate_file(coordinate_path, coordinate_boxes)
    return output_path, coordinate_path, coordinate_boxes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply the saved multi-object patch to multiple images."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=NUM_IMAGES,
        help=f"Override batch_settings.num_images (default: {NUM_IMAGES}).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=START_INDEX,
        help=f"Override batch_settings.start_index (default: {START_INDEX}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("batch_settings.num_images must be greater than zero.")
    if args.start_index < 0:
        raise ValueError("batch_settings.start_index cannot be negative.")
    if APPLY_SCALE <= 0:
        raise ValueError("apply_settings.scale must be greater than zero.")
    if MULTI_MAX_PATCHES <= 0:
        raise ValueError("apply_settings.multi_max_patches must be greater than zero.")
    if not PATCH_PATH.exists():
        raise FileNotFoundError(
            f"Could not find saved patch at {PATCH_PATH}. "
            "Update specific_patch_name or generate the patch first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COORDINATES_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = discover_images(args.start_index, args.num_images)

    print(f"Loading patch from: {PATCH_PATH}")
    patch = torch.from_numpy(np.load(PATCH_PATH)).float().to(device)
    if patch.ndim != 3 or patch.shape[0] != 3:
        raise ValueError(
            f"Expected patch array with shape [3, height, width], got {patch.shape}."
        )

    print(f"Loading model: {config['model_path']}")
    model = YOLO(config["model_path"]).to(device)
    model.model.eval()
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)

    print(
        f"Processing {len(image_paths)} image(s), target_class={TARGET_CLASS}, "
        f"max_patches={MULTI_MAX_PATCHES}, device={device}"
    )
    failures = []
    for position, image_path in enumerate(image_paths, start=1):
        try:
            output_path, coordinate_path, boxes = process_image(
                image_path, patch, model
            )
            print(
                f"[{position}/{len(image_paths)}] {image_path.name}: "
                f"{len(boxes)} patch(es), output={output_path.name}, "
                f"coordinates={coordinate_path.name}"
            )
        except (OSError, ValueError, RuntimeError) as error:
            failures.append((image_path, error))
            print(f"[{position}/{len(image_paths)}] FAILED {image_path.name}: {error}")

    if failures:
        failure_names = ", ".join(path.name for path, _ in failures)
        raise RuntimeError(
            f"{len(failures)} image(s) failed: {failure_names}"
        )

    print(f"Completed successfully. Outputs: {OUTPUT_DIR}")
    print(f"Coordinate files: {COORDINATES_DIR}")


if __name__ == "__main__":
    main()
