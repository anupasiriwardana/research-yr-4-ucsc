"""
Shared code for the three scripts in this folder (generate / apply / verify).

Everything model-specific lives here, so the scripts themselves are identical for
YOLOv5, YOLOv8, YOLOv9 and YOLO11:

    DetModel(cfg, device)
        .raw_preds(x)    [B, 3, 640, 640] in 0..1  ->  [B, A, 4 + nc]
                         columns 0:4 = (cx, cy, w, h) in 640-px space, 4: = per-class confidence.
                         Differentiable w.r.t. x (that is what the patch training needs).
        .detect(x, ...)  -> list of (x1, y1, x2, y2, conf, cls) after NMS (no gradient)
        .names           class id -> name

Two model families (config: model.family):
    "ultralytics"  yolov8 / yolov9 / yolo11 (anything the ultralytics package loads). Anchor-free head,
                   output [B, 4 + nc, A]; class confidence is the class score itself.
    "yolov5"       the ORIGINAL yolov5 repository code (vendored in ./yolov5_src, so no other repo is
                   needed). Anchor-based head with an objectness channel, output [B, A, 5 + nc];
                   class confidence = objectness * class score, which is also what yolov5's own NMS uses.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

HERE = Path(__file__).resolve().parent
INPUT_SIZE = 640
DEFAULT_EXTENSIONS = [".jpg", ".jpeg", ".png"]


# ----------------------------------------------------------------------------- config

def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path if (Path.cwd() / path).exists() else HERE / path
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_dir"] = str(path.parent)
    cfg["_file"] = str(path)
    return cfg


def resolve(cfg, value):
    """Relative paths in the config are relative to the config file's folder."""
    p = Path(value)
    return p if p.is_absolute() else (Path(cfg["_dir"]) / p).resolve()


def section(cfg, name):
    if name not in cfg:
        raise KeyError(f"Config {cfg['_file']} has no '{name}' section.")
    return cfg[name]


def patch_file(cfg, apply_section=None):
    """Full path of the saved patch (.npy): apply.patch_path, else paths.patch_dir / (apply.patch_name or patch_name)."""
    apply_section = apply_section or {}
    if apply_section.get("patch_path"):
        return resolve(cfg, apply_section["patch_path"])
    name = apply_section.get("patch_name") or cfg["patch_name"]
    if not name.endswith(".npy"):
        name += ".npy"
    return resolve(cfg, cfg["paths"]["patch_dir"]) / name


def list_images(directory, start_index=0, num_images=None, extensions=None):
    exts = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}
    files = sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in exts)
    chosen = files[start_index:start_index + num_images] if num_images else files[start_index:]
    if not chosen:
        raise FileNotFoundError(f"No images selected from {directory} (extensions {sorted(exts)}, start_index={start_index}).")
    return chosen


# ----------------------------------------------------------------------------- model

class DetModel:
    def __init__(self, cfg, device, trainable=False):
        m = section(cfg, "model")
        self.family = m.get("family", "ultralytics")
        self.device = device
        weights = resolve(cfg, m["weights"])
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        if self.family == "ultralytics":
            from ultralytics import YOLO
            yolo = YOLO(str(weights)).to(device)
            self.net = yolo.model
            self.names = dict(yolo.names)
        elif self.family == "yolov5":
            src = HERE / "yolov5_src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            from models.experimental import attempt_load
            # inplace=False: the stock yolov5 Detect head edits its sigmoid output in place, which breaks backprop
            self.net = attempt_load(str(weights), device=device, inplace=False, fuse=True)
            self.names = dict(self.net.names)
        else:
            raise ValueError(f"model.family must be 'ultralytics' or 'yolov5', got '{self.family}'.")
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def raw_preds(self, x):
        out = self.net(x)
        if isinstance(out, dict):
            out = out.get("one2many", list(out.values())[0])
        if isinstance(out, (tuple, list)):
            out = out[0]
        if self.family == "yolov5":                      # [B, A, 5 + nc]: cx cy w h obj cls...
            return torch.cat([out[..., :4], out[..., 4:5] * out[..., 5:]], dim=-1)
        return out.transpose(1, 2)                       # [B, 4 + nc, A] -> [B, A, 4 + nc]

    @torch.no_grad()
    def detect(self, x, conf_thresh=0.25, iou_thresh=0.7):
        p = self.raw_preds(x)[0]
        conf, cls = p[:, 4:].max(dim=-1)
        keep = conf > conf_thresh
        p, conf, cls = p[keep], conf[keep], cls[keep]
        if p.shape[0] == 0:
            return []
        xyxy = torch.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                            p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], dim=-1)
        idx = torchvision.ops.batched_nms(xyxy, conf, cls, iou_thresh)
        return [(*xyxy[i].tolist(), float(conf[i]), int(cls[i])) for i in idx]


# ----------------------------------------------------------------------------- images

def load_image(path, device):
    """Image file -> [3, 640, 640] float tensor in 0..1 (stretched to 640x640, as in all other tools here)."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)


def save_image(tensor, path):
    arr = (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------------- targets and patch placement

def select_target_object(preds, conf_thresh=0.25, target_class=None):
    """Training: the single most confident object (of target_class if set) from a CLEAN forward pass.
    Returns (anchor_mask [A], (cx_norm, cy_norm), box_xyxy) or None."""
    boxes = preds[0, :, 0:4]
    max_probs, max_classes = preds[0, :, 4:].max(dim=-1)
    cand = max_probs > conf_thresh
    if target_class is not None:
        cand = cand & (max_classes == target_class)
    if cand.sum() == 0:
        return None
    ids = cand.nonzero(as_tuple=True)[0]
    best = ids[max_probs[ids].argmax()]
    bx, by, bw, bh = boxes[best].tolist()
    x1, y1, x2, y2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ax, ay = boxes[:, 0], boxes[:, 1]
    mask = (ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2) & (max_probs > conf_thresh)
    return mask, ((bx / INPUT_SIZE) * 2 - 1, (by / INPUT_SIZE) * 2 - 1), (x1, y1, x2, y2)


def select_all_target_objects(preds, conf_thresh=0.25, target_class=None, max_objects=4, min_separation=0.3):
    """Applying: up to `max_objects` distinct targets, most confident first, at least `min_separation`
    apart (distance between box centres in the normalised [-1, 1] canvas)."""
    boxes = preds[0, :, 0:4]
    max_probs, max_classes = preds[0, :, 4:].max(dim=-1)
    cand = max_probs > conf_thresh
    if target_class is not None:
        cand = cand & (max_classes == target_class)
    ids = cand.nonzero(as_tuple=True)[0]
    if len(ids) == 0:
        return []
    selected = []
    for i in ids[max_probs[ids].argsort(descending=True)]:
        bx, by, bw, bh = boxes[i].tolist()
        cx, cy = (bx / INPUT_SIZE) * 2 - 1, (by / INPUT_SIZE) * 2 - 1
        if any(((cx - s["cx"]) ** 2 + (cy - s["cy"]) ** 2) ** 0.5 < min_separation for s in selected):
            continue
        selected.append({"cx": cx, "cy": cy, "box": (bx, by, bw, bh), "conf": float(max_probs[i])})
        if len(selected) >= max_objects:
            break
    return selected


def build_patch_overlay(patch, canvas_hw, cx, cy, angle_deg, scale):
    """Differentiably place `patch` (C, ph, pw) on a (C, H, W) canvas: rotated by angle_deg, scaled by
    `scale` (relative to the canvas), centred at normalised (cx, cy) in [-1, 1] (the corrected V2 placement)."""
    C, ph, pw = patch.shape
    H, W = canvas_hw
    theta = torch.deg2rad(torch.tensor(float(angle_deg), device=patch.device))
    cos, sin = torch.cos(theta), torch.sin(theta)
    cx_t = torch.tensor(float(cx), device=patch.device)
    cy_t = torch.tensor(float(cy), device=patch.device)
    tx = -(cos * cx_t + sin * cy_t) / scale
    ty = (sin * cx_t - cos * cy_t) / scale
    affine = torch.stack([
        torch.stack([cos / scale, sin / scale, tx]),
        torch.stack([-sin / scale, cos / scale, ty]),
    ]).unsqueeze(0).float()
    grid = F.affine_grid(affine, size=(1, C, H, W), align_corners=False)
    warped = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False, padding_mode="zeros")
    mask = F.grid_sample(torch.ones((1, 1, ph, pw), device=patch.device), grid, align_corners=False, padding_mode="zeros")
    return warped.squeeze(0), mask.squeeze(0)


def rendered_mask_box(mask):
    """Visible patch mask -> (x1, y1, x2, y2) in 640-px coordinates, or None."""
    ys, xs = np.where(mask.squeeze(0).detach().cpu().numpy() > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def apply_patches(image, patch, targets, scale, rotation):
    """Paste `patch` on every target. Returns (patched image, list of patch boxes)."""
    out, boxes = image.clone(), []
    for t in targets:
        warped, mask = build_patch_overlay(patch, out.shape[1:], t["cx"], t["cy"], rotation, scale)
        out = out * (1 - mask) + warped * mask
        box = rendered_mask_box(mask)
        if box is not None:
            boxes.append(box)
    return out, boxes


def write_coordinate_file(path, boxes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in boxes:
            f.write(f"box=({x1}, {y1}, {x2}, {y2})\n")


def read_coordinate_file(path):
    boxes = []
    if Path(path).exists():
        import re
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            m = re.findall(r"-?\d+", line)
            if len(m) >= 4:
                boxes.append(tuple(int(v) for v in m[:4]))
    return boxes
