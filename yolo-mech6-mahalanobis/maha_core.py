"""
Shared code for Mech-6-Mahalanobis.

Idea: Mech-6 taps backbone layers 1, 2, 3 and profiles ONE number per cell (log energy). This version keeps the
whole feature vector of each cell (C channels) and profiles it like Mech-1 does with the classification-head
feature: a mean vector and a covariance matrix per object class plus one for background. A cell's score is its
Mahalanobis distance to the NEAREST profile:

    d_k(x) = sqrt( (x - mu_k)^T  S_k^-1  (x - mu_k) )          score(x) = min_k d_k(x)

Normal content sits close to one of the profiles (car, person, road/background ...); a patch's texture fits none of
them, so its cells get a large score. The class compared against is chosen by the distance itself, never by the
detector's boxes, so a patch that fools the detector cannot hide from the score.

Differences from Mech-1:
  * three backbone layers (strides 4, 4, 8 for yolov5/v8/v9/11) instead of one head layer;
  * statistics are accumulated as running sums (count, sum x, sum x x^T) in float64, so calibration memory does
    not grow with the number of images (layer 1 alone has 25,600 cells per image);
  * covariance shrinkage S = (1 - a) * cov + a * mean(diag(cov)) * I (config `shrinkage`), because some backbone
    channels are almost constant and the plain covariance is close to singular.
"""

import json
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

INPUT_SIZE = 640
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
BG_ID = -1
NAME_PREFIX_RE = re.compile(r"^(\d+-)?(multi-)?patched[_-]")      # patched image name -> original name
DET_CONF, DET_IOU = 0.25, 0.7                                     # same defaults as ultralytics predict


# ----------------------------------------------------------------------------- config / files

def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_dir"] = str(path.parent)
    return cfg


def resolve(cfg, value):
    """Relative paths in a config are relative to the config file's folder."""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (Path(cfg["_dir"]) / p).resolve()


def profile_path(cfg):
    return resolve(cfg, cfg["profiles_dir"]) / cfg["profile_filename"]


def list_images(directory, start=0, count=None):
    files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No images in {directory}")
    return files[start:start + count] if count else files[start:]


def load_image_tensor(path, device):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
    return rgb, torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0


def gt_boxes(gt_dir, stem):
    """Patch boxes (640-px) for an image: <stem>.txt, or the same name without a 'patched_' style prefix."""
    if not gt_dir:
        return []
    for key in (stem, NAME_PREFIX_RE.sub("", stem)):
        p = Path(gt_dir) / f"{key}.txt"
        if p.exists():
            boxes = []
            for line in p.read_text(encoding="utf-8").splitlines():
                nums = re.findall(r"-?\d+", line)
                if len(nums) >= 4:
                    boxes.append(tuple(int(v) for v in nums[:4]))
            return boxes
    return []


def cell_masks(h, w, stride, boxes, far_px=48):
    """inside: cells whose centre lies in a patch box. far: cells more than far_px from every patch box."""
    ys = (np.arange(h) + 0.5) * stride
    xs = (np.arange(w) + 0.5) * stride
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    inside = np.zeros((h, w), bool)
    near = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in boxes:
        inside |= (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
        near |= (xx >= x1 - far_px) & (xx < x2 + far_px) & (yy >= y1 - far_px) & (yy < y2 + far_px)
    return inside, ~near


# ----------------------------------------------------------------------------- model

class Detector:
    """`.net` = torch module whose `.model[i]` are the layers; `.predict_boxes(x)` -> (xyxy [N,4], cls [N])."""

    def __init__(self, cfg, device):
        kind = cfg.get("model_type", "ultralytics")
        weights = resolve(cfg, cfg["model_path"])
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        if kind == "ultralytics":
            from ultralytics import YOLO
            yolo = YOLO(str(weights)).to(device)
            self.net = yolo.model

            def predict(x):
                b = yolo.predict(source=x, verbose=False)[0].boxes
                return b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        elif kind == "yolov5_repo":
            repo = resolve(cfg, cfg.get("yolov5_repo_dir"))
            if repo is None or not (repo / "models" / "experimental.py").exists():
                raise ValueError("model_type 'yolov5_repo' needs 'yolov5_repo_dir' = folder with the yolov5 models/ and utils/.")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from models.experimental import attempt_load
            from utils.general import non_max_suppression
            self.net = attempt_load(str(weights), device=device, fuse=True).eval()

            def predict(x):
                with torch.no_grad():
                    d = non_max_suppression(self.net(x), DET_CONF, DET_IOU)[0].cpu().numpy()
                return d[:, :4], d[:, 5].astype(int)
        else:
            raise ValueError(f"Unknown model_type '{kind}' (use 'ultralytics' or 'yolov5_repo').")
        self.predict_boxes = predict
        self.name = f"{kind}:{weights.name}"


class BackboneTap:
    """Forward hooks on backbone layers; get_activations(x) -> {layer: [1, C, H, W]}."""

    def __init__(self, net, layers):
        self.net, self.layers, self._acts = net, list(layers), {}
        self._hooks = [net.model[i].register_forward_hook(self._make(i)) for i in self.layers]

    def _make(self, i):
        def hook(module, inp, out):
            self._acts[i] = out.detach()
        return hook

    def get_activations(self, x):
        self._acts = {}
        with torch.no_grad():
            self.net(x)
        missing = [l for l in self.layers if l not in self._acts]
        if missing:
            raise RuntimeError(f"Layers {missing} produced no output; check tap_layers.")
        return dict(self._acts)


def layer_stride(feat):
    return INPUT_SIZE // feat.shape[-1]


# ----------------------------------------------------------------------------- profiles

class MahaProfiles:
    """Loaded from the calibration pickle. Per layer: class ids, means [K, C], inverse covariances [K, C, C],
    plus clean-image distance quantiles (the threshold)."""

    def __init__(self, path, device):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.device = device
        self.layers = sorted(self.data["layers"])
        self.mu, self.prec, self.ids = {}, {}, {}
        for l in self.layers:
            d = self.data["layers"][l]
            self.ids[l] = np.array(d["class_ids"])
            self.mu[l] = torch.from_numpy(d["means"]).float().to(device)
            self.prec[l] = torch.from_numpy(d["inv_covs"]).float().to(device)

    def threshold(self, layer):
        t = self.data["layers"][layer].get("clean_quantiles")
        return None if t is None else t[self.data.get("threshold_quantile_key", "q99.5")]

    def threshold_map(self, layer, cls_map, mode="per_layer"):
        """Threshold for every cell, same shape as cls_map (the nearest-profile id map).
        per_layer: the layer threshold everywhere.
        per_class: the threshold of each cell's nearest profile; profiles without one use the layer threshold."""
        thr = self.threshold(layer)
        if mode == "per_layer":
            return np.full(cls_map.shape, thr, dtype=np.float32)
        if mode != "per_class":
            raise ValueError(f"threshold_mode must be 'per_layer' or 'per_class', got '{mode}'.")
        table = self.data["layers"][layer].get("class_thresholds")
        if table is None:
            raise ValueError("This calibration has no per-class thresholds. Re-run calibrate_mahalanobis.py "
                             "(add --threshold-only to keep the profiles and redo pass 2 only).")
        out = np.full(cls_map.shape, thr, dtype=np.float32)
        for cid, t in table.items():
            out[cls_map == int(cid)] = t["threshold"]
        return out

    def quantile(self, layer, key):
        return self.data["layers"][layer]["clean_quantiles"][key]

    @torch.no_grad()
    def distances(self, layer, feat):
        """feat [1, C, H, W] -> (nearest distance [H, W] float32, nearest class id [H, W] int16), numpy."""
        _, C, H, W = feat.shape
        x = feat[0].reshape(C, -1).T.float()                                # [N, C]
        best = torch.full((x.shape[0],), float("inf"), device=x.device)
        arg = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for k in range(self.mu[layer].shape[0]):
            diff = x - self.mu[layer][k]
            d = ((diff @ self.prec[layer][k]) * diff).sum(1).clamp_min(0).sqrt()
            better = d < best
            best = torch.where(better, d, best)
            arg = torch.where(better, torch.full_like(arg, k), arg)
        ids = self.ids[layer][arg.cpu().numpy()].astype(np.int16)
        return best.reshape(H, W).cpu().numpy().astype(np.float32), ids.reshape(H, W)
