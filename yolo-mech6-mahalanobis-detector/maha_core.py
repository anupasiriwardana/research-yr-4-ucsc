"""
Shared code for the Mech-6-Mahalanobis detector (patch localizer, built like yolo-mech6-detector-lean-v3).

Signal (same as yolo-mech6-mahalanobis): at a backbone layer, every cell's whole C-channel feature vector x is compared
with a Mahalanobis profile (mean vector mu_k + covariance S_k) per object class plus one for background:

    d_k(x) = sqrt( (x - mu_k)^T  S_k^-1  (x - mu_k) )          score(x) = min_k d_k(x)   (nearest profile)

The profile is chosen by the distance itself, never by the detector's boxes. Two localizers share this one score and
differ only in the threshold each cell is compared with:

    panel 2  "per-layer"    one threshold per layer (clean quantile over all cells)
    panel 3  "per-class"    each cell uses the threshold of its NEAREST profile (clean quantile over the clean cells
                            nearest to that profile; profiles with too few such cells use the layer threshold)

Hot cells -> join -> multi-scale thickness -> region mass: the same localizer as lean-v3 (copied unchanged).
The calibrated thresholds and per-scale mass thresholds are written into the config file (calibrate_detector.py), so
they can be edited by hand. The profile .pkl has the same format as yolo-mech6-mahalanobis, so those files can be reused.
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
DET_CONF, DET_IOU = 0.25, 0.7                                   # same defaults as ultralytics predict
NAME_PREFIX_RE = re.compile(r"^(\d+-)?(multi-)?patched[_-]")    # patched image name -> original name
BOX_RE = re.compile(r"(?<![\w])box=\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)")


# ----------------------------------------------------------------------------- config / files

def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_path"] = str(path)
    cfg["_dir"] = str(path.parent)
    return cfg


def save_config(cfg):
    """Write the config back (without the private _ keys). The previous version is kept as <name>.json.bak."""
    path = Path(cfg["_path"])
    path.with_suffix(path.suffix + ".bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path.write_text(json.dumps(clean, indent=4) + "\n", encoding="utf-8")


def resolve(cfg, value):
    """Relative paths in a config are relative to the config file's folder."""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (Path(cfg["_dir"]) / p).resolve()


def list_images(directory, start=0, count=None):
    files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No images in {directory}")
    return files[start:start + count] if count else files[start:]


def load_image_tensor(path, device):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
    return rgb, torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, (w, h)


def read_boxes(path):
    """box=(x1, y1, x2, y2) lines (or plain 'x1 y1 x2 y2' GT lines) -> list of int tuples."""
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    boxes = [tuple(map(int, m.groups())) for m in BOX_RE.finditer(text)]
    if boxes or "box=" in text:
        return boxes
    out = []
    for line in text.splitlines():
        nums = re.findall(r"-?\d+", line)
        if len(nums) >= 4:
            out.append(tuple(int(v) for v in nums[:4]))
    return out


def gt_boxes(gt_dir, stem):
    """GT patch boxes (640 px) for an image: <stem>.txt, or the same name without a 'patched_' style prefix."""
    if not gt_dir:
        return []
    for key in (stem, NAME_PREFIX_RE.sub("", stem)):
        b = read_boxes(Path(gt_dir) / f"{key}.txt")
        if b is not None:
            return b
    return []


# ----------------------------------------------------------------------------- model

class Detector:
    """`.net` = torch module whose `.model[i]` are the layers; `.predict_boxes(x)` -> (xyxy [N,4], cls [N]);
    `.names` = {class id: name}."""

    def __init__(self, cfg, device):
        kind = cfg.get("model_type", "ultralytics")
        weights = resolve(cfg, cfg["model_path"])
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        if kind == "ultralytics":
            from ultralytics import YOLO
            yolo = YOLO(str(weights)).to(device)
            self.net = yolo.model
            self.names = dict(yolo.names)

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
            names = self.net.names
            self.names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)

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


# ----------------------------------------------------------------------------- profiles and score

class MahaProfiles:
    """Profile pickle (same format as yolo-mech6-mahalanobis): {"layers": {layer: {"class_ids", "means" [K, C],
    "inv_covs" [K, C, C], "counts", "stride", "channels"}}, ...}. Any thresholds stored in the pickle are ignored:
    this detector keeps its thresholds in the config."""

    def __init__(self, data, device):
        self.data = data
        self.device = device
        self.layers = sorted(data["layers"])
        self.class_ids, self.mu, self.prec = {}, {}, {}
        for l in self.layers:
            d = data["layers"][l]
            self.class_ids[l] = [int(c) for c in d["class_ids"]]
            self.mu[l] = torch.from_numpy(np.asarray(d["means"])).float().to(device)
            self.prec[l] = torch.from_numpy(np.asarray(d["inv_covs"])).float().to(device)

    @classmethod
    def load(cls, path, device):
        with open(path, "rb") as f:
            return cls(pickle.load(f), device)

    @torch.no_grad()
    def nearest(self, layer, feat):
        """feat [1, C, H, W] -> (nearest-profile distance [H, W] float64, nearest class id [H, W] int), numpy."""
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
        ids = np.asarray(self.class_ids[layer])[arg.cpu().numpy()]
        return best.reshape(H, W).cpu().numpy().astype(np.float64), ids.reshape(H, W)


def class_key(cid, names):
    """Config key for a profile: 'bg' or '<id>_<name>' (e.g. '2_car')."""
    return "bg" if cid == BG_ID else f"{cid}_{names.get(cid, 'cls').replace(' ', '-')}"


def key_to_class(key):
    return BG_ID if key == "bg" else int(str(key).split("_")[0])


def class_threshold_map(cal, ids):
    """Panel-3 per-cell threshold: the threshold of each cell's nearest profile (layer threshold if it has none)."""
    out = np.full(ids.shape, float(cal["layer_threshold"]), dtype=np.float64)
    for key, t in cal.get("class_thresholds", {}).items():
        out[ids == key_to_class(key)] = float(t)
    return out


def panel_maps(cfg, layer, dist, ids):
    """Per-cell thresholds of both panels at one layer from the config's calibrated values.
    Returns {"panel2": (score, thr_map), "panel3": (score, thr_map)} (the score is the same distance map)."""
    c2 = cfg["panel2"].get("calibrated", {}).get(str(layer))
    c3 = cfg["panel3"].get("calibrated", {}).get(str(layer))
    if c2 is None or c3 is None:
        raise ValueError(f"No calibrated thresholds for layer {layer}; run calibrate_detector.py.")
    return {"panel2": (dist, np.full(dist.shape, float(c2["threshold"]))),
            "panel3": (dist, class_threshold_map(c3, ids))}


# ----------------------------------------------------------------------------- localizer (same as lean-v3)

def morph_zero_border(mask, op, r):
    """Closing / opening with a (2r+1)^2 window, treating everything outside the image as NOT hot.
    (OpenCV's default border counts outside cells as hot during erosion, so a thin line along the image edge would
    survive the thickness test and closing would stretch regions along the edge.) The mask is padded with r zero
    cells, processed, and cropped back."""
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    padded = cv2.copyMakeBorder(mask, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    return cv2.morphologyEx(padded, op, k)[r:-r, r:-r]


def regions_at_scale(ratio, hot, joined, stride, thickness, mass_threshold):
    """Regions of one thickness scale: opening with a (2t+1)^2 window on the joined mask, 8-connected components,
    keep a region if its mass (sum of ratio - 1 over its hot cells) > mass_threshold. Overlapping boxes are merged."""
    mask = morph_zero_border(joined, cv2.MORPH_OPEN, thickness) if thickness > 0 else joined
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        hot_c = hot[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == i)
        if not hot_c.any():
            continue
        r = ratio[y:y + h, x:x + w][hot_c]
        mass = float((r - 1.0).sum())
        if mass <= mass_threshold:
            continue
        regions.append({"box": [x * stride, y * stride, (x + w) * stride, (y + h) * stride], "mass": mass,
                        "cells": int(area), "score": float(r.mean()), "peak": float(r.max()), "scale": thickness})
    return _merge_overlaps(regions)


def hot_and_joined(score, thr_map, join_cells):
    """Per-cell ratio score/threshold, the hot mask (ratio > 1) and the joined mask (closing, (2j+1)^2 window)."""
    ratio = score / thr_map
    hot = ratio > 1.0
    joined = hot.astype(np.uint8)
    if join_cells > 0 and hot.any():
        joined = morph_zero_border(joined, cv2.MORPH_CLOSE, join_cells)
    return ratio, hot, joined


def localize(score, thr_map, stride, join_cells, scales):
    """Turn a score map into patch boxes (every thickness scale is tried; the detector picks the scale per region).

      1. hot cells : score > threshold (per cell)
      2. join      : morphological closing, (2*join_cells+1)^2 window, so a speckled patch becomes one solid blob
      3. for EVERY thickness scale t in `scales` ({t: mass_threshold}):
           thickness : morphological opening, (2t+1)^2 window: erase every part thinner than the window
           regions   : 8-connected components of what is left
           mass      : sum over the region's hot cells of (score/threshold - 1) (size x strength, the cluster-mass test)
           keep a region if mass > that scale's mass threshold (calibrated jointly on clean images)
      4. combine the scales, thickest first: a region is added unless it overlaps a region already kept.

    Returns (regions, hot mask). A region = {box (640 px x1 y1 x2 y2), mass, cells, score (mean ratio), peak, scale}.
    """
    ratio, hot, joined = hot_and_joined(score, thr_map, join_cells)
    if not hot.any() or not scales:
        return [], hot
    kept = []
    for t in sorted(scales, reverse=True):
        for r in regions_at_scale(ratio, hot, joined, stride, t, scales[t]):
            if not any(_overlap(r["box"], k["box"]) for k in kept):
                kept.append(r)
    kept.sort(key=lambda r: -r["mass"])
    return kept, hot


def _overlap(A, B):
    return A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]


def _merge_overlaps(regions):
    """Merge overlapping boxes of ONE scale (masses add up: the parts are different cells)."""
    merged = True
    while merged and len(regions) > 1:
        merged = False
        for a in range(len(regions)):
            for b in range(a + 1, len(regions)):
                A, B = regions[a]["box"], regions[b]["box"]
                if _overlap(A, B):
                    ra, rb = regions[a], regions[b]
                    cells = ra["cells"] + rb["cells"]
                    regions[a] = {"box": [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])],
                                  "mass": ra["mass"] + rb["mass"], "cells": cells,
                                  "score": (ra["score"] * ra["cells"] + rb["score"] * rb["cells"]) / cells,
                                  "peak": max(ra["peak"], rb["peak"]), "scale": ra["scale"]}
                    regions.pop(b)
                    merged = True
                    break
            if merged:
                break
    return regions


def region_settings(cfg, panel, layer):
    """(join_cells, {thickness: mass_threshold}) for one panel / layer.
    mass_thresholds 'auto' = the calibrated values in <panel>.calibrated.<layer>.region.mass_thresholds;
    or a dict {"<thickness>": value} set by hand (leave a scale out, or set it to null, to switch it off)."""
    reg = cfg[panel]["region"]
    v = reg.get("mass_thresholds", "auto")
    if v == "auto":
        cal = cfg[panel].get("calibrated", {}).get(str(layer), {}).get("region", {})
        if "mass_thresholds" not in cal:
            raise ValueError(f"{panel}: mass_thresholds is 'auto' but there are no calibrated values for layer {layer}. "
                             "Run calibrate_detector.py or set them by hand.")
        v = cal["mass_thresholds"]
    scales = {int(t): float(m) for t, m in v.items() if m is not None}
    return int(reg.get("join_cells", 1)), scales


def write_detections(path, regions, orig_size):
    """One line per box, same box=( ) format as the GT coordinate files (640 px), plus the original-size box."""
    sx, sy = orig_size[0] / INPUT_SIZE, orig_size[1] / INPUT_SIZE
    lines = []
    for r in regions:
        x1, y1, x2, y2 = r["box"]
        lines.append(f"box=({x1}, {y1}, {x2}, {y2}) mass={r['mass']:.2f} scale={r['scale']} score={r['score']:.3f} "
                     f"peak={r['peak']:.3f} cells={r['cells']} "
                     f"orig_box=({round(x1 * sx)}, {round(y1 * sy)}, {round(x2 * sx)}, {round(y2 * sy)})")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
