"""
Shared code for the lean-v2 Mech-6 detector (patch localizer).

Signal (same as Mech-6): at a backbone layer, energy(cell) = (L1 norm of its feature vector across channels)^2,
used as log-energy. Two scores per cell, one per localizer:

    panel 2  "energy z"      z = (log E - global mean) / global std, clipped at 0      (one threshold per layer)
    panel 3  "nearest"       distance in std units of log-energy to the NEAREST class / background profile
                             nearest_abs = |z_k| of the nearest profile; nearest_up = same, only above it
                             (per-class threshold: each cell uses the threshold of its nearest profile)

The calibrated thresholds and minimum region sizes are written into the config file (see calibrate_detector.py),
so they can be edited by hand.
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

INPUT_SIZE = 640
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
LOG_EPS = 1e-6
STD_FLOOR = 1e-3
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
    """box=(x1, y1, x2, y2) lines -> list of int tuples."""
    p = Path(path)
    if not p.exists():
        return None
    return [tuple(map(int, m.groups())) for m in BOX_RE.finditer(p.read_text(encoding="utf-8"))]


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


def log_energy_map(feat):
    """feat [1, C, H, W] -> log of energy = (L1 norm across channels)^2, float64 [H, W]."""
    e = (feat[0].abs().sum(dim=0) ** 2).cpu().numpy().astype(np.float64)
    return np.log(e + LOG_EPS)


# ----------------------------------------------------------------------------- profiles and scores

class EnergyProfiles:
    """Profile pickle (same format as yolo-mech6): {layer: {"global": {...mean_log, std_log}, class_id: {mean, std, n}}}.
    Class means/stds are of LOG energy."""

    def __init__(self, stats):
        self.stats = stats
        self.layers = sorted(stats)
        self.class_ids = {l: sorted(k for k in v if k != "global") for l, v in stats.items()}
        self.mu = {l: np.array([stats[l][c]["mean"] for c in self.class_ids[l]]) for l in stats}
        self.sd = {l: np.array([max(stats[l][c]["std"], STD_FLOOR) for c in self.class_ids[l]]) for l in stats}

    def energy_z(self, layer, le):
        """Panel 2 score."""
        g = self.stats[layer]["global"]
        return np.clip((le - g["mean_log"]) / g["std_log"], 0, None)

    def nearest(self, layer, le, score="nearest_abs"):
        """Panel 3 score and the nearest profile's class id per cell."""
        # loop over profiles (no [..., K] array): keeps memory at a few copies of `le` even with many profiles
        best = np.full(le.shape, np.inf)
        zn = np.zeros(le.shape)
        idx = np.zeros(le.shape, dtype=np.int32)
        for k, (m, s) in enumerate(zip(self.mu[layer], self.sd[layer])):
            z = (le - m) / s
            better = np.abs(z) < best
            best[better], zn[better], idx[better] = np.abs(z[better]), z[better], k
        ids = np.array(self.class_ids[layer])[idx]
        if score == "nearest_abs":
            return np.abs(zn), ids
        if score == "nearest_up":
            return np.clip(zn, 0, None), ids
        raise ValueError(f"panel3 score must be 'nearest_abs' or 'nearest_up', got '{score}'.")


def class_key(cid, names):
    """Config key for a profile: 'bg' or '<id>_<name>' (e.g. '2_car')."""
    return "bg" if cid == BG_ID else f"{cid}_{names.get(cid, 'cls').replace(' ', '-')}"


def key_to_class(key):
    return BG_ID if key == "bg" else int(str(key).split("_")[0])


def panel3_threshold_map(cal, ids, mode):
    """Per-cell threshold for panel 3 from the calibrated block of one layer."""
    thr = float(cal["layer_threshold"])
    out = np.full(ids.shape, thr, dtype=np.float64)
    if mode == "per_layer":
        return out
    if mode != "per_class":
        raise ValueError(f"panel3 threshold_mode must be 'per_layer' or 'per_class', got '{mode}'.")
    for key, t in cal.get("class_thresholds", {}).items():
        out[ids == key_to_class(key)] = float(t)
    return out


# ----------------------------------------------------------------------------- localizer

def localize(score, thr_map, stride, join_cells, thickness, mass_threshold):
    """Turn a score map into patch boxes (lean-v2: no shape or count assumptions, no hand-set seed or size).

      1. hot cells : score > threshold (per cell)
      2. join      : morphological closing, (2*join_cells+1)^2 window, so a speckled patch becomes one solid blob
      3. thickness : morphological opening, (2*thickness+1)^2 window: erase every part of the blob thinner than that
                     window (lane marks, poles, edges), whatever its angle; any thick shape survives
      4. regions   : 8-connected components of what is left
      5. mass      : sum over the region's hot cells of (score/threshold - 1), i.e. how many cells x how far above
                     the threshold. Size and strength in one number (the "cluster mass" test of brain imaging).
         keep a region if mass > mass_threshold (calibrated on clean images)
      6. overlapping boxes are merged; every remaining region is a box (no cap on the number of boxes).

    Returns (regions, hot mask). A region = {box (640 px x1 y1 x2 y2), mass, cells, score (mean ratio), peak}.
    """
    ratio = score / thr_map
    hot = ratio > 1.0
    if not hot.any():
        return [], hot
    mask = hot.astype(np.uint8)
    if join_cells > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2 * join_cells + 1,) * 2, np.uint8))
    if thickness > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2 * thickness + 1,) * 2, np.uint8))
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
                        "cells": int(area), "score": float(r.mean()), "peak": float(r.max())})
    regions = _merge_overlaps(regions)
    regions.sort(key=lambda r: -r["mass"])
    return regions, hot


def _merge_overlaps(regions):
    merged = True
    while merged and len(regions) > 1:
        merged = False
        for a in range(len(regions)):
            for b in range(a + 1, len(regions)):
                A, B = regions[a]["box"], regions[b]["box"]
                if A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]:
                    ra, rb = regions[a], regions[b]
                    cells = ra["cells"] + rb["cells"]
                    regions[a] = {"box": [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])],
                                  "mass": ra["mass"] + rb["mass"], "cells": cells,
                                  "score": (ra["score"] * ra["cells"] + rb["score"] * rb["cells"]) / cells,
                                  "peak": max(ra["peak"], rb["peak"])}
                    regions.pop(b)
                    merged = True
                    break
            if merged:
                break
    return regions


def region_settings(cfg, panel, layer, mode=None):
    """(join_cells, thickness, mass_threshold) for one panel / layer (/ panel-3 mode).
    'auto' in the region block = the calibrated value from <panel>.calibrated.<layer>.region[.<mode>]."""
    reg = cfg[panel]["region"]
    cal = cfg[panel].get("calibrated", {}).get(str(layer), {}).get("region", {})
    if mode is not None:
        cal = cal.get(mode, {})

    def get(key):
        v = reg.get(key, "auto")
        if v != "auto":
            return v
        if key not in cal:
            raise ValueError(f"{panel}: {key} is 'auto' but there is no calibrated value for layer {layer}"
                             + (f" / {mode}" if mode else "") + ". Run calibrate_detector.py or set a number.")
        return cal[key]

    return int(reg.get("join_cells", 1)), int(get("min_thickness_cells")), float(get("mass_threshold"))


def panel_maps(cfg, prof, layer, le):
    """Scores and per-cell thresholds for both panels at one layer, from the config's calibrated values.
    Returns {"panel2": (score, thr_map), "panel3": (score, thr_map, ids)}."""
    out = {}
    c2 = cfg["panel2"].get("calibrated", {}).get(str(layer))
    if c2 is None:
        raise ValueError(f"panel2 has no calibrated threshold for layer {layer}; run calibrate_detector.py.")
    z = prof.energy_z(layer, le)
    out["panel2"] = (z, np.full(z.shape, float(c2["threshold"])))
    c3 = cfg["panel3"].get("calibrated", {}).get(str(layer))
    if c3 is None:
        raise ValueError(f"panel3 has no calibrated thresholds for layer {layer}; run calibrate_detector.py.")
    score_name = cfg["panel3"].get("score", "nearest_abs")
    if c3.get("score", score_name) != score_name:
        raise ValueError(f"panel3 was calibrated for score '{c3.get('score')}' but the config says '{score_name}'. "
                         "Re-run calibrate_detector.py --thresholds-only.")
    s, ids = prof.nearest(layer, le, score_name)
    out["panel3"] = (s, panel3_threshold_map(c3, ids, cfg["panel3"].get("threshold_mode", "per_class")), ids)
    return out


def write_detections(path, regions, orig_size):
    """One line per box, same box=( ) format as the GT coordinate files (640 px), plus the original-size box."""
    sx, sy = orig_size[0] / INPUT_SIZE, orig_size[1] / INPUT_SIZE
    lines = []
    for r in regions:
        x1, y1, x2, y2 = r["box"]
        lines.append(f"box=({x1}, {y1}, {x2}, {y2}) mass={r['mass']:.2f} score={r['score']:.3f} peak={r['peak']:.3f} cells={r['cells']} "
                     f"orig_box=({round(x1 * sx)}, {round(y1 * sy)}, {round(x2 * sx)}, {round(y2 * sy)})")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
