"""
Shared pieces for Mech-5: multi-scale (P3 + P4) classification-branch profiling.

Per scale this is Mech-1 Option B scoring with Mech-7's per-class thresholds:
every cell is compared against EVERY calibrated profile (objects + background)
of that scale and the nearest one wins; the detector's own boxes are never
consulted. A cell's per-scale score is its "ratio":

    ratio_s = best_distance_s / (threshold_s[nearest class] + adjust_threshold)

ratio > 1 means that scale flags the cell. The scales are then combined on the
finest grid by a RULE (`and` or `confirm`, see README.md), and the combined
score is turned into boxes.
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

INPUT_SIZE = 640
STRIDE_BY_SCALE = {"P3": 8, "P4": 16, "P5": 32}
RULES = ("and", "confirm")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
KERNEL = np.ones((3, 3), np.uint8)


def load_config(path=None):
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ordered_scales(config):
    """Tapped scales, finest grid first (P3, P4, P5)."""
    return sorted(config["scales"], key=lambda s: STRIDE_BY_SCALE[s])


def validated_settings(config):
    """Returns detector_settings after failing loudly on a bad scale list,
    a stride mismatch or an unknown/unconfigured combine rule."""
    scales = config["scales"]
    if len(scales) < 2:
        raise ValueError("Mech-5 is multi-scale: config.scales needs at least two scales (e.g. P3 and P4).")
    for name, entry in scales.items():
        if name not in STRIDE_BY_SCALE:
            raise ValueError(f"scales key must be one of {list(STRIDE_BY_SCALE)}, got {name!r}")
        if int(entry["stride"]) != STRIDE_BY_SCALE[name]:
            raise ValueError(f"Config mismatch: {name} needs stride={STRIDE_BY_SCALE[name]}, "
                             f"but scales.{name}.stride={entry['stride']}.")
    s = config["detector_settings"]
    rule = s.get("combine_rule", "and")
    if rule not in RULES:
        raise ValueError(f"combine_rule must be one of {RULES}, got {rule!r}")
    if rule not in s.get("rules", {}):
        raise ValueError(f"detector_settings.rules has no entry for combine_rule={rule!r}")
    return s


def list_images(directory, max_images=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No images found in {directory}")
    return files[:max_images] if max_images else files


def load_image_tensor(path, device):
    """Returns (resized_rgb uint8 [640,640,3], tensor [1,3,640,640] in 0..1)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return rgb, tensor


class ProfileBank:
    """All class profiles (incl. background) of ONE scale, stacked for batched scoring."""

    def __init__(self, stats, device):
        self.class_ids = list(stats.keys())
        self.means = torch.stack([stats[c]["mean"] for c in self.class_ids]).to(device)         # [K, C]
        self.inv_covs = torch.stack([stats[c]["inv_cov"] for c in self.class_ids]).to(device)   # [K, C, C]

    @classmethod
    def from_file(cls, path, device):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Profile not found at {path}. Run calibrate_profiles.py first.")
        with open(path, "rb") as f:
            return cls(pickle.load(f), device)

    def distances(self, feat_flat):
        """feat_flat [N, C] -> Mahalanobis distance to every profile, [N, K]."""
        # All profiles at once: diff [K, N, C], (diff @ inv_cov) [K, N, C], row-wise dot with diff.
        # Same arithmetic as scoring one profile at a time, without the K-step Python loop.
        diff = feat_flat[None] - self.means[:, None]
        maha_sq = (torch.bmm(diff, self.inv_covs) * diff).sum(dim=-1)        # [K, N]
        return torch.sqrt(torch.clamp(maha_sq, min=0)).T.contiguous()

    def nearest(self, feat_flat):
        """-> (best_distance [N], best_index [N]); index is into self.class_ids."""
        return self.distances(feat_flat).min(dim=1)


class ThresholdTable:
    """Per-profile thresholds of one scale, aligned to its ProfileBank's class order.
    A class with too few cells was given the pooled value when the table was
    learned (learn_thresholds.py), so lookups never branch on sample counts."""

    def __init__(self, data, bank, device, adjust):
        self.adjust = float(adjust)
        pooled, per_class = data["pooled"], data["per_class"]
        thr = [per_class.get(cid, pooled)["threshold"] for cid in bank.class_ids]
        self.thr_k = torch.tensor(thr, device=device)

    @classmethod
    def from_file(cls, path, bank, device, adjust):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Threshold file not found at {path}. Run learn_thresholds.py first.")
        with open(path, "rb") as f:
            return cls(pickle.load(f), bank, device, adjust)

    def threshold(self, best_idx):
        return self.thr_k[best_idx] + self.adjust


def flagged_regions(score, stride, threshold, join, min_cells):
    """score [H, W] (finest grid) -> [{"box": (x1,y1,x2,y2), "score": max score, "cells": n}].

    Cells with score > threshold are flagged. With join > 0 they are first dilated
    `join` times (3x3) so scattered cells of one patch end up in one region. The
    box, the size check and the region score use only the ORIGINAL flagged cells."""
    mask = score > threshold
    if not mask.any():
        return []
    m8 = mask.astype(np.uint8)
    grouped = cv2.dilate(m8, KERNEL, iterations=join) if join else m8
    _, labels = cv2.connectedComponents(grouped, connectivity=8)
    ys, xs = np.nonzero(mask)
    lb = labels[ys, xs]
    order = np.argsort(lb, kind="stable")
    ys, xs, lb = ys[order], xs[order], lb[order]
    vals = score[ys, xs]
    _, starts = np.unique(lb, return_index=True)
    cells = np.diff(np.append(starts, lb.size))
    x0, x1 = np.minimum.reduceat(xs, starts), np.maximum.reduceat(xs, starts)
    y0, y1 = np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts)
    best = np.maximum.reduceat(vals, starts)
    return [{"box": (int(a * stride), int(b * stride), int((c + 1) * stride), int((d + 1) * stride)),
             "score": float(v), "cells": int(n)}
            for n, a, b, c, d, v in zip(cells, x0, y0, x1, y1, best) if n >= min_cells]
