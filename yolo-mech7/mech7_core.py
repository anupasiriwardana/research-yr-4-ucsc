"""
Shared pieces for Mech-7 (per-class thresholding on top of Mech-1 Option B).

Scoring is unchanged from Option B: every cell is compared against EVERY
calibrated profile (objects + background) and the nearest one wins. The
detector's own boxes are never consulted at runtime. The only change is what
happens with the winning distance:

    global    : best_distance > global_threshold              (Mech-1 baseline)
    per_class : best_distance > threshold[best_class]         (Mech-7 base)
    zscore    : (best_distance - mean[best_class]) / std[best_class] > z_threshold
                                                              (Mech-7 Variant B)

`adjust_threshold` is added on top of every learned/global threshold
(raw-distance units for global/per_class, z units for zscore).
"""

import json
import pickle
from pathlib import Path

import cv2
import torch

INPUT_SIZE = 640
STRIDE_BY_SCALE = {"P3": 8, "P4": 16, "P5": 32}
THRESHOLD_MODES = ("global", "per_class", "zscore")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path=None):
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validated_settings(config):
    """Returns detector_settings after failing loudly on a tap_scale/stride
    mismatch (e.g. tap_scale P4 with stride still 8) or an unknown mode."""
    s = config["detector_settings"]
    scale = s.get("tap_scale", "P3")
    if scale not in STRIDE_BY_SCALE:
        raise ValueError(f"tap_scale must be one of {list(STRIDE_BY_SCALE)}, got {scale!r}")
    if int(s["stride"]) != STRIDE_BY_SCALE[scale]:
        raise ValueError(
            f"Config mismatch: tap_scale={scale} needs stride={STRIDE_BY_SCALE[scale]}, "
            f"but detector_settings.stride={s['stride']}."
        )
    mode = s.get("threshold_mode", "per_class")
    if mode not in THRESHOLD_MODES:
        raise ValueError(f"threshold_mode must be one of {THRESHOLD_MODES}, got {mode!r}")
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
    """All class profiles (incl. background) stacked for batched scoring."""

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
        N, K = feat_flat.shape[0], self.means.shape[0]
        out = torch.empty((N, K), device=feat_flat.device)
        for k in range(K):
            diff = feat_flat - self.means[k]
            maha_sq = torch.einsum("nc,cd,nd->n", diff, self.inv_covs[k], diff)
            out[:, k] = torch.sqrt(torch.clamp(maha_sq, min=0))
        return out

    def nearest(self, feat_flat):
        """-> (best_distance [N], best_index [N]); index is into self.class_ids."""
        return self.distances(feat_flat).min(dim=1)


class ThresholdTable:
    """Per-profile thresholds aligned to a ProfileBank's class order.

    A class with too few threshold-pass samples was already given the pooled
    values when the table was learned (see learn_thresholds.py), so lookups
    here never need to branch on sample counts.
    """

    def __init__(self, data, bank, device, mode, adjust, global_threshold):
        self.mode = mode
        self.adjust = float(adjust)
        self.global_threshold = float(global_threshold)
        self.data = data
        if mode == "global":
            return
        if data is None:
            raise FileNotFoundError("threshold_mode needs a learned threshold file; run learn_thresholds.py.")
        pooled = data["pooled"]
        per_class = data["per_class"]
        thr, mean, std = [], [], []
        for cid in bank.class_ids:
            e = per_class.get(cid, pooled)
            thr.append(e["threshold"])
            mean.append(e["mean"])
            std.append(max(e["std"], 1e-6))
        self.thr_k = torch.tensor(thr, device=device)
        self.mean_k = torch.tensor(mean, device=device)
        self.std_k = torch.tensor(std, device=device)
        self.z_threshold = float(data["zscore_threshold"])

    @classmethod
    def from_file(cls, path, bank, device, settings):
        mode = settings.get("threshold_mode", "per_class")
        data = None
        if mode != "global":
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Threshold file not found at {path}. Run learn_thresholds.py first.")
            with open(path, "rb") as f:
                data = pickle.load(f)
        return cls(data, bank, device, mode, settings.get("adjust_threshold", 0.0),
                   settings.get("global_threshold", 17.0))

    def decide(self, best_dist, best_idx):
        """-> (stat [N], threshold [N]); a cell is anomalous when stat > threshold."""
        if self.mode == "global":
            return best_dist, torch.full_like(best_dist, self.global_threshold + self.adjust)
        if self.mode == "per_class":
            return best_dist, self.thr_k[best_idx] + self.adjust
        z = (best_dist - self.mean_k[best_idx]) / self.std_k[best_idx]
        return z, torch.full_like(z, self.z_threshold + self.adjust)
