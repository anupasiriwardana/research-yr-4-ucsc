"""
Shared pieces for Mech-6: per-class feature-energy profiling on Mech-4's backbone signal.

Mech-4 signal: at a backbone layer, energy(cell) = (L1 norm of its feature vector across
channels)^2. Mech-4 flags a cell when its energy is above ONE global threshold per layer
(mean + k*std over all clean cells). Mech-6 instead keeps a 1-D profile (mean/std of the
LOG energy) for every object class and for background, and scores a cell by how far its
energy is from the profile it fits best. The class a cell is compared against is chosen
from the cell's own energy only, never from the detector's boxes.

Tapped layers (same for yolov8n and yolo11n; checked): backbone layer 1 (stride 4),
layer 2 (stride 4), layer 3 (stride 8).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

INPUT_SIZE = 640
LAYER_STRIDE = {1: 4, 2: 4, 3: 8}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
LOG_EPS = 1e-6
STD_FLOOR = 1e-3
BG_ID = -1


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


class BackboneTap:
    """Forward hooks on backbone layers; get_activations -> {layer: [1, C, H, W]}."""

    def __init__(self, model, layers):
        self.layers = list(layers)
        self._acts = {}
        self._model = model
        self._hooks = [model.model[i].register_forward_hook(self._make(i)) for i in self.layers]

    def _make(self, i):
        def hook(module, inp, out):
            self._acts[i] = out.detach()
        return hook

    def get_activations(self, tensor):
        self._acts = {}
        with torch.no_grad():
            self._model(tensor)
        return dict(self._acts)


def energy_map(feat):
    """feat [1, C, H, W] -> energy [H, W] = (L1 norm across channels)^2 (Mech-4 definition)."""
    return feat[0].abs().sum(dim=0) ** 2


def log_energy(e):
    return np.log(np.asarray(e, dtype=np.float64) + LOG_EPS)


class EnergyProfiles:
    """Per layer: class id -> (mean, std) of log-energy, plus the global stats Mech-4 uses."""

    def __init__(self, stats):
        self.stats = stats
        self.class_ids = {l: sorted(k for k in v if k != "global") for l, v in stats.items()}
        self.mu = {l: np.array([stats[l][c]["mean"] for c in self.class_ids[l]]) for l in stats}
        self.sd = {l: np.array([max(stats[l][c]["std"], STD_FLOOR) for c in self.class_ids[l]]) for l in stats}

    def scores(self, layer, e):
        """e: energy map [H, W] -> dict of score maps (all [H, W]):
             E            raw energy (Mech-4 with one global threshold)
             nearest_abs  min over profiles of |z|, z = (log E - mean_c) / std_c
             nearest_up   z of the nearest profile, clipped at 0 (only 'more energy than expected')
        """
        le = log_energy(e)
        z = (le[..., None] - self.mu[layer]) / self.sd[layer]            # [H, W, K]
        idx = np.abs(z).argmin(axis=-1)
        zn = np.take_along_axis(z, idx[..., None], axis=-1)[..., 0]
        return {"E": np.asarray(e, dtype=np.float64), "nearest_abs": np.abs(zn), "nearest_up": np.clip(zn, 0, None)}
