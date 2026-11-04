"""
Model loading for Mech-6, so the same scripts run on different YOLO families.

Config key `model_type`:
    "ultralytics" (default)  yolov8 / yolo11 / yolov9 (ultralytics format, e.g. yolov9c.pt) ...
                             loaded with `ultralytics.YOLO(model_path)`.
    "yolov5_repo"            the ORIGINAL yolov5 repository (github.com/ultralytics/yolov5 cloned locally,
                             e.g. D:\\GitHub\\yolov5_adversarial). Loaded with that repo's own
                             `models.experimental.attempt_load`, so the `yolov5_repo_dir` config key must
                             point at the clone. (The `yolov5su.pt`-style files from the ultralytics package
                             are ordinary "ultralytics" models, not this type.)

Both give a `Detector` with:
    .net                     torch module whose `.model[i]` are the backbone layers (hook targets)
    .predict_boxes(tensor)   -> (xyxy [N, 4] float array in 640x640 pixels, cls [N] int array)
"""

import sys
from pathlib import Path

import numpy as np
import torch

from mech6_energy import INPUT_SIZE, LAYER_STRIDE, BackboneTap

DET_CONF = 0.25     # same defaults as ultralytics predict (conf 0.25, NMS IoU 0.7)
DET_IOU = 0.7


class Detector:
    def __init__(self, net, predict_fn, name):
        self.net = net
        self._predict = predict_fn
        self.name = name

    def predict_boxes(self, tensor):
        return self._predict(tensor)


def _load_ultralytics(cfg, device):
    from ultralytics import YOLO
    yolo = YOLO(cfg["model_path"]).to(device)

    def predict(tensor):
        boxes = yolo.predict(source=tensor, verbose=False)[0].boxes
        return boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)

    return Detector(yolo.model, predict, f"ultralytics:{Path(cfg['model_path']).name}")


def _load_yolov5_repo(cfg, device):
    repo = cfg.get("yolov5_repo_dir")
    if not repo or not (Path(repo) / "models" / "experimental.py").exists():
        raise ValueError("model_type 'yolov5_repo' needs 'yolov5_repo_dir' pointing at a clone of the yolov5 "
                         "repository (the folder that contains models/ and utils/).")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))            # yolov5's own `models` / `utils` packages
    from models.experimental import attempt_load
    from utils.general import non_max_suppression

    net = attempt_load(cfg["model_path"], device=device, fuse=True).eval()

    def predict(tensor):
        with torch.no_grad():
            out = net(tensor)
            det = non_max_suppression(out, DET_CONF, DET_IOU)[0]      # [N, 6] x1 y1 x2 y2 conf cls
        det = det.cpu().numpy()
        return det[:, :4], det[:, 5].astype(int)

    return Detector(net, predict, f"yolov5_repo:{Path(cfg['model_path']).name}")


def load_detector(cfg, device):
    kind = cfg.get("model_type", "ultralytics")
    if kind == "ultralytics":
        return _load_ultralytics(cfg, device)
    if kind == "yolov5_repo":
        return _load_yolov5_repo(cfg, device)
    raise ValueError(f"Unknown model_type '{kind}' (use 'ultralytics' or 'yolov5_repo').")


def verify_strides(tap: BackboneTap, device):
    """Mech-6 assumes backbone layers 1, 2, 3 sit at strides 4, 4, 8. Check it on this model."""
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    acts = tap.get_activations(dummy)
    for layer in tap.layers:
        if layer not in acts:
            raise RuntimeError(f"Layer {layer} produced no output; check tap_layers for this model.")
        stride = INPUT_SIZE // acts[layer].shape[-1]
        if stride != LAYER_STRIDE[layer]:
            raise RuntimeError(f"Layer {layer} has stride {stride}, Mech-6 expects {LAYER_STRIDE[layer]}. "
                               f"This model's backbone is laid out differently; choose other tap_layers "
                               f"(and update LAYER_STRIDE in mech6_energy.py).")
