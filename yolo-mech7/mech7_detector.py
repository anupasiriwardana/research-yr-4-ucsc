"""
Mech-7 runtime detector: Option B nearest-profile scoring with per-class
thresholds. Never reads the detector's own boxes to decide what to score
(`results.boxes` is used only to draw green reference boxes in the overlay).

Output contract is identical to the Mech-1 detector, so the existing batch /
metrics pipeline and the downstream recovery module can consume it unchanged:

    {"is_attack", "score", "bounding_boxes": [{"box": (x1,y1,x2,y2), "score"}],
     "detection_latency_ms", "visualization_latency_ms"}

Boxes are in the 640x640 input space (cell index * stride).
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from mech7_core import (INPUT_SIZE, load_config, validated_settings, load_image_tensor,
                        ProfileBank, ThresholdTable)
from ultralytics_cls_head_adapter import UltralyticsClsHeadAdapter


class Mech7Detector:
    def __init__(self, config=None, visualizations_dir=None):
        self.config = config or load_config()
        self.settings = validated_settings(self.config)
        self.scale = self.settings.get("tap_scale", "P3")
        self.stride = int(self.settings["stride"])
        self.mode = self.settings.get("threshold_mode", "per_class")
        self.min_region_cells = self.settings.get("min_region_cells", 1)
        self.visualizations_dir = visualizations_dir or self.config.get("batch_settings", {}).get("visualizations_dir")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.config["model_path"]).to(self.device)
        self.adapter = UltralyticsClsHeadAdapter(self.model.model, scales=(self.scale,))
        print(self.adapter.describe())

        profiles_dir = Path(self.config["profiles_dir"])
        self.bank = ProfileBank.from_file(profiles_dir / self.config["calibration_profile_filename"], self.device)
        self.thresholds = ThresholdTable.from_file(profiles_dir / self.config["threshold_filename"],
                                                   self.bank, self.device, self.settings)
        print(f"Mech-7: {len(self.bank.class_ids)} profiles, tap={self.scale}, stride={self.stride}, "
              f"threshold_mode={self.mode}, adjust={self.thresholds.adjust}")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def score_from_activations(self, feat, resized_img=None, results=None, img_name="output",
                               save_visualization=False):
        B, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
        best, idx = self.bank.nearest(flat)
        stat, thr = self.thresholds.decide(best, idx)
        stat, thr = stat.reshape(H, W), thr.reshape(H, W)
        raw = best.reshape(H, W)
        mask = stat > thr

        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        n_labels, labels, cc, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        raw_np = raw.cpu().numpy()
        boxes = []
        for i in range(1, n_labels):
            if cc[i, cv2.CC_STAT_AREA] < self.min_region_cells:
                continue
            x, y = int(cc[i, cv2.CC_STAT_LEFT]), int(cc[i, cv2.CC_STAT_TOP])
            w, h = int(cc[i, cv2.CC_STAT_WIDTH]), int(cc[i, cv2.CC_STAT_HEIGHT])
            boxes.append({"box": (x * self.stride, y * self.stride, (x + w) * self.stride, (y + h) * self.stride),
                          "score": float(raw_np[labels == i].max())})

        viz_ms = 0.0
        if save_visualization and resized_img is not None:
            t0 = time.perf_counter()
            self._save_overlay((stat - thr).cpu().numpy(), resized_img, results, boxes, img_name)
            viz_ms = (time.perf_counter() - t0) * 1000.0

        return {"is_attack": len(boxes) > 0,
                "score": max((b["score"] for b in boxes), default=0.0),
                "bounding_boxes": boxes,
                "visualization_latency_ms": viz_ms}

    def _save_overlay(self, margin, resized_img, results, boxes, img_name):
        # Heatmap shows margin over the cell's own threshold (>0 = flagged).
        norm = cv2.normalize(margin, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heat = cv2.applyColorMap(cv2.resize(norm, (INPUT_SIZE, INPUT_SIZE)), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), 0.5, heat, 0.5, 0)
        if results is not None and len(results.boxes) > 0:
            for b in results.boxes.xyxy:
                x1, y1, x2, y2 = b.int().tolist()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for e in boxes:
            x1, y1, x2, y2 = e["box"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(overlay, f"PATCH ({e['score']:.1f})", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        out_dir = Path(self.visualizations_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"mech7_{img_name}"), overlay)

    def detect(self, img_path, save_visualization=True):
        """Latency covers feature extraction + scoring + box extraction only
        (same definition as the Mech-1 batch pipeline): image loading, the
        separate YOLO prediction and visualization are excluded."""
        resized, tensor = load_image_tensor(img_path, self.device)
        results = self.model.predict(source=tensor, verbose=False)[0] if save_visualization else None
        self._sync()
        t0 = time.perf_counter()
        feat = self.adapter.get_activations(tensor)[self.scale]
        result = self.score_from_activations(feat, resized, results, Path(img_path).name, save_visualization)
        self._sync()
        result["detection_latency_ms"] = max(
            0.0, (time.perf_counter() - t0) * 1000.0 - result["visualization_latency_ms"])
        return result
