"""
Mech-5 runtime detector: P3 + P4 classification-branch profiling, combined.

One forward pass; the adapter taps every configured scale. Per scale each cell
gets a ratio (nearest-profile distance / that profile's own threshold). The
ratio maps are brought to the finest grid (a P4 cell is repeated 2x2 on the P3
grid) and merged by the configured rule:

    and     : min over scales    -- a cell counts only if EVERY scale flags it
    confirm : the finest scale decides, but a cell only counts if every coarser
              scale also flags a cell within one cell (3x3) of it

Cells with score > 1 + offset are flagged and grouped into boxes (optionally
joining nearby cells first). The detector's own boxes are never used to decide
anything (`results.boxes` only draws green reference boxes in the overlay).

Output contract is identical to the Mech-1 / Mech-7 detectors, so the existing
batch / metrics pipeline and the downstream recovery module consume it unchanged:

    {"is_attack", "score", "bounding_boxes": [{"box": (x1,y1,x2,y2), "score"}],
     "detection_latency_ms", "visualization_latency_ms"}

Boxes are in the 640x640 input space.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from mech5_core import (INPUT_SIZE, STRIDE_BY_SCALE, load_config, validated_settings, ordered_scales,
                        load_image_tensor, ProfileBank, ThresholdTable, flagged_regions)
from ultralytics_cls_head_adapter import UltralyticsClsHeadAdapter


class Mech5Detector:
    def __init__(self, config=None, visualizations_dir=None):
        self.config = config or load_config()
        self.settings = validated_settings(self.config)
        self.scales = ordered_scales(self.config)                       # finest first
        self.strides = {s: STRIDE_BY_SCALE[s] for s in self.scales}
        self.fine_stride = self.strides[self.scales[0]]
        self.rule = self.settings.get("combine_rule", "and")
        r = self.settings["rules"][self.rule]
        self.offset, self.join, self.min_cells = float(r["offset"]), int(r["join"]), int(r["min_region_cells"])
        adjust = self.settings.get("adjust_threshold", 0.0)
        self.visualizations_dir = visualizations_dir or self.config.get("batch_settings", {}).get("visualizations_dir")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.config["model_path"]).to(self.device)
        self.adapter = UltralyticsClsHeadAdapter(self.model.model, scales=tuple(self.scales))
        print(self.adapter.describe())

        profiles_dir = Path(self.config["profiles_dir"])
        self.banks, self.thresholds = {}, {}
        for s in self.scales:
            entry = self.config["scales"][s]
            self.banks[s] = ProfileBank.from_file(profiles_dir / entry["profile_filename"], self.device)
            self.thresholds[s] = ThresholdTable.from_file(profiles_dir / entry["threshold_filename"],
                                                          self.banks[s], self.device, adjust)
        print(f"Mech-5: scales={self.scales}, rule={self.rule}, offset={self.offset}, join={self.join}, "
              f"min_region_cells={self.min_cells}, adjust={adjust}")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def combined_score(self, activations):
        """{scale: [1, C, H, W]} -> combined ratio map [H0, W0] on the finest grid."""
        ratios = []
        for s in self.scales:
            feat = activations[s]
            _, C, H, W = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
            best, idx = self.banks[s].nearest(flat)
            ratio = (best / self.thresholds[s].threshold(idx).clamp(min=1e-6)).reshape(H, W)
            f = self.strides[s] // self.fine_stride
            if f > 1:
                ratio = ratio.repeat_interleave(f, 0).repeat_interleave(f, 1)
            ratios.append(ratio)
        if len({tuple(r.shape) for r in ratios}) != 1:
            raise RuntimeError(f"Scale grids do not line up after upsampling: {[tuple(r.shape) for r in ratios]}")
        if self.rule == "and":
            return torch.stack(ratios).min(dim=0).values
        # confirm: coarse ratios are max-filtered 3x3 (flag within one cell), then every one must pass
        coarse = [F.max_pool2d(r[None, None], 3, stride=1, padding=1)[0, 0] for r in ratios[1:]]
        return torch.minimum(ratios[0], torch.stack(coarse).min(dim=0).values)

    def score_from_activations(self, activations, resized_img=None, results=None, img_name="output",
                               save_visualization=False):
        score = self.combined_score(activations).cpu().numpy()
        threshold = 1.0 + self.offset
        regions = flagged_regions(score, self.fine_stride, threshold, self.join, self.min_cells)
        boxes = [{"box": r["box"], "score": r["score"]} for r in regions]

        viz_ms = 0.0
        if save_visualization and resized_img is not None:
            t0 = time.perf_counter()
            self._save_overlay(score - threshold, resized_img, results, boxes, img_name)
            viz_ms = (time.perf_counter() - t0) * 1000.0
        return {"is_attack": len(boxes) > 0,
                "score": max((b["score"] for b in boxes), default=0.0),
                "bounding_boxes": boxes,
                "visualization_latency_ms": viz_ms}

    def _save_overlay(self, margin, resized_img, results, boxes, img_name):
        # Heatmap shows the combined score minus its flag threshold (>0 = flagged).
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
            cv2.putText(overlay, f"PATCH ({e['score']:.2f})", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        out_dir = Path(self.visualizations_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"mech5_{img_name}"), overlay)

    def detect(self, img_path, save_visualization=True):
        """Latency covers feature extraction (one forward pass, all scales) + scoring
        + box extraction only (same definition as the Mech-1/Mech-7 batch pipelines):
        image loading, the separate YOLO prediction and visualization are excluded."""
        resized, tensor = load_image_tensor(img_path, self.device)
        results = self.model.predict(source=tensor, verbose=False)[0] if save_visualization else None
        self._sync()
        t0 = time.perf_counter()
        acts = self.adapter.get_activations(tensor)
        result = self.score_from_activations(acts, resized, results, Path(img_path).name, save_visualization)
        self._sync()
        result["detection_latency_ms"] = max(
            0.0, (time.perf_counter() - t0) * 1000.0 - result["visualization_latency_ms"])
        return result
