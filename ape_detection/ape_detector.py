import logging
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
from yolo_feature_extractor import YOLOFeatureExtractor
from adversarial_patch_extractor import AdversarialPatchExtractor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("ApeDetector")

class ApeDetector:
    def __init__(self, config_data: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        if config_data is not None:
            self.config = config_data
        else:
            cfg_file = Path(config_path) if config_path else Path(__file__).parent / "config.json"
            with open(cfg_file, "r") as f:
                self.config = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # Core Config
        input_size = self.config.get("input_size", [640, 384])
        self.input_width, self.input_height = input_size[0], input_size[1]
        self.target_classes = self.config.get("target_classes", None)
        
        # Defense Config
        defense_cfg = self.config.get("defense_settings", {})
        self.defense_enabled = defense_cfg.get("enabled", True)
        self.defense_layer_idx = defense_cfg.get("layer_index", 2)
        self.detection_output_dir = Path(self.config.get("detection_output_dir", "runs/detect"))
        self.heatmap_dir = Path(defense_cfg.get("heatmap_dir", "runs/defense_heatmaps"))
        
        # Initialize sub-modules
        self.model = YOLO(self.config["model_path"]).to(self.device)
        self.target_layer_name = f"model.{self.defense_layer_idx}"
        self.adapter = YOLOFeatureExtractor(self.model, layers=[self.target_layer_name])
        
        # Automatically pass the nested patch_extractor settings to the new class
        extractor_cfg = defense_cfg.get("patch_extractor", {})
        self.patch_extractor = AdversarialPatchExtractor(config=extractor_cfg)
        self.anomaly_floor = self.patch_extractor.floor

    def _compute_z_score_map(self, feat: torch.Tensor) -> np.ndarray:
        """Compute a robust MAD-based Z-score map from layer activations."""
        B, C, H, W = feat.shape
        energy = torch.sum(feat ** 2, dim=1, keepdim=True)
        energy_flat = energy.view(B, -1)
        median_energy = energy_flat.median(dim=1, keepdim=True)[0]
        abs_dev = torch.abs(energy_flat - median_energy)
        mad = abs_dev.median(dim=1, keepdim=True)[0]
        mad = torch.clamp(mad, min=1e-6)
        z_score = (0.6745 * (energy_flat - median_energy) / mad).view(B, 1, H, W)
        return z_score

    def _save_detection_overlay(self, resized_img, results, patch_boxes, is_attack, img_stem):
        """Draw YOLO detections and adversarial patch boxes on the resized image and save."""
        yolo_vis = resized_img.copy()
        if len(results.boxes) > 0:
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                x1, y1, x2, y2 = box.int().tolist()
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(yolo_vis, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if self.defense_enabled and is_attack:
            for (px1, py1, px2, py2) in patch_boxes:
                cv2.rectangle(yolo_vis, (px1, py1), (px2, py2), (0, 0, 255), 3)
                cv2.putText(yolo_vis, "PATCH", (px1, max(py1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.detection_output_dir.mkdir(parents=True, exist_ok=True)
        yolo_out_path = self.detection_output_dir / f"yolo_{img_stem}.jpg"
        cv2.imwrite(str(yolo_out_path), cv2.cvtColor(yolo_vis, cv2.COLOR_RGB2BGR))
        LOGGER.info(f"[Visualizer] YOLO detection saved to: {yolo_out_path}")

    def _save_heatmap_and_csv(self, z_map: np.ndarray, img_stem: str):
        """Render the Z-score map as a JET heatmap and export the raw values as CSV."""
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        max_val = max(float(np.max(z_map)), self.anomaly_floor * 2.0)
        z_clipped = np.clip(z_map, 0, max_val)
        z_norm = (z_clipped / max_val * 255).astype(np.uint8)
        
        heatmap_resized = cv2.resize(z_norm, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
        colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        heatmap_out_path = self.heatmap_dir / f"heatmap_{img_stem}.jpg"
        cv2.imwrite(str(heatmap_out_path), colored_heatmap)
        LOGGER.info(f"[Visualizer] Heatmap saved to: {heatmap_out_path}")

        z_score_dir = self.heatmap_dir.parent / "z_score_maps"
        z_score_dir.mkdir(parents=True, exist_ok=True)
        z_score_path = z_score_dir / f"z_score_{img_stem}.csv"
        np.savetxt(str(z_score_path), z_map, fmt="%.4f", delimiter=",")
        LOGGER.info(f"[Visualizer] Z-score map saved to: {z_score_path}")

    def detect(self, img_path: str, save_visualization: bool = True):
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img_stem = Path(img_path).stem
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img_rgb, (self.input_width, self.input_height))
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        self.adapter.activations.clear()
        results = self.model.predict(source=img_tensor, classes=self.target_classes, verbose=False)[0]
        
        defense_stats = {}
        is_attack = False
        patch_boxes = []
        max_z_val = 0.0

        if self.defense_enabled:
            feat = self.adapter.activations[self.target_layer_name]
            z_score = self._compute_z_score_map(feat)
            z_map_2d = z_score[0, 0].detach().cpu().numpy()
            max_z_val = z_map_2d.max()
            
            # Extract patches using the dedicated OOP class
            stride = self.input_width // feat.shape[3]
            patch_boxes = self.patch_extractor.extract(z_map_2d, stride=stride, frame=orig_img)
            is_attack = len(patch_boxes) > 0

            if is_attack:
                LOGGER.warning(f"⚠️ [Defense] Adversarial patch detected! {len(patch_boxes)} patch(es) found (Max Z: {max_z_val:.2f}).")

            defense_stats = {
                "patch_detected": is_attack,
                "patch_boxes": patch_boxes,
                "z_score_map": z_map_2d if save_visualization else None
            }

        if save_visualization:
            self._save_detection_overlay(resized_img, results, patch_boxes, is_attack, img_stem)
            if self.defense_enabled and defense_stats.get("z_score_map") is not None:
                self._save_heatmap_and_csv(defense_stats["z_score_map"], img_stem)

        return {
            "image": Path(img_path).name,
            "is_attack": is_attack,
            "detections_count": len(results.boxes),
            "defense_stats": {k: v for k, v in defense_stats.items() if k != "z_score_map"},
            "max_z_score": max_z_val,
        }

    def close(self):
        """Removes PyTorch forward hooks."""
        if hasattr(self, "adapter") and self.adapter:
            self.adapter.teardown()