import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import json
import numpy as np
import torch

from feature_extractor import FeatureExtractor
from adversarial_patch_extractor import AdversarialPatchExtractor
from detection_backends import get_backend, Detection

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

        # Defense Config
        defense_cfg = self.config.get("defense_settings", {})
        self.defense_enabled = defense_cfg.get("enabled", True)
        self.detection_output_dir = Path(self.config.get("detection_output_dir", "runs/detect"))
        self.heatmap_dir = Path(defense_cfg.get("heatmap_dir", "runs/defense_heatmaps"))

        # Model family: "yolo" (default, backward-compatible) or "torchvision_rcnn"
        self.model_family = self.config.get("model_family", "yolo")
        self.backend = get_backend(self.model_family, score_thresh=self.config.get("score_thresh", 0.5))

        # Load the model through whichever backend applies
        self.model = self.backend.load(
            self.config.get("model_path"),
            self.device,
            (self.input_width, self.input_height),
            architecture=self.config.get("architecture"),
            num_classes=self.config.get("num_classes"),
        )

        # Resolve which layer to tap. "layer_name" (a dotted string path) is the
        # new, family-agnostic way to say this; the old integer "layer_index"
        # is still honoured for YOLO configs ("model.{idx}"), and if neither
        # is given, the backend's own sensible default is used (model.2 for
        # YOLO, backbone.body.layer1 for torchvision R-CNN -- both are the
        # first stride-4 backbone stage for their respective architecture).
        self.target_layer_name = defense_cfg.get("layer_name")
        if self.target_layer_name is None:
            layer_idx = defense_cfg.get("layer_index")
            self.target_layer_name = f"model.{layer_idx}" if layer_idx is not None else self.backend.default_layer

        self.adapter = FeatureExtractor(self.model, layers=[self.target_layer_name])

        # Automatically pass the nested patch_extractor settings to the new class
        extractor_cfg = defense_cfg.get("patch_extractor", {})
        self.patch_extractor = AdversarialPatchExtractor(config=extractor_cfg)

        # Class names, resolved once at load time. target_classes in config
        # may be given as names (recommended -- portable across backends,
        # e.g. "car") or raw integer ids (backward-compatible, but tied to
        # whichever label space the specific loaded model uses: YOLO's COCO-80
        # vs torchvision's COCO-91-with-background -- the SAME integer means
        # a DIFFERENT class across those two, so prefer names when mixing
        # model families).
        self._class_names = self.backend.class_names(self.model)
        name_overrides = self.config.get("class_names")
        if name_overrides:
            if isinstance(name_overrides, dict):
                self._class_names.update({int(k): v for k, v in name_overrides.items()})
            else:
                self._class_names.update({i: n for i, n in enumerate(name_overrides)})

        raw_targets = self.config.get("target_classes")
        self.target_classes = self._resolve_target_classes(raw_targets)

    def _resolve_target_classes(self, raw_targets) -> Optional[List[int]]:
        if not raw_targets:
            return None
        name_to_id = {v: k for k, v in self._class_names.items()}
        resolved = []
        for t in raw_targets:
            if isinstance(t, str):
                if t not in name_to_id:
                    raise ValueError(
                        f"Unknown class name '{t}' for this model (family='{self.model_family}'). "
                        f"Known names: {sorted(name_to_id)[:20]}{'...' if len(name_to_id) > 20 else ''}"
                    )
                resolved.append(name_to_id[t])
            else:
                resolved.append(int(t))
        return resolved

    def _compute_z_score_map(self, feat: torch.Tensor) -> np.ndarray:
        """Compute a robust MAD-based Z-score map from layer activations.

        Unchanged from the original, YOLO-only version. It only assumes a
        generic [B, C, H, W] activation tensor -- it has no idea whether that
        tensor came from a YOLO backbone block or a Faster R-CNN ResNet
        stage, and it doesn't need to.
        """
        B, C, H, W = feat.shape
        energy = torch.sum(feat ** 2, dim=1, keepdim=True)
        energy_flat = energy.view(B, -1)
        median_energy = energy_flat.median(dim=1, keepdim=True)[0]
        abs_dev = torch.abs(energy_flat - median_energy)
        mad = abs_dev.median(dim=1, keepdim=True)[0]
        mad = torch.clamp(mad, min=1e-6)
        z_score = (0.6745 * (energy_flat - median_energy) / mad).view(B, 1, H, W)
        return z_score

    def _save_detection_overlay(self, resized_img, detections: List[Detection], patch_boxes, is_attack, img_stem):
        """Draw detector boxes and adversarial patch boxes on the resized image and save."""
        yolo_vis = resized_img.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            label = f"{det.label} {det.score:.2f}"
            cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(yolo_vis, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.defense_enabled and is_attack:
            for (px1, py1, px2, py2) in patch_boxes:
                cv2.rectangle(yolo_vis, (px1, py1), (px2, py2), (0, 0, 255), 3)
                cv2.putText(yolo_vis, "PATCH", (px1, max(py1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.detection_output_dir.mkdir(parents=True, exist_ok=True)
        yolo_out_path = self.detection_output_dir / f"yolo_{img_stem}.jpg"
        cv2.imwrite(str(yolo_out_path), cv2.cvtColor(yolo_vis, cv2.COLOR_RGB2BGR))
        LOGGER.info(f"[Visualizer] Detection overlay saved to: {yolo_out_path}")

    def _save_heatmap_and_csv(self, z_map: np.ndarray, img_stem: str, feat: torch.tensor):
        """Render the Z-score map as a JET heatmap and export the raw values as CSV."""
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        max_val = float(np.percentile(z_map, 99.9))
        z_clipped = np.clip(z_map, 0, max_val)
        z_norm = (z_clipped / max_val * 255).astype(np.uint8)

        heatmap_resized = cv2.resize(z_norm, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
        colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        heatmap_out_path = self.heatmap_dir / f"heatmap_{img_stem}.jpg"
        cv2.imwrite(str(heatmap_out_path), colored_heatmap)
        LOGGER.info(f"[Visualizer] Heatmap saved to: {heatmap_out_path}")

        z_score_dir = self.heatmap_dir.parent / "z_score_maps"
        z_score_dir.mkdir(parents=True, exist_ok=True)
        z_score_path = z_score_dir / f"z_score_attacked{img_stem}.csv"
        np.savetxt(str(z_score_path), z_map, fmt="%.4f", delimiter=",")
        LOGGER.info(f"[Visualizer] Z-score map saved to: {z_score_path}")

        # Save raw feature map [B, C, H, W] as .npy
        # feat_dir = self.heatmap_dir.parent / "feature_maps"
        # feat_dir.mkdir(parents=True, exist_ok=True)
        # feat_path = feat_dir / f"feat_{img_stem}.npy"
        # np.save(str(feat_path), feat.detach().cpu().numpy())
        # LOGGER.info(f"[Visualizer] Feature map saved to: {feat_path}")

    def detect(self, img_path: str, save_visualization: bool = True):
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img_stem = Path(img_path).stem
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img_rgb, (self.input_width, self.input_height))

        model_input = self.backend.preprocess(resized_img, self.device)

        self.adapter.activations.clear()
        raw_output = self.backend.infer(self.model, model_input, target_classes=self.target_classes)
        detections = self.backend.parse_detections(raw_output, self._class_names)

        defense_stats = {}
        is_attack = False
        patch_boxes = []
        max_z_val = 0.0

        if self.defense_enabled:
            feat = self.adapter.activations[self.target_layer_name]
            if isinstance(feat, dict):
                raise TypeError(
                    f"Layer '{self.target_layer_name}' returned multiple feature maps "
                    f"({list(feat.keys())}), not a single tensor -- this happens if you "
                    f"point layer_name at a whole FPN (e.g. 'backbone' or 'backbone.fpn'). "
                    f"Point it at a single-output stage instead, e.g. 'backbone.body.layer1', "
                    f"or select one pyramid level explicitly, e.g. 'backbone.fpn.inner_blocks.0'."
                )
            z_score = self._compute_z_score_map(feat)
            z_map_2d = z_score[0, 0].detach().cpu().numpy()
            max_z_val = z_map_2d.max()

            stride = self.backend.compute_stride(self.input_width, self.input_height, feat.shape)
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
            self._save_detection_overlay(resized_img, detections, patch_boxes, is_attack, img_stem)
            if self.defense_enabled and defense_stats.get("z_score_map") is not None:
                self._save_heatmap_and_csv(defense_stats["z_score_map"], img_stem, feat)

        return {
            "image": Path(img_path).name,
            "is_attack": is_attack,
            "detections_count": len(detections),
            "defense_stats": {k: v for k, v in defense_stats.items() if k != "z_score_map"},
            "max_z_score": max_z_val,
        }

    def close(self):
        """Removes PyTorch forward hooks."""
        if hasattr(self, "adapter") and self.adapter:
            self.adapter.teardown()