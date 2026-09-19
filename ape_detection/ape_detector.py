import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import json
import torch
from ultralytics import YOLO
from yolo_feature_extractor import YOLOFeatureExtractor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("ApeDetector")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ApeDetector:
    def __init__(self, config_data: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        if config_data is not None:
            self.config = config_data
        else:
            cfg_file = Path(config_path) if config_path else Path(__file__).parent / "config.json"
            with open(cfg_file, "r") as f:
                self.config = json.load(f)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # Image Dimensions
        input_size = self.config.get("input_size", [640, 384])
        self.input_width = input_size[0]
        self.input_height = input_size[1]

        # Test directory
        self.test_images_dir = Path(self.config.get("test_images_dir", "data/test_images"))

        # Read target classes from config (defaults to None, meaning detect all)
        self.target_classes = self.config.get("target_classes", None)
        
        self.model = YOLO(self.config["model_path"]).to(self.device)
        
        defense_cfg = self.config.get("defense_settings", {})
        self.defense_enabled = defense_cfg.get("enabled", True)
        self.anomaly_tau = defense_cfg.get("anomaly_tau", 4.0)
        self.defense_layer_idx = defense_cfg.get("layer_index", 16)
        self.detection_output_dir = Path(self.config.get("detection_output_dir", "runs/detect"))
        self.heatmap_dir = Path(defense_cfg.get("heatmap_dir", "runs/defense_heatmaps"))
        
        # Initialize the feature extractor
        self.target_layer_name = f"model.{self.defense_layer_idx}"
        self.adapter = YOLOFeatureExtractor(self.model, layers=[self.target_layer_name])

    def detect(self, img_path: str, save_visualization: bool = True):
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img_stem = Path(img_path).stem

        # 1. Preprocessing
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img_rgb, (self.input_width, self.input_height))
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        # 2. Forward Pass (YOLO runs at full speed, Adapter silently copies Layer 16)
        self.adapter.activations.clear()
        results = self.model.predict(
            source=img_tensor, 
            classes=self.target_classes, 
            verbose=False
        )[0]
        
        defense_stats = {}
        is_attack = False

        # 3. Post-Inference Anomaly Detection
        if self.defense_enabled:
            # Grab the detached Layer features captured by the adapter
            feat = self.adapter.activations[self.target_layer_name]  # [B, C, H, W]
            B, C, H, W = feat.shape

            # Compute Math (Now it doesn't block the YOLO forward pass)
            energy = torch.sum(feat ** 2, dim=1, keepdim=True)
            energy_flat = energy.view(B, -1)

            median_energy = energy_flat.median(dim=1, keepdim=True)[0]
            abs_dev = torch.abs(energy_flat - median_energy)
            mad = abs_dev.median(dim=1, keepdim=True)[0]
            mad = torch.clamp(mad, min=1e-6)

            z_score = (0.6745 * (energy_flat - median_energy) / mad).view(B, 1, H, W)
            mask = z_score > self.anomaly_tau
            
            is_attack = mask.any().item()

            if is_attack:
                max_z = z_score.max().item()
                anomalous_pixels = mask.sum().item()
                LOGGER.warning(f"⚠️ [Defense] Adversarial patch detected! {anomalous_pixels} extreme pixels found (Max Z: {max_z:.2f}).")

            defense_stats = {
                "patch_detected": is_attack,
                "z_score_map": z_score.detach().cpu().numpy() if save_visualization else None
            }

        # 4. Visualization
        if save_visualization:
            # -------------------------------------------------------------
            # 4.1 Save YOLO Detection Output (Boxes + Labels on Clean Image)
            # -------------------------------------------------------------
            yolo_vis = resized_img.copy()
            if len(results.boxes) > 0:
                for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                    x1, y1, x2, y2 = box.int().tolist()
                    label = f"{self.model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(yolo_vis, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            yolo_out_path = self.detection_output_dir / f"yolo_{img_stem}.jpg"
            cv2.imwrite(str(yolo_out_path), cv2.cvtColor(yolo_vis, cv2.COLOR_RGB2BGR))
            LOGGER.info(f"[Visualizer] YOLO detection saved to: {yolo_out_path}")

            # -------------------------------------------------------------
            # 4.2 Save Anomaly Heatmap Output (Z-Score Map)
            # -------------------------------------------------------------
            if self.defense_enabled and defense_stats.get("z_score_map") is not None:
                self.heatmap_dir.mkdir(parents=True, exist_ok=True)
                
                z_map = defense_stats["z_score_map"][0, 0]
                z_norm = cv2.normalize(z_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Use INTER_NEAREST to maintain the distinct feature grid blocks
                heatmap_resized = cv2.resize(z_norm, (self.input_width, self.input_height))
                
                # applyColorMap outputs BGR directly
                colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                heatmap_out_path = self.heatmap_dir / f"heatmap_{img_stem}.jpg"
                
                # Save directly without converting, as it is already in BGR format
                cv2.imwrite(str(heatmap_out_path), colored_heatmap)
                LOGGER.info(f"[Visualizer] Heatmap saved to: {heatmap_out_path}")

        return {
            "image": Path(img_path).name,
            "is_attack": is_attack,
            "detections_count": len(results.boxes),
            "defense_stats": {k: v for k, v in defense_stats.items() if k != "z_score_map"},
        }

    def detect_folder(self, folder_path: Optional[str] = None, save_visualization: bool = True) -> List[Dict[str, Any]]:
        """Processes all supported images located in the specified folder or self.test_images_dir."""
        target_dir = Path(folder_path) if folder_path else self.test_images_dir
        if not target_dir.exists() or not target_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {target_dir}")

        image_files = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
        LOGGER.info(f"📂 Processing {len(image_files)} images from: {target_dir}")

        all_results = []
        for img_path in image_files:
            res = self.detect(str(img_path), save_visualization=save_visualization)
            all_results.append(res)

        return all_results

    def close(self):
        """Removes PyTorch forward hooks."""
        if hasattr(self, "adapter") and self.adapter:
            self.adapter.teardown()