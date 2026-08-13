import torch
import cv2
import json
import pickle
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from cls_head_adapter import YOLOv8ClsHeadAdapter

# 1. Load Configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

INPUT_SIZE = 640  # keep in sync with calibrate_cls_head.py


class ClsHeadMahalanobisDetector:
    def __init__(self, config_data=config):
        self.config = config_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(self.config["model_path"]).to(self.device)
        self.adapter = YOLOv8ClsHeadAdapter(self.model.model)
        self.threshold = self.config["detector_settings"]["threshold"]
        self.stride = self.config["detector_settings"]["stride"]

        # Load the offline calibration artifact
        profile_path = Path(self.config["profiles_dir"]) / self.config["calibration_profile_filename"]
        if not profile_path.exists():
            raise FileNotFoundError(f"Calibration profile not found at {profile_path}. Run calibrate_cls_head.py first.")

        with open(profile_path, 'rb') as f:
            self.stats = pickle.load(f)

        # Move stats to GPU for fast runtime matrix multiplication
        for cls_id in self.stats:
            self.stats[cls_id]['mean'] = self.stats[cls_id]['mean'].to(self.device)
            self.stats[cls_id]['inv_cov'] = self.stats[cls_id]['inv_cov'].to(self.device)

    def detect(self, img_path, save_visualization=True):
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # --- FIX 1: one tensor, shared by prediction and activation
        # extraction. Everything downstream (boxes, class grid, feat,
        # heatmap, visualization) now lives in this same 640x640 space --
        # there is only ever one coordinate system in play.
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img_rgb, (INPUT_SIZE, INPUT_SIZE))
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        results = self.model.predict(source=img_tensor, verbose=False)[0]
        print(f"size : {results.boxes.xyxy.max()}")
        activations = self.adapter.get_activations(img_tensor)
        feat = activations["P3"]  # [1, C, H, W]
        B, C, H, W = feat.shape

        # Class-assignment grid is sized to match feat exactly -- no more
        # separately-computed img_h/img_w // stride that could disagree
        # with feat's actual spatial dimensions.
        predicted_classes = torch.full((H, W), -1, dtype=torch.long, device=self.device)

        if len(results.boxes) > 0:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = (box / self.stride).int().tolist()
                x1 = min(max(x1, 0), W - 1)
                x2 = min(max(x2, 0), W)
                y1 = min(max(y1, 0), H - 1)
                y2 = min(max(y2, 0), H)
                if x2 > x1 and y2 > y1:
                    predicted_classes[y1:y2, x1:x2] = int(cls.item())

        feat_flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
        scores = torch.zeros(H, W, device=self.device)

        # Compute Mahalanobis distance strictly on regions with predicted objects
        for idx in range(H * W):
            y, x = divmod(idx, W)
            cls_id = predicted_classes[y, x].item()

            if cls_id in self.stats:
                mean = self.stats[cls_id]['mean']
                inv_cov = self.stats[cls_id]['inv_cov']
                diff = feat_flat[idx] - mean
                scores[y, x] = torch.sqrt(torch.matmul(torch.matmul(diff, inv_cov), diff.unsqueeze(-1))).squeeze()

        # Free VRAM
        torch.cuda.empty_cache()

        # Masking and extraction
        mask = (scores > self.threshold)
        is_attack = mask.any().item()
        score_val = scores.max().item()
        bounding_box = (0, 0, 0, 0)

        if is_attack:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Scale up from grid coordinates to pixel coordinates using
                # the stride -- these pixel coordinates are in the 640x640
                # space, matching the image this detector actually operated on.
                bounding_box = (x * self.stride, y * self.stride, (x + w) * self.stride, (y + h) * self.stride)

        # --- VISUALIZATION OVERLAY GENERATOR ---
        if save_visualization:
            scores_np = scores.cpu().numpy()

            # Normalize Mahalanobis distance scores to [0, 255] range for display
            scores_norm = cv2.normalize(scores_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Resize heatmap grid up to the 640x640 canvas -- the same
            # space the tensor/boxes/feat all share, so no separate
            # original-resolution tracking is needed anymore.
            heatmap_resized = cv2.resize(scores_norm, (INPUT_SIZE, INPUT_SIZE))
            colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

            # Blend 50% heatmap over the resized (640x640) image
            overlay = cv2.addWeighted(resized_img, 0.5, colored_heatmap, 0.5, 0)

            if is_attack and bounding_box != (0, 0, 0, 0):
                x1, y1, x2, y2 = bounding_box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(overlay, f"PATCH DETECTED (Score: {score_val:.1f})",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            output_dir = Path(self.config["detection_output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            out_visualization_path = output_dir / f"clean_detected_cls_head_{Path(img_path).name}"
            cv2.imwrite(str(out_visualization_path), overlay)
            print(f"\n[Visualizer] Heatmap saved to: {out_visualization_path}")

        return {
            "is_attack": is_attack,
            "score": float(score_val),
            "bounding_box": bounding_box
        }


if __name__ == "__main__":
    detector = ClsHeadMahalanobisDetector(config)

    PATCHED_DIR = Path(config["patched_data_dir"])

    if config["specific_test_image"]:
        test_image = str(PATCHED_DIR / config["specific_test_image"])
    else:
        patched_files = list(PATCHED_DIR.glob("*.jpg")) + list(PATCHED_DIR.glob("*.png"))
        patched_files = [f for f in patched_files if "detected_cls_head" not in f.name]
        test_image = str(patched_files[0]) if patched_files else None

    if test_image and Path(test_image).exists():
        print(f"Running detection on target image: {test_image}")
        result = detector.detect(test_image, save_visualization=True)
        print("\n--- Decoupled Head Middleware Output ---")
        print(f"Attack Detected: {result['is_attack']}")
        print(f"Anomaly Score:   {result['score']:.2f}")
        print(f"Bounding Box:    {result['bounding_box']}")
    else:
        print(f"Target test image not found. Checked: {test_image}")