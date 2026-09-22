"""
Runtime anomaly detector: Decoupled-Head (cv3 classification branch)
Mahalanobis distance middleware -- OPTION B scoring.

WHAT CHANGED FROM THE BOX-GATED VERSION:
The old version built a `predicted_classes` grid entirely from
`results.boxes` (the detector's OWN, possibly-attacked output), and
only computed a Mahalanobis distance for cells inside a predicted box.
A successful hiding attack removes the box, which silently left the
hidden object's cells scored at exactly 0 -- the detector never even
looked there. See box-dependency-problem-and-solutions.md.

This version never looks at `results.boxes` to decide what to score.
Every cell in the P3 grid is compared against EVERY calibrated class
(including a background class -- see calibrate_cls_head.py), and the
MINIMUM distance across all of them becomes that cell's score: "how
well does this cell's feature vector fit its best available
explanation, whatever that turns out to be." `results.boxes` is still
computed, but only for optional visual comparison in the heatmap
overlay, never for scoring.

NOTE ON THE THRESHOLD: the scoring rule fundamentally changed (min
distance across many classes, rather than a lookup against one
specific predicted class), so any threshold value tuned for the old
box-gated detector is not meaningful here. Re-derive it from a fresh
clean-image run before trusting `is_attack` output from this version.
"""

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

        profile_path = Path(self.config["profiles_dir"]) / self.config["calibration_profile_filename"]
        if not profile_path.exists():
            raise FileNotFoundError(f"Calibration profile not found at {profile_path}. Run calibrate_cls_head.py first.")

        with open(profile_path, 'rb') as f:
            self.stats = pickle.load(f)

        # Stack every class's (mean, inv_cov) -- including background --
        # into single tensors once, up front, so scoring at inference
        # time is a handful of batched matrix ops instead of a Python
        # loop over every (cell, class) pair.
        self.class_ids = list(self.stats.keys())
        self.means = torch.stack([self.stats[c]['mean'] for c in self.class_ids]).to(self.device)        # [K, C]
        self.inv_covs = torch.stack([self.stats[c]['inv_cov'] for c in self.class_ids]).to(self.device)   # [K, C, C]
        print(f"Loaded {len(self.class_ids)} calibrated distributions "
              f"(including background) for Option B scoring.")

    def _min_mahalanobis_scores(self, feat_flat):
        """feat_flat: [N, C] -- one row per grid cell, N = H*W.
        Returns scores: [N], the MINIMUM Mahalanobis distance across
        ALL calibrated classes for each cell. The loop here is over K
        classes (typically a handful to a few dozen), never over N
        cells individually -- each iteration scores every cell in the
        image against one class in a single batched operation."""
        N = feat_flat.shape[0]
        K = self.means.shape[0]
        best = torch.full((N,), float('inf'), device=feat_flat.device)

        for k in range(K):
            diff = feat_flat - self.means[k]                                  # [N, C]
            maha_sq = torch.einsum('nc,cd,nd->n', diff, self.inv_covs[k], diff)  # [N]
            dist = torch.sqrt(torch.clamp(maha_sq, min=0))
            best = torch.minimum(best, dist)

        return best

    def detect(self, img_path, save_visualization=True):
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img_rgb, (INPUT_SIZE, INPUT_SIZE))
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        # Still computed -- but ONLY for optional visual comparison
        # below (drawing what the detector itself currently reports).
        # Not used anywhere in the scoring path.
        results = self.model.predict(source=img_tensor, verbose=False)[0]

        activations = self.adapter.get_activations(img_tensor)
        feat = activations["P3"]  # [1, C, H, W]
        B, C, H, W = feat.shape

        feat_flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
        scores_flat = self._min_mahalanobis_scores(feat_flat)
        scores = scores_flat.reshape(H, W)

        torch.cuda.empty_cache()

        mask = (scores > self.threshold)
        is_attack = mask.any().item()
        score_val = scores.max().item()
        bounding_box = (0, 0, 0, 0)

        if is_attack:
            # --- Tighter box extraction, decoupled from the detection
            # decision above -- so tightening the box never costs you
            # detection sensitivity. Two knobs:
            #  1. A stricter "core" threshold just for drawing the box:
            #     cells only moderately above `threshold` are often the
            #     receptive-field halo around the true patch, not the
            #     patch itself.
            #  2. Light morphological erosion on the resulting mask, to
            #     trim a further cell or two of that halo from each edge.
            box_multiplier = self.config["detector_settings"].get("box_core_threshold_multiplier", 1.3)
            erosion_cells = self.config["detector_settings"].get("box_erosion_cells", 1)

            core_mask = (scores > self.threshold * box_multiplier)
            core_mask_np = core_mask.cpu().numpy().astype(np.uint8) * 255

            if erosion_cells > 0:
                kernel = np.ones((erosion_cells * 2 + 1, erosion_cells * 2 + 1), np.uint8)
                core_mask_np = cv2.erode(core_mask_np, kernel, iterations=1)

            contours, _ = cv2.findContours(core_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # Stricter mask eroded away entirely -- fall back to the
                # original (looser) mask so we still report a box rather
                # than none, at the cost of it being less tight this time.
                fallback_np = mask.cpu().numpy().astype(np.uint8) * 255
                contours, _ = cv2.findContours(fallback_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bounding_box = (x * self.stride, y * self.stride, (x + w) * self.stride, (y + h) * self.stride)

        if save_visualization:
            scores_np = scores.cpu().numpy()
            scores_norm = cv2.normalize(scores_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_resized = cv2.resize(scores_norm, (INPUT_SIZE, INPUT_SIZE))
            colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(resized_img, 0.5, colored_heatmap, 0.5, 0)

            # Purely informational: draw what the (possibly-fooled)
            # detector itself currently reports, in green, so you can
            # visually compare "what YOLO sees" against the heatmap
            # underneath -- this is exactly how you'll SEE the box
            # missing over a successfully hidden object, while the
            # heatmap underneath still lights up there.
            if len(results.boxes) > 0:
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = box.int().tolist()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if is_attack and bounding_box != (0, 0, 0, 0):
                x1, y1, x2, y2 = bounding_box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(overlay, f"PATCH DETECTED (Score: {score_val:.1f})",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            output_dir = Path(self.config["detection_output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            out_visualization_path = output_dir / f"optionB_detected_cls_head_{Path(img_path).name}"
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
        print("\n--- Decoupled Head Middleware Output (Option B) ---")
        print(f"Attack Detected: {result['is_attack']}")
        print(f"Anomaly Score:   {result['score']:.2f}")
        print(f"Bounding Box:    {result['bounding_box']}")
    else:
        print(f"Target test image not found. Checked: {test_image}")