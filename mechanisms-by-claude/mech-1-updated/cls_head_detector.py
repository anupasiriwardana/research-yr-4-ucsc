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

NOTE ON THE BOUNDING BOX: this script reports a simple contour-derived
box straight from the thresholded score mask -- it tends to be
noticeably larger than the actual patch (a "halo" from the tapped
layer's receptive field, plus grid quantization). See
cls_head_detector_tight.py for a tightened version of the box
(same detection scoring, different box-extraction step), and
mech-1-optionB-tightPatchBox.md for why the halo happens and how the
tightening works.
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
        """Standalone constructor: loads its OWN model/adapter from
        config. Used when Mechanism 1 runs alone. For combined mode
        (sharing a model/forward pass with Mechanism 4), build via
        from_stats() instead -- see run_detection.py."""
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
            stats = pickle.load(f)
        self._load_stats(stats)

    @classmethod
    def from_stats(cls, stats: dict, device, threshold: float, stride: int,
                    detection_output_dir: str):
        """Lightweight constructor for combined mode: no model/adapter
        of its own -- activations are supplied externally via
        score_from_activations(), from a shared forward pass."""
        self = cls.__new__(cls)
        self.config = {"detection_output_dir": detection_output_dir}
        self.device = device
        self.model = None
        self.adapter = None
        self.threshold = threshold
        self.stride = stride
        self._load_stats(stats)
        return self

    def _load_stats(self, stats: dict):
        self.stats = stats
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

    def score_from_activations(self, cls_head_activation, resized_img, results,
                                img_name="output", save_visualization=True):
        """Pure scoring step: takes an ALREADY-COMPUTED cls_head
        activation tensor (plus the resized image and model results,
        needed only for the visualization overlay) and returns the
        detection result. Has no dependency on how the activation was
        obtained -- this is what lets Mechanism 1 be scored from a
        shared, combined forward pass (see yolov8_combined_adapter.py)
        as easily as from its own standalone one.
        """
        feat = cls_head_activation  # [1, C, H, W]
        B, C, H, W = feat.shape

        feat_flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
        scores_flat = self._min_mahalanobis_scores(feat_flat)
        scores = scores_flat.reshape(H, W)

        torch.cuda.empty_cache()

        mask = (scores > self.threshold)

        # Report EVERY separate connected component as its own box, not
        # just the single largest one. With multiple, separately-placed
        # patches on different objects, each one produces its own
        # distinct anomalous region -- discarding all but the biggest
        # silently drops every additional patch from the output, which
        # is exactly the "only one patch detected" symptom this fixes.
        # Uses exact pixel-count area (connectedComponentsWithStats),
        # not cv2.findContours + contourArea's polygon approximation,
        # for the same reason this was already fixed in Mechanism 4.
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        num_labels, labels_im, stats_cc, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        min_region_cells = self.config.get("detector_settings", {}).get("min_region_cells", 1)

        bounding_boxes = []
        labels_im_t = torch.from_numpy(labels_im).to(scores.device)
        for i in range(1, num_labels):  # label 0 is background
            if stats_cc[i, cv2.CC_STAT_AREA] < min_region_cells:
                continue
            x = int(stats_cc[i, cv2.CC_STAT_LEFT])
            y = int(stats_cc[i, cv2.CC_STAT_TOP])
            w = int(stats_cc[i, cv2.CC_STAT_WIDTH])
            h = int(stats_cc[i, cv2.CC_STAT_HEIGHT])
            region_score = scores[labels_im_t == i].max().item()
            bounding_boxes.append({
                "box": (x * self.stride, y * self.stride, (x + w) * self.stride, (y + h) * self.stride),
                "score": region_score,
            })

        # is_attack now reflects whether anything SURVIVED filtering,
        # not just whether any single cell crossed the threshold --
        # the same class of bug min_region_cells filtering can cause if
        # checked in the wrong order (see mech-4-feature-energy.md's
        # rolled-back min_layer_agreement section for the earlier case
        # of this).
        is_attack = len(bounding_boxes) > 0
        score_val = max((b["score"] for b in bounding_boxes), default=0.0)

        if save_visualization:
            scores_np = scores.cpu().numpy()
            scores_norm = cv2.normalize(scores_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_resized = cv2.resize(scores_norm, (INPUT_SIZE, INPUT_SIZE))
            colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(resized_img, 0.5, colored_heatmap, 0.5, 0)

            if results is not None and len(results.boxes) > 0:
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = box.int().tolist()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if is_attack:
                for entry in bounding_boxes:
                    x1, y1, x2, y2 = entry["box"]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(overlay, f"PATCH (Score: {entry['score']:.1f})",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            output_dir = Path(self.config["detection_output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            out_visualization_path = output_dir / f"16-clean-mech1_{img_name}"
            cv2.imwrite(str(out_visualization_path), overlay)
            print(f"\n[Visualizer] Heatmap saved to: {out_visualization_path}")

        return {
            "is_attack": is_attack,
            "score": float(score_val),
            "bounding_boxes": bounding_boxes,   # list of {"box": (x1,y1,x2,y2), "score": float} -- see note below
        }

    def detect(self, img_path, save_visualization=True):
        """Standalone entry point: loads the image itself, runs its
        OWN forward pass via its OWN adapter, then scores. Used when
        Mechanism 1 runs alone (see run_detection.py for the combined,
        shared-pass alternative)."""
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img_rgb, (INPUT_SIZE, INPUT_SIZE))
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        results = self.model.predict(source=img_tensor, verbose=False)[0]
        activations = self.adapter.get_activations(img_tensor)

        return self.score_from_activations(
            activations["P3"], resized_img, results,
            img_name=Path(img_path).name, save_visualization=save_visualization,
        )


if __name__ == "__main__":
    detector = ClsHeadMahalanobisDetector(config)

    PATCHED_DIR = Path(config["patched_data_dir"])
    CLEAN_DIR = Path(config["clean_data_dir"])

    if config["specific_test_image"]:
        test_image = str(CLEAN_DIR / config["specific_test_image"])
    else:
        patched_files = list(PATCHED_DIR.glob("*.jpg")) + list(PATCHED_DIR.glob("*.png"))
        patched_files = [f for f in patched_files if "detected_cls_head" not in f.name]
        test_image = str(patched_files[0]) if patched_files else None

    if test_image and Path(test_image).exists():
        print(f"Running detection on target image: {test_image}")
        result = detector.detect(test_image, save_visualization=True)
        print("\n--- Decoupled Head Middleware Output (Option B) ---")
        print(f"Attack Detected: {result['is_attack']}")
        print(f"Max Score:       {result['score']:.2f}")
        print(f"Patches found:   {len(result['bounding_boxes'])}")
        for i, entry in enumerate(result["bounding_boxes"]):
            print(f"  [{i+1}] box={entry['box']}  score={entry['score']:.2f}")
    else:
        print(f"Target test image not found. Checked: {test_image}")