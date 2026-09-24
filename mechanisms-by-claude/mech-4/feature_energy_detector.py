"""
Mechanism 4: Adversarial Patch-Feature Energy (APE-masking), matching
Kim, Yu & Ro (2022) -- see mech-4-feature-energy.md for the full
explanation and citation, and calibrate_energy_stats.py for why this
now requires a calibration step after all.

WHAT CHANGED FROM THE ORIGINAL VERSION:
  1. Energy formula corrected: (L1 norm across channels)^2, not the
     sum-of-squares (squared L2 norm) used before.
  2. Threshold is now loaded from an OFFLINE calibration profile
     (mean/std of energy from a clean dataset, per layer) instead of
     being recomputed from each test image's own statistics. This is
     the fix for k needing to change per image -- a single image's
     own local statistics aren't a stable reference; a dataset-wide
     one is.
  3. Supports MULTIPLE tap layers at once, matching the paper's
     mask-aggregation design: each layer's partial proposal (computed
     against ITS OWN calibrated threshold) is upsampled via nearest-
     neighbor interpolation ("spatial unpooling") to a common
     resolution and summed into one aggregated mask.

Still model-agnostic in the same sense as before: this file never
imports ultralytics directly and only consumes whatever dict of
{layer_idx: Tensor} activations the adapter provides.
"""

import json
import pickle
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path


class FeatureEnergyDetector:
    def __init__(self, adapter, device, tap_layers: list, stride_per_layer: dict,
                 calibration_stats: dict, sigma_multiplier: float = 3.5,
                 min_region_cells: int = 4, input_size: int = 640,
                 detection_output_dir: str = None):
        self.adapter = adapter
        self.device = device
        self.tap_layers = tap_layers
        self.stride_per_layer = stride_per_layer          # {layer_idx: stride}
        self.stats = calibration_stats                     # {layer_idx: {'mean', 'std'}}, from calibrate_energy_stats.py
        self.sigma_multiplier = sigma_multiplier
        self.min_region_cells = min_region_cells
        self.input_size = input_size
        self.detection_output_dir = Path(detection_output_dir) if detection_output_dir else None

        # Aggregate at the resolution of the SHALLOWEST tapped layer
        # (highest spatial resolution) -- matches the paper's "upsample
        # the coarser, deeper proposals up to the shallower layers'
        # resolution," not the other way around.
        self.base_layer = min(self.tap_layers, key=lambda l: self.stride_per_layer[l])
        self.base_stride = self.stride_per_layer[self.base_layer]
        self.base_size = self.input_size // self.base_stride

    def _compute_energy_map(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [1, C, H, W]. Energy = (L1 norm across channels)^2 at
        each spatial location -- the paper's definition, not a
        sum-of-squares."""
        l1_norm = feat[0].abs().sum(dim=0)  # [H, W]
        return l1_norm ** 2

    def _partial_proposal(self, energy_map: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Binary mask for ONE layer: 1 where energy exceeds that
        layer's OWN calibrated threshold (mean + sigma_multiplier*std,
        both fixed, loaded from calibration -- never recomputed from
        the current image)."""
        mean = self.stats[layer_idx]["mean"]
        std = self.stats[layer_idx]["std"]
        threshold = mean + self.sigma_multiplier * std
        return (energy_map > threshold).float()

    def detect(self, img_path, save_visualization=True):
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img_rgb, (self.input_size, self.input_size))
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        activations = self.adapter.get_activations(img_tensor)

        # --- Per-layer partial proposals, spatially unpooled to a
        # common resolution, then summed -- the paper's aggregation. ---
        aggregated_mask = torch.zeros((self.base_size, self.base_size), device=self.device)
        for layer_idx in self.tap_layers:
            feat = activations[layer_idx]
            energy_map = self._compute_energy_map(feat)
            proposal = self._partial_proposal(energy_map, layer_idx)  # [H_l, W_l]

            if proposal.shape[0] != self.base_size:
                proposal = F.interpolate(
                    proposal.unsqueeze(0).unsqueeze(0),
                    size=(self.base_size, self.base_size), mode="nearest",
                ).squeeze(0).squeeze(0)

            aggregated_mask += proposal

        # Continuous APE-mask -> strict binary mask
        mask = (aggregated_mask > 0)
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255

        # Filter isolated single-cell noise before treating anything as
        # a genuine detection.
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [c for c in contours if cv2.contourArea(c) >= self.min_region_cells]

        is_attack = len(significant) > 0
        score_val = aggregated_mask.max().item()  # how many layers agreed at the worst cell
        bounding_box = (0, 0, 0, 0)

        if is_attack:
            largest_contour = max(significant, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bounding_box = (x * self.base_stride, y * self.base_stride,
                             (x + w) * self.base_stride, (y + h) * self.base_stride)

        if save_visualization:
            agg_np = aggregated_mask.cpu().numpy()
            agg_norm = cv2.normalize(agg_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_resized = cv2.resize(agg_norm, (self.input_size, self.input_size))
            colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(resized_img, 0.5, colored_heatmap, 0.5, 0)

            if is_attack:
                x1, y1, x2, y2 = bounding_box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(overlay, f"PATCH DETECTED (layers agreeing: {int(score_val)})",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if self.detection_output_dir:
                self.detection_output_dir.mkdir(parents=True, exist_ok=True)
                out_path = self.detection_output_dir / f"135-clean-mech4_{Path(img_path).name}"
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), overlay_bgr)
                print(f"[Visualizer] Heatmap saved to: {out_path}")

        return {"is_attack": is_attack, "score": float(score_val), "bounding_box": bounding_box}


if __name__ == "__main__":
    from ultralytics import YOLO
    from stem_activation_adapter import YOLOv8StemAdapter

    CONFIG_PATH = Path(__file__).parent / "config.json"
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    ds = config["detector_settings"]
    tap_layers = ds["tap_layers"]
    stride_per_layer = {int(k): v for k, v in ds["stride_per_layer"].items()}

    profile_path = Path(config["profiles_dir"]) / config["calibration_profile_filename"]
    if not profile_path.exists():
        raise FileNotFoundError(f"Calibration profile not found at {profile_path}. Run calibrate_energy_stats.py first.")
    with open(profile_path, "rb") as f:
        calibration_stats = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(config["model_path"]).to(device)
    model.model.eval()

    adapter = YOLOv8StemAdapter(model.model, layer_indices=tap_layers)

    detector = FeatureEnergyDetector(
        adapter=adapter,
        device=device,
        tap_layers=tap_layers,
        stride_per_layer=stride_per_layer,
        calibration_stats=calibration_stats,
        sigma_multiplier=ds.get("sigma_multiplier", 3.5),
        min_region_cells=ds.get("min_region_cells", 4),
        detection_output_dir=config["detection_output_dir"],
    )

    PATCHED_DIR = Path(config["patched_data_dir"])
    CLEAN_DIR = Path(config["clean_data_dir"])
    if config["specific_test_image"]:
        test_image = str(CLEAN_DIR / config["specific_test_image"])
    else:
        patched_files = list(PATCHED_DIR.glob("*.jpg")) + list(PATCHED_DIR.glob("*.png"))
        patched_files = [f for f in patched_files if "mech4_detected" not in f.name]
        test_image = str(patched_files[0]) if patched_files else None

    if test_image and Path(test_image).exists():
        print(f"Running Mechanism 4 (feature-energy) detection on: {test_image}")
        result = detector.detect(test_image, save_visualization=True)
        print("\n--- Feature-Energy Middleware Output ---")
        print(f"Attack Detected: {result['is_attack']}")
        print(f"Score (layers agreeing at worst cell): {result['score']:.2f}")
        print(f"Bounding Box:    {result['bounding_box']}")
    else:
        print(f"Target test image not found. Checked: {test_image}")