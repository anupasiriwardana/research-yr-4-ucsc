# Mechanism 1 — Option B (Box-Independent Detection) — README

Covers both `cls_head_detector.py` (base Option B) and `cls_head_detector_for_tightPatch.py` (Option B with a tightened bounding box), which share `cls_head_adapter.py`, `calibrate_cls_head.py`, and this `config.json`.

For the reasoning behind each design decision, see:
- `mech-1-updated.md` — the box-dependency problem and why Option B's minimum-Mahalanobis scoring solves it.
- `mech-1-updated-for-tightPatchBox.md` — the oversized-box problem and the moment-based tightening solution.

---

## 1. Architecture

```
Input image (clean or patched)
        │
        ▼
YOLOv8ClsHeadAdapter (cls_head_adapter.py)
  -- forward hook on cv3[-2] (classification branch,
     penultimate conv block, P3 scale, stride 8)
        │
        ▼
Offline calibration (calibrate_cls_head.py)          Runtime detection (cls_head_detector.py /
  -- fits per-class Mahalanobis stats                    cls_head_detector_for_tightPatch.py)
     (mean, inv_cov) from clean images                -- scores EVERY grid cell against ALL
  -- ALSO fits a background class from                   calibrated classes (+ background),
     cells outside any detected box                      takes the minimum distance
        │                                                     │
        └──────────────► cls_head_calibration_p3_optionB.pkl ◄┘
                                                                │
                                                                ▼
                                                     AnomalyResult-style output:
                                                     is_attack, score, bounding_box
```

**Why two detector scripts, not one:** `cls_head_detector.py` is the base Option B scoring with a simple contour-derived box (larger than the actual patch — see the tightening doc for why). `cls_head_detector_for_tightPatch.py` runs the *identical* detection scoring and only replaces the final box-extraction step with a moment-based (weighted centroid + spread) box. Keeping them separate makes it easy to demonstrate/compare the effect of the tightening step in isolation, and either can be used standalone.

**What stayed constant throughout this whole redesign:** both scripts still tap only the classification branch (`cv3`), never the neck or box-regression branch (`cv2`) — this separation from box-regression signal was the original reason Mechanism 1 exists, and nothing in the Option B or tightening changes touches that.

---

## 2. Code Files

| File | Role |
|---|---|
| `cls_head_adapter.py` | Model-agnostic adapter — registers a forward hook on the YOLOv8 classification branch and returns its activations. Unchanged since the original Mechanism 1 design. |
| `calibrate_cls_head.py` | Offline script. Runs over a clean image directory, fits per-class Mahalanobis statistics (mean, inverse covariance) plus a background class, and saves the result to a `.pkl` profile. Run once before either detector script. |
| `cls_head_detector.py` | Runtime detector, base Option B: scores every grid cell via minimum Mahalanobis distance across all calibrated classes (no box-gating), simple contour bounding box. |
| `cls_head_detector_for_tightPatch.py` | Same detection scoring as above; the bounding box is instead derived from the weighted centroid/spread of the detected region's score mass, giving a tighter, symmetric box more suitable for downstream recovery/masking. |
| `config.json` | Shared configuration for calibration and both detector scripts (see below). |

---

## 3. Configuration Reference

```json
{
    "model_path": "yolov8n.pt",
    "clean_data_dir": "...",
    "profiles_dir": "...",
    "calibration_profile_filename": "cls_head_calibration_p3_optionB.pkl",
    "patched_data_dir": "...",
    "detection_output_dir": "...",
    "specific_test_image": "...",
    "detector_settings": {
        "threshold": 25.0,
        "stride": 8,
        "min_calibration_samples": 20,
        "background_class_id": -1,
        "background_samples_per_image": 50,
        "box_spread_multiplier": 1.5
    }
}
```

| Key | Used by | Meaning |
|---|---|---|
| `model_path` | calibration, both detectors | YOLOv8 checkpoint to load (e.g. `yolov8n.pt`). Must be the exact model/weights being protected — a mismatch here silently invalidates calibration. |
| `clean_data_dir` | calibration | Directory of clean (unpatched) images used to fit the per-class and background Mahalanobis statistics. |
| `profiles_dir` | calibration, both detectors | Directory where the calibration `.pkl` is written/read. |
| `calibration_profile_filename` | calibration, both detectors | Filename of the calibration artifact. Kept distinct from any pre-Option-B profile name (`..._optionB.pkl`) deliberately — an old profile has no background class and is not compatible with either detector script here; mixing them up fails silently rather than raising an error. |
| `patched_data_dir` | both detectors | Directory the detector scripts look in by default for a test image (see `specific_test_image` below). |
| `detection_output_dir` | both detectors | Where heatmap visualization overlays are saved. |
| `specific_test_image` | both detectors | If set, this exact file (inside `patched_data_dir`) is used as the detection target. If empty, the script falls back to the first `.jpg`/`.png` found in that directory. |
| `detector_settings.threshold` | both detectors | The anomaly-score cutoff for the `is_attack` decision. **Must be re-derived whenever the calibration profile or scoring rule changes** — a threshold tuned for one scoring rule (e.g. the pre-Option-B box-gated version) is not meaningful for another. See the statistical thresholding discussion referenced in `mech-1-optionB.md` for how to derive this from a clean-image validation run rather than guessing a constant. |
| `detector_settings.stride` | calibration, both detectors | The P3 feature map's stride (8 for a 640×640 input with YOLOv8's default architecture) — used to map between grid coordinates and pixel coordinates. |
| `detector_settings.min_calibration_samples` | calibration | Minimum number of collected feature vectors required before a class (or background) is included in the fitted profile. Classes/background with fewer samples than this are skipped, with a warning printed during calibration. |
| `detector_settings.background_class_id` | calibration, both detectors | The sentinel class id used to store/reference the background distribution (default `-1`). Must match between calibration and detection — it's just a dictionary key, not a real COCO class id. |
| `detector_settings.background_samples_per_image` | calibration | Caps how many non-object grid cells are randomly sampled as background candidates per calibration image. Exists specifically to prevent background data (which vastly outnumbers object-class data in any typical image) from ballooning calibration memory usage across a large dataset. |
| `detector_settings.box_spread_multiplier` | `cls_head_detector_tight.py` only | Controls how far the tightened box extends from the weighted centroid, in units of the score distribution's own standard deviation (roughly, "how many std-devs of score mass to include on each side"). Not used by the base `cls_head_detector.py`, which uses a plain contour box instead. |

---

## 4. Environment Setup

This module runs in the primary research environment (`yolo_adv`).

```bash
#create the enviorenment
conda deactivate
conda create -n yolo_adv=3.10 -y
conda activate yolo_adv

# Core dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y   # match your CUDA version if using GPU
pip install ultralytics opencv-python numpy
```

A CUDA-capable GPU is used automatically if available (`torch.cuda.is_available()`); both scripts fall back to CPU otherwise, at a significant speed cost for calibration in particular (which runs the model over every image in `clean_data_dir`).

No dependency on ART (Adversarial Robustness Toolbox) is required for anything in this mech-1 folder — that's only used by the separate patch-generation scripts (`gen-adv-patch/`).

---

## 5. Execution Guide

**Step 1 — Calibrate (run once, or whenever `clean_data_dir` or `model_path` changes):**

```bash
python calibrate_cls_head.py
```

This populates `profiles_dir/cls_head_calibration_p3_optionB.pkl`. Watch the console output — it prints the sample count per class (and for background), and warns if any class fell below `min_calibration_samples` and was skipped.

**Step 2 — Re-derive `threshold` (recommended before trusting results):**

Run the detector across a batch of held-out clean images and inspect the resulting score distribution before deciding on a threshold value in `config.json` — don't reuse a threshold from a different scoring rule or profile. See `mech-1-optionB.md` for the reasoning.

**Step 3 — Run detection:**

```bash
# Base Option B (simple box)
python cls_head_detector.py

# Option B with tightened box (recommended for recovery-facing use)
python cls_head_detector_for_tightPatch.py
```

Either prints a summary to the console:

```
--- Decoupled Head Middleware Output (Option B) ---
Attack Detected: True
Anomaly Score:   46.15
Bounding Box:    (144, 288, 328, 488)
```

and saves a heatmap overlay (with YOLO's own current detection boxes drawn in green for comparison, and the anomaly bounding box in red if an attack was flagged) to `detection_output_dir`.

**Validating against a hiding attack specifically:** open the saved overlay for an image where the patch successfully hid its target object. The green box (YOLO's own output) should be missing over that object, while the heatmap underneath should still show an elevated, red-boxed region there — that combination confirms the box-independence fix is working as intended.