# Mechanism 4 — Self-Referential Feature-Energy Detection — README

Covers `stem_activation_adapter.py` and `feature_energy_detector.py`, sharing `config.json`.

For the reasoning behind this mechanism and the paper it's adapted from, see `mech-4-feature-energy.md`.

---

## 1. Architecture

```
Input image (clean or patched)
        │
        ▼
YOLOv8StemAdapter (stem_activation_adapter.py)
  -- forward hook on the backbone's first conv block(s)
     (before PANet fusion, before the cls/box head split)
        │
        ▼
FeatureEnergyDetector (feature_energy_detector.py)
  -- computes per-cell feature energy from THIS image alone
  -- derives threshold = mean + k*std from THIS image's own
     energy map -- no offline calibration, no persisted profile
        │
        ▼
Output: is_attack, score, bounding_box
```

**No calibration script exists for this mechanism, and that's intentional** — unlike Mechanism 1, there's no reference dataset to collect or profile to fit. Every run is self-contained. See `mech-4-feature-energy.md` Section 1 for why this matters for deployment specifically.

**Adapter/detector separation, and why it matters for reuse:** `FeatureEnergyDetector` never imports `ultralytics` and has no YOLO-specific logic anywhere in it — it only calls `adapter.get_activations(image)` and expects a `{"stem": Tensor[1,C,H,W]}` dict back. To use this exact detection logic with a different model or YOLO version later, write a new adapter class (or extend `STEM_LAYER_INDEX`) with that same interface; `feature_energy_detector.py` itself never needs to change.

---

## 2. Code Files

| File | Role |
|---|---|
| `stem_activation_adapter.py` | The only model-specific file. Registers a forward hook on the backbone's stem layer and returns its activations under the `{"stem": ...}` key contract. |
| `feature_energy_detector.py` | Model-agnostic detection logic: energy computation, per-image adaptive thresholding, connected-component noise filtering, heatmap visualization. Also contains the runnable `__main__` entry point. |
| `config.json` | Shared configuration (see below). |

---

## 3. Environment Setup

No new environment is needed — this mechanism uses exactly the same dependencies as Mechanism 1 (`torch`, `opencv-python`, `ultralytics`, `numpy`). Reuse your existing `yolo_adv` conda environment:

```bash
conda activate yolo_adv
```

If you're setting this up fresh for some reason, the same install steps as Mechanism 1's README apply — nothing extra to add.

---

## 4. Configuration Reference

```json
{
    "model_path": "yolov8n.pt",
    "model_version_key": "yolov8",
    "patched_data_dir": "...",
    "detection_output_dir": "...",
    "specific_test_image": "",
    "detector_settings": {
        "stride": 2,
        "k": 3.5,
        "min_region_cells": 4
    }
}
```

| Key | Meaning |
|---|---|
| `model_path` | YOLOv8 checkpoint to load. |
| `model_version_key` | Looks up the stem layer index in `stem_activation_adapter.py`'s `STEM_LAYER_INDEX` dict. Add a new entry there for a different model/version rather than hardcoding a new index inline. |
| `patched_data_dir` | Directory the detector looks in by default for a test image. |
| `detection_output_dir` | Where heatmap overlays are saved. |
| `specific_test_image` | If set, this exact file (inside `patched_data_dir`) is used. If empty, falls back to the first `.jpg`/`.png` found. |
| `detector_settings.stride` | The tapped stem layer's downsampling factor relative to the 640×640 input (e.g. `2` if the stem is a single stride-2 conv). **Must be verified against your actual checkpoint** — see Step 1 below — since an incorrect value silently produces a wrong pixel-space bounding box (it's only used for the final grid→pixel scaling, not for detection itself, so a wrong value won't crash anything, it'll just misplace the reported box). |
| `detector_settings.k` | The threshold multiplier (`mean + k·std`). Starts at `3.5`, following the reconstructed APE formulation — tune empirically per `new-mech-4-detection-process.md`'s guidance, since this hasn't been validated against the primary paper's exact value. |
| `detector_settings.min_region_cells` | Minimum contiguous flagged-cell area (in grid units) before a region counts as a detection, filtering single-pixel noise. |

---

## 5. Execution Guide

**Step 1 — Verify the stem layer index and stride (one-time, per model/version):**

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print(model.model.model[0])   # inspect the layer STEM_LAYER_INDEX["yolov8"] points to

import torch
x = torch.zeros(1, 3, 640, 640)
out = model.model.model[0](x)
print(out.shape)   # e.g. [1, C, 320, 320] -> stride = 640/320 = 2
```

Update `STEM_LAYER_INDEX` in `stem_activation_adapter.py` and `detector_settings.stride` in `config.json` to match what you actually find — the values shipped here are starting assumptions, not guarantees for every checkpoint.

**Step 2 — Run detection directly (no calibration step):**

```bash
python feature_energy_detector.py
```

```
--- Feature-Energy Middleware Output ---
Attack Detected: True
Energy Score:    812.44
Bounding Box:    (146, 292, 330, 486)
```

A heatmap overlay is saved to `detection_output_dir`, following the same visual convention as Mechanism 1 (JET colormap, red box if an attack was flagged).

**Step 3 — Tune `k` and `min_region_cells` against your actual data:**

Run across a handful of clean images first and confirm nothing (or only small, filtered-out noise) gets flagged, before moving to patched images. Since there's no calibration step to catch a misconfiguration early the way Mechanism 1's calibration console output does, this manual clean-image sanity pass is the equivalent check here.

**Step 4 — Compare against Mechanism 1 on the same test images.** Since this mechanism is meant to complement, not replace, Mechanism 1's (fixed) Option B detector, running both on identical clean and patched images and comparing where they agree or diverge is the natural next validation step — see `new-mech-4-detection-process.md` Section 4.