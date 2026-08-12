# Mechanism 1: Decoupled-Head Classification-Branch Tap

## Module Overview
Mechanism 1 implements Module 1 (Anomaly Detection) of the Runtime Security Middleware. It resolves the severe **multi-scale fusion and box/class entanglement** issues found in standard PANet neck taps by tapping directly into YOLOv8's **decoupled classification head (`cv3`)**.

By evaluating Mahalanobis distances strictly on intermediate classification activations *conditioned on predicted object classes*, this module completely eliminates false positives caused by real bounding-box edges.

---

## 1. Environment Setup

This module runs in the primary research environment (`yolo_adv`).

```bash
conda deactivate
conda activate yolo_adv
pip install ultralytics opencv-python torch torchvision pandas
```

---

## 2. Configuration Schema (`config.json`)

All execution variables, profiles, thresholds, and target folders are controlled via `config.json`.

```json
{
  "model_path": "yolov8n.pt",
  "clean_data_dir": "D:\\GitHub\\experiment_data\\bdd100k_images_10k\\train",
  "profiles_dir": "D:\\GitHub\\yr-4-research\\mechanisms-by-claude\\mech-1\\profiles",
  "calibration_profile_filename": "cls_head_calibration_p3.pkl",
  "patched_data_dir": "D:\\GitHub\\yr-4-research\\mechanisms-by-claude\\gen-adv-patch\\bdd100k_patched",
  "detection_output_dir": "D:\\GitHub\\yr-4-research\\mechanisms-by-claude\\mech-1\\detections",
  "specific_test_image": "",
  "detector_settings": {
    "threshold": 15.0,
    "stride": 8,
    "min_calibration_samples": 5
  }
}
```

### Configuration Field Definitions
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `model_path` | `string` | Weights file for the target YOLOv8 model. |
| `clean_data_dir` | `string` | Folder containing uncorrupted clean images for offline calibration. |
| `profiles_dir` | `string` | Destination folder for storing calibration artifacts (`.pkl`). |
| `calibration_profile_filename` | `string` | Name of the serialized calibration pickle file (e.g., `cls_head_calibration_p3.pkl`). |
| `patched_data_dir` | `string` | Input folder containing adversarial patch images to test. |
| `detection_output_dir` | `string` | Output directory for saving heatmap overlays and detection visuals. |
| `specific_test_image` | `string` | *(Optional)* Target test filename. If `""`, automatically selects the first patched image found. |
| `detector_settings.threshold` | `float` | Mahalanobis distance gating threshold (e.g., `15.0`). Scores exceeding this trigger an anomaly. |
| `detector_settings.stride` | `int` | Spatial downsampling stride of the target head scale (`8` for P3 high-resolution scale). |
| `detector_settings.min_calibration_samples` | `int` | Minimum required instances of a class in the clean dataset to compute its covariance. |

---

## 3. Code Files & Architecture

### 1. `cls_head_adapter.py` (`YOLOv8ClsHeadAdapter`)
* **Role:** Model-Agnostic Adapter.
* **Mechanism:** Hooks `model.model[-1].cv3[0][-2]` (the second-to-last conv block of scale P3's classification branch).
* **Benefit:** Extracts intermediate hidden feature representations before final $1 \times 1$ logit projection, completely bypassing regression loss contamination.

### 2. `calibrate_cls_head.py`
* **Role:** Offline Calibration Pipeline.
* **Mechanism:** 
  * Passes clean images through YOLOv8.
  * Resizes frames to $640 \times 640$ to prevent downsampling integer division mismatches.
  * Maps predicted bounding-box centers to the P3 grid.
  * Computes class-conditioned Mean ($\mu_c$) and Inverse Covariance ($\Sigma_c^{-1}$) with $1e-6 \cdot I$ regularization.
  * Exports `cls_head_calibration_p3.pkl`.

### 3. `cls_head_detector.py` (`ClsHeadMahalanobisDetector`)
* **Role:** Runtime Anomaly Detector (Strategy).
* **Mechanism:**
  * Loads `cls_head_calibration_p3.pkl` onto GPU memory.
  * Calculates per-cell Mahalanobis distance:
    $$D_M(x) = \sqrt{(x - \mu_c)^T \Sigma_c^{-1} (x - \mu_c)}$$
  * Applies `threshold` gating.
  * Extracts the largest anomaly contour and scales stride-8 grid coordinates to full pixel dimensions.
  * Returns strict `AnomalyResult` contract (`is_attack`, `score`, `bounding_box`).
  * Exports JET colormap visualization `detected_cls_head_<filename>.jpg`.

---

## 4. Execution Guide

### Step 1: Run Offline Calibration
```bash
python calibrate_cls_head.py
```
*Output:* Creates `cls_head_calibration_p3.pkl` inside `profiles_dir`.

### Step 2: Run Anomaly Detection & Visualization
```bash
python cls_head_detector.py
```
*Output:* Prints `AnomalyResult` dictionary to stdout and exports heatmap overlay image to `detection_output_dir`.