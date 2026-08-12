# Adversarial Patch Generator (YOLOv8 + ART)

## Module Overview
This module provides a white-box adversarial patch generation and verification pipeline tailored specifically for **YOLOv8**. It utilizes the **Adversarial Robustness Toolbox (ART)** and **Expectation Over Transformation (EoT)** to generate physical-world robust patches capable of blinding YOLOv8 object detectors.

---

## 1. Environment Setup

To avoid dependency clashes with the security middleware runtime, this module operates inside its own isolated Conda environment (`patch_gen_yolov8`).

### Step 1: Create Environment
```bash
conda deactivate
conda create -n patch_gen_yolov8 python=3.10 -y
conda activate patch_gen_yolov8
```

### Step 2: Install PyTorch with CUDA Support
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Step 3: Install ART and Dependencies
```bash
pip install "adversarial-robustness-toolbox[pytorch]>=1.20.0" ultralytics opencv-python matplotlib pandas
```

---

## 2. Configuration Schema (`config.json`)

All directory paths, file overrides, and attack hyperparameters are centralized in `config.json`.

```json
{
  "model_path": "yolov8n.pt",
  "clean_dir": "D:\\GitHub\\experiment_data\\bdd100k_images_10k\\train",
  "output_dir": "D:\\GitHub\\yr-4-research\\mechanisms-by-claude\\gen-adv-patch\\bdd100k_patched",
  "specific_clean_image": "",
  "specific_patched_image": "",
  "patch_settings": {
    "patch_scale": 0.3,
    "max_iter": 500,
    "batch_size": 4,
    "max_images": 16
  }
}
```

### Configuration Field Definitions
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `model_path` | `string` | Target YOLOv8 weights file (e.g., `yolov8n.pt`). |
| `clean_dir` | `string` | Directory containing uncorrupted baseline training/test images. |
| `output_dir` | `string` | Destination folder for `art_patch.npy` and patched images. |
| `specific_clean_image` | `string` | *(Optional)* Filename (e.g., `"ac9be3fe-790d1f8e.jpg"`). If `""`, picks the first image automatically. |
| `specific_patched_image`| `string` | *(Optional)* Filename for verification scripts. If `""`, picks the latest generated patched image. |
| `patch_scale` | `float` | Scale ratio of the patch relative to the image size (e.g., `0.3` = 30% of image size). |
| `max_iter` | `int` | Number of optimization iterations for gradient descent (e.g., `500`). |
| `batch_size` | `int` | Training batch size for patch generation. |
| `max_images` | `int` | Number of clean calibration images used to optimize the patch over EoT transformations. |

---

## 3. Code Files & Functionality

### 1. `generate_art_patch.py`
* **Purpose:** Runs the complete white-box optimization loop.
* **Key Mechanisms:**
  * Uses `ART_YOLOv8_Wrapper` to force internal YOLOv8 into `.eval()` mode during gradient passes, preventing output dictionary crashes.
  * Computes a **Proxy Loss** by minimizing maximum class probabilities to blind object detections globally.
  * Exports `art_patch.npy` and the composite image `art_patched_<filename>.jpg`.

### 2. `apply_saved_patch.py`
* **Purpose:** Applies an already generated `art_patch.npy` to any new clean image without re-running the 7+ minute optimization process.
* **Key Mechanisms:**
  * Loads `art_patch.npy` directly from `output_dir`.
  * Renders the patch onto the target specified by `specific_clean_image` using the configured `patch_scale`.

### 3. `verify_patch_efficacy.py`
* **Purpose:** Evaluates whether the generated patch successfully degraded YOLOv8's detection performance.
* **Key Mechanisms:**
  * Runs unmodified YOLOv8 on both clean and patched images.
  * Prints side-by-side detection counts, bounding boxes, and confidence scores.
  * Exports an annotated comparison image `efficacy_comparison.jpg`.

---

## 4. Step-by-Step Execution Workflow

### Step 1: Generate Patch
```bash
python generate_art_patch.py
```
*Output:* Saves `art_patch.npy` and `art_patched_<filename>.jpg` in `output_dir`.

### Step 2: Verify Patch Efficacy
```bash
python verify_patch_efficacy.py
```
*Output:* Inspect `efficacy_comparison.jpg` to confirm that objects beneath or near the patch have disappeared or dropped in confidence.

### Step 3: Apply Saved Patch to Other Images (Optional)
Set `"specific_clean_image": "another_image.jpg"` in `config.json` and run:
```bash
python apply_saved_patch.py
```