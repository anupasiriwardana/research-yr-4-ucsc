# Adversarial Patch Generator (YOLOv8, Object-Aware)

## Module Overview
This module provides a white-box adversarial patch generation and application pipeline targeting **YOLOv8**, designed to suppress detection confidence specifically at a real target object's location (a "hiding" attack), rather than a generic whole-image disruption. See `patch-generation-process.md` for the full history of why this module looks the way it does, and the supporting literature.

**Note for anyone who used the earlier version of this module:** this no longer depends on ART (Adversarial Robustness Toolbox) at all. Both `generate_hiding_patch.py` and `apply_saved_patch.py` were rewritten as direct, transparent PyTorch code — no `PyTorchYolo` estimator, no `AdversarialPatchPyTorch` attack class, and no version-fragile `attack_losses` key naming to track. If you have an old `patch_gen_yolov8` environment with ART installed, it isn't required for anything in this module anymore.

---

## 1. Environment Setup

```bash
conda deactivate
conda create -n patch_gen_yolov8 python=3.10 -y
conda activate patch_gen_yolov8

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install ultralytics opencv-python numpy
```

No `adversarial-robustness-toolbox` package is needed.

---

## 2. Configuration Schema (`config.json`)

```json
{
    "model_path": "yolov8n.pt",
    "clean_dir": "D:\\GitHub\\experiment_data\\bdd100k_images_10k\\train",
    "output_dir": "D:\\GitHub\\yr-4-research\\mechanisms-by-claude\\gen-adv-patch-hiding\\bdd100k_patched",
    "patch_detection_dir": "D:\\GitHub\\yr-4-research\\mechanisms-by-claude\\gen-adv-patch-hiding\\patch_detections",
    "specific_clean_image": "00e9be89-00001885.jpg",
    "specific_patched_image": "sample_image2.jpg",
    "specific_patch_name": "art_patch.npy",
    "patch_settings": {
        "patch_scale": 0.2,
        "patch_size": 150,
        "num_steps": 500,
        "learning_rate": 0.02,
        "conf_thresh": 0.25,
        "rotation_max": 22.5,
        "scale_min": 0.2,
        "scale_max": 0.4,
        "placement_jitter": 0.1,
        "target_class": 2,
        "gpu_batch_size": 4,
        "max_images": 2000
    },
    "apply_settings": {
        "placement_mode": "auto",
        "manual_x": 320,
        "manual_y": 400,
        "scale": 0.3,
        "rotation": 0.0
    }
}
```

### `patch_settings` (used by `generate_hiding_patch.py`)

| Field | Description |
|---|---|
| `patch_scale` | Fallback scale used only for the script's own end-of-run demo composite (training itself samples scale randomly between `scale_min`/`scale_max`, per step, as part of EoT). |
| `patch_size` | Patch's own pixel dimensions (square), e.g. `150` → a 150×150 patch. |
| `num_steps` | Number of training steps. Each step samples `gpu_batch_size` images. |
| `learning_rate` | Adam optimizer learning rate for the patch pixels. |
| `conf_thresh` | Confidence threshold (on the *clean* image) for deciding an anchor represents a real, pseudo-labeled target object — `0.25` matches YOLO's own conventional detection threshold. |
| `rotation_max` | EoT rotation range in degrees (±) applied to the patch each training step. |
| `scale_min` / `scale_max` | EoT scale range (relative to canvas) sampled each training step. |
| `placement_jitter` | How far (in normalized canvas units) the patch's training position is allowed to wander from the target object's own detected center each step — keeps training realistic without losing the object-targeting the whole redesign was for. |
| `target_class` | COCO class id to attack specifically (e.g. `2` = car). Set to `null`/omit to target whichever object the clean model is most confident about, regardless of class. |
| `gpu_batch_size` | How many images are loaded from disk and moved to GPU at once, per training step. Lower this (e.g. to `1`) if you hit `CUDA error: out of memory`. |
| `max_images` | Size of the candidate training pool (filenames only — safe to set this large; see `patch-generation-redesign.md` Section 2.3). |

### `apply_settings` (used by `apply_saved_patch.py`)

| Field | Description |
|---|---|
| `placement_mode` | `"auto"` (default): re-detects `target_class` on the target image and centers the patch there automatically. `"manual"`: use `manual_x`/`manual_y` instead. |
| `manual_x` / `manual_y` | Pixel coordinates (in the 640×640 working space) for manual placement. Ignored in `"auto"` mode. |
| `scale` | Patch scale (relative to canvas) at application time. Falls back to `patch_settings.patch_scale` if omitted. |
| `rotation` | Rotation in degrees applied at application time (default `0` — a deployed/static test usually doesn't want extra rotation on top of what EoT already trained for). |

### Other top-level fields

| Field | Description |
|---|---|
| `model_path` | Target YOLOv8 weights file. Must match the model your detection mechanism actually protects. |
| `clean_dir` | Directory of clean images — training pool for `generate_hiding_patch.py`, and the source directory `apply_saved_patch.py` looks in for its target image. |
| `patch_dir` | Where the patch (`.npy` and standalone `.png`) are saved. |
| `output_dir` | Where all composited demo/patch applied images are saved. |
| `patch_detection_dir` | Used by `verify_patch_efficacy.py` to save the results. |
| `specific_clean_image` | If set, controls ONLY which image the finished patch is demonstrated on / applied to — never which images are used for training. If empty, falls back to the first file found. |
| `specific_patched_image` | Used by `verify_patch_efficacy.py` to pick which patched image to compare against its clean counterpart. |
| `specific_patch_name` | Filename (inside `patch_dir`) of the saved `.npy` patch that `apply_saved_patch.py` loads. |

---

## 3. Code Files & Functionality

### `generate_hiding_patch.py`
- Trains a patch via a direct PyTorch loop (no ART): on each step, pseudo-labels the real target object from a clean forward pass, places the patch (randomly transformed, jittered around that object's location) via a differentiable affine-warp compositor, and minimizes detection confidence specifically at that object's anchors.
- Streams training images from disk in small batches (`gpu_batch_size`) rather than preloading the whole dataset to GPU.
- Saves two outputs to `patch_dir`: the raw patch array (`art_patch.npy`), a standalone patch image (`art_patch_standalone.png`, for manual use in an image editor), and a demo composite (`art_patched_<filename>.jpg`) with the patch placed on the target object of whichever image `specific_clean_image` names to `output_dir`.

### `apply_saved_patch.py`
- Loads a previously-saved `.npy` patch and applies it to any image, using the *same* compositing function training used (`build_patch_overlay`) — not ART's `apply_patch`, and not a manual image-editor paste, both of which risk applying the patch outside the geometric/quality envelope it was actually optimized for.
- Two placement modes: `"auto"` (re-detects the target object on the given image, same logic as training) or `"manual"` (explicit pixel coordinates).

### `verify_patch_efficacy.py`
- Unchanged in purpose from the original module: runs unmodified YOLOv8 on a clean/patched image pair and reports detection differences side by side. Use this to confirm a patch is actually suppressing the target object's detection (not just adding spurious detections elsewhere) before feeding it into the detection-mechanism test pipeline.

---

## 4. Step-by-Step Execution Workflow

**Step 1 — Generate a patch:**
```bash
python generate_hiding_patch.py
```
Watch the printed `mean targeted confidence` value across steps — it should trend downward, since that's exactly the quantity being minimized.

**Step 2 — Verify it actually suppresses the target object:**
```bash
python verify_patch_efficacy.py
```
Confirm the target object's detection is missing or substantially lower-confidence in the patched image — not just that new, unrelated detections appeared elsewhere.

**Step 3 — Apply the saved patch to other images:**
```bash
python apply_saved_patch.py
```
Leave `apply_settings.placement_mode` as `"auto"` unless you specifically need to test a non-detected location — auto-placement reuses the exact detection logic the patch was trained against, so it's both the easiest option and the most consistent with the patch's own training conditions.

**Step 4 — Feed the result into the detection mechanism:**
Point `mech-1-updated`'s `config.json` (`patched_data_dir` / `specific_test_image`) at whatever `apply_saved_patch.py` or `generate_hiding_patch.py` produced, and run `cls_head_detector.py` / `cls_head_detector_for_tightPatch.py` as documented in that module's own `README.md`.