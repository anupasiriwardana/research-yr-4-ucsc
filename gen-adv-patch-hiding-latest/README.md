# gen-adv-patch-hiding-latest

Self-contained patch toolkit for **YOLOv5, YOLOv8, YOLOv9 and YOLO11**. Nothing here depends on another repo or
folder except the clean image folder you point to in the config. The original scripts in
`mechanisms-by-claude/gen-adv-patch-hiding/` are untouched.

## What is in this folder

| Item | What it is |
|---|---|
| `generate_hiding_patch.py` | trains a patch against one model (V2 method: corrected placement, targeted loss) |
| `apply_patch_batch.py` | pastes a saved patch on many images (up to `max_patches` per image), writes the patched images and a coordinate file per image |
| `verify_patch_efficacy_batch.py` | compares many patched images with their originals and reports how many target objects were hidden |
| `patch_common.py` | library: model loading for all four families, patch placement, target selection (never run directly) |
| `config_yolov5.json`, `config_yolov8.json`, `config_yolov9.json`, `config_yolo11.json` | one config per model; every name, folder and setting lives here |
| `requirements.txt` | the Python packages (except PyTorch) for the `yolo_adv` environment |
| `weights/` | the original weights: `yolov5s.pt` (identical to the official release, checked by SHA-256), `yolov8n.pt`, `yolo11n.pt`, `yolov9c.pt` (from the ultralytics assets release v8.4.0) |
| `yolov5_src/` | the original yolov5 repository code (`models/`, `utils/`, AGPL-3.0 licence) needed to load `yolov5s.pt` |

Run everything inside the `yolo_adv` conda environment (set-up steps right below). YOLOv5 needs `tqdm`, `seaborn`, `scipy`; they are in `requirements.txt`.

## Set up the conda environment `yolo_adv` (NVIDIA GPU)

Works on Windows or Linux; the commands are the same. Do the steps in this order: PyTorch first, then the rest, otherwise pip may pull a
CPU-only PyTorch.

**1. Install conda** (skip if `conda --version` already works): install Miniconda from https://docs.conda.io/en/latest/miniconda.html, then open a
new terminal (on Windows: "Anaconda Prompt" or any shell where `conda` works).

**2. Check the NVIDIA driver and the CUDA version it supports.**
```bash
nvidia-smi
```
Look at the top right: `CUDA Version: 12.x`. That is the newest CUDA your driver can run (it is not a separate CUDA install; PyTorch brings its own
CUDA runtime). If `nvidia-smi` is not found, install or update the NVIDIA driver first.

**3. Create and activate the environment** (Python 3.10).
```bash
conda create -n yolo_adv python=3.10 -y
conda activate yolo_adv
```

**4. Install PyTorch with the CUDA build that matches step 2.** Pick one line (all use PyTorch 2.5.1, the tested version):

| `nvidia-smi` shows | Command |
|---|---|
| CUDA 12.4 or newer (most current drivers) | `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.1 to 12.3 | `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 to 12.0 (older drivers; this is what the author uses) | `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118` |
| RTX 50-series (Blackwell) cards | PyTorch 2.5.1 does not support these. Use a newer build instead: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` (not tested with these scripts, but the code does not depend on a PyTorch version) |

If a driver is older than CUDA 11.8, update the driver.

**5. Install the other packages** (run inside this folder, where `requirements.txt` is).
```bash
pip install -r requirements.txt
```

**6. Check that PyTorch sees the GPU.**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The third value must be `True` and the last one must be the GPU name. If it prints `False` or a version ending in `+cpu`, a CPU build was installed:
run `pip uninstall -y torch torchvision`, then repeat step 4.

**7. Check the toolkit** (the weights are already in `weights/`). First set `paths.clean_dir` in the config you use to a folder of street images on this
computer (the configs ship with the author's `D:/...` path; relative paths are relative to the config file). Then:
```bash
python generate_hiding_patch.py --config config_yolov8.json --steps 5 --max-images 20
```
It should print the model name, `device cuda`, a few `step ... mean targeted confidence` lines and the saved patch paths in `patches/yolov8/`. Delete
that test patch afterwards.

Notes: training one patch needs a few GB of GPU memory. If it stops with `CUDA out of memory`, lower `generate.gpu_batch_size` in the config.
Send the whole folder to the friend (about 80 MB with the weights; the `patched/` and `efficacy/` folders are not needed).

## The three-step workflow

```bash
python generate_hiding_patch.py        --config config_yolov8.json
python apply_patch_batch.py            --config config_yolov8.json
python verify_patch_efficacy_batch.py  --config config_yolov8.json
```

Swap the config for `config_yolov5.json`, `config_yolov9.json` or `config_yolo11.json`. A quick test of the whole chain:
`generate_hiding_patch.py --config ... --steps 20 --max-images 40`, then `apply_patch_batch.py --config ... --num-images 10`, then verify.

1. **generate** trains the patch and saves `patches/<model>/<patch_name>.npy` (the array), `.png` (the picture), `.json` (the settings used) and
   `_demo.jpg` (the patch pasted on one image).
2. **apply** finds the target objects in each clean image with the same model, pastes the patch on up to `apply.max_patches` of them and writes
   `patched/<model>/images/<prefix><original name>` plus `patched/<model>/coordinates/<original stem>.txt`
   (`box=(x1, y1, x2, y2)` per patch, 640-px coordinates; the same format as the existing tools use).
3. **verify** takes every patched image whose name starts with the prefix, **removes the prefix to find the original image** in `clean_dir`, runs the model
   on both and saves a side-by-side picture (left clean, right patched, cyan boxes = where the patches are) plus `efficacy_summary.csv` / `.json`.

## The config file (every setting)

Relative paths are relative to the config file's folder. Use `/` or `\\` in absolute Windows paths.

**`model`**

| Key | Meaning |
|---|---|
| `family` | `"ultralytics"` for v8 / v9 / 11, `"yolov5"` for the original yolov5 code |
| `weights` | weights file, e.g. `weights/yolov8n.pt` |

**Top level**: `patch_name` is the name (no extension) of the patch that `generate` writes and `apply` reads.

**`paths`**

| Key | Meaning |
|---|---|
| `clean_dir` | folder of clean images (used for training, applying and as the originals in verify) |
| `patch_dir` | where patches are saved |
| `patched_dir` | where patched images are written |
| `coordinates_dir` | where the patch coordinate files are written |
| `efficacy_dir` | where the comparison pictures and summaries are written |

**`generate`**

| Key | Meaning |
|---|---|
| `target_class` | COCO class id to hide (2 = car); null = most confident object of any class |
| `patch_size` | patch resolution in pixels |
| `num_steps`, `learning_rate` | training length and Adam learning rate |
| `conf_thresh` | an object counts as a target above this confidence |
| `rotation_max`, `scale_min`, `scale_max`, `placement_jitter` | random transforms during training (degrees, fraction of the image, fraction of the canvas) |
| `gpu_batch_size` | images per training step; lower it if the GPU runs out of memory (defaults: 20 for v8/11n, 8 for v5s, 2 for v9c) |
| `max_images` | how many images of `clean_dir` (sorted, from the start) are used for training |
| `patch_scale` | size of the patch in the `_demo.jpg` |
| `demo_image` | file name for the demo image (null = first image with a target) |
| `seed` | integer for a repeatable run (null = random) |

**`apply`**

| Key | Meaning |
|---|---|
| `patch_name` / `patch_path` | use another patch than the top-level `patch_name` (a name in `patch_dir`, or a full path, e.g. to test a patch from another model); null = default |
| `prefix` | text put in front of each patched image name, e.g. `multi-patched_` |
| `max_patches` | **maximum number of patches per image** (fewer if fewer targets are found) |
| `min_separation` | minimum distance between two patched objects (normalised canvas units) |
| `scale`, `rotation` | patch size (fraction of the image) and rotation in degrees at apply time |
| `target_class`, `conf_thresh` | which objects get a patch, and the confidence needed |
| `start_index`, `num_images` | which slice of the sorted `clean_dir` to process |
| `coordinates_use_prefix` | false (default): coordinate file named after the original image (what the existing tools expect); true: named after the patched image |
| `image_extensions` | file types to read |

**`verify`**

| Key | Meaning |
|---|---|
| `patched_prefix` | prefix to strip to find the original; null = use `apply.prefix` |
| `start_index`, `num_images` | slice of the patched images (null = all) |
| `target_class` | class whose objects are counted as hidden (null = all classes) |
| `conf_thresh`, `iou_thresh` | detection confidence and NMS IoU |
| `match_iou` | IoU needed to call a clean and a patched detection "the same object" |
| `save_visualizations` | true/false: write the side-by-side pictures |
| `output_prefix` | prefix of the comparison picture names, e.g. `efficacy_` |

Every setting that is likely to change between runs can also be given on the command line (`--max-patches`, `--prefix`, `--num-images`,
`--start-index`, `--patch-name`, `--steps`, `--max-images`, `--patched-dir`, `--out-dir`, `--save-visualizations/--no-save-visualizations`);
the command line wins over the config.

## How to read the verify summary

- **hidden object**: a clean detection of the target class with no patched detection of the same class at IoU >= `match_iou`. A car that turns into a
  "truck" counts as hidden for the car class.
- **new detection**: a patched detection with no clean detection of the same class at IoU >= `match_iou` (spurious boxes, including a box on the patch itself).
- `hiding_rate` = hidden / clean target objects over all compared images. `images_all_targets_hidden` = images where every clean target vanished.

## Notes and choices worth knowing

- **Same 640x640 pipeline everywhere.** Images are stretched to 640x640 before the model (as in the training and in the detection tools). The old
  `verify_patch_efficacy.py` fed the clean image at its original size (letterboxed) and the patched one at 640x640; here both use 640x640, so the comparison is fair.
- **YOLOv5 confidence = objectness x class score.** YOLOv5 has an objectness output that the anchor-free models do not; the attack loss and the
  detections both use the product, which is what yolov5's own NMS uses.
- **Default train/test split.** The configs train on the first 1500 images of `clean_dir` (`generate.max_images`) and apply patches from index 1500
  (`apply.start_index`), so the patched test images were not seen during training. The old setup trained on all images and applied to the first
  100; set `max_images` and `start_index` back if you want that.
- **Prefix and the detection tools.** The existing detection / metric tools strip `patched_`, `multi-patched_` (optionally with a number in front) from
  image names when they look for the coordinate file, so the default (coordinate file named after the original image) works with prefixes like
  `multi-patched_`. If you choose a prefix that does not end in `patched_` / `patched-`, those tools cannot strip it; set `apply.coordinates_use_prefix`
  to true so the coordinate file carries the same name as the patched image. generate / apply / verify here work with any prefix either way.
- **Tested** on all four models: a 6-step training run, apply on 6 images, verify, and a gradient check (the patch receives gradients through each
  model's head). No full-length training run has been done, so no real hiding numbers exist yet.
