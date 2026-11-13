# Mech-6-Mahalanobis

Mech-6 with the feature vector instead of the energy. It taps the same backbone layers (1, 2, 3; strides 4, 4, 8) on
**YOLOv5, YOLOv8, YOLOv9 and YOLO11**, but profiles them the way Mech-1 (`yolo-transferable-mech1`) profiles the
classification-head feature: a **Mahalanobis** profile (mean vector + covariance matrix) per object class plus one for
background. Each cell gets the distance to its **nearest** profile.

## The idea in short

- **Mech-6:** a cell is one number, `log energy` = how strongly the layer fires there. One mean/std per class.
- **This version:** a cell is its whole C-channel feature vector, e.g. 32 channels for yolov8n layer 1 or 128 for yolov9c layer 1.
  Each class (car, person, ..., background) gets a mean vector `mu_k` and a covariance matrix `S_k`.
- **Score:** `score(cell) = min_k sqrt((x - mu_k)^T S_k^-1 (x - mu_k))`. This is the Mahalanobis distance to the closest profile,
  measured in "standard deviations" that take the correlations between channels into account.
  - Normal content (road, car, sky) is close to one of the profiles.
  - A patch's texture is far from all of them, so it lights up.
- **No reliance on the boxes:** the profile a cell is compared with is picked by the distance itself, never by the detector's boxes.
  A patch that fools the detector can't change which profile it's compared against.
- **What changes compared with Mech-6:** Mech-6 can only notice "too much / too little activity". Mahalanobis also notices "the
  wrong *kind* of activity", i.e. an unusual mix of channels at normal strength.
- **What changes compared with Mech-1:** Mech-1 works on one head layer at stride 8. This version works on early backbone
  layers, where a patch's high-frequency texture is still visible at stride 4.

## Files

| File | What it does | Run it? |
|---|---|---|
| `calibrate_mahalanobis.py` | builds the profiles and the clean threshold for one model (once per model) | yes |
| `extract_mahalanobis_maps.py` | scores a folder of images: saves distance maps (`.npz`) and the 2-panel pictures | yes |
| `diagnose_mahalanobis.py` | cell-level numbers from saved maps: AUROC, TPR at 0.5% FPR, % above threshold | yes |
| `maha_core.py` | model loading (ultralytics + original yolov5 code), layer hooks, profiles, distance | no (library) |
| `maha_viz.py` | draws the 2-panel picture | no (library) |
| `config_yolov5.json`, `config_yolov8.json`, `config_yolov9.json`, `config_yolo11.json` | one config per model | edit |

The models come from `../gen-adv-patch-hiding-latest/weights/` (the original weights, the same files the patches were trained
against). The YOLOv5 code comes from `../gen-adv-patch-hiding-latest/yolov5_src/`. Use the `yolo_adv` conda env.

## Calibration (yes, every model needs its own)

The profiles describe one model's features, so each model needs its own calibration. A model's profile can't be reused for
another model: even the channel counts differ. The calibration does two passes:

1. **Profiles (pass 1).** Uses the first `max_calibration_images` images of `profile_images_dir` (BDD `train`).
   - The model's own detections (offline only) are projected onto each layer's grid. Every cell inside a box adds to that class.
   - `background_samples_per_image` random uncovered cells add to the background profile.
   - Statistics are running sums (count, sum, sum of outer products), so memory stays flat however many images you use.
   - A class is kept only with at least `max(min_calibration_samples, 2 x channels)` cells.
   - The covariance is shrunk towards a scaled identity (`shrinkage`, default 0.01). Some backbone channels are almost constant,
     and without this the inverse explodes. Mech-1 used only a tiny ridge, which is fine for the 80-dim head feature but not here.
2. **Clean threshold (pass 2).** Uses the first `threshold_num_images` images of `threshold_images_dir` (BDD `test`, so
   different images from pass 1).
   - It takes the nearest-profile distance of every cell and stores the quantiles q50, q90, q99, q99.5, q99.9 and max per layer.
   - The threshold is the `threshold_quantile` (default 0.995): only 0.5% of clean cells go above it.
   - The pictures and `diagnose_mahalanobis.py` use it.

```bash
python calibrate_mahalanobis.py --config config_yolov8.json
```

Output: `profiles/maha_profiles_yolov8.pkl`. With 2000 images this takes a few minutes for the nano models and longer for yolov9c.
`--max-images 60 --threshold-images 20` makes a quick test version. `--threshold-only` redoes pass 2 without rebuilding the profiles.

## Step by step: calibrate, run on patched and clean images, look at the pictures

Example with YOLOv5. For other models, swap in `config_yolov8.json`, `config_yolov9.json` or `config_yolo11.json`.

**1. Calibrate** (once per model).
```bash
python calibrate_mahalanobis.py --config config_yolov5.json
```

**2. Patched images.** Each config already points `input_dir` at that model's patched set and `gt_coordinates_dir` at its patch
coordinates, so this needs no flags:
```bash
python extract_mahalanobis_maps.py --config config_yolov5.json
```
Maps go to `maps/yolov5/patched/` and pictures to `viz/yolov5/patched/`.

**3. Clean images** (no patches, so they show the false alarms). BDD `val` images 0–99:
```bash
python extract_mahalanobis_maps.py --config config_yolov5.json --input-dir D:/GitHub/experiment_data/bdd100k_images_10k/val --num-images 100 --out-dir maps/yolov5/clean --visualizations-dir viz/yolov5/clean
```

**4. Look at the pictures** in `viz/yolov5/patched/` and `viz/yolov5/clean/` (see below).

**5. (Optional) numbers.** This reads the maps from steps 2 and 3 (folders set in the config's `diagnose` section):
```bash
python diagnose_mahalanobis.py --config config_yolov5.json --out results/yolov5_cells.json
```

To look at layer 2 or 3 instead, set `viz_layer` (or pass `--viz-layer 3`). The maps always hold all three layers.

## The picture (2 panels)

- **Left:** the image, with the ground-truth patch boxes in cyan when a coordinate file is found.
- **Right:** the nearest-profile distance at layer `viz_layer`.
  - The colour scale is per model and layer, taken from the calibration. **Blue** is the clean median (q50), **red** is
    `viz_scale_max_ratio` x threshold (default 2x).
  - The **white outline** marks cells above the threshold, i.e. rarer than 99.5% of clean cells. Turn it off with `viz_outline: false`.
  - The patch boxes are not drawn here, so the heatmap stays clear.
- **Header (patched image with GT):**
  - median and max of distance/threshold inside the patch;
  - % of patch cells above the threshold;
  - % of far cells (more than 48 px from the patch) above it.
- **Header (clean image):** overall numbers.

A good result is a solid red/white block on the patch, with the rest blue and only scattered white specks.

## How to read `diagnose_mahalanobis.py`

- **AUROC:** the chance that a random patch cell scores higher than a random normal cell. 1.0 is perfect, 0.5 is guessing.
  The two columns differ only in what counts as normal: "clean" uses every cell of the clean images, and "far" uses cells more than 48 px
  from the patch in the patched images.
- **TPR@0.5%:** the share of patch cells caught when the threshold lets through only 0.5% of normal cells. This is the same measure as in
  Mech-6's `findings.md`, so you can compare directly.
- **patch>thr, clean cells>thr:** the same idea at the calibrated threshold.
- **clean imgs>thr:** the share of clean images with at least one cell above the threshold. It is close to 1.0 at the cell level because
  a 160x160 grid has 25,600 cells and 0.5% of them is still about 128 cells. A real detector needs a region rule on top: grouping cells
  and a minimum size, as in Mech-5/7.

## Config keys

| Key | Meaning |
|---|---|
| `model_type` | `"ultralytics"` (v8 / v9 / 11) or `"yolov5_repo"` (original yolov5 code) |
| `yolov5_repo_dir` | yolov5 only: folder with the yolov5 `models/` and `utils/` |
| `model_path` | weights |
| `tap_layers` | backbone layers to profile (default 1, 2, 3; strides are read from the model) |
| `profile_images_dir`, `max_calibration_images` | pass-1 images and how many |
| `background_samples_per_image` | random background cells per image in pass 1 |
| `min_calibration_samples` | a class needs at least this many cells (and 2 x channels) to get a profile |
| `shrinkage` | covariance shrinkage towards identity, 0..1 (default 0.01) |
| `threshold_images_dir`, `threshold_num_images`, `threshold_quantile` | pass-2 clean images, how many, which quantile is the threshold |
| `profiles_dir`, `profile_filename` | where the calibration is saved |
| `input_dir`, `out_dir`, `start_index`, `num_images` | images to score, where the maps go, which slice (`null` = all) |
| `save_maps` | true/false: write the `.npz` maps |
| `save_visualizations`, `visualizations_dir` | true/false and where the pictures go |
| `viz_layer` | layer shown in the picture |
| `viz_scale_max_ratio` | red end of the colour scale = this x threshold |
| `viz_outline` | true/false: white outline of cells above the threshold |
| `gt_coordinates_dir` | patch coordinate files. They are found by image name, with or without the `multi-patched_` / `patched_` prefix |
| `diagnose.patched_maps_dir`, `diagnose.clean_maps_dir` | map folders that `diagnose_mahalanobis.py` reads by default |

Relative paths are relative to the config file's folder. Every folder or switch can also be given as a flag on the command line,
and the flag wins.

## Patched sets the configs point to

| Model | Patched images | Coordinates |
|---|---|---|
| YOLOv5 | `gen-adv-patch-hiding-latest/patched/yolov5-images` | `.../yolov5-coordinates` |
| YOLOv8 | `test-patch-det-multi-img/gen-patched-img/yolov8-patched-imgs` | `.../yolov8-patch_coordinates` |
| YOLOv9 | `gen-adv-patch-hiding-latest/patched/yolov9-images` | `.../yolov9-coordinates` |
| YOLO11 | `test-patch-det-multi-img/gen-patched-img/yolov11-patched-imgs` | `.../yolov11-patch_coordinates` |

## Status

Tested end to end on all four models with a small calibration (60 profile images, 20 threshold images) and 5 patched / 5 clean images
each. No full calibration has been run yet, so there are no real results to report.
