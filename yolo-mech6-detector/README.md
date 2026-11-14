# Mech-6 detector (patch localizer)

Turns the two Mech-6 heatmaps into **two separate patch localizers** that output boxes:

| | Panel 2 localizer | Panel 3 localizer |
|---|---|---|
| Score per cell | energy z = (log E − global mean) / global std, clipped at 0 | Mech-6 nearest-profile distance: \|z\| to the nearest class / background log-energy profile (`nearest_abs`), or only above it (`nearest_up`) |
| Threshold | one per layer | **per class** (default) or one per layer (`threshold_mode`) |

Energy = (L1 norm of the cell's feature vector across channels)², at backbone layer 1, 2 or 3 (strides 4, 4, 8) of
**YOLOv5, YOLOv8, YOLOv9 or YOLO11**. Use the `yolo_adv` conda env. Weights and the YOLOv5 code come from
`../gen-adv-patch-hiding-latest/`.

**All calibrated values (thresholds, per-class thresholds, minimum region sizes) are written into the config file**, so
you can read them, edit them by hand and re-run `detect_patches.py` to test other values. Only the class profiles
(log-energy mean/std per class) stay in a `.pkl` file.

## Files

| File | What it does |
|---|---|
| `calibrate_detector.py` | profiles (pass 1), thresholds (pass 2) and minimum region sizes; writes the values into the config |
| `detect_patches.py` | runs both localizers on a folder: box files per panel, 3-panel pictures, `summary.csv` |
| `evaluate_detector.py` | box precision / recall / F1 / IoU on patched images and false alarms on clean images, panel 2 vs panel 3 |
| `mech6_core.py` | model loading, energy, profiles, scores, the localizer (library) |
| `detector_viz.py` | the 3-panel picture (library) |
| `config_yolov5.json`, `config_yolov8.json`, `config_yolov9.json`, `config_yolo11.json` | one config per model |

## How the thresholds are set (calibration)

1. **Pass 1, profiles** (`calibration.profile_images_dir`, BDD `train`). The model's own detections say which cells are
   which class; random uncovered cells are background. Per layer and class: mean/std of log-energy, plus the global
   mean/std (for panel 2). Saved to `profile_file` (same format as `yolo-mech6`, so those `.pkl` files can be reused).
2. **Pass 2, thresholds** (`calibration.threshold_images_dir`, BDD `test`, different images). With
   `threshold_quantile` = 0.995, only 0.5% of clean cells go above a threshold:
   - **panel 2:** the 99.5% point of the clean energy z, per layer. This replaces Mech-6's fixed `3.5`.
   - **panel 3, `layer_threshold`:** the 99.5% point of the clean nearest-profile score, per layer.
   - **panel 3, `class_thresholds`:** for each profile k, the 99.5% point over the clean cells whose **nearest profile
     is k**. When detecting, each cell is compared with the threshold of its own nearest profile. A profile with fewer
     than `min_threshold_cells` clean cells is left out and uses the layer threshold.
3. **Region sizes** (same clean images). The localizer runs with every rule except the minimum size, and the size of
   the largest region in each clean image is recorded. `min_region_cells` is then set so that only
   `target_clean_image_fpr` (2%) of clean images still keep a region, never below `min_region_cells_floor`.
   It is computed for each layer, for panel 2, and for panel 3 in both modes.

## How a patch is found (the localizer)

The same steps for both panels, each with its own score, thresholds and `region` settings:

1. **Hot cells:** score > threshold. In panel 3 `per_class` mode each cell has its own threshold.
2. **Join:** a morphological closing with a (2·`join_cells`+1)² window. Patch textures have dark pixels, so their hot
   cells come out speckled; this fills the gaps so one patch becomes one blob.
3. **Regions:** 8-connected groups of the joined cells.
4. **Region rules.** A region is a patch only if all of these hold:
   - **size:** cells ≥ `min_region_cells` (`"auto"` = the calibrated value; or a number). At layer 1 one cell is 4×4 px.
   - **fill:** hot cells / bounding-box area ≥ `min_fill`. Rejects thin lines and edges.
   - **shape:** long side / short side ≤ `max_aspect`. Rejects strips such as lane markings and poles.
   - **seed:** its strongest cell has score/threshold ≥ `seed_ratio`. The region must contain at least one clearly
     abnormal cell; weak clusters that only just pass the threshold are dropped (seed-and-grow).
   - **strength:** mean score/threshold over its hot cells ≥ `min_mean_ratio`.
5. **Boxes:** the region's bounding box (cells × stride) in 640×640 pixels. Overlapping boxes are merged; the best
   `max_detections` (by mean score/threshold) are kept. An image is "patched" if at least one box remains.

## Step by step (example YOLOv8)

**1. Calibrate** (writes into `config_yolov8.json`; the old version is kept as `config_yolov8.json.bak`):
```bash
python calibrate_detector.py --config config_yolov8.json
```
- To reuse an existing Mech-6 profile, set `profile_file` to e.g. `../yolo-mech6/profiles/mech6_energy_profiles_yolov8-v2.pkl`
  and add `--thresholds-only`.
- After changing a panel's `region` settings (join, fill, seed ...), recompute only the minimum sizes:
  `--regions-only` (keeps your thresholds, even hand-edited ones).
- `--max-images 60 --threshold-images 30` makes a quick test version.

**2. Patched images** (the config points `detect.input_dir` at the patched set and `detect.gt_coordinates_dir` at its coordinates):
```bash
python detect_patches.py --config config_yolov8.json
```

**3. Clean images** (false alarms), BDD `val` 0–99:
```bash
python detect_patches.py --config config_yolov8.json --input-dir D:/GitHub/experiment_data/bdd100k_images_10k/val --num-images 100 --detections-dir detections/yolov8/clean --visualizations-dir viz/yolov8/clean
```

**4. Numbers:**
```bash
python evaluate_detector.py --config config_yolov8.json --out results/yolov8.json
```

`--panel3-mode per_layer` on `detect_patches.py` compares the two panel-3 modes without editing the config
(use a different `--detections-dir`).

## Outputs

- `detections/<model>/<set>/panel2/<stem>.txt` and `.../panel3/<stem>.txt`: one line per box,
  `box=(x1, y1, x2, y2) score=… peak=… cells=… fill=… orig_box=(…)`. `box` is in 640 px like the GT files; `orig_box` in
  the original image size. Empty file = no patch.
- `viz/<model>/<set>/det_<stem>.png`:
  - left: image, GT patches in cyan;
  - middle: energy z, white outline = hot cells, **green boxes** = panel-2 patches;
  - right: nearest-profile distance, white outline = hot cells, **magenta boxes** = panel-3 patches;
  - colour scale per panel: blue = 0, red = `viz_scale_max_ratio` × that panel's layer threshold;
  - header: number of boxes per panel with their score (mean score/threshold) and size.

## `evaluate_detector.py` columns

- **precision / recall / F1:** a predicted box matches a GT patch when IoU ≥ `match_iou` (0.3; one-to-one).
- **mean_iou:** over matched pairs. Boxes are on the layer grid, so they are a few px coarse at stride 4.
- **centre_hit:** share of GT patches whose centre is inside some predicted box (looser than IoU).
- **image_recall:** share of patched images with at least one box.
- **clean_image_fpr:** share of clean images with at least one box (should be near `target_clean_image_fpr`).

## Config keys

| Key | Meaning |
|---|---|
| `model_type`, `yolov5_repo_dir`, `model_path` | model (`ultralytics` for v8/v9/11, `yolov5_repo` for the original YOLOv5) |
| `tap_layers` | layers calibrated (1, 2, 3) |
| `detect_layer` | layer the localizers run on (default 1; `--layer` overrides) |
| `profile_file` | class profiles (.pkl) |
| `calibration.*` | pass-1 images, pass-2 images, `threshold_quantile`, `min_threshold_cells`, `target_clean_image_fpr`, `min_region_cells_floor` |
| `panel2.region`, `panel3.region` | localizer settings per panel: `min_region_cells`, `seed_ratio`, `join_cells`, `min_fill`, `max_aspect`, `min_mean_ratio`, `max_detections` |
| `panel2.calibrated.<layer>` | **written by calibration, editable:** `threshold`, `min_region_cells` |
| `panel3.score` | `nearest_abs` (default, as in the Mech-6 panel 3) or `nearest_up`. Changing it needs `--thresholds-only` |
| `panel3.threshold_mode` | `per_class` (default) or `per_layer` |
| `panel3.calibrated.<layer>` | **written by calibration, editable:** `layer_threshold`, `class_thresholds` (`"bg"`, `"<id>_<name>"`), `min_region_cells` per mode. You can add a class by hand |
| `calibration_info` | when and how the values were calibrated |
| `viz_outline`, `viz_scale_max_ratio` | picture settings |
| `detect.*` | input folder, slice, output folders, GT folder |
| `evaluate.*` | detection folders, GT folder, `match_iou` |

Relative paths are relative to the config file's folder.

## A note on per-class thresholds for panel 3

Energy is a single number, so "nearest profile" cuts the log-energy axis into ranges, one per profile. A profile whose
mean sits between two close neighbours (e.g. car, bus, truck, whose means differ by about 0.1) owns only a narrow range,
so a cell assigned to it can never be far from it: its per-class threshold comes out tiny (e.g. 0.05), but so does every
score in that range. The profiles at the two ends of the axis (lowest and highest mean) own open-ended ranges, and
**patch cells, which have very high energy, always land in the highest profile.** So in practice the panel-3
per-class decision for patches is made by that top profile's threshold, which is usually close to the layer threshold.
Expect `per_class` and `per_layer` to give similar results at layer 1; the evaluation will show it.

## Status

Smoke-tested on YOLOv8 and YOLOv5 with tiny calibrations (40 / 60 and 20 / 20 images): calibration writes the config,
both localizers find the patches in the sample images, evaluation runs. No full calibration yet.
