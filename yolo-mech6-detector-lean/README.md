# Mech-6 detector, lean version (patch localizer)

A copy of `yolo-mech6-detector` with the settings that assumed something about the patches removed. It turns the two
Mech-6 heatmaps into **two separate patch localizers** that output boxes, for **YOLOv5, YOLOv8, YOLOv9 and YOLO11**.

**What changed from `yolo-mech6-detector`:**

| Removed | Why |
|---|---|
| `max_detections` | assumed at most 5 patches per image |
| `max_aspect` | assumed patches are roughly square |
| `min_fill` | assumed patches are compact blobs that fill their bounding box |
| `min_mean_ratio` | did nothing: hot cells are above the threshold by definition |

| Added | Why |
|---|---|
| `min_thickness_cells` | one thickness test that rejects thin lines at any angle without assuming a shape (replaces `max_aspect` and `min_fill`) |

Everything else is the same. The original folder is untouched, so the two can be compared on the same images.

## The two localizers

| | Panel 2 localizer | Panel 3 localizer |
|---|---|---|
| Score per cell | **energy z** = (log E − global mean) / global std, clipped at 0 | **Mech-6 nearest-profile distance**: \|z\| to the nearest class / background log-energy profile |
| Threshold | one per layer | one per class (default) or one per layer |
| Box colour in the picture | green | magenta |

Energy E = (L1 norm of the cell's feature vector across channels)², taken at one backbone layer (`detect_layer`,
default 1 = stride 4, so one cell is 4×4 px of the 640×640 input).

## Folder contents

| File | What it does | Run it? |
|---|---|---|
| `calibrate_detector.py` | builds the class profiles, the thresholds and the minimum region sizes from clean images, and **writes them into the config file** | yes, once per model |
| `detect_patches.py` | runs both localizers on a folder of images: box files per panel, 3-panel pictures, `summary.csv` | yes |
| `evaluate_detector.py` | scores the boxes against the GT patch boxes and counts false alarms on clean images | yes |
| `mech6_core.py` | model loading, energy, profiles, scores, the localizer | no (library) |
| `detector_viz.py` | the 3-panel picture | no (library) |
| `config_yolov5.json`, `config_yolov8.json`, `config_yolov9.json`, `config_yolo11.json` | one config per model | edit |
| `profiles/` | class profiles (`.pkl`) written by calibration | no |

## Setup

- Use the `yolo_adv` conda env (setup steps: `../gen-adv-patch-hiding-latest/README.md`).
- Weights come from `../gen-adv-patch-hiding-latest/weights/`, and the YOLOv5 code from
  `../gen-adv-patch-hiding-latest/yolov5_src/`, so nothing else needs to be installed.
- Clean images: BDD100K at `D:/GitHub/experiment_data/bdd100k_images_10k/` (`train` for profiles, `test` for
  thresholds, `val` for false-alarm checks, so the three never overlap).

## How the thresholds are set (calibration)

Calibration uses **clean images only**. Patched images are used only for evaluation, so no value is fitted to the
current patches.

1. **Pass 1, profiles.** On `profile_images_dir` (BDD `train`) the model's own detections say which cells belong to which
   class; random uncovered cells are background. For each layer and class it stores the mean and std of log-energy,
   plus the global mean/std (used by panel 2). Saved to `profile_file`.
2. **Pass 2, thresholds.** On `threshold_images_dir` (BDD `test`):
   - **panel 2:** the `threshold_quantile` (99.5%) point of the clean energy z, per layer.
   - **panel 3, layer threshold:** the 99.5% point of the clean nearest-profile score, per layer.
   - **panel 3, class thresholds:** for each profile, the 99.5% point over the clean cells whose nearest profile is
     that class. A class with fewer than `min_threshold_cells` such cells gets no entry and uses the layer threshold.
3. **Region sizes.** On the same clean images, the localizer runs with every rule except the minimum size, and the
   size of the largest region in each image is recorded. `min_region_cells` is set so that only
   `target_clean_image_fpr` (2%) of clean images still keep a region. It is computed per layer, for panel 2, and for
   panel 3 in both threshold modes, with the region settings (`join_cells`, `min_thickness_cells`, `seed_ratio`) in the
   config at that time.

All of these values go into the config (`panel2.calibrated` and `panel3.calibrated`), so they can be read and edited
by hand. The previous config is kept as `config_<model>.json.bak`.

## How a region becomes a patch

The same steps run for each panel separately, with that panel's score, thresholds and `region` settings.

1. **Hot cells.** Every cell whose score is above its threshold. In panel 3 `per_class` mode, each cell is compared with
   the threshold of its nearest profile.
2. **Join** (`join_cells`). A morphological closing: hot cells that are up to about 2×`join_cells` cells apart are
   joined. A patch's texture has dark pixels, so its hot cells come out speckled; this turns them into one solid blob.
3. **Thickness** (`min_thickness_cells`). A morphological opening: the blob is shrunk by `min_thickness_cells` cells on
   every side, then grown back. Any part thinner than 2×`min_thickness_cells`+1 cells disappears and does not come back.
   This erases lane markings, poles, wires and object edges **whatever their angle**, while a thick region of **any
   shape** (square, rotated, round, irregular) survives.
   In both the join and the thickness step, everything outside the image counts as *not* hot, so a thin strip along the
   image edge is erased like any other thin line, and a patch touching the edge keeps its full size.
4. **Regions.** The cells that are left are grouped into 8-connected regions.
5. **Region rules.** A region is a patch only if both hold:
   - **size:** it has at least `min_region_cells` cells (calibrated so clean images rarely reach it);
   - **seed:** its strongest cell is at least `seed_ratio` × its threshold, i.e. it contains at least one clearly
     abnormal cell, not just a cluster of cells that barely pass.
6. **Boxes.** Each surviving region becomes its bounding box (cells × stride, in 640×640 px). Overlapping boxes are
   merged into one. There is **no limit** on the number of boxes. An image counts as "patched" if it has at least one box.

Each box gets a **score** = the mean score/threshold ratio of its hot cells (1.0 = just at the threshold) and a **peak** =
the highest ratio.

## Settings: meaning and effect

**Model and files**

| Setting | Meaning | Effect of changing it |
|---|---|---|
| `model_type` | `"ultralytics"` (v8 / v9 / 11) or `"yolov5_repo"` (original YOLOv5 code) | must match the weights |
| `yolov5_repo_dir` | YOLOv5 only: folder with the YOLOv5 `models/` and `utils/` | — |
| `model_path` | weights file | — |
| `profile_file` | where the class profiles are saved / read | point it at another profile file to reuse it (then use `--thresholds-only`) |

**Layers and scores**

| Setting | Meaning | Effect of changing it |
|---|---|---|
| `tap_layers` | layers that get calibrated (1, 2, 3) | only calibrated layers can be used as `detect_layer` |
| `detect_layer` | layer the localizers run on (default 1) | layer 1 carried almost all the signal in the Mech-6 findings. Layers 2 and 3 are coarser or weaker |
| `panel3.score` | `nearest_abs` (distance to the nearest profile in both directions) or `nearest_up` (only more energy than the profile) | the thresholds belong to one score, so changing it needs `--thresholds-only` |
| `panel3.threshold_mode` | `per_class` (each cell uses its nearest profile's threshold) or `per_layer` (one threshold) | changes which cells are hot in panel 3. Both are calibrated, so switching needs no recalibration |

**Calibration (`calibration` section)**

| Setting | Meaning | Effect of changing it |
|---|---|---|
| `profile_images_dir`, `max_calibration_images` | clean images for the profiles, and how many | more images = steadier profiles; slower pass 1 |
| `background_samples_per_image` | random background cells taken per image for the background profile | only the precision of the background profile; little effect |
| `min_calibration_samples` | a class needs at least this many cells to get a profile | higher = fewer, better-supported profiles. Too low gives many rare-class profiles that crowd the energy axis and make per-class thresholds tiny |
| `threshold_images_dir`, `threshold_num_images` | clean images for thresholds and region sizes, and how many | more images = steadier thresholds and region sizes |
| `threshold_quantile` | cell threshold: share of clean cells kept below it (0.995 = 0.5% above) | higher = fewer hot cells: fewer false alarms but weaker patches are missed. Lower = more hot cells, and calibration then raises `min_region_cells` |
| `min_threshold_cells` | panel 3 per-class: a class needs this many clean cells to get its own threshold | higher = more classes fall back to the layer threshold |
| `target_clean_image_fpr` | share of clean images allowed to keep a region (0.02 = 2%). **The main operating point** | lower = bigger `min_region_cells`: fewer false alarms, small patches missed. Higher = the opposite |
| `min_region_cells_floor` | the calibrated minimum region size never goes below this | only matters when clean images almost never produce regions |

**Localizer (`panel2.region` and `panel3.region`, set separately for each panel)**

| Setting | Meaning | Effect of changing it |
|---|---|---|
| `min_region_cells` | smallest region (in cells) that counts as a patch. `"auto"` = the calibrated value; a number overrides it | higher = fewer false alarms but small patches missed. At layer 1, 25 cells ≈ a 20×20 px area |
| `seed_ratio` | a region must contain a cell at least this many times its threshold | higher = only regions with a clearly abnormal core are kept: fewer false alarms, faint patches missed. 1.0 = no seed rule |
| `join_cells` | how far apart hot cells can be and still be joined (closing window 2×`join_cells`+1) | higher = speckled patches become one region more easily, but nearby clutter can also merge into bigger regions. 0 = no joining |
| `min_thickness_cells` | regions must be at least 2×`min_thickness_cells`+1 cells thick (opening window). 1 → 3 cells = 12 px at layer 1 | higher = removes more thin clutter, but also patches thinner than the window. 0 = no thickness test |

After changing any `region` setting, run `calibrate_detector.py --regions-only` so `min_region_cells` is recalibrated
with the new settings (otherwise the clean false-alarm rate is no longer at the target).

**Calibrated values (written by calibration; you may edit them)**

| Setting | Meaning |
|---|---|
| `panel2.calibrated.<layer>.threshold` | panel-2 energy-z threshold |
| `panel2.calibrated.<layer>.min_region_cells` | panel-2 calibrated minimum region size |
| `panel3.calibrated.<layer>.layer_threshold` | panel-3 threshold for the whole layer (used in `per_layer` mode, and for classes without their own) |
| `panel3.calibrated.<layer>.class_thresholds` | panel-3 per-class thresholds, keys `"bg"` or `"<id>_<name>"` (e.g. `"2_car"`). You can add or remove a class by hand |
| `panel3.calibrated.<layer>.min_region_cells` | panel-3 calibrated minimum region size for `per_layer` and `per_class` |
| `panel3.calibrated.<layer>.score` | which score the thresholds belong to (checked when detecting) |
| `calibration_info` | when and how the values were calibrated |

**Running, pictures and evaluation**

| Setting | Meaning |
|---|---|
| `detect.input_dir`, `start_index`, `num_images` | images to run on, and which slice (`null` = all) |
| `detect.detections_dir` | where the box files go |
| `detect.save_visualizations`, `visualizations_dir` | whether to save the pictures, and where |
| `detect.gt_coordinates_dir` | GT patch coordinate files, only for drawing the cyan GT boxes |
| `viz_outline` | draw the thin white outline of hot cells |
| `viz_scale_max_ratio` | red end of each heatmap = this × that panel's layer threshold (blue = 0) |
| `evaluate.patched_detections_dir`, `clean_detections_dir` | detection folders read by `evaluate_detector.py` |
| `evaluate.gt_coordinates_dir` | GT patch boxes |
| `evaluate.match_iou` | IoU needed for a predicted box to count as finding a GT patch (0.3) |

Relative paths are relative to the config file's folder. Most folders and switches can also be given as command-line
flags, and the flag wins.

## Step by step (example: YOLOv8)

**1. Calibrate** (once per model; writes the values into `config_yolov8.json`):
```bash
python calibrate_detector.py --config config_yolov8.json
```
Options:
- `--thresholds-only`: keep the existing profile file and redo the thresholds and region sizes. Use it to reuse the profiles
  of `yolo-mech6-detector` or `yolo-mech6` (set `profile_file` to that `.pkl`). The YOLOv5 profile from
  `yolo-mech6-detector` is already copied into `profiles/`.
- `--regions-only`: keep the profiles and the thresholds in the config (even hand-edited ones) and redo only
  `min_region_cells`. Use it after changing any `region` setting.
- `--max-images 60 --threshold-images 30`: a quick test calibration.

The values calibrated by `yolo-mech6-detector` were **not** copied: its minimum region sizes were made with the old
rules (fill / aspect) and do not fit the lean localizer.

**2. Patched images.** The config points `detect.input_dir` at the model's patched set:
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

**5. (Optional) Compare panel-3 modes** without editing the config:
```bash
python detect_patches.py --config config_yolov8.json --panel3-mode per_layer --detections-dir detections/yolov8/patched-perlayer
```

**6. (Optional) Compare with `yolo-mech6-detector`:** run the same patched and clean images through both folders and
compare the two `evaluate_detector.py` tables.

For other models, use `config_yolov5.json`, `config_yolov9.json` or `config_yolo11.json`.

## Outputs

- `detections/<model>/<set>/panel2/<stem>.txt` and `.../panel3/<stem>.txt`: one line per box,
  `box=(x1, y1, x2, y2) score=… peak=… cells=… orig_box=(…)`. `box` is in 640×640 px, like the GT files; `orig_box` is in the
  original image size. An empty file means no patch was found.
- `detections/<model>/<set>/summary.csv`: number of boxes per panel for every image.
- `viz/<model>/<set>/det_<stem>.png`:
  - left: the image, with GT patches in cyan;
  - middle: energy z, a thin white outline around hot cells, **green boxes** = panel-2 patches;
  - right: nearest-profile distance, a thin white outline around hot cells, **magenta boxes** = panel-3 patches;
  - the header shows the number of boxes per panel with their score and size.

## Reading `evaluate_detector.py`

| Column | Meaning |
|---|---|
| precision | share of predicted boxes that match a GT patch (IoU ≥ `match_iou`, one-to-one) |
| recall | share of GT patches matched by a predicted box |
| f1 | balance of precision and recall |
| mean_iou | how well matched boxes fit. Boxes follow the layer grid, so they are a few px coarse |
| centre_hit | share of GT patches whose centre lies inside some predicted box (looser than IoU) |
| image_recall | share of patched images with at least one box |
| clean_image_fpr | share of clean images with at least one box (should be near `target_clean_image_fpr`) |
| boxes_per_clean_image | average number of false boxes per clean image |

## Assumptions that remain (stated, not removable)

- **The patch changes the backbone's energy locally.** This is the hypothesis being tested. A patch that doesn't change
  energy at `detect_layer` is invisible to this detector.
- **Panel 2 counts only higher-than-usual energy** (z clipped at 0). A low-energy patch would show only in panel 3,
  whose `nearest_abs` score is two-sided.
- **The patch is thicker than 2×`min_thickness_cells`+1 cells and larger than `min_region_cells`.** That is the smallest
  detectable patch at the chosen false-alarm rate, and it is worth reporting in px with the results.
- **Calibration describes one domain** (clean BDD images). For another domain (night, rain, a different camera) it has
  to be redone on clean images from that domain. Only clean images are needed.

## A note on per-class thresholds for panel 3

Energy is a single number, so "nearest profile" cuts the log-energy axis into ranges, one per profile. Profiles with
very close means each own only a narrow range, so their per-class thresholds come out tiny, but so does every score in
that range. Patch cells have very high energy and land in the highest profile, so the panel-3 per-class decision for
patches is made mostly by that top profile's threshold. Many rare-class profiles make this stronger; raising
`min_calibration_samples` keeps only well-supported classes.

## Status

Smoke-tested on YOLOv8 with a small calibration. No full calibration has been run yet.
