# Mech-6-Mahalanobis detector

`yolo-mech6-mahalanobis` turned into a patch **detector**, built the same way as `yolo-mech6-detector-lean-v3`.
It works on **YOLOv5, YOLOv8, YOLOv9 and YOLO11**.
- **Signal** (from `yolo-mech6-mahalanobis`): each cell's whole feature vector, scored by its Mahalanobis distance to the nearest class or background profile.
- **Localizer** (from lean-v3): hot cells → join → multi-scale thickness test → region mass, with mass limits calibrated jointly on clean images.

`yolo-mech6-mahalanobis` itself is unchanged. That folder makes heatmaps only; this one outputs patch boxes.

## The idea in short

1. **Score per cell.** At backbone layer `detect_layer` (default 1, stride 4), a cell is its C-channel feature vector `x`.
   Each object class and background has a profile: mean vector `mu_k` and covariance `S_k`.
   `score(x) = min_k sqrt((x - mu_k)^T S_k^-1 (x - mu_k))` is the distance to the closest profile, in "standard deviations"
   that account for the correlations between channels.
   - Normal content is close to some profile; a patch's texture is far from all of them.
   - The profile is picked by the distance itself, never by the detector's boxes, so a patch that fools the detector can't
     choose its own comparison.
2. **Two localizers on that one score.** They differ only in which threshold a cell is compared with:

   | Panel | Threshold | Where it comes from |
   |---|---|---|
   | **panel 2** | **per-layer**: one threshold for every cell | the 99.5% point of all clean cells' distances |
   | **panel 3** | **per-class**: each cell uses its nearest profile's threshold | the 99.5% point over the clean cells nearest to that profile (a profile with fewer than `min_threshold_cells` such cells uses the layer threshold) |

   These are the two `threshold_mode`s of `yolo-mech6-mahalanobis`, now run side by side. The distance is computed once
   per image, so the second panel costs only a threshold map and one more localization.
3. **Boxes.** This is lean-v3's localizer, copied unchanged:
   - hot cells: `ratio = distance / threshold > 1`;
   - join: closing, 3×3 window;
   - for each thickness `t` in `thickness_scales` [0, 1, 2, 3]: opening with a (2t+1)² window, then 8-connected
     regions, then `mass = Σ(ratio − 1)` over the region's hot cells; keep the region if the mass is above that scale's limit;
   - combine the scales thickest first, skipping regions that overlap one already kept.

**Compared with lean-v3:** the only change is the score.
- Lean-v3 panel 2 is the energy z: how *strongly* the layer fires.
- Lean-v3 panel 3 is the log-energy distance to the nearest class: one number per cell.
- Here both panels use the full feature vector, so they also catch the wrong *kind* of activity at a normal strength.

## Files

| File | What it does | Run it? |
|---|---|---|
| `calibrate_detector.py` | profiles + thresholds + per-scale mass limits for one model, **written into the config** | yes, once per model |
| `detect_patches.py` | runs both localizers on a folder: box files, `summary.csv`, 3-panel pictures | yes |
| `evaluate_detector.py` | quick IoU-based check of one run (precision, recall, clean-image FPR) | optional |
| `measure_latency.py` | per-image latency of each panel, writes `latency/<model>.json` | yes, for the charts |
| `maha_core.py` | model loading, layer hook, Mahalanobis profiles, localizer | no (library) |
| `detector_viz.py` | draws the 3-panel picture | no (library) |
| `config_yolov5.json`, `config_yolov8.json`, `config_yolov9.json`, `config_yolo11.json` | one per model | edit |

Use the `yolo_adv` conda env. The weights come from `../gen-adv-patch-hiding-latest/weights/`, and the YOLOv5 code from
`../gen-adv-patch-hiding-latest/yolov5_src/`.

## Calibration

Each config's results go into `panel2.calibrated` and `panel3.calibrated`, and the old config is kept as `<config>.json.bak`.
1. **Profiles (pass 1).** The first `max_calibration_images` images of BDD `train`.
   - The model's own boxes (offline only) assign cells to classes; `background_samples_per_image` random uncovered cells go to background.
   - Statistics are kept as running sums, so memory stays flat.
   - A class needs at least `max(min_calibration_samples, 2 × channels)` cells.
   - The covariance is shrunk towards a scaled identity (`shrinkage`, 0.01) before it is inverted.
   - The result is saved to `profile_file`. The `.pkl` has the same format as `yolo-mech6-mahalanobis`.
2. **Thresholds (pass 2).** The first `threshold_num_images` images of BDD `test`, which are different images.
   - Panel 2: `threshold` = the `threshold_quantile` (0.995) of every clean cell's distance.
   - Panel 3: `class_thresholds` = the same quantile per nearest profile. `layer_threshold` is the fallback.
   - `clean_median` sets the blue end of the picture's colour scale.
3. **Mass limits.** The same clean images, the same method as lean-v3, per panel.
   - For each clean image and scale, take the heaviest region.
   - Each scale's limit is the (k+1)-th largest of those values, and k is raised as far as possible while at most
     `target_clean_image_fpr` (2%) of clean images are flagged at any scale.

```bash
python calibrate_detector.py --config config_yolov8.json                     # everything
python calibrate_detector.py --config config_yolov8.json --thresholds-only   # keep the profiles, redo 2 + 3
python calibrate_detector.py --config config_yolov8.json --regions-only      # keep profiles + thresholds, redo 3
```

**Profiles already copied in.** Two full profile files from `yolo-mech6-mahalanobis/profiles/` (7000 images) are in `profiles/`:
- `maha_profiles_yolov5.pkl` is a copy of `maha_profiles_yolov5-v2.pkl`. It was built with `min_calibration_samples` 100,
  so it has 53 profiles at layer 1.
- `maha_profiles_yolov8.pkl` is a copy of `maha_profiles_yolov8-v3.pkl`. It was built with 20000, so it has 14 profiles at layer 1.

For these two models, `--thresholds-only` is enough. YOLOv5 then uses the old 53-profile setting; for the same setting as
the other models (20000, as in lean-v3), run the full calibration. YOLOv9 and YOLO11 need the full calibration.

## Step by step (YOLOv5 example; swap the config for the other models)

**1. Calibrate.**
```bash
python calibrate_detector.py --config config_yolov5.json --thresholds-only
```

**2. Patched images.** `detect.input_dir` and `detect.gt_coordinates_dir` already point at the model's patched set:
```bash
python detect_patches.py --config config_yolov5.json
```
Boxes go to `detections/yolov5/patched-layer1/panel2|panel3/`, and pictures to `viz/yolov5/patched-layer1/`.

**3. Clean images** (BDD `val` 0–99, for the false-positive rate):
```bash
python detect_patches.py --config config_yolov5.json --input-dir D:/GitHub/experiment_data/bdd100k_images_10k/val --num-images 100 --detections-dir detections/yolov5/clean-layer1 --visualizations-dir viz/yolov5/clean-layer1
```

**4. Latency:**
```bash
python measure_latency.py --config config_yolov5.json --input-dir ../gen-adv-patch-hiding-latest/patched/yolov5-images --out latency/yolov5.json
```

**5. Metrics and charts for all four models.** This uses the same script and metric as lean-v3: a box is correct if
≥ 50% of it lies inside the patches, and a patch counts as found if its correct boxes cover ≥ 25% of it.
```bash
python ../test-patch-det-multi-img/calc-metrics/compare_mech6_models.py --config ../test-patch-det-multi-img/calc-metrics/compare_mech6_mahalanobis_config.json
```
Output goes to `test-patch-det-multi-img/calc-metrics/patch-detection-metrics/mech-6-mahalanobis-detector/`:
`metrics.json`, `metrics.csv` and one chart per panel.

| Model | Patched images | Coordinates |
|---|---|---|
| YOLOv5 | `gen-adv-patch-hiding-latest/patched/yolov5-images` | `.../yolov5-coordinates` |
| YOLOv8 | `test-patch-det-multi-img/gen-patched-img/yolov8-patched-imgs` | `.../yolov8-patch_coordinates` |
| YOLOv9 | `gen-adv-patch-hiding-latest/patched/yolov9-images` | `.../yolov9-coordinates` |
| YOLO11 | `test-patch-det-multi-img/gen-patched-img/yolov11-patched-imgs` | `.../yolov11-patch_coordinates` |

## Outputs

- **Box files**, one line per box:
  `box=(x1, y1, x2, y2) mass=.. scale=.. score=.. peak=.. cells=.. orig_box=(..)`.
  - `box` is in 640×640 px, the same as the GT files; `orig_box` is in the original image size.
  - `scale` is the thickness that found the box.
  - An empty file means no patch was found.
- **Picture**, three panels:
  - the image with GT patches in cyan;
  - the distance map with per-layer hot cells outlined in white and panel-2 boxes in green;
  - the same map with per-class hot cells and panel-3 boxes in magenta.

  Both heatmaps share one colour scale: blue = clean median, red = `viz_scale_max_ratio` × layer threshold.
  Box labels show `mass t<scale>`.
- **Latency JSON**, using the same definition as Mech-1 and lean-v3.
  - Timed: forward pass + distance (shared), then that panel's threshold map + localization.
  - Not timed: image loading, pictures and file writes.
  - The GPU is synced before each clock read, and the first 5 images are a warm-up.

## Config keys

| Key | Meaning |
|---|---|
| `model_type`, `yolov5_repo_dir`, `model_path` | which model and how to load it |
| `tap_layers` | layers profiled and calibrated (1, 2, 3) |
| `detect_layer` | layer used for detection (default 1); `--layer` overrides it |
| `profile_file` | Mahalanobis profiles (.pkl) |
| `calibration.profile_images_dir`, `max_calibration_images`, `background_samples_per_image`, `min_calibration_samples`, `shrinkage` | pass 1 |
| `calibration.threshold_images_dir`, `threshold_num_images`, `threshold_quantile`, `min_threshold_cells` | pass 2 |
| `calibration.target_clean_image_fpr`, `thickness_scales` | mass limits |
| `panel2.region` / `panel3.region` | `join_cells` (closing window = 2j+1) and `mass_thresholds`. `"auto"` uses the calibrated values; you can also give a dict `{"0": .., "1": ..}`, and a scale left out or set to `null` is switched off |
| `panel2.calibrated.<layer>` | `threshold`, `clean_median`, `region` (written by calibration) |
| `panel3.calibrated.<layer>` | `layer_threshold`, `class_thresholds`, `region` (written by calibration) |
| `detect.*` | input folder, output folders, slice, pictures on/off, GT folder |
| `evaluate.*` | default folders for `evaluate_detector.py` |
| `viz_outline`, `viz_scale_max_ratio` | picture settings |

Relative paths are relative to the config file. Command-line flags override the config.

## Where the ideas come from

The localizer, thresholds, mass and metric are the same as lean-v3, so the supporting papers in
`yolo-mech6-detector-lean-v3/README.md` ("Where each idea comes from") apply here too. The changes in this version are:

| Component | Supporting paper |
|---|---|
| Per-class Gaussian profiles of hidden features, score = Mahalanobis distance to the closest class | Lee et al., *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*, NeurIPS 2018 |
| Per-location Gaussian of CNN features from normal images only, Mahalanobis distance as an anomaly map | Defard et al., *PaDiM*, ICPR Workshops 2021 |
| Covariance shrunk towards a scaled identity so the inverse is stable | Ledoit & Wolf, *A well-conditioned estimator for large-dimensional covariance matrices*, J. Multivariate Analysis 2004 |

## Status

The code was smoke-tested on 2026-11-15, and all smoke files were removed afterwards.
- **YOLOv5:** full calibration with a tiny setting (30 profile images, 40 clean images), detection on 4 patched and 4 clean
  images, latency, and `evaluate_detector.py`.
- **YOLOv8:** `--thresholds-only` with the copied profile, then detection.

No full calibration or evaluation has been run, so there are no results yet.
