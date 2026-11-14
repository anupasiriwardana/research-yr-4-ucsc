# Mech-6 detector, lean-v3 (patch localizer)

A copy of `yolo-mech6-detector-lean-v2` in which the **thickness is no longer chosen during calibration**. The detector
tries every thickness scale on every image and keeps a region if it passes at any scale. It turns the two Mech-6
heatmaps into **two separate patch localizers** that output boxes, for **YOLOv5, YOLOv8, YOLOv9 and YOLO11**.

**What changed from `yolo-mech6-detector-lean-v2`:**

| lean-v2 | lean-v3 | Why |
|---|---|---|
| calibration picks **one** thickness (`min_thickness_cells`) | **every** scale in `thickness_scales` is used at detection | the image decides the scale: a small, strong patch is caught at a thin scale, a big, mild one at a thick scale |
| one `mass_threshold` (for the chosen thickness) | **`mass_thresholds`**, one per scale, set **together** | trying several scales gives clean images several chances to trigger, so the limits are tightened jointly to keep the overall clean false-alarm rate at the target |
| `reference_excess`, `smallest_detectable_side_px`, `search` | removed | they were only needed to choose a thickness. `reference_excess` also hid an assumption (that a patch's cells are only as abnormal as typical clean hot cells) |
| `thickness_candidates` | `thickness_scales` | same list, but now these are the scales the detector uses, not options to choose from |

Everything else (scores, thresholds, per-class thresholds, both panel-3 modes, `join_cells`, `detect_layer`, the outputs)
is the same. The other folders are untouched, so all versions can be compared on the same images.

**Lineage:** `yolo-mech6-detector` (fill, aspect, max detections, seed, size) → `-lean` (thickness instead of fill and
aspect; no box limit) → `-lean-v2` (mass instead of seed and size; thickness calibrated) → `-lean-v3` (all thickness
scales at detection; mass limits calibrated jointly).

## The two localizers

| | Panel 2 localizer | Panel 3 localizer |
|---|---|---|
| Score per cell | **energy z** = (log E − global mean) / global std, clipped at 0 | **Mech-6 nearest-profile distance**: \|z\| to the nearest class / background log-energy profile (`nearest_abs`), or only above it (`nearest_up`) |
| Threshold | one per layer | one per class (default) or one per layer |
| Box colour in the picture | green | magenta |

Energy E = (L1 norm of the cell's feature vector across channels)², at one backbone layer (`detect_layer`, default
1 = stride 4, so one cell is 4×4 px of the 640×640 input).

## Folder contents

| File | What it does | Run it? |
|---|---|---|
| `calibrate_detector.py` | builds the class profiles, the thresholds and the per-scale mass thresholds from clean images, and **writes them into the config file** | yes, once per model |
| `detect_patches.py` | runs both localizers on a folder of images: box files per panel, 3-panel pictures, `summary.csv` | yes |
| `evaluate_detector.py` | scores the boxes against the GT patch boxes and counts false alarms on clean images | yes |
| `mech6_core.py` | model loading, energy, profiles, scores, the localizer | no (library) |
| `detector_viz.py` | the 3-panel picture | no (library) |
| `config_yolov5.json`, `config_yolov8.json`, `config_yolov9.json`, `config_yolo11.json` | one config per model | edit |
| `profiles/` | class profiles (`.pkl`); the YOLOv5 and YOLOv8 profiles from `-lean-v2` are copied in | no |

## Setup

- Use the `yolo_adv` conda env (setup steps: `../gen-adv-patch-hiding-latest/README.md`).
- Weights come from `../gen-adv-patch-hiding-latest/weights/`, and the YOLOv5 code from
  `../gen-adv-patch-hiding-latest/yolov5_src/`.
- Clean images: BDD100K at `D:/GitHub/experiment_data/bdd100k_images_10k/`. `train` is used for profiles, `test` for
  thresholds and region settings, and `val` for false-alarm checks.

## Calibration (clean images only)

Patched images are used only for evaluation, so no value is fitted to the current patches.

1. **Pass 1, profiles** (`profile_images_dir`, BDD `train`). The model's own detections say which cells belong to which
   class; random uncovered cells are background. Per layer and class: mean and std of log-energy, plus the global
   mean/std (for panel 2). Saved to `profile_file`.
2. **Pass 2, thresholds** (`threshold_images_dir`, BDD `test`):
   - **panel 2:** the `threshold_quantile` (99.5%) point of the clean energy z, per layer.
   - **panel 3, layer threshold:** the 99.5% point of the clean nearest-profile score, per layer.
   - **panel 3, class thresholds:** for each profile, the 99.5% point over the clean cells whose nearest profile is that
     class. A class with fewer than `min_threshold_cells` such cells uses the layer threshold.
3. **Per-scale mass thresholds** (same clean images). Done separately per layer, for panel 2, and for panel 3 in both
   threshold modes:
   1. **For every clean image and every scale t in `thickness_scales`:** run the localizer with no mass rule and record
      the largest region mass (0 if no region survives that scale). This gives a table: images × scales.
   2. **Same strictness at every scale:** each scale's threshold is set so that roughly the same number k of clean
      images is above it at every scale.
   3. **Tighten together:** k is raised as far as possible while the share of clean images that would be flagged at
      **any** scale stays ≤ `target_clean_image_fpr` (2%).

   So the 2% is the false-alarm rate of the **whole** detector, not of each scale. No thickness is chosen: every scale
   keeps its own threshold and detection uses them all.

   A scale's threshold can come out as **0**: no clean image had any region that survived that scale. Then any region
   that survives that thickness counts as a patch. This is a result of the data, not a setting.

All values go into the config (`panel2.calibrated` and `panel3.calibrated`), with the clean flag rate overall and per
scale. The previous config is kept as `config_<model>.json.bak`.

## How a region becomes a patch

The same steps run for each panel separately, with that panel's score, thresholds and region settings.

1. **Hot cells.** Every cell whose score is above its threshold. In panel 3 `per_class` mode, each cell is compared with
   the threshold of its nearest profile.
2. **Join** (`join_cells`). A morphological closing: hot cells up to about 2×`join_cells` cells apart are joined. A patch's
   texture has dark pixels, so its hot cells come out speckled; this turns them into one solid blob.
3. **For every thickness scale t** (`thickness_scales`, default 0, 1, 2, 3):
   1. **Thickness.** A morphological opening: the blob is shrunk by t cells on every side, then grown back. Any part
      thinner than 2t+1 cells disappears. This erases lane markings, poles, wires and object edges **whatever their
      angle**, while a thick region of **any shape** survives. t = 0 means no thickness test.
   2. **Regions.** The cells that are left are grouped into 8-connected regions.
   3. **Mass.** For each region: mass = the sum over its hot cells of (score/threshold − 1), i.e. how many cells are hot
      and how far above the threshold they are. A large region of moderately abnormal cells and a small region of very
      abnormal cells can both reach the same mass; a few barely-hot cells give a tiny mass.
   4. A region is a patch **at this scale** if **mass > that scale's mass threshold**. This is the "cluster mass" test
      used in brain imaging (fMRI) to decide whether a blob of active voxels is real.

   In both the join and the thickness step, everything outside the image counts as *not* hot, so a thin strip along the
   image edge is erased like any other thin line, and a patch touching the edge keeps its full size.
4. **Combine the scales, thickest first.** A region is kept unless it overlaps a region already kept from a thicker
   scale. The thicker scale wins because its box has thin attachments (a lane line touching the patch) trimmed off,
   while a solid patch keeps its full box after an opening.
   - small, strong patch (thinner than the thick window): erased at thick scales, **found at a thin scale**;
   - big, mild patch: too light at thin scales (high limit), **found at a thick scale** (low limit);
   - thin clutter: erased at thick scales and too light at thin ones, so **dropped**.
5. **Boxes.** Each kept region becomes its bounding box (cells × stride, in 640×640 px). Within one scale, overlapping
   boxes are merged (their masses add up). There is **no limit** on the number of boxes. An image counts as "patched" if
   it has at least one box.

Each box reports its **mass**, its **scale** (the thickness that found it), its **score** (mean score/threshold of its
hot cells) and its **peak** (highest ratio).

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
| `detect_layer` | layer the localizers run on (default 1) | layer 1 carried almost all the signal in the Mech-6 findings |
| `panel3.score` | `nearest_abs` (distance to the nearest profile in both directions) or `nearest_up` (only more energy) | the thresholds belong to one score, so changing it needs `--thresholds-only` |
| `panel3.threshold_mode` | `per_class` or `per_layer` | changes which cells are hot in panel 3. Both are calibrated, so switching needs no recalibration |

**Calibration (`calibration` section)**

| Setting | Meaning | Effect of changing it |
|---|---|---|
| `profile_images_dir`, `max_calibration_images` | clean images for the profiles, and how many | more = steadier profiles; slower pass 1 |
| `background_samples_per_image` | random background cells per image for the background profile | little effect |
| `min_calibration_samples` | a class needs at least this many cells to get a profile | higher = fewer, better-supported profiles. Too low gives many rare-class profiles that make per-class thresholds tiny |
| `threshold_images_dir`, `threshold_num_images` | clean images for thresholds and region settings, and how many | more = steadier values. The 2% point rests on about 2% of these images, so use many (2000) |
| `threshold_quantile` | cell threshold: share of clean cells kept below it (0.995 = 0.5% above) | higher = fewer hot cells: weaker patches missed. Lower = more hot cells; the mass thresholds then rise to compensate |
| `min_threshold_cells` | panel 3 per-class: a class needs this many clean cells for its own threshold | higher = more classes use the layer threshold |
| `target_clean_image_fpr` | share of clean images allowed to keep a region at any scale (0.02 = 2%). **The main operating point** | lower = higher mass thresholds: fewer false alarms, small or faint patches missed. Higher = the opposite |
| `thickness_scales` | the thickness scales the detector tries (default [0, 1, 2, 3]) | a range, not a tuned value. More scales = more patch sizes covered, but each scale shares the same 2% budget, so every limit rises a little |

**Localizer (`panel2.region` and `panel3.region`, separately for each panel)**

| Setting | Meaning | Effect of changing it |
|---|---|---|
| `join_cells` | how far apart hot cells can be and still be joined (closing window 2×`join_cells`+1) | higher = speckled patches join more easily, but nearby clutter merges into bigger regions too. 0 = no joining |
| `mass_thresholds` | `"auto"` = the calibrated values; or set by hand as `{"0": 40.9, "1": 13.1, "2": 10.3, "3": 7.1}`. Leave a scale out, or set it to `null`, to switch it off | higher = fewer false alarms at that scale, small or faint patches missed |

After changing `join_cells`, `threshold_quantile`, `thickness_scales` or a threshold by hand, run
`calibrate_detector.py --regions-only` so the mass thresholds are recalibrated. Otherwise the clean false-alarm rate is
no longer at the target.

**Calibrated values (written by calibration; you may edit them)**

| Setting | Meaning |
|---|---|
| `panel2.calibrated.<layer>.threshold` | panel-2 energy-z threshold |
| `panel3.calibrated.<layer>.layer_threshold` | panel-3 threshold for the whole layer (`per_layer` mode, and classes without their own) |
| `panel3.calibrated.<layer>.class_thresholds` | panel-3 per-class thresholds, keys `"bg"` or `"<id>_<name>"` (e.g. `"2_car"`) |
| `panel3.calibrated.<layer>.score` | which score the thresholds belong to (checked when detecting) |
| `…calibrated.<layer>.region` (panel 2) and `…calibrated.<layer>.region.<mode>` (panel 3) | `mass_thresholds` (one per scale), `clean_images_flagged` (share of clean images flagged at any scale; should be ≤ the target), `clean_images_flagged_per_scale` |
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

## Step by step (example: YOLOv5)

The profiles and thresholds of `-lean-v2` were copied into the YOLOv5 and YOLOv8 configs (they are computed the same
way). The v2 region values were **removed**, because v3 stores per-scale thresholds instead. So for those two models
start at `--regions-only`.

**1. Calibrate** (writes the values into `config_yolov5.json`):
```bash
python calibrate_detector.py --config config_yolov5.json --regions-only
```
Options:
- (no flag): everything, including new profiles. Needed for YOLOv9 and YOLO11.
- `--thresholds-only`: keep the existing profile file and redo the thresholds and the mass thresholds (needed after
  changing `panel3.score`).
- `--regions-only`: keep the profiles and thresholds (even hand-edited ones) and redo only the per-scale mass thresholds.
- `--max-images 60 --threshold-images 100`: a quick test calibration.

**2. Patched images:**
```bash
python detect_patches.py --config config_yolov5.json
```
(uses `detect.input_dir`, `detect.detections_dir` and `detect.visualizations_dir`; or pass `--input-dir`,
`--detections-dir`, `--visualizations-dir`)

**3. Clean images** (false alarms), BDD `val` 0–99:
```bash
python detect_patches.py --config config_yolov5.json --input-dir D:/GitHub/experiment_data/bdd100k_images_10k/val --num-images 100 --detections-dir detections/yolov5/clean-layer1 --visualizations-dir viz/yolov5/clean-layer1
```

**4. Numbers** (reads `evaluate.patched_detections_dir` and `evaluate.clean_detections_dir`):
```bash
python evaluate_detector.py --config config_yolov5.json --out results/yolov5.json
```

**5. (Optional) Compare panel-3 threshold modes** without editing the config:
```bash
python detect_patches.py --config config_yolov5.json --panel3-mode per_layer --detections-dir detections/yolov5/patched-perlayer
```

**Comparing panel-3 scores (`nearest_abs` vs `nearest_up`):** the thresholds belong to one score, so each needs its own
`--thresholds-only` calibration, its own patched run **and its own clean run**. A clean run made with one score does not
give the false-alarm rate of the other.

**6. (Optional) Compare with `-lean-v2`:** run the same patched and clean images through both folders and compare the
`evaluate_detector.py` tables. The `scale` column in the box files shows which thickness found each patch.

For other models, use `config_yolov8.json`, `config_yolov9.json` or `config_yolo11.json`.

## Outputs

- `detections/<model>/<set>/panel2/<stem>.txt` and `.../panel3/<stem>.txt`: one line per box,
  `box=(x1, y1, x2, y2) mass=… scale=… score=… peak=… cells=… orig_box=(…)`. `box` is in 640×640 px, like the GT files;
  `orig_box` is in the original image size. An empty file means no patch was found.
- `detections/<model>/<set>/summary.csv`: number of boxes, the largest mass and its scale per panel for every image.
- `viz/<model>/<set>/det_<stem>.png`:
  - left: the image, with GT patches in cyan;
  - middle: energy z, a thin white outline around hot cells, **green boxes** = panel-2 patches (with mass and scale);
  - right: nearest-profile distance, a thin white outline around hot cells, **magenta boxes** = panel-3 patches (with mass and scale);
  - the header shows the number of boxes per panel with their mass, scale and size.

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

- **The patch changes the backbone's energy locally.** This is the hypothesis being tested.
- **Panel 2 counts only higher-than-usual energy** (z clipped at 0). A low-energy patch would show only in panel 3.
- **A patch is one solid region** that, at some scale in `thickness_scales`, is thick enough to survive and has enough
  mass. Patches thinner than 1 cell after joining, or so faint that even the thinnest scale's limit is not reached, are
  missed.
- **Calibration describes one domain** (clean BDD images). For another domain it has to be redone on clean images from
  that domain. Only clean images are needed.

## A note on per-class thresholds for panel 3

Energy is a single number, so "nearest profile" cuts the log-energy axis into ranges, one per profile. Profiles with
very close means each own only a narrow range, so their per-class thresholds come out tiny. Patch cells have very high
energy and land in the highest profile, so the panel-3 per-class decision for patches is made mostly by that top
profile's threshold. Many rare-class profiles make this stronger; raising `min_calibration_samples` keeps only
well-supported classes.

## Status

Smoke-tested on YOLOv5 (layer 1) with a small region calibration (150 clean images) on 20 patched images: both panels
found patches in 19/20 images, all at scale 3 for these patch sizes. A synthetic test confirmed the multi-scale rule: a
4×4-cell strong patch was found at scale 1 (a single scale 3, as in v2, misses it), a 20×20 mild patch at scale 3 with
its full box, and a 2-cell-thick streak was dropped. No full calibration has been run yet: run `--regions-only` with
2000 clean images first.
