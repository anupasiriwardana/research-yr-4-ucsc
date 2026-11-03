# Mech-5: multi-scale (P3 + P4) patch detection

Mech-5 finds adversarial patches by asking **two YOLO feature scales at the same time** whether a part of the image
looks abnormal, and only trusting the answer when they agree. It outputs patch boxes in 640x640 pixel space, in the
same format as Mech-1 and Mech-7, so the metrics pipeline and the later recovery module work unchanged.

It is a separate folder from `yolo-mech7`; nothing here imports from it. (The `profiles/` and `yolo11n.pt` were
copied from there so this folder runs on its own.)

## The idea in simple terms

YOLO looks at the image on a grid, at several resolutions: **P3** (one cell = 8x8 px, good for small things),
**P4** (16x16 px) and **P5** (32x32 px, big things). For each cell, the classification branch produces an
80-number feature vector.

Before this mechanism ran, we learned from clean images what that vector looks like for each object class and for
plain background (a **profile**: mean + covariance). At runtime, for each cell we measure the **Mahalanobis distance**
to every profile and keep the smallest one. A normal cell is close to some profile. A patch cell looks like nothing
the model knows, so its distance is large.

Mech-1 did this at P3 only with one threshold. Mech-7 tried per-class thresholds and showed that the threshold was not
the real problem: the patch cells are only slightly above normal at any single scale, and the false alarms are
mostly isolated cells at one scale. Mech-5 uses that: a real patch is large, so it should stand out at **both** P3
and P4, while a false alarm usually only shows up at one. Agreement between scales cuts false alarms without
raising the threshold so high that the patch is lost.

## How it works, step by step

1. **One forward pass.** The adapter hooks the classification branch at P3 and P4 (the penultimate block of
   `cv3`, before the 1x1 projection to class logits). The detector's own boxes are never used.
2. **Ratio per scale.** For every cell: `ratio = nearest-profile distance / that profile's own threshold`
   (thresholds are learned per class and per scale on a clean image set; `adjust_threshold` is added to them).
   `ratio > 1` means "this scale flags the cell". Dividing by the threshold puts P3 and P4 on the same footing.
3. **Align the grids.** P4 has a 40x40 grid and P3 an 80x80 grid. Each P4 cell is repeated 2x2, so all maps are on the
   finest (P3) grid.
4. **Combine with a rule** (`and` or `confirm`, below). The result is one combined score per cell.
5. **Flag cells.** A cell is flagged when `combined score > 1 + offset`.
6. **Join and size filter.** Flagged cells are dilated `join` times (3x3) so scattered cells of one patch merge into one
   region. Each region becomes a box from its **original** flagged cells; regions with fewer than `min_region_cells`
   original cells are dropped.
7. **Output.** Box = bounding box of the region (cell index x 8 px). Box score = highest combined score in it.

## The two rules (switchable)

Set `detector_settings.combine_rule` to `"and"` or `"confirm"`, or override per run with `--rule and|confirm`.
Each rule has its own tuned numbers in `detector_settings.rules`.

### `and`: every scale must flag the cell
The combined score is the **smaller** of the two ratios. A cell counts only if P3 *and* P4 both exceed their thresholds.
- Simple, strict.
- Example: a cell has P3 ratio 1.3 and P4 ratio 0.7. Combined = 0.7, so it is not flagged.
- Good recall among the two rules and the best F1 in our offline test.

### `confirm`: P3 decides, P4 confirms nearby
P3 gives the fine, precise location. A P3 cell only counts if P4 also flags something **within one cell of it** (3x3
neighbourhood on the P3 grid). The P4 map is max-filtered first, so a P4 hit "reaches" the cells around it.
- Example: the same cell with P3 ratio 1.3 and P4 ratio 0.7 is not flagged, but if a neighbouring cell has P4 ratio 1.1, it is.
  This forgives the small misalignment between the 8 px and 16 px grids.
- Slightly fewer false boxes and better box placement (higher IoU) than `and`, a little lower recall.

Rule of thumb: start with `and`; try `confirm` if you want cleaner, more precise boxes.

## Settings (`config.json`)

| key | meaning |
|---|---|
| `scales` | the tapped scales: `stride`, profile file, threshold file per scale. P3+P4 by default (at least two scales; P5 is possible but needs its own profile and threshold files) |
| `detector_settings.combine_rule` | `and` or `confirm` |
| `rules.<rule>.offset` | flag when score > 1 + offset. Negative = more sensitive (more patch cells, more false alarms) |
| `rules.<rule>.join` | how many times flagged cells are dilated before grouping (0 = no joining) |
| `rules.<rule>.min_region_cells` | smallest region kept, counted in finest-grid (8 px) cells |
| `adjust_threshold` | added to every learned threshold (distance units); changes both scales at once |
| `threshold_percentile`, `min_class_samples_for_own_threshold`, `background_*`, `min_calibration_samples` | calibration settings (only used by the two calibration scripts) |
| `batch_settings.save_coordinates` | write the per-image `<stem>.txt` box files (off = only `run_summary.json`) |
| `batch_settings.save_visualizations` | write an overlay PNG per image: heat = combined score minus its flag threshold, red = detected boxes, green = YOLO's own boxes |

The default numbers come from an offline sweep on saved P3/P4 score maps (see Results).

## Workflow

The P3/P4 profiles and thresholds in `profiles/` are ready to use, so you can skip steps 1-2.

1. `python calibrate_profiles.py`: builds the per-class (+ background) profiles of every scale from `profile_images_dir`.
2. `python learn_thresholds.py`: learns per-class thresholds of every scale from `threshold_images_dir` (a **different** clean set).
3. `python detect_mech5_multi_img.py`: runs a folder, writes `<stem>.txt` per image (Mech-1 format) and `run_summary.json`.
   - patched images: `--input-dir <patched> --coordinates-dir <out>`, then `calc-metrics/compare_patch_detection_metrics.py`
   - clean images: `images_with_detection / images` in `run_summary.json` is the image-level false-positive rate
   - `--rule confirm`, `--no-visualizations`, `--no-save-coordinates`, `--num-images`, `--start-index` as needed
4. Metrics: point `calc-metrics/metrics_config.json` at the output folder.

## Results so far (YOLO11n, BDD100K, 100 patched images with 171 patches; 400 clean images not used in tuning)

Settings chosen by an offline sweep (`yolo-mech7/combine_scales.py`) that optimises object F1 under a 20% clean-image
false-positive cap. Held-out check: 20 random 50/50 splits gave F1 about 0.53-0.55 for these rules vs 0.31 for Mech-7 P3.

| | precision | recall | F1 | mean IoU | boxes / patched img | clean images with any box |
|---|---|---|---|---|---|---|
| `yolo-transferable-mech1` (baseline) | 0.376 | **0.854** | 0.522 | 0.322 | 3.88 | high (5 of 5 on the small clean set) |
| Mech-5 `and` | 0.779 | 0.433 | **0.556** | 0.616 | 0.95 | 80 / 400 = 20% |
| Mech-5 `confirm` | **0.820** | 0.427 | **0.562** | **0.636** | 0.89 | 77 / 400 = 19% |

How to read this honestly:
- Mech-5 fixes the YOLO11 **false-positive** problem (precision roughly doubles, boxes per image drop from 3.9 to under 1,
  and the boxes are much better placed) but it **misses more than half of the patches** (recall 0.43 vs 0.85). F1 is about
  equal only because precision rose as much as recall fell. Which is better depends on the downstream recovery module:
  if missing a patch is worse than checking a few extra boxes, the baseline's operating point may still be preferable.
- A lower `offset` / smaller `min_region_cells` moves Mech-5 towards higher recall and more false alarms. Re-run the offline
  sweep with a recall-oriented objective to pick another point.
- YOLOv8 has not been tested with Mech-5 (no profiles built for it yet); the original problem was YOLO11.
- Image-level metrics on the 5 clean images of the patch set are too small to mean anything; use the 400-image clean run.
- Latency: one forward pass plus two scales of scoring, about 50 ms median on the RTX 2050 laptop GPU (the first image is
  slower because of GPU warm-up, which inflates the mean).

## Files

| file | purpose |
|---|---|
| `config.json` | all paths and settings |
| `mech5_core.py` | config, image loading, `ProfileBank` (distance to all profiles at once), `ThresholdTable`, `flagged_regions` (join + size filter + boxes) |
| `mech5_detector.py` | `Mech5Detector`: tap scales, ratios, combine rule, boxes, overlay |
| `ultralytics_cls_head_adapter.py` | hooks the classification branch of the YOLO head per scale |
| `calibrate_profiles.py`, `learn_thresholds.py` | steps 1 and 2 |
| `detect_mech5_multi_img.py` | batch run + `run_summary.json` |
| `profiles/` | P3 and P4 profile + threshold files for `yolo11n.pt` |
