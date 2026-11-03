# Mech-7: per-class thresholding on Mech-1 Option B

Same nearest-profile scoring as `yolo-transferable-mech1` (each P-scale cell is compared against every
class profile + background; box-independent). Only the decision changes: the winning profile's own
threshold is used instead of one global number. See `newAlternatives.md`.

## Workflow (all paths in `config.json`)
1. `profile_images_dir` -> `python calibrate_profiles.py` builds the profiles.
2. `threshold_images_dir` (a DISJOINT clean set) -> `python learn_thresholds.py` learns per-class thresholds.
3. `python detect_mech7_multi_img.py` runs a folder and writes `<stem>.txt` files in the Mech-1 format plus
   `run_summary.json` (use `--input-dir/--coordinates-dir/--num-images` for a clean-only run: 
   `images_with_detection / images` is then the image-level false-positive rate).
4. Metrics: point `calc-metrics/metrics_config.json` at the output folder (patched-image run only).

## `detector_settings`
- `tap_scale` / `stride`: P3=8, P4=16, P5=32. A mismatch fails at load time.
- `threshold_mode`: `global` (baseline, uses `global_threshold`), `per_class`, `zscore` (Variant B).
- `threshold_percentile`: per-class percentile of clean best-distances (default 99.5).
- `adjust_threshold`: added at detection time to every threshold (raw-distance units; z units for `zscore`). No re-learning needed.
- `min_class_samples_for_own_threshold`: classes with fewer cells in the threshold set use the pooled values.
- Changing `tap_scale`/`threshold_percentile` needs steps 1-2 re-run (use a different `*_filename` per model/scale).

- `batch_settings.save_coordinates` (or `--save-coordinates/--no-save-coordinates` on the batch script): writes the per-image `<stem>.txt` box files into `coordinates_dir`. On by default. When off, only `run_summary.json` is written there (it still counts boxes, so the false-positive rate is still available).
- `batch_settings.save_score_heatmaps` (+ `score_heatmaps_dir`, `gt_coordinates_dir`, `heatmap_max_distance`, or `--save-score-heatmaps/--no-save-score-heatmaps`, `--score-heatmaps-dir`, `--gt-coordinates-dir` on the batch script): writes `heatmap_<image stem>.png` per image with three panels (image with GT + detected boxes | raw distance on a fixed 0..`heatmap_max_distance` scale | margin over each cell's own threshold). If a GT file is found in `gt_coordinates_dir` (names like `patched_`/`multi-patched_` prefixes are stripped), the header reports the raw median/max and % flagged inside the patch vs far from it; otherwise it just draws the maps. Off by default; drawn after the latency timer.
- `batch_settings.save_score_maps` (+ `score_maps_dir`, or `--save-score-maps/--no-save-score-maps` on the batch script): also writes `<image stem>.npz` per image with `raw` (min distance per cell), `best_class`, `bg_only`, `class_ids`, `stride`, `tap_scale`. Off by default; written after the latency timer, so latency is unaffected.

## Threshold modes (`detector_settings.threshold_mode`)

All three modes start the same way: for every cell, find the nearest profile (background or an object class) by Mahalanobis distance, and call that distance `d` (this is `raw` in the heatmap / `.npz`). They differ only in **what they compare `d` against** to decide if the cell is flagged. `adjust_threshold` is added on top in every mode.

| mode | flagged when | needs learned thresholds file | `adjust_threshold` unit |
|---|---|---|---|
| `global` | `d > global_threshold + adjust` | no | distance |
| `per_class` | `d > T[nearest class] + adjust` | yes | distance |
| `zscore` | `(d - mean[class]) / std[class] > z_thr + adjust` | yes | std-devs (z) |

- **`global`**: one fixed number for every cell, whatever class it is nearest to. This is the Mech-1 baseline, so use it as a control. Set `global_threshold` by hand (default 17, close to the learned pooled value of about 16.7). It does not read the thresholds `.pkl`.
- **`per_class`**: each profile has its own threshold `T`, learned by `learn_thresholds.py` as the `threshold_percentile` (default 99.5) of the distances of cells that were nearest to that class, on the clean threshold-image set. A cell "near cars" is judged against the car threshold, a cell "near background" against the (lower) background threshold. A class with fewer than `min_class_samples_for_own_threshold` cells falls back to the pooled threshold. Learned thresholds sit in a narrow band (about 13-20), so this behaves like `global` with small per-class tweaks.
- **`zscore`**: instead of comparing the raw distance, first normalise it per class: how many standard deviations above that class's typical (mean) clean distance is this cell? Then one single z threshold (the same percentile of all pooled z-scores) is used. The idea is to treat classes with a wide spread and classes with a tight spread equally. Be careful with `adjust_threshold` here: it is in z units, so `-2.0` is a huge shift compared to `-2.0` in the other two modes; try small values like -0.5 or -1.0 first.

Which to try: run `global` first (the control), then `per_class` (done), then `zscore`. In our separability check (`diagnose_separability.py`), `zscore` and `ratio` separated patch cells from normal cells *worse* than the plain raw distance (AUROC about 0.90 vs 0.96 at P3), so do not expect `zscore` to win; the per-class modes help mainly if a specific class is producing the false alarms. The heatmap and `.npz` always show the raw distance, and the margin panel is `stat - threshold` in the active mode's units.

Switching mode needs no re-learning (the thresholds file already holds the per-class values, the pooled value and the z threshold), only a change of `threshold_mode` in `config.json`.

## Reading the heatmap PNG (`save_score_heatmaps`)

Each cell of the grid is a small square of the image (8x8 px at P3, 16x16 at P4, 32x32 at P5). Mech-7 gives every cell one number: how far that cell's internal features are from the nearest "normal" thing the detector knows (the background or one of the object classes). **Low = looks normal, high = looks like nothing it knows.** A cell is *flagged* when its number is above the threshold of its nearest class. Flagged cells that touch are merged into a box, and boxes smaller than `min_region_cells` are dropped.

The PNG has a header line and three panels:

1. **Left: the image.** Cyan boxes are where the patches really are (ground truth, only drawn if a GT file is found). Red boxes are what Mech-7 reported. Ideal: every cyan box is covered by a red box.
2. **Middle: raw distance.** The number above, drawn as colour on a fixed scale (blue = about 0, red = `heatmap_max_distance`, default 30), so different images can be compared. White outlines are the flagged cells. Ideal: the patch is clearly redder than everything around it.
3. **Right: margin.** The same thing, but relative to each cell's own threshold: distance minus threshold. Blue = below the threshold (normal), red = above it (flagged), white/pale = close to it. Ideal: red only inside the cyan boxes. A patch that is pale instead of red is raised but not enough to be flagged, so a lower `adjust_threshold` could catch it (at the price of more false alarms elsewhere).

Header line (only when a GT file is found):
- `patch cells` / `far cells`: how many cells are inside the GT boxes, and how many are more than 48 px away from all of them (the ring in between is ignored).
- `raw median` / `max`: typical and highest distance inside the patch (and typical outside).
- `flagged`: percent of those cells above threshold. Good detection = high inside the patch, near 0 outside. Without GT it just shows the overall median, max and flagged-cell count.

## Offline multi-scale combination (`combine_scales.py`, the Mech-5 idea)

Tests "what if we use P3 and P4 together, and build boxes more cleverly" on saved score maps, with no model run (about 20 s for ~90 images).

**Prepare the maps** (one batch run per scale, same images each time, `save_score_maps` on, a different `score_maps_dir` per run):
- the patched images at P3 and at P4 (and P5 if wanted) -> e.g. `score-maps/yolov11/p3-patched`, `.../p4-patched`
- a set of CLEAN images at the same scales -> `.../p3-clean`, `.../p4-clean` (use a few hundred; without them the false-positive rate is not measured)
- each scale needs its own config (`tap_scale`, `stride`, profile + threshold filenames), as for `diagnose_separability.py`

**Run:**
```
python combine_scales.py --patched P3=<dir> P4=<dir> --clean P3=<dir> P4=<dir>
```

**How it works, simply.** Each cell's distance is divided by its own class threshold (`ratio`; above 1 = flagged), so P3 and P4 are on the same footing. P4 cells are 16 px, so each is repeated 2x2 to line up with the 8 px P3 grid. Then one *rule* turns the scales into one score:

| rule | meaning |
|---|---|
| `P3`, `P4`, `P5` | that scale alone (baselines) |
| `and` | the smaller ratio: a cell counts only if every scale flags it (fewer false alarms, may miss) |
| `or` | the larger ratio: any scale can flag it (more recall, more false alarms) |
| `mean` | average of the ratios |
| `confirm` | P3 decides, but only where P4 is also flagged within one cell |

Sweeps: `--offsets` (flag when ratio > 1 + offset, like `adjust_threshold` but unitless), `--joins` (grow flagged cells by N cells before grouping, so scattered patch cells merge into one box; the box and size check still use only the original flagged cells), `--min-cells` (smallest region kept, counted on the finest grid).

**Output:** per rule, the config with the best object F1 whose clean-image false-positive rate (% of clean images with any box) is <= `--max-clean-fpr` (default 0.2), plus the detector-as-today reference row. Precision / recall / IoU use the same definition as `compare_patch_detection_metrics.py`. Every config is in `diagnostics/combine_scales.csv` (`--out` to change). `--mode global --global-threshold 17` uses one number instead of the learned per-class thresholds.

Caveat: the best config is picked on the same images it is scored on, so the numbers are optimistic. Treat the ranking of rules as the result, and confirm the winner on different images.

**Holdout (honest numbers):** add `--holdout-splits 20` (optionally `--train-frac 0.5`, `--seed 0`). It repeats N random splits of the patched and clean images: pick the best config per rule on the tuning part, then score that one config on the held-out part. It prints the held-out mean +- std per rule, next to the fixed "today" config on the same held-out parts, and which config was chosen most often (a config chosen in only a few splits is unstable). The "best per rule on ALL images" table above it is still the optimistic one. Widen `--offsets`, `--joins` and `--min-cells` if a chosen value sits at the edge of its range.
- Metric and term definitions (precision, recall, F1, IoU, Mahalanobis distance, ...): see `yolo-mech5/README.md`, section Glossary.
