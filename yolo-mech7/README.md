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
- `batch_settings.save_score_maps` (+ `score_maps_dir`, or `--save-score-maps/--no-save-score-maps` on the batch script): also writes `<image stem>.npz` per image with `raw` (min distance per cell), `best_class`, `bg_only`, `class_ids`, `stride`, `tap_scale`. Off by default; written after the latency timer, so latency is unaffected.
