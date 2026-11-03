# Findings: Mech-7 (per-class thresholding) and the per-scale diagnostics

Status as of 2026-11-03. Model: `yolo11n.pt` (YOLOv8n for the old baseline only). Everything here is one attack setup
and one dataset, so read it as "what we measured", not "what is universally true" (see Caveats at the end).
The multi-scale follow-up that grew out of these results (Mech-5: combining P3/P4/P5, the offset / join / min-cell sweeps and the runtime detector)
has its own findings in `../yolo-mech5/findings.md`.

## 1. Setup

- **Detector signal.** For every grid cell, the Mahalanobis distance from its 80-number classification-branch feature vector to the
  nearest of 33 profiles (32 object classes + background). A cell is abnormal when that distance is high. Boxes are built
  from connected abnormal cells, in 640x640 pixel space.
- **Data (BDD100K).** Profiles built on `train`. Thresholds learned on `test`. Clean evaluation images from `val`
  (the first 100 for the offline analysis; `val` images 100-499 for the unseen-clean runtime check described in `../yolo-mech5/findings.md`).
- **Patched images.** 100 images from `test-patch-det-multi-img/gen-patched-img/yolov11-patched-imgs`: 95 contain patches
  (171 patches in total) and 5 contain none.
- **Metrics.** Same definitions as `calc-metrics/compare_patch_detection_metrics.py`: a detected box matches a real patch when they
  overlap with positive area, one-to-one, highest IoU first. Terms (precision, recall, F1, IoU, ...) are defined in
  `yolo-mech5/README.md`, section Glossary.
- **Clean-image false-positive rate (FPR).** Percentage of patch-free images that get at least one box.

## 2. The starting problem: the Mech-1 baseline (`yolo-transferable-mech1`)

| model | precision | recall | F1 | mean IoU | boxes / patched img | image-level FPR |
|---|---|---|---|---|---|---|
| YOLOv8n | 0.691 | 0.889 | 0.777 | 0.670 | 2.20 | 0 of 6 clean (tiny set) |
| YOLO11n | 0.376 | 0.854 | 0.522 | 0.322 | 3.88 | 5 of 5 clean (tiny set) |

Settings: one global threshold of 17.0 on the P3 distance, stride 8, minimum region 9 cells, profiles from `cls_head_calibration_p3_yolo11-v2.pkl`
(values read from `yolo-transferable-mech1/config.json`; an earlier note recalled a threshold of 15 for the YOLO11 run, which was not verified).

The baseline is fine on YOLOv8. On YOLO11 recall stays high but precision collapses: about 3.9 boxes per image, most of them wrong.
That is the problem the rest of this document tries to fix. (The "clean" sets inside the patched-image metrics have only 5-6 images, so
their FPR means little; the Mech-5 findings use 100-400 clean images.)

## 3. Mech-7: per-class thresholds (a dead end)

Idea: compare each cell's best distance with the threshold of the class it is nearest to, instead of one global number.

YOLO11, P3, per-class mode, `adjust_threshold` -2.0 (the only run found on disk):

| | precision | recall | F1 | mean IoU | boxes / patched img |
|---|---|---|---|---|---|
| Mech-1 baseline | 0.376 | 0.854 | 0.522 | 0.322 | 3.88 |
| Mech-7 per-class, adjust -2.0 | 0.273 | 0.497 | 0.353 | 0.229 | 3.11 |

Settings of the Mech-7 row: `threshold_mode` per_class, `adjust_threshold` -2.0 (distance units, added to every class's threshold), `threshold_percentile` 99.5,
P3 (stride 8), `min_region_cells` as in that run's config (9 or 10). The `adjust_threshold` 0.0 run you did was not on disk when this was written, so it is not listed.

Worse on every axis. Why, from the diagnostics below:

- **The learned thresholds barely differ between classes.** At P3 they sit in a band of about 13-20 (pooled 16.7, background 11.9),
  so per-class thresholds behave almost like one global threshold.
- **Normalising by class hurts the signal** (next section).

## 4. Does the score separate patch cells from normal cells at all? (`diagnose_separability.py`)

Settings: one config per scale (P3 stride 8, P4 16, P5 32; their own profile and threshold files), halo 48 px. Patch cells = cell centre inside a ground-truth patch box. Normal cells = more than 48 px from any patch. 95 patched images, 171 patches.
AUROC: chance that a random patch cell scores higher than a random normal cell (0.5 = coin flip, 1.0 = perfect).
TPR@0.5% = share of patch cells caught by the threshold that flags only 0.5% of normal cells.

| score | P3 AUROC | P3 TPR@0.5% | P4 AUROC | P4 TPR@0.5% | P5 AUROC | P5 TPR@0.5% |
|---|---|---|---|---|---|---|
| `raw` (nearest-profile distance) | 0.958 | 0.196 | 0.977 | 0.479 | **0.968** | **0.514** |
| `ratio` (distance / class threshold) | 0.888 | 0.085 | 0.905 | 0.150 | 0.913 | 0.215 |
| `zscore` (per-class standardised) | 0.902 | 0.102 | 0.934 | 0.224 | 0.940 | 0.295 |
| `bg_only` (distance to background profile only) | 0.975 | 0.227 | 0.978 | 0.336 | 0.963 | 0.366 |

Using clean-image cells from `val` as the "normal" set gave nearly the same numbers (for example P5 `raw` AUROC 0.964).

What this says:

1. **The raw score is good.** AUROC is 0.96-0.98 at every scale, so the problem is not the signal itself.
2. **Per-class normalisation (`ratio`, `zscore`) is clearly worse than raw** at every scale. This is the main reason per-class thresholds
   did not help. Mech-7 as a thresholding idea is a dead end.
3. **Coarser scales are better at a tight false-alarm budget.** At 0.5% FPR, P3 catches 20% of patch cells, P4 48%, P5 51%.
   P3 is the weakest at this budget, even though it is the scale the baseline uses.
4. **`bg_only` is not a free win.** Better at P3, worse at P4 and P5 at tight FPR, and it would also flag real objects (cars, people),
   which the class profiles normally explain away. Not pursued.

What a single score map looked like (P3, `patched_b503cd8a`): the patch cells have distances of 22-27 against a normal median of
6 and thresholds of 13-20, but only the lower-left part of the patch was above threshold, with isolated single flagged cells elsewhere. The
patch is raised but mostly sits just under the thresholds, and the 10-cell connected-region rule then drops the scattered pieces or keeps only a corner.
The weak point is turning abnormal cells into boxes, together with the single-scale signal being only partly separable.

## 5. What came next: Mech-5 (moved)

The diagnostics above suggested dropping per-class thresholds and using the raw distances of several scales together, since the coarser scales
(P4, P5) separate patch cells better than P3 and the errors of different scales differ. That became Mech-5, built in `../yolo-mech5/`.
Headline (held-out, about 20% of clean images getting a box): P3 alone F1 0.32, P3+P4 0.53-0.55, **P3+P4+P5 0.76-0.77** (precision 0.875, recall 0.69, mean IoU 0.655),
with the settings (offset, join, minimum region) for every configuration listed in `../yolo-mech5/findings.md`.

## 6. Caveats (for the Mech-7 findings above)

- One model, one attack setup (class-2 raw anchors above 0.25 confidence, top 2, fixed patch scale 0.15), one dataset.
- The shared patch-placement code can pick small or edge-clipped targets and skip large ones, so some patches are unusually hard or easy.
- The 5-6 patch-free images inside the patched sets are too few for a false-positive rate; use clean-folder runs.
- Matching is lenient (any positive-area overlap), so mean IoU is reported to show box quality.

## 7. Files

| item | where |
|---|---|
| per-scale score maps (patched / clean, P3-P5) | `score-maps/yolov11/{p3,p4,p5}-{patched,clean}/` |
| separability results | `diagnostics/separability_{P3,P4,P5}.json` (`diagnose_separability.py`) |
| combination sweeps (every config, per scale set) | `diagnostics/combine_scales_*.csv` (`combine_scales.py`; analysed in `../yolo-mech5/findings.md`) |
