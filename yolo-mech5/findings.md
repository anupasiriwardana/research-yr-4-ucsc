# Mech-5 findings: combining feature scales (P3 / P4 / P5)

Status as of 2026-11-03. Model `yolo11n.pt`, BDD100K images, one attack setup. The earlier work this builds on (the Mech-1 baseline,
Mech-7 per-class thresholds and the per-scale separability diagnostic) is in `../yolo-mech7/findings.md`; the numbers needed here are repeated below.

## 0. Short version

- Requiring several feature scales to **agree** works far better than any single scale. P3 alone: held-out F1 0.32. P3+P4: 0.53-0.55. **P3+P4+P5: 0.76-0.77**
  (precision 0.875, recall 0.69, mean IoU 0.655, about 19% of clean images with a box).
- **Adding P5 was the largest single gain** (recall about 0.4 to about 0.7 at the same false-alarm level).
- Best rule: **`confirm`** (settings: offset -0.3, join 1, min region 15 cells), then `and` (offset -0.4, join 0, min region 30).
- Measured end-to-end only for P3+P4 so far; the three-scale settings above are from the offline sweep on saved score maps and are now the defaults in `config.json`.
- Recall (0.69) is still below the Mech-1 YOLO11 baseline (0.85); precision is much higher (0.875 vs 0.376).

## 1. Setup and how to read the settings

- **Data.** Profiles built on BDD100K `train`, thresholds learned on `test`, clean evaluation images from `val` (the first 100 for the offline sweeps;
  `val` images 100-499 as 400 unseen clean images for the runtime check). Patched images: 100 from `test-patch-det-multi-img/gen-patched-img/yolov11-patched-imgs`:
  95 contain patches (171 patches in total), 5 contain none.
- **Metrics.** Same definitions as `calc-metrics/compare_patch_detection_metrics.py` (positive-area overlap, one-to-one, highest IoU first); terms are in the Glossary of `README.md`.
  Clean FPR = percentage of patch-free images that get at least one box.
- **Per-scale score ("ratio").** `ratio = nearest-profile distance / (that class's threshold + adjust_threshold)`. Above 1 = that scale flags the cell.
- **Settings used in every table below:**
  - `offset`: a cell is flagged when the combined score is above `1 + offset`. `-0.3` means "flag at 70% of the threshold" (more sensitive); `+0.2` means "flag at 120%" (stricter).
  - `join`: how many times flagged cells are dilated (3x3) before grouping, so scattered patch cells merge into one region (0 = off). The box and size check use only the original cells.
  - `min cells`: smallest region kept, counted in cells of the **finest** scale used (P3 cell = 8 px, P4 = 16 px, P5 = 32 px). The values are not comparable across scale sets.
  - `adjust_threshold` was 0.0 everywhere in this document (the offset does the same job without touching the learned thresholds).
  - Thresholds: `per_class`, learned as the 99.5th percentile of clean best-distances per class and scale (pooled value for classes with fewer than 30 cells).
- **Rules** (how the scales are merged into one score): single scale; `and` = smallest ratio (every scale must flag the cell); `or` = largest (any scale);
  `mean`; `confirm` = the finest scale decides, but every coarser scale must flag a cell within one cell of it; `vote2` = at least two of three scales.

## 2. Starting point (from `../yolo-mech7/findings.md`)

| detector | settings | precision | recall | F1 | mean IoU | boxes / patched img | clean FPR |
|---|---|---|---|---|---|---|---|
| Mech-1 baseline, YOLO11 | global threshold 17.0, P3, min region 9 cells (values from `yolo-transferable-mech1/config.json`; an earlier note recalled 15, not verified) | 0.376 | 0.854 | 0.522 | 0.322 | 3.88 | 5 of 5 on a tiny set |
| Mech-1 baseline, YOLOv8 | same | 0.691 | 0.889 | 0.777 | 0.670 | 2.20 | 0 of 6 on a tiny set |
| Mech-7 per-class, P3 | `adjust_threshold` -2.0 (distance units), min region cells as in that run's config (9-10) | 0.273 | 0.497 | 0.353 | 0.229 | 3.11 | 4 of 5 on a tiny set |
| Mech-7 P3 as run "today" (the reference row in the sweeps) | per_class, offset 0, join 0, min cells 10 | 0.526 (held-out) | 0.215 | 0.305 | 0.171 | 0.74 | 0.314 |

(The Mech-7 `adjust_threshold` 0.0 run was done by you but its results were not on disk when this was written, so it is not listed.)

What the per-scale separability check says (details in the Mech-7 findings): the raw nearest-profile distance separates patch cells from normal cells at every scale,
AUROC 0.958 (P3), 0.977 (P4), 0.968 (P5), but the share of patch cells caught at 0.5% false alarms is 0.196 (P3), 0.479 (P4), 0.514 (P5). The coarser scales are
more discriminative at a tight false-alarm budget, which is why adding P4 and P5 helps. Dividing by the class threshold (`ratio`) or standardising per class (`zscore`) made separation
worse at every scale, which is why Mech-7's per-class thresholds did not help; Mech-5 uses the ratio only to put scales on a common footing and relies on cross-scale agreement.

## 3. Offline sweep on saved score maps (`../yolo-mech7/combine_scales.py`)

Per rule the tool tries offsets -0.4 to +0.3 (10 values), join 0-4, min cells 1-30, and picks the config with the best object F1 whose clean-image FPR is at most 20%.
"Held-out" = 20 random 50/50 splits (seed 0): pick the config on one half (47 patched / 50 clean images), score it on the other half; mean +- std over splits.
Because every split can pick a different config, the settings column shows the config chosen most often and how many of the 20 splits chose it.

### 3.1 Held-out results and the settings behind them (FPR cap 20%)

| scales | rule | held-out F1 | precision | recall | mean IoU | clean FPR | settings chosen most often: offset / join / min cells (splits of 20) |
|---|---|---|---|---|---|---|---|
| P3 | today (fixed) | 0.305 | 0.526 | 0.215 | 0.171 | 0.314 | 0.0 / 0 / 10 (no tuning) |
| P3 | tuned | 0.322 | 0.764 | 0.208 | 0.473 | 0.218 | -0.1 / 2 / 30 (3) |
| P4 | tuned | 0.402 | 0.684 | 0.287 | 0.450 | 0.215 | -0.05 / 0 / 30 (4) |
| P5 | tuned | 0.283 | 0.609 | 0.191 | 0.354 | 0.207 | +0.2 / 2 / 1 (4) |
| P3+P4 | `and` | 0.549 | 0.800 | 0.419 | 0.562 | 0.187 | -0.2 / 2 / 20 (6) |
| P3+P4 | `confirm` | 0.528 | 0.842 | 0.387 | 0.606 | 0.173 | -0.2 / 2 / 30 (10) |
| P3+P4 | `mean` | 0.502 | 0.758 | 0.378 | 0.511 | 0.186 | -0.05 / 4 / 15 (6) |
| P3+P4 | `or` | 0.387 | 0.692 | 0.271 | 0.504 | 0.219 | 0.0 / 0 / 30 (6) |
| P3+P5 | `and` | 0.723 | 0.855 | 0.630 | 0.672 | 0.198 | -0.3 / 2 / 20 (8) |
| P3+P5 | `confirm` | 0.748 | 0.904 | 0.639 | 0.714 | 0.156 | -0.3 / 1 / 30 (16) |
| P4+P5 | `and` | 0.709 | 0.852 | 0.611 | 0.589 | 0.206 | -0.3 / 1 / 10 (11) |
| P4+P5 | `confirm` | 0.690 | 0.819 | 0.598 | 0.597 | 0.231 | -0.2 / 0 / 6 (10) |
| **P3+P4+P5** | `and` | 0.757 | 0.851 | 0.683 | 0.639 | 0.194 | -0.4 / 0 / 30 (12) |
| **P3+P4+P5** | **`confirm`** | **0.770** | **0.875** | **0.690** | **0.655** | 0.187 | **-0.3 / 1 / 15 (16)** |
| P3+P4+P5 | `vote2` (2 of 3) | 0.711 | 0.857 | 0.610 | 0.675 | 0.189 | -0.2 / 1 / 30 (16) |
| P3+P4+P5 | `mean` | 0.699 | 0.784 | 0.633 | 0.633 | 0.190 | -0.2 / 1 / 20 (7) |
| P3+P4+P5 | `or` | 0.305 | 0.566 | 0.214 | 0.451 | 0.195 | +0.2 / 3 / 20 (6) |

Spread across splits: F1 about +-0.04, clean FPR about +-0.05 to +-0.07. For P4+P5 the finest scale is P4, so its "min cells" is in 16 px cells; for P5 alone it is in 32 px cells.

### 3.2 Best settings when tuned and scored on all images (optimistic)

The same sweep, one config per rule, tuned and scored on the same 95 patched / 100 clean images (so the numbers are flattering, but these are the settings you would copy).

| scales | rule | offset | join | min cells | precision | recall | F1 | mean IoU | boxes / patched img | clean FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| P3 | single | -0.10 | 1 | 30 | 0.875 | 0.205 | 0.332 | 0.644 | 0.42 | 0.19 |
| P4 | single | +0.05 | 1 | 20 | 0.653 | 0.275 | 0.387 | 0.368 | 0.76 | 0.19 |
| P5 | single | +0.20 | 2 | 1 | 0.565 | 0.152 | 0.240 | 0.283 | 0.48 | 0.18 |
| P3+P4 | `and` | -0.20 | 2 | 20 | 0.796 | 0.433 | 0.561 | 0.616 | 0.98 | 0.18 |
| P3+P4 | `confirm` | -0.20 | 2 | 30 | 0.859 | 0.427 | 0.570 | 0.636 | 0.89 | 0.18 |
| P3+P4 | `mean` | -0.10 | 4 | 20 | 0.761 | 0.392 | 0.517 | 0.476 | 0.93 | 0.20 |
| P3+P4 | `or` | 0.00 | 0 | 30 | 0.733 | 0.257 | 0.381 | 0.516 | 0.63 | 0.16 |
| P3+P5 | `and` | -0.30 | 2 | 20 | 0.864 | 0.667 | 0.752 | 0.698 | 1.39 | 0.19 |
| P3+P5 | `confirm` | -0.30 | 1 | 30 | 0.911 | 0.655 | 0.762 | 0.730 | 1.29 | 0.15 |
| P4+P5 | `and` | -0.30 | 1 | 10 | 0.846 | 0.643 | 0.731 | 0.611 | 1.37 | 0.20 |
| P4+P5 | `confirm` | -0.30 | 1 | 20 | 0.894 | 0.544 | 0.676 | 0.609 | 1.09 | 0.15 |
| P3+P4+P5 | `and` | -0.40 | 0 | 30 | 0.863 | 0.702 | 0.774 | 0.656 | 1.46 | 0.18 |
| **P3+P4+P5** | **`confirm`** | **-0.30** | **1** | **15** | 0.879 | 0.719 | **0.791** | 0.658 | 1.47 | 0.19 |
| P3+P4+P5 | `vote2` | -0.20 | 1 | 30 | 0.870 | 0.626 | 0.728 | 0.681 | 1.29 | 0.17 |
| P3+P4+P5 | `mean` | -0.20 | 4 | 20 | 0.818 | 0.632 | 0.713 | 0.591 | 1.39 | 0.17 |
| P3+P4+P5 | `or` | +0.20 | 3 | 20 | 0.579 | 0.193 | 0.289 | 0.437 | 0.60 | 0.15 |

Reference (Mech-7 P3 as it ran, offset 0 / join 0 / min cells 10): precision 0.514, recall 0.211, F1 0.299, mean IoU 0.171, 0.74 boxes per patched image, clean FPR 0.31.

### 3.3 What this says

1. **Agreement between scales is what works.** Any single scale, and `or`, stay at F1 0.28-0.40; requiring agreement (`and`, `confirm`) is far better.
2. **Adding P5 is the big jump.** P3+P4 gave held-out F1 about 0.53-0.55 with recall about 0.4; adding P5 lifts it to about 0.77 with recall 0.69 at the same ~19% clean FPR.
   Even P3+P5 (0.75) beats P3+P4. The good configs all use a much lower offset (-0.2 to -0.4, each scale flags at 60-80% of its own threshold), so each scale on its own is permissive and the
   agreement of the scales supplies the precision. (This explanation fits the numbers but was not tested directly.)
3. **`confirm` is slightly better than `and`, and its settings are chosen more stably** (the same config in 16 of 20 splits for three scales, vs 12 of 20 for `and`).
   `vote2` (any two of three) is clearly worse than requiring all three.
4. **Box building matters.** Join 1-2 and a minimum of 15-30 cells appear in nearly every good config.
5. **Edge-of-range check.** `and` chose the lowest offset (-0.4) often, so the three-scale sweep was repeated with offsets down to -0.7 and min cells up to 40
   (`and` -0.4 / join 0 / min 30, `confirm` -0.3 / join 1 / min 15 again; chosen 7 and 16 of 20 times, `vote2` -0.2 / 1 / 30 16 times, `mean` -0.2 / 1 / 20 7 times). The optimum is inside the range.

### 3.4 Where you land on the precision / recall trade-off (P3+P4+P5)

The clean-FPR cap is the dial. Offsets -0.7 to +0.2, join 0-4, min cells 4-40 were swept for each cap.

| clean-FPR cap | rule | best settings on all images (offset / join / min cells) | held-out precision | recall | F1 | mean IoU | held-out clean FPR | settings chosen most often (splits of 20) |
|---|---|---|---|---|---|---|---|---|
| 5% | `confirm` | -0.3 / 1 / 40 | 0.942 | 0.491 | 0.640 | 0.701 | 0.049 | -0.3 / 1 / 40 (8) |
| 5% | `and` | -0.3 / 1 / 30 | 0.940 | 0.492 | 0.641 | 0.659 | 0.049 | -0.3 / 1 / 30 (8) |
| 10% | `confirm` | -0.3 / 1 / 30 | 0.902 | 0.589 | 0.708 | 0.689 | 0.104 | -0.3 / 1 / 20 (9) |
| 10% | `and` | -0.4 / 0 / 40 | 0.892 | 0.636 | 0.741 | 0.656 | 0.125 | -0.4 / 0 / 40 (8) |
| 20% | `confirm` | -0.3 / 1 / 15 | 0.875 | 0.690 | 0.770 | 0.655 | 0.187 | -0.3 / 1 / 15 (16) |
| 20% | `and` | -0.4 / 0 / 30 | 0.855 | 0.676 | 0.754 | 0.649 | 0.182 | -0.4 / 0 / 30 (7) |
| 30% | `confirm` | -0.3 / 1 / 15 | 0.856 | 0.705 | 0.772 | 0.658 | 0.225 | -0.3 / 1 / 15 (14) |
| 30% | `and` | -0.4 / 1 / 30 | 0.840 | 0.696 | 0.760 | 0.649 | 0.231 | -0.4 / 1 / 30 (12) |

Recall flattens around 0.69-0.70 above a 20% cap, so 20% is a reasonable point; accepting 10% still keeps recall near 0.6, and 5% keeps about 0.49. A stricter cap mostly pushes the minimum region size up (30-40 cells).

## 4. Runtime detector (this folder), P3+P4 only so far

The runtime detector reproduces the offline P3+P4 results exactly (identical boxes on all 100 patched images for both rules). End to end, 100 patched images and 400 unseen clean `val` images
(images 100-499), with the P3+P4 settings that were the defaults at the time:

| scales | rule | settings (offset / join / min cells) | precision | recall | F1 | mean IoU | boxes / patched img | clean images with any box | median latency |
|---|---|---|---|---|---|---|---|---|---|
| P3+P4 | `and` | -0.2 / 2 / 20 | 0.779 | 0.433 | 0.556 | 0.616 | 0.95 | 80 / 400 = 20% | about 50 ms |
| P3+P4 | `confirm` | -0.2 / 2 / 30 | 0.820 | 0.427 | 0.562 | 0.636 | 0.89 | 77 / 400 = 19% | about 50 ms |

The unseen-clean FPR (19-20%) matches the 20% cap the settings were tuned for, so the cap carried over to new images. (Latency was 212 ms before the profile-distance computation was batched into one matrix
operation, 52 ms after, with identical output; a 5-image check of the three-scale config ran at about 20-27 ms per image after warm-up.)

**Not yet measured end to end:** P3+P4+P5. `config.json` now holds P5 (stride 32, copied P5 profile and threshold files) and the three-scale defaults: `confirm` -0.3 / join 1 / min 15,
`and` -0.4 / join 0 / min 30, `adjust_threshold` 0.0. The offline expectation to compare a run against: precision about 0.875, recall about 0.69, mean IoU about 0.655, about 19% of clean images with a box (`confirm`).

(Your own later experiments with other offsets and rule modes, for example `and` -0.3 and `confirm` -0.25 with P3+P4, were not recorded in these results.)

## 5. Caveats

- **Tuning and scoring share data.** Settings were tuned on the same 95 patched / 100 clean images; the held-out splits (47/50 images each) correct for this, but they are small and the clean-FPR estimate swings by
  about +-5 points between splits. The 400 unseen clean images are the stronger check and exist only for P3+P4.
- **Low offsets make the three-scale detector sensitive to calibration.** Flagging at 60-80% of each scale's threshold relies on thresholds being right for this model and image domain.
- **One model, one attack setup.** YOLO11n only (no YOLOv8 profiles for Mech-5 yet), one patch-placement protocol (class-2 raw anchors above 0.25 confidence, top 2, fixed scale 0.15), one dataset.
  The shared placement code can pick small or edge-clipped targets and skip large ones, so some patches may be unusually hard or easy.
- **Lenient matching.** Any positive-area overlap counts as a hit, so a box much bigger or smaller than the patch still matches; mean IoU is reported to show box quality.
- **The 5 patch-free images** inside the patched set are too few to say anything about false positives; use the clean-folder runs.

## 6. Suggested next steps

1. Run P3+P4+P5 `confirm` (then `and`) with `detect_mech5_multi_img.py` on the 100 patched and the 400 unseen clean images and compare with the offline numbers above.
2. Check YOLOv8: build P3-P5 profiles and thresholds for it and repeat (the original comparison was YOLOv8 vs YOLO11).
3. Decide the operating point (section 3.4) with the recovery module in mind: a stricter cap trades recall for precision.
4. Only then consider Mech-6 (per-class energy on Mech-4's backbone signal). Per-class normalisation hurt in Mech-7, so the per-class part is doubtful; a complementary second signal
   is the only part worth testing, and the same offline checks (separability, then combination with the P3+P4+P5 score) should come first.

## 7. Files and how to reproduce

| item | where |
|---|---|
| per-scale score maps (patched / clean, P3-P5) | `../yolo-mech7/score-maps/yolov11/{p3,p4,p5}-{patched,clean}/` |
| offline sweeps, every config per scale set | `../yolo-mech7/diagnostics/combine_scales_{P3P4,P3P4P5,P4P5,P3P5,P3P4P5_wide}.csv` |
| sweep tool | `../yolo-mech7/combine_scales.py` (`--holdout-splits 20 --seed 0`, `--max-clean-fpr`) |
| runtime detector | this folder (`detect_mech5_multi_img.py`, `config.json`) |

The three-scale row of 3.1 / 3.2 (run from `yolo-mech7`):

```
python combine_scales.py --holdout-splits 20 --seed 0 --joins 0 1 2 3 4 --min-cells 1 2 4 6 10 15 20 30 \
  --offsets -0.4 -0.3 -0.2 -0.1 -0.05 0.0 0.05 0.1 0.2 0.3 \
  --patched P3=score-maps/yolov11/p3-patched P4=score-maps/yolov11/p4-patched P5=score-maps/yolov11/p5-patched \
  --clean   P3=score-maps/yolov11/p3-clean   P4=score-maps/yolov11/p4-clean   P5=score-maps/yolov11/p5-clean
```
