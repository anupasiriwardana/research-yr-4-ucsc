# Findings: Mech-7 and the multi-scale (Mech-5) follow-up

Status as of 2026-11-03. Model: `yolo11n.pt` (YOLOv8n for the old baseline only). Everything here is one attack setup
and one dataset, so read it as "what we measured", not "what is universally true" (see Caveats at the end).

## 1. Setup

- **Detector signal.** For every grid cell, the Mahalanobis distance from its 80-number classification-branch feature vector to the
  nearest of 33 profiles (32 object classes + background). A cell is abnormal when that distance is high. Boxes are built
  from connected abnormal cells, in 640x640 pixel space.
- **Data (BDD100K).** Profiles built on `train`. Thresholds learned on `test`. Clean evaluation images from `val`
  (the first 100 for everything offline; `val` images 100-499 for the one unseen-clean runtime check in section 6).
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

The baseline is fine on YOLOv8. On YOLO11 recall stays high but precision collapses: about 3.9 boxes per image, most of them wrong.
That is the problem the rest of this document tries to fix. (The "clean" sets inside the patched-image metrics have only 5-6 images, so
their FPR means little; section 6 uses 100-400 clean images.)

## 3. Mech-7: per-class thresholds (a dead end)

Idea: compare each cell's best distance with the threshold of the class it is nearest to, instead of one global number.

YOLO11, P3, per-class mode, `adjust_threshold` -2.0 (the only run found on disk):

| | precision | recall | F1 | mean IoU | boxes / patched img |
|---|---|---|---|---|---|
| Mech-1 baseline | 0.376 | 0.854 | 0.522 | 0.322 | 3.88 |
| Mech-7 per-class, adjust -2.0 | 0.273 | 0.497 | 0.353 | 0.229 | 3.11 |

Worse on every axis. Why, from the diagnostics below:

- **The learned thresholds barely differ between classes.** At P3 they sit in a band of about 13-20 (pooled 16.7, background 11.9),
  so per-class thresholds behave almost like one global threshold.
- **Normalising by class hurts the signal** (next section).

## 4. Does the score separate patch cells from normal cells at all? (`diagnose_separability.py`)

Patch cells = cell centre inside a ground-truth patch box. Normal cells = more than 48 px from any patch. 95 patched images, 171 patches.
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

## 5. Mech-5 offline: combining scales and better box building (`combine_scales.py`)

Works on saved score maps (no model run). Each scale's distance is divided by its own class threshold (`ratio`, above 1 = flagged),
maps are put on the finest grid, a **rule** merges the scales, cells with score > 1 + offset are flagged, flagged cells are optionally
joined (dilated `join` times), and regions with fewer than `min_cells` original cells are dropped.

Rules: single scale; `and` (all scales must flag the cell); `or` (any); `mean`; `confirm` (the finest scale decides, but every coarser scale must
flag a cell within one cell of it); `vote2` (at least two of three scales).

**Method.** For every rule, a grid of offsets (-0.4 to +0.3), joins (0-4) and min cells (1-30) is tried; the config with the best object F1 whose
clean-image FPR is at most 20% is picked. "Held-out" = 20 random 50/50 splits: pick the config on one half (47 patched / 50 clean images),
score it on the other half; mean +- std over splits. "today" = the Mech-7 P3 detector as it ran (offset 0, join 0, min cells 10, no tuning).

### 5.1 Held-out results by scale combination (FPR cap 20%)

| scales | rule | held-out F1 | precision | recall | mean IoU | clean FPR |
|---|---|---|---|---|---|---|
| P3 | today (fixed) | 0.305 | 0.526 | 0.215 | 0.171 | 0.314 |
| P3 | tuned | 0.322 | 0.764 | 0.208 | 0.473 | 0.218 |
| P4 | tuned | 0.402 | 0.684 | 0.287 | 0.450 | 0.215 |
| P5 | tuned | 0.283 | 0.609 | 0.191 | 0.354 | 0.207 |
| P3+P4 | `and` | 0.549 | 0.800 | 0.419 | 0.562 | 0.187 |
| P3+P4 | `confirm` | 0.528 | 0.842 | 0.387 | 0.606 | 0.173 |
| P3+P5 | `and` | 0.723 | 0.855 | 0.630 | 0.672 | 0.198 |
| P3+P5 | `confirm` | 0.748 | 0.904 | 0.639 | 0.714 | 0.156 |
| P4+P5 | `and` | 0.709 | 0.852 | 0.611 | 0.589 | 0.206 |
| P4+P5 | `confirm` | 0.690 | 0.819 | 0.598 | 0.597 | 0.231 |
| **P3+P4+P5** | `and` | 0.757 | 0.851 | 0.683 | 0.639 | 0.194 |
| **P3+P4+P5** | **`confirm`** | **0.770** | **0.875** | **0.690** | **0.655** | 0.187 |
| P3+P4+P5 | `vote2` (2 of 3) | 0.711 | 0.857 | 0.610 | 0.675 | 0.189 |
| P3+P4+P5 | `mean` | 0.699 | 0.784 | 0.633 | 0.633 | 0.190 |
| P3+P4+P5 | `or` | 0.305 | 0.566 | 0.214 | 0.451 | 0.195 |

(Spread across splits: F1 about +-0.04, FPR about +-0.05 to +-0.07. Full per-rule tables are in the CSV files named below.)

### 5.2 What this says

1. **Agreement between scales is what works.** Any single scale, and `or`, stay at F1 0.28-0.40. Requiring agreement (`and`, `confirm`) is far better.
2. **Adding P5 is the big jump.** P3+P4 gave held-out F1 about 0.53-0.55 with recall about 0.4. Adding P5 lifts that to 0.77 with recall 0.69 at the
   same ~19% clean FPR: recall goes from about 0.4 to about 0.7 without more false alarms. Even P3+P5 (0.75) beats P3+P4. This is the opposite of what a stricter
   rule would suggest; the best configs use a much lower offset (-0.3 to -0.4, i.e. each scale flags at 60-70% of its own threshold),
   so each scale on its own is permissive and the agreement of all three supplies the precision. (This explanation fits the numbers but was not tested directly.)
3. **`confirm` is slightly better than `and` and chosen more stably.** In the 20 splits `confirm` picked the same config (offset -0.3, join 1, min 15) 16 times;
   `and`'s top config was picked 12 times (and 7 when the sweep was widened).
   `vote2` (any two of three) is clearly worse than requiring all three: forgiving one scale did not help here.
4. **Box building matters.** Joining nearby flagged cells (join 1-2) and requiring larger regions (15-30 cells) are chosen in almost every good config.
5. **The edge of the sweep was checked.** `and` chose the lowest offset (-0.4) often, so the sweep was repeated down to -0.7 and min cells up to 40.
   The best configs did not move (`and` -0.4 / `confirm` -0.3), so the optimum is inside the range.

### 5.3 Where you land on the precision / recall trade-off (P3+P4+P5, held-out, 20 splits)

The FPR cap is the dial. Same method, different cap:

| clean-FPR cap | rule | held-out precision | recall | F1 | mean IoU | held-out clean FPR |
|---|---|---|---|---|---|---|
| 5% | `confirm` | 0.942 | 0.491 | 0.640 | 0.701 | 0.049 |
| 5% | `and` | 0.940 | 0.492 | 0.641 | 0.659 | 0.049 |
| 10% | `confirm` | 0.902 | 0.589 | 0.708 | 0.689 | 0.104 |
| 10% | `and` | 0.892 | 0.636 | 0.741 | 0.656 | 0.125 |
| 20% | `confirm` | 0.875 | 0.690 | 0.770 | 0.655 | 0.187 |
| 20% | `and` | 0.855 | 0.676 | 0.754 | 0.649 | 0.182 |
| 30% | `confirm` | 0.856 | 0.705 | 0.772 | 0.658 | 0.225 |
| 30% | `and` | 0.840 | 0.696 | 0.760 | 0.649 | 0.231 |

Recall flattens around 0.69-0.70 above 20% FPR, so a 20% cap is a reasonable point; accepting 10% still keeps recall about 0.6.

## 6. Mech-5 runtime detector (`yolo-mech5/`), P3+P4 only so far

The runtime detector reproduces the offline P3+P4 results exactly (identical boxes on all 100 patched images for both rules).
End-to-end with the default P3+P4 settings at the time (`and`: offset -0.2, join 2, min 20; `confirm`: offset -0.2, join 2, min 30),
100 patched images, and 400 clean `val` images not used for tuning:

| | precision | recall | F1 | mean IoU | boxes / patched img | clean images with any box | median latency |
|---|---|---|---|---|---|---|---|
| `yolo-transferable-mech1`, YOLO11 | 0.376 | 0.854 | 0.522 | 0.322 | 3.88 | high (5 of 5, tiny set) | about 25 ms mean |
| Mech-5 `and` (P3+P4) | 0.779 | 0.433 | 0.556 | 0.616 | 0.95 | 80 / 400 = 20% | about 50 ms |
| Mech-5 `confirm` (P3+P4) | 0.820 | 0.427 | 0.562 | 0.636 | 0.89 | 77 / 400 = 19% | about 50 ms |

The unseen-clean FPR (19-20%) matches the 20% cap the settings were tuned for, so the cap carried over to new images.
The three-scale configuration from section 5 has NOT yet been run through the runtime detector.

## 7. Summary

- **The YOLO11 false-positive problem was not a threshold problem.** Per-class and z-score thresholds (Mech-7) were worse than a plain threshold;
  the per-class normalisation reduces the separation between patch cells and normal cells.
- **What helped:** (a) requiring several feature scales to agree, (b) adding P5, which was the largest single gain,
  (c) joining scattered flagged cells and enforcing a minimum region size, (d) a lower per-scale offset once three scales vote.
- **Best result so far (offline, held-out):** P3+P4+P5 with `confirm`: precision 0.875, recall 0.690, F1 0.770, mean IoU 0.655, clean FPR about 19%.
  For reference, the old YOLOv8 baseline had precision 0.691, recall 0.889, F1 0.777, IoU 0.670, so on YOLO11 this reaches a comparable F1
  and IoU, with lower recall and much higher precision than the YOLO11 baseline (precision 0.376, F1 0.522).
- **Not solved:** recall is still below the baseline's (0.69 vs 0.85); the baseline remains better if finding every patch matters more than false boxes.

## 8. Caveats

- **Tuning and scoring share data.** Settings were tuned on the same 95 patched / 100 clean images; the held-out splits (47/50 images each)
  correct for this, but they are small and the clean-FPR estimate swings by about +-5 points between splits. Section 6's 400 unseen clean images are the stronger check, and
  it exists only for P3+P4.
- **Low offsets make the three-scale detector sensitive to calibration.** Flagging at 60-70% of each scale's threshold relies on the thresholds being right for that model and image domain.
- **One model, one attack setup.** YOLO11n only (no YOLOv8 profiles exist for Mech-5/7), one patch-placement protocol
  (class-2 raw anchors above 0.25 confidence, top 2, fixed scale 0.15), one dataset. A placement quirk was found earlier
  (the shared selection can pick small or edge-clipped targets and skip large ones), so some patches may be unusually hard or easy.
- **Lenient matching.** Any positive-area overlap counts as a hit, so a box much bigger or smaller than the patch still matches. Mean IoU is reported to show box quality.
- **`min_cells` is in finest-grid cells,** so its meaning changes when the finest scale changes (P3 cell = 8 px, P4 = 16 px); do not compare it across scale sets.
- **The 5 patch-free images** inside the patched set are too few to say anything about false positives; use the clean-folder runs.

## 9. Suggested next steps

1. **Run P3+P4+P5 `confirm` through the Mech-5 runtime detector.** It is a config change (add a `P5` entry with stride 32 and its profile and threshold files under `scales`;
   the code already handles more than two scales for `and` and `confirm`; `vote2` is offline-only and not needed). Check it on the 400 unseen clean images and compare the metrics file against the offline numbers.
2. **Check YOLOv8.** Build P3-P5 profiles for YOLOv8 and repeat, since the original comparison was YOLOv8 vs YOLO11.
3. **Decide the operating point** with the recovery module in mind (section 5.3): a stricter FPR cap trades recall for precision.
4. **Only then consider Mech-6** (per-class energy on Mech-4's backbone signal). Per-class normalisation hurt here, so the per-class part is doubtful; a complementary second signal is the only part worth testing, and the same offline check
   (separability, then combination with the P3+P4+P5 score) should come first.

## 10. Files and how to reproduce

| item | where |
|---|---|
| per-scale score maps (patched / clean, P3-P5) | `score-maps/yolov11/{p3,p4,p5}-{patched,clean}/` |
| separability results | `diagnostics/separability_{P3,P4,P5}.json` (`diagnose_separability.py`) |
| combination sweeps (every config, per scale set) | `diagnostics/combine_scales_{P3P4,P3P4P5,P4P5,P3P5,P3P4P5_wide}.csv` |
| combination tool | `combine_scales.py` (`--holdout-splits 20 --seed 0`, `--max-clean-fpr`) |
| runtime detector | `../yolo-mech5/` |

Example (the P3+P4+P5 row of 5.1):

```
python combine_scales.py --holdout-splits 20 --seed 0 --joins 0 1 2 3 4 --min-cells 1 2 4 6 10 15 20 30 \
  --offsets -0.4 -0.3 -0.2 -0.1 -0.05 0.0 0.05 0.1 0.2 0.3 \
  --patched P3=score-maps/yolov11/p3-patched P4=score-maps/yolov11/p4-patched P5=score-maps/yolov11/p5-patched \
  --clean   P3=score-maps/yolov11/p3-clean   P4=score-maps/yolov11/p4-clean   P5=score-maps/yolov11/p5-clean
```
