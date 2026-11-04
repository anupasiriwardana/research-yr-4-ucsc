# Mech-6 findings: feature-energy profiling

Status as of 2026-11-04. Models `yolo11n.pt` and `yolov8n.pt`, BDD100K images, the same patched images, ground truth and clean images as the Mech-5 findings
(`../yolo-mech5/findings.md`; 95 / 94 patched images with 171 patches, 100 clean `val` images per model: indices 0-99 for YOLO11, 100-199 for YOLOv8). Metric definitions and the meaning of
`offset / join / min cells` are in that file and in the Mech-5 README glossary.

## 0. Short version

1. **Per-class profiling (the Mech-6 idea) does not help.** At every layer the per-class scores are equal to or worse than plain energy. This repeats what Mech-7 found for the distance-based signal.
2. **But plain energy at backbone layer 1 (the Mech-4 signal) separates THIS benchmark's patches almost perfectly,** much better than any Mech-5 scale: held-out F1 **0.985 (YOLO11)** and **0.989 (YOLOv8)**, with clean FPR about 3% / 0.5%.
   (Mech-5: 0.77 on YOLO11 and 0.93 on YOLOv8.)
3. **That probably says more about the patches than about the detector.** The benchmark patches are saturated, high-frequency textures, which is exactly what an energy detector is built to catch. A smoother, lower-contrast or
   natural-looking adversarial patch has low energy and would not be caught this way. Mech-5 (classification-feature abnormality) and energy look at different things, which is the argument for combining them,
   but nothing here shows how either does on other kinds of patches. See section 5.
4. Combining energy with Mech-5 gives about the same numbers as energy alone on this benchmark, because energy already saturates it.

## 1. Setup

- Tapped backbone layers 1, 2, 3 (strides 4, 4, 8), checked on both models. Energy = (L1 across channels)^2.
- Profiles: log-energy mean/std per class and for background, per layer, from 2000 clean `train` images (boxes projected on each layer's grid; same procedure as the Mech-1 profiles).
- Scores per layer: `E` (raw energy, one global threshold, i.e. Mech-4), `nearest_abs` (distance to the nearest profile in std units of log-energy), `nearest_up` (same, but only energy above the profile counts).
- Cell-level test: patch cells (centre inside a ground-truth patch) vs normal cells (more than 48 px from any patch), same as `yolo-mech7/diagnose_separability.py`.
- Box-level test: layer-1 energy as a pseudo scale `E3` (z-score of log-energy, clipped at 0, max-pooled 2x2 to the 80x80 P3 grid), threshold = 99.5th percentile over every cell of 300 clean `test` images,
  then the same sweep and 20-split held-out protocol as the Mech-5 findings (offsets -0.7 to +2.0, join 0-3, min cells 4-60, seed 0, 20% clean-FPR cap).

## 2. Cell level: does per-class profiling help? (no) Which layer carries the signal? (layer 1)

AUROC / share of patch cells caught at 0.5% false alarms (against the normal cells of the patched images; clean-image cells gave nearly the same).

| layer (stride) | score | YOLOv8 | YOLO11 |
|---|---|---|---|
| L1 (4) | `E` (Mech-4) | **0.990 / 0.645** | **0.997 / 0.867** |
| L1 (4) | `nearest_abs` (Mech-6) | 0.988 / 0.645 | 0.994 / 0.867 |
| L1 (4) | `nearest_up` (Mech-6) | 0.990 / 0.645 | 0.996 / 0.867 |
| L2 (4) | `E` | 0.795 / 0.150 | 0.796 / 0.079 |
| L2 (4) | `nearest_abs` | 0.725 / 0.117 | 0.582 / 0.050 |
| L2 (4) | `nearest_up` | 0.783 / 0.150 | 0.737 / 0.079 |
| L3 (8) | `E` | 0.836 / 0.037 | 0.967 / 0.483 |
| L3 (8) | `nearest_abs` | 0.586 / 0.036 | 0.880 / 0.440 |
| L3 (8) | `nearest_up` | 0.726 / 0.037 | 0.938 / 0.483 |

All layers merged on the 80x80 grid (each layer first turned into its percentile among clean cells; AUROC / catch rate at 0.5%):

| merge | YOLOv8 `E` | YOLO11 `E` |
|---|---|---|
| any layer (max) | 0.992 / 0.718 | 0.998 / 0.931 |
| every layer (min) | 0.916 / 0.165 | 0.957 / 0.430 |
| mean | 0.956 / 0.230 | 0.979 / 0.534 |

What this says:
- Per-class normalisation never beats plain energy (at best it ties). The reason it ties at layer 1 is that patch cells have more energy than every class profile, so the one-sided score is just a monotone copy of energy.
  The Mech-6 hypothesis (a white car and a patch differ once you know what "normal" is for the cell's class) is not supported by this data.
- Nearly all of the signal sits in **layer 1** (the first stride-4 convolution, before any block that mixes in object-level features). The layer-2 and layer-3 signal is much weaker, except layer 3 on YOLO11.
- Using any layer is no better than layer 1 alone, and requiring all layers to agree is much worse.

## 3. Box level, with settings (held-out, 20 splits, 20% clean-FPR cap)

Settings column = most often chosen config (offset / join / min cells; splits out of 20). Offsets are on the "ratio" scale (score / threshold; the threshold is the 99.5th percentile of clean cells).
For E3 "today" means the fixed config offset 0 / join 0 / min cells 10, no tuning.

| model | scales | rule | settings (splits) | held-out F1 | precision | recall | mean IoU | clean FPR |
|---|---|---|---|---|---|---|---|---|
| YOLO11 | E3 | single | +0.2 / 0 / 15 (10) | **0.985** | 0.989 | 0.982 | 0.868 | 0.027 |
| YOLO11 | E3 | fixed, no tuning | 0 / 0 / 10 | 0.913 | 0.854 | 0.982 | 0.855 | 0.208 |
| YOLO11 | P3+P4+P5+E3 | `or` | +0.2 / 0 / 60 (19) | 0.969 | 0.962 | 0.976 | 0.885 | 0.043 |
| YOLO11 | P3+P4+P5+E3 | `mean` | -0.3 / 0 / 60 (13) | 0.947 | 0.961 | 0.933 | 0.805 | 0.057 |
| YOLO11 | P3+P4+P5+E3 | `and` | -0.5 / 0 / 60 (10) | 0.886 | 0.955 | 0.828 | 0.729 | 0.048 |
| YOLO11 | P3+P4+P5+E3 | `confirm` | -0.3 / 1 / 10 (10) | 0.801 | 0.898 | 0.727 | 0.642 | 0.138 |
| YOLO11 | P3+P4+E3 | `or` | +0.2 / 0 / 60 (11) | 0.980 | 0.984 | 0.976 | 0.874 | 0.002 |
| YOLO11 | P3+P4+E3 | `and` | -0.3 / 1 / 15 (18) | 0.828 | 0.937 | 0.743 | 0.637 | 0.076 |
| YOLO11 | *Mech-5 best (P3+P4+P5)* | `confirm` | -0.3 / 1 / 15 (16) | 0.770 | 0.875 | 0.690 | 0.655 | 0.187 |
| YOLOv8 | E3 | single | +0.3 / 1 / 4 (9) | **0.989** | 0.993 | 0.984 | 0.657 | 0.005 |
| YOLOv8 | E3 | fixed, no tuning | 0 / 0 / 10 | 0.922 | 0.963 | 0.841 | 0.108 | 0.108 |
| YOLOv8 | P3+E3 | `confirm` | +0.3 / 0 / 20 (19) | **0.997** | 0.996 | 0.998 | 0.623 | 0.000 |
| YOLOv8 | P3+E3 | `and` | +0.3 / 1 / 4 (13) | 0.993 | 0.996 | 0.990 | 0.585 | 0.000 |
| YOLOv8 | P3+P4+E3 | `confirm` | +0.3 / 0 / 20 (19) | 0.997 | 0.996 | 0.998 | 0.620 | 0.000 |
| YOLOv8 | P3+P4+E3 | `mean` | +0.5 / 0 / 10 (9) | 0.977 | 0.975 | 0.980 | 0.624 | 0.000 |
| YOLOv8 | *Mech-5 best (P3+P4)* | `confirm` | +0.6 / 0 / 15 (10) | 0.934 | 0.989 | 0.886 | 0.624 | 0.000 |

Best settings tuned and scored on all images (optimistic), precision / recall / F1 / IoU / clean FPR:
YOLO11 E3 +0.20 / 0 / 15: 1.000 / 0.988 / 0.994 / 0.892 / 0.00. YOLOv8 E3 +0.30 / 1 / 4: 0.988 / 1.000 / 0.994 / 0.443 / 0.00. YOLOv8 P3+E3 `confirm` +0.30 / 0 / 20: 0.994 / 1.000 / 0.997 / 0.618 / 0.00.
Full per-rule tables are in the logs that produced these numbers (rerun with the commands in `README.md`).

Reading it:
- **Energy alone reaches F1 0.985-0.989 and beats Mech-5 on both models.** On YOLO11 it also gives much better recall and box placement (IoU 0.87 vs 0.66) at a lower clean FPR.
- **Adding Mech-5 scales to energy does not improve on energy alone** (YOLO11: 0.98 at best, with `or`), and the stricter rules (`and`, `confirm`) are worse than energy alone because P3/P4/P5 miss patch cells that energy catches.
  On YOLOv8, `confirm` with P3 reaches 0.997 (F1) at 0% clean FPR; that is a small gain over energy alone, within what these sample sizes can resolve.
- Without tuning, the percentile threshold gives 11-21% clean FPR (recall 0.84-0.98): the offset and the minimum region size are what bring the false alarms down, as with Mech-5.
- YOLOv8 energy boxes are poorly placed on average (IoU 0.44-0.66, with a large spread across splits); the config with the smallest minimum region (4 cells) picks up stray cells.

## 4. Why not trust this as a general result

Energy is not looking at what makes a patch adversarial. It measures how strong the early-layer feature response is. The benchmark's patches are optimised textures with full colour saturation and strong edges, so they have very high
energy. Three consequences:
- A patch designed or constrained to look natural (smoothness or total-variation penalty, limited colour range, blended or low-contrast placement) would have much lower energy. Nothing in this experiment tests that.
- Real scenes contain high-energy content (high-contrast signs, text, foliage edges, a white car). Here clean FPR is low because layer 1 is compared against the global distribution of a clean set and regions must be large; on other data it can differ.
- An adaptive attacker who knows the detector can add an energy penalty to the patch objective; this is the standard weakness of a threshold on a simple statistic.

The classification-feature signal in Mech-5 is not tied to contrast in the same way, which is the reason to expect the two to be complementary. That is a hypothesis; the experiment below would test it.

## 5. Suggested next experiment: attenuated patches

Keep the same patched images but weaken the patch region (blur, reduce contrast toward the patch mean, partially blend with the underlying image, JPEG/downscale), at several strengths, and measure two things:
1. whether the attack still works (the target car stays hidden, i.e. the detector still misses it),
2. how recall of Mech-5 and of the energy detector fall as the patch gets smoother.

If energy recall collapses while the attack still works and Mech-5 holds up, then the combination is worth building with the offset chosen per detector from clean images only (see the clean-only calibration discussion). If both collapse, neither covers
low-contrast attacks. A second, independent check is a patch generated with a smoothness/total-variation penalty.

## 6. Caveats

- Same limits as the Mech-5 findings: one attack family, one dataset, small clean sets (100 images), settings tuned on the evaluation images (held-out splits reduce but do not remove this).
- The energy threshold uses a clean-percentile with global statistics; it is clean-only like Mech-5, but the offset, join and min-cell choices were again made with patched images. The clean-only calibration of the operating point still has to be done.
- Per-class profiling was judged at cell level (section 2) and not swept at box level, because it showed no cell-level gain over plain energy.
- Profiles were built from the first 2000 `train` images (the profile code takes `max_calibration_images`); the per-class numbers could change with more, but the layer-1 result does not depend on them (it equals plain energy).
